#!/usr/bin/env python3
# build_prompts_tsv.py
import argparse
import csv
import sys
from pathlib import Path
import re

AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$", re.IGNORECASE)

def read_fasta_or_txt(p: Path) -> str | None:
    if not p.exists():
        return None
    txt = p.read_text().strip()
    if txt.startswith(">"):
        # FASTA
        lines = [l.strip() for l in txt.splitlines() if l and not l.startswith(">")]
        seq = "".join(lines).strip()
    else:
        seq = txt.splitlines()[0].strip()
    return seq if seq else None

def find_peptide_seq(sample_dir: Path) -> str | None:
    for name in ("peptide.seq", "peptide.txt", "peptide.fa", "peptide.fasta"):
        seq = read_fasta_or_txt(sample_dir / name)
        if seq:
            return seq
    return None

def is_sequence(s: str) -> bool:
    return bool(s) and len(s) >= 3 and bool(AA_RE.match(s))

def gather_candidates(cands_dir: Path) -> list[Path]:
    return sorted([p for p in cands_dir.glob("*.pdb") if p.is_file()])

def mode_scan(dataset_root: Path, min_k: int, skip_if_too_few: bool) -> list[tuple[str, str, str]]:
    rows = []
    for sample_dir in sorted([d for d in dataset_root.iterdir() if d.is_dir()]):
        receptor_pdb = sample_dir / "receptor.pdb"
        cands_dir = sample_dir / "cands"
        if not receptor_pdb.exists():
            # 兼容其他命名
            pdbs = list(sample_dir.glob("*.pdb"))
            receptor_pdb = pdbs[0] if pdbs else receptor_pdb
        if not receptor_pdb.exists():
            print(f"[WARN] skip {sample_dir}: no receptor PDB found.", file=sys.stderr)
            continue

        # 候选目录名称兜底
        if not cands_dir.exists():
            for alt in ("candidates", "cand", "ligands", "peptides"):
                if (sample_dir / alt).exists():
                    cands_dir = sample_dir / alt
                    break
        if not cands_dir.exists() or not cands_dir.is_dir():
            print(f"[WARN] skip {sample_dir}: no candidates dir.", file=sys.stderr)
            continue

        cand_files = gather_candidates(cands_dir)
        if len(cand_files) < min_k and skip_if_too_few:
            print(f"[WARN] skip {sample_dir}: only {len(cand_files)} candidates (<{min_k}).", file=sys.stderr)
            continue

        pep_seq = find_peptide_seq(sample_dir)
        if not is_sequence(pep_seq or ""):
            print(f"[WARN] skip {sample_dir}: peptide sequence missing/invalid.", file=sys.stderr)
            continue

        rows.append((str(receptor_pdb.resolve()), pep_seq, str(cands_dir.resolve())))
    return rows

def mode_map(map_tsv: Path, receptors_dir: Path, cands_root: Path,
             min_k: int, skip_if_too_few: bool, receptor_suffix: str) -> list[tuple[str, str, str]]:
    rows = []
    with map_tsv.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"id", "peptide_seq"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{map_tsv} 必须包含列: {required}")
        for r in reader:
            _id = r["id"].strip()
            pep = r["peptide_seq"].strip()
            if not is_sequence(pep):
                print(f"[WARN] skip id={_id}: invalid peptide_seq.", file=sys.stderr)
                continue
            receptor_pdb = receptors_dir / f"{_id}{receptor_suffix}"
            if not receptor_pdb.exists():
                print(f"[WARN] skip id={_id}: receptor not found: {receptor_pdb}", file=sys.stderr)
                continue
            cands_dir = cands_root / _id
            if not cands_dir.exists():
                print(f"[WARN] skip id={_id}: candidates dir not found: {cands_dir}", file=sys.stderr)
                continue
            cand_files = gather_candidates(cands_dir)
            if len(cand_files) < min_k and skip_if_too_few:
                print(f"[WARN] skip id={_id}: only {len(cand_files)} candidates (<{min_k}).", file=sys.stderr)
                continue
            rows.append((str(receptor_pdb.resolve()), pep, str(cands_dir.resolve())))
    return rows

def write_tsv(rows: list[tuple[str, str, str]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["receptor_pdb", "peptide_seq", "candidates_dir"])
        for r in rows:
            w.writerow(r)
    print(f"[OK] wrote {out_path} with {len(rows)} rows.")

def main():
    ap = argparse.ArgumentParser(description="Build prompts.tsv for DPO preference mining.")
    sub = ap.add_subparsers(dest="mode", required=True)

    # 模式A：扫描逐样本子目录
    ap_scan = sub.add_parser("scan", help="scan dataset_root/<id>/{receptor.pdb, peptide.seq/fa, cands/*.pdb}")
    ap_scan.add_argument("--dataset-root", type=Path, default='/root/autodl-tmp/train_data')
    ap_scan.add_argument("--min-k", type=int, default=12)
    ap_scan.add_argument("--skip-if-too-few", action="store_true",
                         help="少于 min-k 的样本直接跳过（默认保留）。")
    ap_scan.add_argument("--out", type=Path, default='/root/autodl-tmp/Peptide_3D/utils/dpo/prompts.tsv')

    # 模式B：映射表 + 统一目录
    ap_map = sub.add_parser("map", help="use a map tsv: id,peptide_seq + receptors_dir + cands_root")
    ap_map.add_argument("--map-tsv", type=Path, required=True,
                        help="TSV with header: id, peptide_seq")
    ap_map.add_argument("--receptors-dir", type=Path, required=True)
    ap_map.add_argument("--cands-root", type=Path, required=True)
    ap_map.add_argument("--receptor-suffix", type=str, default=".pdb",
                        help="例如 .pdb 或 _receptor.pdb")
    ap_map.add_argument("--min-k", type=int, default=12)
    ap_map.add_argument("--skip-if-too-few", action="store_true")
    ap_map.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()

    if args.mode == "scan":
        rows = mode_scan(args.dataset_root, args.min_k, args.skip_if_too_few)
        write_tsv(rows, args.out)
    else:
        rows = mode_map(args.map_tsv, args.receptors_dir, args.cands_root,
                        args.min_k, args.skip_if_too_few, args.receptor_suffix)
        write_tsv(rows, args.out)

if __name__ == "__main__":
    main()

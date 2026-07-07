#!/usr/bin/env python3
"""
将 ``PPDbench/<pdb_id>/multi_cands`` 中 **HDOCK 亲和力最优** 的单个 ``pep_*.pdb`` 写入 ``Table_S1``。

对每个合法靶点目录：
- 读取 ``multi_cands/cands_hdock_scores.json``（路径 → HDOCK 打分）；
- **取数值最小** 的条目作为亲和力最优（HDOCK 能量越低通常表示结合越强）；
- 将 ``<target>/receptor.pdb`` 与该最优肽合并为双链复合物 PDB（写入本工程
  ``intermediate/ppdbench_merged_complexes/``），供 03 界面分析读取；
- ``source`` 设为 ``ours:PPDbench:<pep_pdb>``，``usable_for_interface_analysis=True``，
  ``complex_structure_path`` 指向合并文件。

追加后请在工程根目录依次运行：``02_analyze_free_peptides.py`` → ``03`` → ``04`` →
``05_build_summary_scores.py`` → ``06_make_figures.py``。
（若刚跑过 ``01_build_master_table.py`` 覆盖了 S1，需在本脚本之后重新 ingest。）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB import PPBuilder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _seq_from_peptide_pdb(pdb_path: Path) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    ppb = PPBuilder()
    peptides: list = []
    for model in structure:
        peptides.extend(ppb.build_peptides(model))
    if not peptides:
        return ""
    peptides.sort(key=lambda p: len(p))
    return str(peptides[0].get_sequence())


def _element_from_atom_line(line: str) -> str:
    if len(line) > 76:
        el = line[76:78].strip()
        if el:
            return el.upper()[:1]
    name = line[12:16].strip()
    if not name:
        return ""
    if len(name) >= 2 and name[0].isdigit():
        return name[1].upper()
    return name[0].upper()


def _heavy_atom_lines(pdb_path: Path) -> list[str]:
    """返回 ATOM 行列表（去掉氢），保留原始行内容（含换行）。"""
    out: list[str] = []
    for raw in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.startswith("ATOM"):
            continue
        el = _element_from_atom_line(raw)
        if el == "H":
            continue
        out.append(raw + "\n")
    return out


def _merge_receptor_peptide(receptor_path: Path, peptide_path: Path, out_path: Path) -> None:
    rec_lines = _heavy_atom_lines(receptor_path)
    pep_lines = _heavy_atom_lines(peptide_path)
    if not rec_lines or not pep_lines:
        raise ValueError(f"empty ATOM records: rec={len(rec_lines)} pep={len(pep_lines)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        w.writelines(rec_lines)
        w.write("TER\n")
        w.writelines(pep_lines)
        w.write("END\n")


def _load_hdock_scores(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[str, float] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _pick_best_peptide_path(multi_cands_dir: Path, scores: dict[str, float]) -> tuple[Path | None, float | None]:
    """
    返回 (最优 pep 路径, 对应 HDOCK 分数)。分数 **越小** 表示模型越优。
    若无有效 json，则在存在的 pep_*.pdb 中取字典序第一个。
    """
    if not scores:
        cand = sorted(multi_cands_dir.glob("pep_*.pdb"))
        return (cand[0].resolve() if cand else None, None)
    best_path: Path | None = None
    best_score: float | None = None
    for key, sc in scores.items():
        p = Path(key).expanduser()
        if not p.is_file():
            alt = multi_cands_dir / Path(key).name
            if alt.is_file():
                p = alt.resolve()
            else:
                continue
        p = p.resolve()
        if best_score is None or sc < best_score or (sc == best_score and str(p) < str(best_path or "")):
            best_score = sc
            best_path = p
    if best_path is None:
        cand = sorted(multi_cands_dir.glob("pep_*.pdb"))
        return (cand[0].resolve() if cand else None, None)
    return best_path, best_score


def _discover_targets(ppdbench_root: Path, *, only_targets: set[str] | None) -> list[Path]:
    out: list[Path] = []
    root = ppdbench_root.expanduser().resolve()
    if not root.is_dir():
        return out
    for target_dir in sorted(root.iterdir()):
        if not target_dir.is_dir():
            continue
        tid = target_dir.name.strip().lower()
        if not re.match(r"^[0-9][a-z0-9]{3}$", tid):
            continue
        if only_targets is not None and tid not in only_targets:
            continue
        mc = target_dir / "multi_cands"
        if not mc.is_dir():
            continue
        out.append(target_dir)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ppdbench-root",
        type=Path,
        default=Path("/root/autodl-tmp/PPDbench"),
        help="PPDbench 根目录（其下为 <pdb_id>/multi_cands/…）。",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=ROOT / "tables",
        help="含 Table_S1 的 tables 目录。",
    )
    p.add_argument(
        "--merge-out-dir",
        type=Path,
        default=ROOT / "intermediate" / "ppdbench_merged_complexes",
        help="receptor+最优肽 合并 PDB 输出目录。",
    )
    p.add_argument("--dry-run", action="store_true", help="只打印将写入行数，不写盘。")
    p.add_argument("--max-targets", type=int, default=0, help="最多处理靶点数，0 表示不限制。")
    p.add_argument(
        "--targets",
        type=str,
        default="",
        help="逗号分隔的 PDB id（如 `1cjr,4gyw`）；为空则导入全部合法靶点。",
    )
    p.add_argument(
        "--replace-ppdbench",
        action="store_true",
        help="写盘前删除本脚本曾写入的行（notes=`ingested_from_PPDbench_multi_cands`），便于重导。",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    s1_path = args.tables_dir / "Table_S1_master_sequence_table.csv"
    if not s1_path.exists():
        print(f"Missing {s1_path}", file=sys.stderr)
        return 2

    existing = pd.read_csv(s1_path)

    only: set[str] | None = None
    if str(args.targets).strip():
        only = {x.strip().lower() for x in str(args.targets).split(",") if x.strip()}

    if args.replace_ppdbench and "notes" in existing.columns:
        n0 = len(existing)
        mask_ppdbench = existing["notes"].astype(str).str.startswith("ingested_from_PPDbench_multi_cands")
        existing = existing.loc[~mask_ppdbench].copy()
        print(f"--replace-ppdbench: dropped {n0 - len(existing)} prior PPDbench-ingested rows")

    merge_root = args.merge_out_dir.expanduser().resolve()
    target_dirs = _discover_targets(args.ppdbench_root, only_targets=only)
    if args.max_targets and args.max_targets > 0:
        target_dirs = target_dirs[: int(args.max_targets)]

    rows: list[dict] = []
    for target_dir in target_dirs:
        tid = target_dir.name.strip().lower()
        mc = target_dir / "multi_cands"
        rec = target_dir / "receptor.pdb"
        score_path = mc / "cands_hdock_scores.json"
        scores = _load_hdock_scores(score_path)
        pep_path, hdock = _pick_best_peptide_path(mc, scores)
        if pep_path is None:
            print(f"skip (no pep): {tid}", file=sys.stderr)
            continue
        if not rec.is_file():
            print(f"skip (no receptor.pdb): {tid}", file=sys.stderr)
            continue

        stem = pep_path.stem
        peptide_id = f"{tid}_ppdbench_{stem}"
        src = f"ours:PPDbench:{pep_path}"
        seq = _seq_from_peptide_pdb(pep_path)
        if not seq:
            print(f"skip (no sequence): {pep_path}", file=sys.stderr)
            continue

        merged = merge_root / f"{tid}_receptor_best_pep.pdb"
        if not args.dry_run:
            try:
                _merge_receptor_peptide(rec, pep_path, merged)
            except Exception as e:
                print(f"skip (merge failed {tid}): {e}", file=sys.stderr)
                continue

        score_note = f"hdock_score={hdock}" if hdock is not None else "hdock_score=na"
        rows.append(
            {
                "target_id": tid,
                "peptide_id": peptide_id,
                "group": "generated",
                "sequence": seq,
                "length": len(seq),
                "sequence_source_path": str(pep_path),
                "free_structure_path": str(pep_path),
                "complex_structure_path": str(merged.resolve()) if not args.dry_run else str(merged),
                "rank": 1,
                "source": src,
                "usable_for_free_structure_analysis": True,
                "usable_for_interface_analysis": True,
                "usable_for_solubility_analysis": True,
                "notes": f"ingested_from_PPDbench_multi_cands|{score_note}|complex={merged.name}",
            }
        )

    if not rows:
        print("No PPDbench best-peptide rows produced.", file=sys.stderr)
        return 1

    new_df = pd.DataFrame(rows)
    ex = existing.copy()
    drop_mask = pd.Series(False, index=ex.index)
    for _, r in new_df.iterrows():
        m = (ex["target_id"].astype(str).str.lower() == r["target_id"]) & (ex["peptide_id"] == r["peptide_id"])
        drop_mask |= m
    ex = ex.loc[~drop_mask]
    merged_tbl = pd.concat([ex, new_df], ignore_index=True)

    print(
        f"PPDbench targets processed: {len(rows)}; "
        f"merged complexes under {merge_root}; table rows {len(existing)} -> {len(merged_tbl)}"
    )
    if args.dry_run:
        return 0

    merged_tbl.to_csv(s1_path, index=False)
    print(f"Wrote {s1_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

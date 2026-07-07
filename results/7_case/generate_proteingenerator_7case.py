#!/usr/bin/env python3
"""
使用 RosettaCommons protein_generator（序列-结构扩散）对 7_case 三个靶点各生成若干条设计。

优先使用项目内副本：
  <Peptide_3D>/results/5_robustness/baseline/repos/protein_generator
若不存在则克隆到：
  <本目录>/external/protein_generator

对 each 靶点：将靶链提取为链 A、残基 1..N，contig 采用「固定靶标 + 生成段」形式：
  A1-N,<pep_min>-<pep_max>
与 examples/motif_scaffolding.sh 中模板+生成段写法一致。

依赖：按仓库 environment.yml 配置环境，并下载 README 中列出的 checkpoint（--checkpoint）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CASE_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("3V2A-vegf", "3V2A-vegf.pdb", "R"),
    ("6LML-GPCR", "6LML-GPCR.pdb", "R"),
    ("7OUN-PD-L1", "7OUN-PD-L1.pdb", "A"),
)

PG_GIT = "https://github.com/RosettaCommons/protein_generator.git"


def peptide_3d_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_repo() -> Path:
    bundled = peptide_3d_root() / "results" / "5_robustness" / "baseline" / "repos" / "protein_generator"
    if bundled.is_dir() and (bundled / "inference.py").is_file():
        return bundled
    return Path(__file__).resolve().parent / "external" / "protein_generator"


def ensure_repo(repo: Path) -> Path:
    if (repo / "inference.py").is_file():
        return repo
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", PG_GIT, str(repo)],
        check=True,
    )
    return repo


def write_chain_only_renumbered(pdb_path: Path, chain_id: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping: dict[tuple[str, str], int] = {}
    n = 0
    lines_out: list[str] = []
    serial = 0
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 27:
                continue
            ch = line[21]
            if ch.strip() != chain_id.strip():
                continue
            resseq = line[22:26]
            icode = line[26] if len(line) > 26 else " "
            key = (resseq, icode)
            if key not in mapping:
                n += 1
                mapping[key] = n
            new_res = mapping[key]
            serial += 1
            new_line = f"{line[:6]}{serial:5d}{line[11:21]}A{new_res:4d}{line[26:].rstrip()}\n"
            lines_out.append(new_line)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines_out)
        handle.write("END\n")
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="7_case 三靶点 Protein Generator 批跑。")
    p.add_argument("--repo-root", type=str, default="", help="protein_generator 仓库根目录。")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline/repos/protein_generator/checkpoints/SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt",
        help="SEQDIFF 权重路径；为空则尝试 <repo>/checkpoints/SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt",
    )
    p.add_argument("--pep-len-min", type=int, default=15)
    p.add_argument("--pep-len-max", type=int, default=30)
    p.add_argument("--num-designs", type=int, default=3)
    p.add_argument("--T", type=int, default=25, help="扩散步数（与官方示例一致可设为 25）。")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = Path(__file__).resolve().parent
    repo = Path(args.repo_root).resolve() if args.repo_root else default_repo()
    if not (repo / "inference.py").is_file():
        print(f"未找到 protein_generator，正在克隆到: {repo}")
        if args.dry_run:
            print(f"[dry-run] git clone {PG_GIT} {repo}")
        else:
            ensure_repo(repo)

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    if ckpt is None or not ckpt.is_file():
        cand = repo / "checkpoints" / "SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt"
        ckpt = cand if cand.is_file() else Path("")
    if not ckpt.is_file() and not args.dry_run:
        print(
            "警告: 未找到 checkpoint。请从 README 链接下载 SEQDIFF_*.pt 并传入 --checkpoint。\n"
            f"  期望路径示例: {repo / 'checkpoints' / 'SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt'}",
            file=sys.stderr,
        )

    prep_dir = case_dir / "proteingenerator_inputs"
    out_root = case_dir / "proteingenerator_runs"

    for stem, pdb_name, chain in CASE_TARGETS:
        raw_pdb = case_dir / pdb_name
        if not raw_pdb.is_file():
            raise FileNotFoundError(raw_pdb)
        clean_pdb = prep_dir / f"{stem}_target_A.pdb"
        n_res = write_chain_only_renumbered(raw_pdb, chain, clean_pdb)
        contig = f"A1-{n_res},{args.pep_len_min}-{args.pep_len_max}"
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(repo / "inference.py"),
            "--pdb",
            str(clean_pdb.resolve()),
            "--contigs",
            contig,
            "--num_designs",
            str(args.num_designs),
            "--out",
            str(out_dir),
            "--T",
            str(args.T),
            "--save_best_plddt",
        ]
        if ckpt.is_file():
            cmd.extend(["--checkpoint", str(ckpt.resolve())])

        print("\n>>>", stem, "| contig =", contig)
        print(" ", " \\\n  ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(repo), check=False)


if __name__ == "__main__":
    main()

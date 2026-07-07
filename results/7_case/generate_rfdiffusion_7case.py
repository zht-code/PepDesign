#!/usr/bin/env python3
"""
使用 RFdiffusion 对 7_case 三个靶点各跑一批 binder/多肽扩散设计。

优先使用项目内副本：
  <Peptide_3D>/results/5_robustness/baseline/repos/RFdiffusion
若不存在则克隆到：
  <本目录>/external/RFdiffusion

依赖：需在 RFdiffusion 目录下按官方说明安装环境，并下载权重（如 models/Complex_base_ckpt.pt）。
参见仓库内 scripts/download_models.sh。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# (输出名前缀, 原始 PDB 文件名, 作为靶标的链 ID —— 与结构 HEADER 一致)
CASE_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("3V2A-vegf", "3V2A-vegf.pdb", "R"),
    ("6LML-GPCR", "6LML-GPCR.pdb", "R"),
    ("7OUN-PD-L1", "7OUN-PD-L1.pdb", "A"),
)

RFDIFFUSION_GIT = "https://github.com/RosettaCommons/RFdiffusion.git"


def peptide_3d_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_repo() -> Path:
    bundled = peptide_3d_root() / "results" / "5_robustness" / "baseline" / "repos" / "RFdiffusion"
    if bundled.is_dir() and (bundled / "scripts" / "run_inference.py").is_file():
        return bundled
    return Path(__file__).resolve().parent / "external" / "RFdiffusion"


def ensure_repo(repo: Path) -> Path:
    if (repo / "scripts" / "run_inference.py").is_file():
        return repo
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", RFDIFFUSION_GIT, str(repo)],
        check=True,
    )
    return repo


def write_chain_only_renumbered(pdb_path: Path, chain_id: str, out_path: Path) -> int:
    """仅保留指定链，残基从 1 连续编号，链 ID 改为 A。返回残基数 N。"""
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


def pick_hotspot_pdb_nums(n_res: int, k: int = 3) -> list[int]:
    if n_res <= 0:
        return [1]
    idxs = sorted({max(1, min(n_res, int(round(x)))) for x in [n_res * 0.25, n_res * 0.5, n_res * 0.75]})
    while len(idxs) < k and len(idxs) < n_res:
        for j in range(1, n_res + 1):
            if j not in idxs:
                idxs.append(j)
                break
    return idxs[:k]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="7_case 三靶点 RFdiffusion 批跑（每靶点独立输出目录）。")
    p.add_argument("--repo-root", type=str, default="/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline/repos/RFdiffusion", help="RFdiffusion 仓库根目录（默认自动探测或克隆到 external）。")
    p.add_argument("--ckpt", type=str, default="/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline/repos/RFdiffusion/models/Complex_base_ckpt.pt", help="Complex 权重路径，默认 <repo>/models/Complex_base_ckpt.pt")
    p.add_argument("--binder-len", type=str, default="15-30", help="binder 长度范围，如 15-30（与官方 contig 一致）。")
    p.add_argument("--num-designs", type=int, default=3, help="每个靶点的扩散样本数。")
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不执行。")
    p.add_argument("--skip-prepare", action="store_true", help="跳过链提取（假定已自行准备好 input pdb）。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = Path(__file__).resolve().parent
    repo = Path(args.repo_root).resolve() if args.repo_root else default_repo()
    if not (repo / "scripts" / "run_inference.py").is_file():
        print(f"未找到 RFdiffusion，正在克隆到: {repo}")
        if args.dry_run:
            print(f"[dry-run] git clone {RFDIFFUSION_GIT} {repo}")
        else:
            ensure_repo(repo)

    ckpt = Path(args.ckpt).resolve() if args.ckpt else repo / "models" / "Complex_base_ckpt.pt"
    if not ckpt.is_file() and not args.dry_run:
        print(
            f"警告: 未找到权重文件 {ckpt}。请先运行仓库内 scripts/download_models.sh 或手动下载 Complex_base_ckpt.pt。\n"
            "仍将继续尝试运行（若你已指定其他 --ckpt 可忽略）。",
            file=sys.stderr,
        )

    prep_dir = case_dir / "rfdiffusion_inputs"
    out_root = case_dir / "rfdiffusion_runs"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}{os.pathsep}{env.get('PYTHONPATH', '')}"

    for stem, pdb_name, chain in CASE_TARGETS:
        raw_pdb = case_dir / pdb_name
        if not raw_pdb.is_file():
            raise FileNotFoundError(raw_pdb)
        if args.skip_prepare:
            clean_pdb = raw_pdb.resolve()
            n_res = 0
            for line in open(clean_pdb, encoding="utf-8", errors="ignore"):
                if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21].strip() == "A":
                    n_res += 1
        else:
            clean_pdb = prep_dir / f"{stem}_target_A.pdb"
            n_res = write_chain_only_renumbered(raw_pdb, chain, clean_pdb)

        hs = pick_hotspot_pdb_nums(n_res)
        hs_str = ",".join(f"A{r}" for r in hs)
        contig = f"A1-{n_res}/0 {args.binder_len}"
        prefix = (out_root / stem / "run").as_posix()

        cmd = [
            sys.executable,
            str(repo / "scripts" / "run_inference.py"),
            f"inference.output_prefix={prefix}",
            f"inference.input_pdb={clean_pdb.as_posix()}",
            f"contigmap.contigs=[{contig}]",
            f"ppi.hotspot_res=[{hs_str}]",
            f"inference.num_designs={args.num_designs}",
            f"inference.ckpt_override_path={ckpt.as_posix()}",
            "denoiser.noise_scale_ca=0",
            "denoiser.noise_scale_frame=0",
        ]
        print("\n>>>", stem)
        print(" ", " \\\n  ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(repo), env=env, check=False)


if __name__ == "__main__":
    main()

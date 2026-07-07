#!/usr/bin/env python3
"""
对 `generate_rfdiffusion_7case.py` 生成的复合物 PDB 运行 ProteinMPNN，为 binder 链恢复序列。

RFdiffusion 输出一般为：链 A = 靶标（固定序列），链 B = binder（多为 GLY 占位），
故默认仅设计链 B（`--pdb-path-chains B`）。

依赖：
  - 项目内副本：<Peptide_3D>/results/5_robustness/baseline/repos/ProteinMPNN
  - 需在上述目录下按官方说明安装 PyTorch 等，并下载权重到
    `vanilla_model_weights/`（如 v_48_020.pt），参见该仓库 README。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 与 generate_rfdiffusion_7case.py 中 rfdiffusion_runs/<stem>/ 目录名一致
CASE_STEMS: tuple[str, ...] = ("3V2A-vegf", "6LML-GPCR", "7OUN-PD-L1")


def peptide_3d_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_mpnn_repo() -> Path:
    return (
        peptide_3d_root()
        / "results"
        / "5_robustness"
        / "baseline"
        / "repos"
        / "ProteinMPNN"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="7_case：对 RFdiffusion 输出批量跑 ProteinMPNN 序列设计。")
    p.add_argument(
        "--mpnn-repo",
        type=str,
        default=str(default_mpnn_repo()),
        help="ProteinMPNN 仓库根目录（含 protein_mpnn_run.py）",
    )
    p.add_argument(
        "--rfdiffusion-runs",
        type=str,
        default="",
        help="RFdiffusion 输出根目录，默认与本脚本同级的 rfdiffusion_runs",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="",
        help="MPNN 输出根目录，默认 <本目录>/proteinmpnn_runs",
    )
    p.add_argument(
        "--pdb-glob",
        type=str,
        default="run_*.pdb",
        help="每个靶点目录下要处理的 PDB glob（相对该 stem 子目录）",
    )
    p.add_argument(
        "--pdb-path-chains",
        type=str,
        default="B",
        help="要重新设计序列的链 ID，空格分隔多链；默认 B（binder）",
    )
    p.add_argument(
        "--path-to-model-weights",
        type=str,
        default="",
        help="含 .pt 权重的目录，默认 <mpnn-repo>/vanilla_model_weights",
    )
    p.add_argument("--model-name", type=str, default="v_48_020", help="权重文件名（不含 .pt）")
    p.add_argument("--num-seq-per-target", type=int, default=2, help="每个结构生成的序列条数")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sampling-temp", type=str, default="0.1")
    p.add_argument("--backbone-noise", type=float, default=0.0)
    p.add_argument("--ca-only", action="store_true", help="使用 CA-only 模型（需 ca_model_weights）")
    p.add_argument("--use-soluble-model", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="0 表示每次随机种子")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--stems",
        type=str,
        default="",
        help="只处理指定 stem，逗号分隔；默认处理全部 CASE_STEMS",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = Path(__file__).resolve().parent
    mpnn_repo = Path(args.mpnn_repo).resolve()
    runner = mpnn_repo / "protein_mpnn_run.py"
    if not runner.is_file():
        raise FileNotFoundError(f"未找到 ProteinMPNN 入口: {runner}")

    run_root = Path(args.rfdiffusion_runs).resolve() if args.rfdiffusion_runs else case_dir / "rfdiffusion_runs"
    out_root = Path(args.out_root).resolve() if args.out_root else case_dir / "proteinmpnn_runs"

    if args.path_to_model_weights:
        weights_dir = Path(args.path_to_model_weights).resolve()
    else:
        if args.ca_only:
            sub = "ca_model_weights"
        elif args.use_soluble_model:
            sub = "soluble_model_weights"
        else:
            sub = "vanilla_model_weights"
        weights_dir = mpnn_repo / sub

    ckpt = weights_dir / f"{args.model_name}.pt"
    if not ckpt.is_file() and not args.dry_run:
        print(
            f"警告: 未找到权重 {ckpt}。请从 ProteinMPNN 官方说明下载模型到 {weights_dir}。\n"
            "仍打印/尝试命令（若你已指定其他路径可忽略）。",
            file=sys.stderr,
        )

    stems = tuple(s.strip() for s in args.stems.split(",")) if args.stems.strip() else CASE_STEMS

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{mpnn_repo}{os.pathsep}{env.get('PYTHONPATH', '')}"

    for stem in stems:
        subdir = run_root / stem
        if not subdir.is_dir():
            print(f"跳过（目录不存在）: {subdir}", file=sys.stderr)
            continue
        pdbs = sorted(subdir.glob(args.pdb_glob))
        if not pdbs:
            print(f"跳过（无匹配 {args.pdb_glob}）: {subdir}", file=sys.stderr)
            continue
        for pdb_path in pdbs:
            out_folder = (out_root / stem / pdb_path.stem).resolve()
            out_folder.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(runner),
                f"--pdb_path={pdb_path.as_posix()}",
                f"--pdb_path_chains={args.pdb_path_chains}",
                f"--out_folder={out_folder.as_posix()}",
                f"--num_seq_per_target={args.num_seq_per_target}",
                f"--batch_size={args.batch_size}",
                f"--sampling_temp={args.sampling_temp}",
                f"--backbone_noise={args.backbone_noise}",
                f"--model_name={args.model_name}",
                f"--seed={args.seed}",
                f"--path_to_model_weights={weights_dir.as_posix()}",
            ]
            if args.ca_only:
                cmd.append("--ca_only")
            if args.use_soluble_model:
                cmd.append("--use_soluble_model")

            print("\n>>>", stem, pdb_path.name)
            print(" ", " \\\n  ".join(cmd))
            if args.dry_run:
                continue
            subprocess.run(cmd, cwd=str(mpnn_repo), env=env, check=False)


if __name__ == "__main__":
    main()

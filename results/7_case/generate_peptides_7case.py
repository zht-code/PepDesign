#!/usr/bin/env python3
"""
针对 results/7_case 目录下三个复合物 PDB 靶点，生成模型采样的多肽全原子 PDB。

流程与 utils/reference/train_data_generate_top10.py 的 worker 一致：序列生成 → interface 重排 →
α-螺旋初始全原子构象 → 口袋附近刚体摆放 → OpenMM 约束最小化。

每个靶点的输出目录为：<本脚本目录>/<PDB 主文件名>/cands/pep_01.pdb, pep_02.pdb, ...
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

CASE_DIR = Path(__file__).resolve().parent
PEPTIDE_3D_ROOT = CASE_DIR.parent.parent
REF_DIR = PEPTIDE_3D_ROOT / "utils" / "reference"

for p in (str(PEPTIDE_3D_ROOT), str(REF_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_data_generate_top10 as gen_ref

# 与目录中三个案例 PDB 一一对应（主文件名不含路径）
CASE_RECEPTOR_PDBS = (
    "3V2A-vegf.pdb",
    "6LML-GPCR.pdb",
    "7OUN-PD-L1.pdb",
)


def build_prot_list(case_dir: Path) -> list[tuple[str, str]]:
    """返回 (sample_dir, receptor_pdb) 列表；sample_dir 用于存放该靶点的 cands/ 输出。"""
    pairs: list[tuple[str, str]] = []
    for fname in CASE_RECEPTOR_PDBS:
        pdb_path = case_dir / fname
        if not pdb_path.is_file():
            raise FileNotFoundError(f"未找到受体 PDB：{pdb_path}")
        stem = pdb_path.stem
        sample_dir = case_dir / stem
        pairs.append((str(sample_dir), str(pdb_path)))
    return pairs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="为 7_case 三个靶点生成多肽 PDB（复用 train_data_generate_top10.worker）。")
    ap.add_argument(
        "--ckpt-path",
        default="/root/autodl-tmp/Peptide_3D/logs_Ranger_no_DPO/best_model_epoch_72_loss_2.0048.pth",
        help="ProteinPeptideModel 权重路径。",
    )
    ap.add_argument("--num-per-protein", type=int, default=5, help="每个靶点保留的多肽条数。")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--oversample-factor", type=int, default=3)
    ap.add_argument("--num-gpus", type=int, default=1, help="使用的 GPU 数量（多进程）。")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    prot_list = build_prot_list(CASE_DIR)
    print(f"将处理 {len(prot_list)} 个靶点，输出各自子目录下的 cands/：")
    for sample_dir, pdb_path in prot_list:
        print(f"  {pdb_path} -> {sample_dir}/cands/")

    avail = torch.cuda.device_count()
    if avail == 0:
        print("未检测到 CUDA，使用 CPU 单进程。")
        world_size = 1
        shards = [prot_list]
    else:
        world_size = min(args.num_gpus, avail)
        indices = np.array_split(np.arange(len(prot_list)), world_size)
        shards = [[prot_list[i] for i in idx.tolist()] for idx in indices]

    cfg = dict(
        ckpt_path=args.ckpt_path,
        num_per_protein=args.num_per_protein,
        top_k=args.top_k,
        max_len=args.max_len,
        temperature=args.temperature,
        num_gpus=world_size,
        oversample_factor=args.oversample_factor,
    )

    if world_size == 1:
        gen_ref.worker(0, shards[0], cfg)
        return

    ctx = get_context("spawn")
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=gen_ref.worker, args=(rank, shards[rank], cfg), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()


'''

python /root/autodl-tmp/Peptide_3D/results/7_case/generate_peptides_7case.py --num-per-protein 5 --num-gpus 1

'''
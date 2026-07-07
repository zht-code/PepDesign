#!/usr/bin/env python3
"""
Generate peptides for both 2_SOTA test sets with multi-GPU workers.

This script reuses the worker implementation in test_data_generate_top10.py.
"""

from __future__ import annotations

import argparse
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

import test_data_generate_top10 as gen_ref


def find_receptor_pdbs(root: Path) -> list[tuple[str, str]]:
    """Return (sample_dir, receptor.pdb) pairs under root/*."""
    pairs: list[tuple[str, str]] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        receptor = d / "receptor.pdb"
        if receptor.is_file():
            pairs.append((str(d), str(receptor)))
    return pairs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--protein-root",
        default="/root/autodl-tmp/Peptide_3D/results/2_SOTA/protein_level_test",
        help="Directory containing protein_level_test sample folders.",
    )
    ap.add_argument(
        "--family-root",
        default="/root/autodl-tmp/Peptide_3D/results/2_SOTA/family_level_test",
        help="Directory containing family_level_test sample folders.",
    )
    ap.add_argument(
        "--ckpt-path",
        default="/root/autodl-tmp/Peptide_3D/logs_Ranger_dpo_multi/policy_dpo_multi_epoch5_loss_0.6073.pth",
        help="Model checkpoint path.",
    )
    ap.add_argument("--num-per-protein", type=int, default=5, help="Peptides per protein.")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--oversample-factor", type=int, default=3)
    ap.add_argument("--num-gpus", type=int, default=2, help="How many GPUs to use.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    protein_root = Path(args.protein_root)
    family_root = Path(args.family_root)
    if not protein_root.is_dir():
        raise FileNotFoundError(f"Missing directory: {protein_root}")
    if not family_root.is_dir():
        raise FileNotFoundError(f"Missing directory: {family_root}")

    prot_list = find_receptor_pdbs(protein_root) + find_receptor_pdbs(family_root)
    if not prot_list:
        raise RuntimeError("No receptor.pdb files found in both test roots.")

    print(
        f"Found {len(prot_list)} proteins total "
        f"(protein_level_test={len(find_receptor_pdbs(protein_root))}, "
        f"family_level_test={len(find_receptor_pdbs(family_root))})."
    )

    avail = torch.cuda.device_count()
    if avail == 0:
        print("No CUDA device found; running on CPU with a single process.")
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
python /root/autodl-tmp/Peptide_3D/utils/reference/generate_peptides_for_2sota_tests.py \
  --num-per-protein 5 \
  --num-gpus 2 \
  --ckpt-path /root/autodl-tmp/Peptide_3D/logs_Ranger_dpo_multi/policy_dpo_multi_epoch5_loss_0.6073.pth


'''
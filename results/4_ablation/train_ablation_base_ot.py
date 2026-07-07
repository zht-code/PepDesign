#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融：Base + OT（多任务 SFT，无 DPO）
- 默认从 Base 的权重继续训练：--init_ckpt logs_base/sft_best.pth
- 数据与超参与 train_DPO_multi_objective / train_ablation_base 默认一致（三份 jsonl + λ + normalize）。
  若 OT 体现在「增强集 PDB」，请先用增强数据重建 jsonl，再通过 --aff-jsonl / --stab-jsonl / --sol-jsonl 传入。
- 默认绑定物理 GPU 1。

输出：{PROJECT}/logs_base_ot/sft_best.pth
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[2]
_TRAIN_SFT = _PROJECT / "results" / "3_Pareto_improved" / "train_SFT_multi_objective.py"
_SAVE_DIR = _PROJECT / "logs_base_ot"
_DEFAULT_INIT = _PROJECT / "logs_base" / "sft_best.pth"


def main():
    ap = argparse.ArgumentParser(description="Ablation: Base+OT (SFT from base checkpoint; override jsonl for OT data).")
    ap.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES for this run.")
    ap.add_argument("--init-ckpt", type=str, default=str(_DEFAULT_INIT), help="Usually logs_base/sft_best.pth after train_ablation_base.")
    ap.add_argument("--aff-jsonl", type=str, default=str(_PROJECT / "utils/dpo/affinity_pairs_cleaned.jsonl"))
    ap.add_argument("--stab-jsonl", type=str, default=str(_PROJECT / "utils/dpo/stability_pairs.jsonl"))
    ap.add_argument("--sol-jsonl", type=str, default=str(_PROJECT / "utils/dpo/solubility_pairs.jsonl"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    args, rest = ap.parse_known_args()

    if not _TRAIN_SFT.is_file():
        raise FileNotFoundError(f"Missing SFT script: {_TRAIN_SFT}")
    if not os.path.isfile(args.init_ckpt):
        raise FileNotFoundError(
            f"init_ckpt not found: {args.init_ckpt}\nRun train_ablation_base.py first (produces logs_base/sft_best.pth)."
        )

    cmd = [
        sys.executable,
        "-u",
        str(_TRAIN_SFT),
        "--aff_jsonl",
        args.aff_jsonl,
        "--stab_jsonl",
        args.stab_jsonl,
        "--sol_jsonl",
        args.sol_jsonl,
        "--lambda_aff",
        "1.0",
        "--lambda_stab",
        "0.35",
        "--lambda_sol",
        "0.35",
        "--normalize_lambda",
        "--init_ckpt",
        args.init_ckpt,
        "--save_dir",
        str(_SAVE_DIR),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--grad_accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--optimizer",
        "ranger",
        "--use_amp",
        "--save_every_epoch",
    ]
    cmd.extend(rest)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs(_SAVE_DIR, exist_ok=True)
    print("[ablation base+OT] CUDA_VISIBLE_DEVICES=", args.gpu)
    print("[ablation base+OT] ", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=str(_PROJECT), env=env)
    raise SystemExit(ret.returncode)


if __name__ == "__main__":
    main()


'''

# 2) Base+OT — GPU 1（需先有 logs_base/sft_best.pth）
nohup python -u /root/autodl-tmp/Peptide_3D/results/4_ablation/train_ablation_base_ot.py \
  > /root/autodl-tmp/Peptide_3D/logs_base_ot/nohup_train_base_ot.log 2>&1 &

'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融：Base + DPO（无 OT 阶段：从 Base 的 SFT 权重做多目标 DPO）
- 直接调用 utils/dpo/train_DPO_multi_objective.py，超参与 utils/dpo/train_DPO _multi.sh 对齐。
- 默认 --init_ckpt 为 logs_base/sft_best.pth（需先跑完 train_ablation_base.py）。
- 默认绑定物理 GPU 2。

输出：{PROJECT}/logs_base_dpo/policy_dpo_multi_best.pth（及各 epoch 可选）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[2]
_TRAIN_DPO = _PROJECT / "utils" / "dpo" / "train_DPO_multi_objective.py"
_SAVE_DIR = _PROJECT / "logs_base_dpo"
_DEFAULT_INIT = _PROJECT / "logs_base" / "sft_best.pth"


def main():
    ap = argparse.ArgumentParser(description="Ablation: Base+DPO (multi-objective DPO from base SFT ckpt).")
    ap.add_argument("--gpu", type=str, default="1", help="CUDA_VISIBLE_DEVICES for this run.")
    ap.add_argument("--init-ckpt", type=str, default=str(_DEFAULT_INIT))
    ap.add_argument("--aff-jsonl", type=str, default=str(_PROJECT / "utils/dpo/affinity_pairs_cleaned.jsonl"))
    ap.add_argument("--stab-jsonl", type=str, default=str(_PROJECT / "utils/dpo/stability_pairs.jsonl"))
    ap.add_argument("--sol-jsonl", type=str, default=str(_PROJECT / "utils/dpo/solubility_pairs.jsonl"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    args, rest = ap.parse_known_args()

    if not _TRAIN_DPO.is_file():
        raise FileNotFoundError(f"Missing DPO script: {_TRAIN_DPO}")
    if not os.path.isfile(args.init_ckpt):
        raise FileNotFoundError(
            f"init_ckpt not found: {args.init_ckpt}\nRun train_ablation_base.py first (produces logs_base/sft_best.pth)."
        )

    cmd = [
        sys.executable,
        "-u",
        str(_TRAIN_DPO),
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
        "--dpo_mode",
        "soft",
        "--kl_coef",
        "0.01",
        "--kl_pairs",
        "2048",
        "--use_amp",
        "--save_every_epoch",
    ]
    cmd.extend(rest)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs(_SAVE_DIR, exist_ok=True)
    print("[ablation base+DPO] CUDA_VISIBLE_DEVICES=", args.gpu)
    print("[ablation base+DPO] ", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=str(_PROJECT), env=env)
    raise SystemExit(ret.returncode)


if __name__ == "__main__":
    main()


'''

# 3) Base+DPO — GPU 2（需先有 logs_base/sft_best.pth）
nohup python -u /root/autodl-tmp/Peptide_3D/results/4_ablation/train_ablation_base_dpo.py \
  > /root/autodl-tmp/Peptide_3D/logs_base_dpo/nohup_train_base_dpo.log 2>&1 &

'''
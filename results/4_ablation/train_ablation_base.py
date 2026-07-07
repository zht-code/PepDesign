#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融：Base（多任务 SFT，无 OT、无 DPO）
- 训练方式对齐 utils/dpo/train_DPO_multi_objective.py：同一套 aff/stab/sol jsonl 与 λ、normalize_lambda、
  epochs / batch / grad_accum / use_amp / save_every_epoch。
- 使用 results/3_Pareto_improved/train_SFT_multi_objective.py（结构 CE + 多任务加权）。
- 不加载 init_ckpt（避免默认落到 logs_data_augmentation），从零开始仅依赖 ESM 预训练权重。
- 默认绑定物理 GPU 0（通过子进程环境变量 CUDA_VISIBLE_DEVICES=0）。

输出：{PROJECT}/logs_base/sft_best.pth
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
_SAVE_DIR = _PROJECT / "logs_base"
# 故意使用不存在路径，使 train_SFT 跳过 load_state_dict（其逻辑为 init_ckpt 存在才加载）
_SKIP_INIT = _PROJECT / "results" / "4_ablation" / ".no_init_checkpoint"


def main():
    ap = argparse.ArgumentParser(description="Ablation: Base (multi-task SFT only).")
    ap.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES for this run.")
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
        str(_SKIP_INIT),
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
    print("[ablation base] CUDA_VISIBLE_DEVICES=", args.gpu)
    print("[ablation base] ", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=str(_PROJECT), env=env)
    raise SystemExit(ret.returncode)


if __name__ == "__main__":
    main()


'''
mkdir -p /root/autodl-tmp/Peptide_3D/logs_base /root/autodl-tmp/Peptide_3D/logs_base_ot /root/autodl-tmp/Peptide_3D/logs_base_dpo

# 1) Base — GPU 0（脚本内已设子进程 CUDA_VISIBLE_DEVICES=0，此处可省略外层）
nohup python -u /root/autodl-tmp/Peptide_3D/results/4_ablation/train_ablation_base.py \
  > /root/autodl-tmp/Peptide_3D/logs_base/nohup_train_base.log 2>&1 &


'''
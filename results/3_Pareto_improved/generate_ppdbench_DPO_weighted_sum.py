#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 PPDbench 各靶点生成多肽 PDB；加载 train_DPO_weighted_sum（1:1:1 加权和 DPO）权重。
输出子目录：generated_dpo_weighted_sum/
指定 GPU：--gpu N；或 CUDA_VISIBLE_DEVICES=N python ... --gpu 0
"""
import sys
from pathlib import Path

_PEPTIDE_ROOT = Path("/root/autodl-tmp/Peptide_3D")
_HERE = Path(__file__).resolve().parent
for _p in (_PEPTIDE_ROOT, _HERE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import ppdbench_generate_core as _core

DEFAULT_CKPT = str(_PEPTIDE_ROOT / "logs_weighted_sum/policy_dpo_weighted_sum_best.pth")
OUT_SUBDIR = "generated_dpo_weighted_sum"

if __name__ == "__main__":
    raise SystemExit(
        _core.run_ppdbench_with_defaults(
            default_ckpt=DEFAULT_CKPT,
            default_out_subdir=OUT_SUBDIR,
        )
    )
'''

nohup python /root/autodl-tmp/Peptide_3D/results/3_Pareto_improved/generate_ppdbench_DPO_weighted_sum.py \
  --gpu 2
  > /root/autodl-tmp/Peptide_3D/results/3_Pareto_improved/generated_dpo_weighted_sum.log 2>&1 &

'''
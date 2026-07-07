#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 PPDbench 各靶点（receptor.pdb）生成多肽 PDB；加载 train_DPO_affinity_only 对应权重。
默认每靶点 3 条，输出到 <靶点目录>/generated_dpo_affinity_only/pep_01.pdb ...
指定 GPU：加参数 --gpu N（如 --gpu 2）；或先 export CUDA_VISIBLE_DEVICES=2 再 --gpu 0。
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

DEFAULT_CKPT = str(_PEPTIDE_ROOT / "log_affinity_only/policy_dpo_affinity_best.pth")
OUT_SUBDIR = "generated_dpo_affinity_only"

if __name__ == "__main__":
    raise SystemExit(
        _core.run_ppdbench_with_defaults(
            default_ckpt=DEFAULT_CKPT,
            default_out_subdir=OUT_SUBDIR,
        )
    )
'''

nohup python /root/autodl-tmp/Peptide_3D/results/3_Pareto_improved/generate_ppdbench_DPO_affinity_only.py \
  --gpu 0
  > /root/autodl-tmp/Peptide_3D/results/3_Pareto_improved/generated_dpo_affinity_only.log 2>&1 &

'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPDbench：Ablation Base+OT 推理生成（每靶点生成多肽 PDB）。
权重：logs_base_ot/sft_best.pth
输出子目录：generated_ablation_base_ot/
"""
import sys
from pathlib import Path

_PEPTIDE_ROOT = Path("/root/autodl-tmp/Peptide_3D")
_GEN_ROOT = _PEPTIDE_ROOT / "results" / "3_Pareto_improved"
_HERE = Path(__file__).resolve().parent

for _p in (_PEPTIDE_ROOT, _GEN_ROOT, _HERE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import ppdbench_generate_core as _core  # noqa: E402

DEFAULT_CKPT = str(_PEPTIDE_ROOT / "logs_base_ot" / "sft_best.pth")
OUT_SUBDIR = "generated_ablation_base_ot"

if __name__ == "__main__":
    raise SystemExit(
        _core.run_ppdbench_with_defaults(
            default_ckpt=DEFAULT_CKPT,
            default_out_subdir=OUT_SUBDIR,
        )
    )


'''

python /root/autodl-tmp/Peptide_3D/results/4_ablation/generate_ppdbench_ablation_base_ot.py \
  --bench-root /root/autodl-tmp/PPDbench \
  --want-gpus 2 --gpu 0 \
  --num-per-target 3

'''
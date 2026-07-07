#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五种方法各靶点 **1 条肽（top-1）** 的三张散点图（HDOCK–稳定性、HDOCK–溶解性、稳定性–溶解性），
300 dpi PNG + 矢量 PDF。

- DPO×3 + SFT：在 json 中对齐且三项齐全的多肽里，选 **HDOCK 最低** 的一条作为该靶点代表。
- multi_cands：在 bench HDOCK + sol/stab json 的全部候选上，靶点内 min-max 后
  **综合分 = norm(HDOCK→越大越好) + norm(stab) + norm(sol)**，取 **综合分最高** 的一条
  （与 plot_ppdbench_methods_scatter.py 中 multi_cands 的打分一致，仅 top_k=1）。

输出文件名带 `top1_` 前缀，避免覆盖 mean-top-3 版本。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_ppdbench_methods_scatter as base

DEFAULT_DATA = HERE.parent
DEFAULT_BENCH = Path("/root/autodl-tmp/PPDbench")

# 图例名称（与 mean-top3 脚本区分）
METHOD_LABELS: List[Tuple[str, str, str, str, Optional[str]]] = [
    ("DPO (affinity)", "ppdbench_hdock_dpo_affinity_only.json", "ppdbench_solubility_dpo_affinity_only.json", "ppdbench_stability_dpo_affinity_only.json", None),
    ("DPO (stability)", "ppdbench_hdock_dpo_stability_only.json", "ppdbench_solubility_dpo_stability_only.json", "ppdbench_stability_dpo_stability_only.json", None),
    ("DPO (1:1:1)", "ppdbench_hdock_dpo_weighted_sum.json", "ppdbench_solubility_dpo_weighted_sum.json", "ppdbench_stability_dpo_weighted_sum.json", None),
    ("SFT (multi-objective)", "ppdbench_hdock_sft_multi_objective.json", "ppdbench_solubility_sft_multi_objective.json", "ppdbench_stability_sft_multi_objective.json", None),
    (
        "Multi-cands",
        "",
        "ppdbench_solubility_multi_cands.json",
        "ppdbench_stability_multi_cands.json",
        base.MULTI_CANDS_BENCH_MO3,
    ),
]


def best_hdock_per_target(
    aff: Dict[str, float],
    sol: Dict[str, float],
    stab: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每靶点：aff/stab/sol 齐全的肽中 HDOCK 最低者（并列时按键名）。"""
    groups: Dict[str, List[Tuple[str, float, float, float]]] = defaultdict(list)
    for key in aff:
        if key not in sol or key not in stab:
            continue
        tid = key.split("/", 1)[0]
        groups[tid].append((key, aff[key], stab[key], sol[key]))

    xs, ys, zs = [], [], []
    for tid in sorted(groups.keys()):
        rows = groups[tid]
        if not rows:
            continue
        best = min(rows, key=lambda r: (r[1], r[0]))
        _, a, st, so = best
        xs.append(float(a))
        ys.append(float(st))
        zs.append(float(so))
    if not xs:
        return np.array([]), np.array([]), np.array([])
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--out-dir", type=str, default=str(HERE))
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--bench-root", type=str, default=str(DEFAULT_BENCH))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    bench_root = Path(args.bench_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    triplets: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for name, f_aff, f_sol, f_stab, f_flag in METHOD_LABELS:
        sol = base.prop_dict(base.load_json(data_dir / f_sol))
        stab = base.prop_dict(base.load_json(data_dir / f_stab))
        if f_flag == base.MULTI_CANDS_BENCH_MO3:
            aff = base.affinity_from_bench_hdock(bench_root)
            a, st, so = base.mean_per_target_multi_cands_three_objective_topk(
                aff, sol, stab, top_k=1
            )
            print(f"[{name}] targets n={a.size}")
        elif f_flag:
            aff = base.affinity_from_top3_json(base.load_json(data_dir / f_flag))
            a, st, so = best_hdock_per_target(aff, sol, stab)
            print(f"[{name}] targets n={a.size}")
        else:
            aff = base.affinity_from_hdock_json(base.load_json(data_dir / f_aff))
            a, st, so = best_hdock_per_target(aff, sol, stab)
            print(f"[{name}] targets n={a.size}")
        triplets.append((name, a, st, so))

    base.plot_panel(
        [(n, a, st) for n, a, st, _ in triplets],
        xlabel="HDOCK score (lower is better)",
        ylabel="Stability score (FoldX, higher is better)",
        title="Top-1 peptide per target",
        out_base=out_dir / "ppdbench_scatter_top1_hdock_vs_stability",
        dpi=args.dpi,
    )
    base.plot_panel(
        [(n, a, so) for n, a, _, so in triplets],
        xlabel="HDOCK score (lower is better)",
        ylabel="Solubility (Protein-Sol scaled-sol, higher is better)",
        title="Top-1 peptide per target",
        out_base=out_dir / "ppdbench_scatter_top1_hdock_vs_solubility",
        dpi=args.dpi,
    )
    base.plot_panel(
        [(n, st, so) for n, _, st, so in triplets],
        xlabel="Stability score (FoldX, higher is better)",
        ylabel="Solubility (Protein-Sol scaled-sol, higher is better)",
        title="Top-1 peptide per target",
        out_base=out_dir / "ppdbench_scatter_top1_stability_vs_solubility",
        dpi=args.dpi,
    )
    print(f"[FIN] wrote top-1 PNG+PDF to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五种方法在 (HDOCK, 稳定性, 溶解性) 空间中的 **三维散点图**，各靶点 1 个点。

生成两套图：
  - **top3**：DPO×3 + SFT 对对齐肽三项分别取平均（通常 3 条）；
            multi_cands 用靶点内三目标 min-max 综合分取前 3 再平均（与 plot_ppdbench_methods_scatter.py 一致）。
  - **top1**：DPO×3 + SFT 取 HDOCK 最低的一条肽；
            multi_cands 用同一套三目标综合分取 **top-1**（与 plot_ppdbench_methods_scatter_top1.py 一致）。

输出：``ppdbench_scatter_3d_top1`` / ``ppdbench_scatter_3d_top3`` 的 PNG（300 dpi）与 PDF。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — 注册 3d projection

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_ppdbench_methods_scatter as base
from plot_illustrator_export import savefig_png_then_pdf

DEFAULT_DATA = HERE.parent
DEFAULT_BENCH = Path("/root/autodl-tmp/PPDbench")

MULTI_LEGEND = {
    "top1": "multi_cands (top-1 by 3-objective)",
    "top3": "multi_cands (top-3 by 3-objective)",
}


def best_hdock_per_target(
    aff: Dict[str, float],
    sol: Dict[str, float],
    stab: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每靶点：三项齐全肽中 HDOCK 最低者（并列按键名）。"""
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


def build_triplets(
    data_dir: Path,
    bench_root: Path,
    mode: str,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    assert mode in ("top1", "top3")
    out: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for name, f_aff, f_sol, f_stab, f_flag in base.METHODS:
        label = MULTI_LEGEND[mode] if f_flag == base.MULTI_CANDS_BENCH_MO3 else name
        sol = base.prop_dict(base.load_json(data_dir / f_sol))
        stab = base.prop_dict(base.load_json(data_dir / f_stab))

        if f_flag == base.MULTI_CANDS_BENCH_MO3:
            aff = base.affinity_from_bench_hdock(bench_root)
            k = 1 if mode == "top1" else 3
            a, st, so = base.mean_per_target_multi_cands_three_objective_topk(
                aff, sol, stab, top_k=k
            )
        elif f_flag:
            aff = base.affinity_from_top3_json(base.load_json(data_dir / f_flag))
            if mode == "top1":
                a, st, so = best_hdock_per_target(aff, sol, stab)
            else:
                a, st, so = base.mean_per_target(aff, sol, stab)
        else:
            assert f_aff is not None
            aff = base.affinity_from_hdock_json(base.load_json(data_dir / f_aff))
            if mode == "top1":
                a, st, so = best_hdock_per_target(aff, sol, stab)
            else:
                a, st, so = base.mean_per_target(aff, sol, stab)

        print(f"[{mode}] {label} n={a.size}")
        out.append((label, a, st, so))
    return out


def plot_3d_panel(
    triplets: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    *,
    title: str,
    out_base: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(9.0, 8.0), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")

    for i, (name, xd, yd, zd) in enumerate(triplets):
        if xd.size == 0:
            continue
        ax.scatter(
            xd,
            yd,
            zd,
            s=36,
            alpha=0.55,
            c=base.COLORS[i % len(base.COLORS)],
            edgecolors="none",
            depthshade=True,
            label=f"{name} (n={xd.size})",
            rasterized=True,
        )

    ax.set_xlabel("HDOCK (lower is better)", fontsize=10, labelpad=8)
    ax.set_ylabel("Stability / FoldX (higher is better)", fontsize=10, labelpad=8)
    ax.set_zlabel("Solubility scaled-sol (higher is better)", fontsize=10, labelpad=8)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.view_init(elev=22, azim=-55)

    savefig_png_then_pdf(fig, out_base, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--out-dir", type=str, default=str(HERE))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--bench-root", type=str, default=str(DEFAULT_BENCH))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    bench_root = Path(args.bench_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in ("top1", "top3"):
        tri = build_triplets(data_dir, bench_root, mode)
        plot_3d_panel(
            tri,
            title=f"PPDbench 3D: HDOCK × stability × solubility ({mode})",
            out_base=out_dir / f"ppdbench_scatter_3d_{mode}",
            dpi=args.dpi,
        )

    print(f"[FIN] wrote ppdbench_scatter_3d_top1/top3 PNG+PDF to {out_dir}")


if __name__ == "__main__":
    main()




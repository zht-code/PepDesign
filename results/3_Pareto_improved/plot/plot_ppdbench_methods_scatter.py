#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将五种方法（DPO affinity / DPO stability / DPO weighted-sum / SFT / multi_cands）
在 PPDbench 上已对齐的 HDOCK（亲和力）、FoldX 稳定性、Protein-Sol 溶解性（scaled-sol）
绘制三张散点图（每图 5 种颜色），输出 300 dpi PNG 与矢量 PDF。
输出文件名带 ``top3_`` 前缀（如 ``ppdbench_scatter_top3_hdock_vs_stability.png``），与 top-1 脚本区分。
PDF 经 ``plot_illustrator_export``：TrueType 文字（fonttype 42）+ 导出前关闭栅格化，便于 Adobe Illustrator 编辑。

合并键：\"<target_id>/<peptide_basename>\"（与 sol/stab json 一致）。

五种方法统一：每个靶点 1 个点 = 对 **3 条肽** 的 HDOCK / 稳定性 / 溶解性 **分别取算术平均**。

前四种（DPO×3 + SFT）：对齐的肽通常即生成的 top-3，直接对这三条取平均。

multi_cands：在「bench 的 HDOCK + json 的 stab/sol」齐全的 **全部候选** 上，将三项在靶点内
各自 min-max 到 [0,1]（HDOCK 越低越好 → 归一化为越大越好），**综合分 = 三者之和**，
按综合分从高到低取前 3 条（不足 3 条则全取），再对这三条的 **原始** HDOCK/stab/sol 取平均。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from plot_illustrator_export import savefig_png_then_pdf

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent
DEFAULT_BENCH = Path("/root/autodl-tmp/PPDbench")

# 第五种方法：从 PPDbench 读 HDOCK，并按三目标综合分选 top-3 再平均
MULTI_CANDS_BENCH_MO3: str = "__MULTI_CANDS_BENCH_MO3__"

METHODS: List[Tuple[str, str, str, str, Optional[str]]] = [
    (
        "DPO (affinity)",
        "ppdbench_hdock_dpo_affinity_only.json",
        "ppdbench_solubility_dpo_affinity_only.json",
        "ppdbench_stability_dpo_affinity_only.json",
        None,
    ),
    (
        "DPO (stability)",
        "ppdbench_hdock_dpo_stability_only.json",
        "ppdbench_solubility_dpo_stability_only.json",
        "ppdbench_stability_dpo_stability_only.json",
        None,
    ),
    (
        "DPO (1:1:1)",
        "ppdbench_hdock_dpo_weighted_sum.json",
        "ppdbench_solubility_dpo_weighted_sum.json",
        "ppdbench_stability_dpo_weighted_sum.json",
        None,
    ),
    (
        "SFT (multi-objective)",
        "ppdbench_hdock_sft_multi_objective.json",
        "ppdbench_solubility_sft_multi_objective.json",
        "ppdbench_stability_sft_multi_objective.json",
        None,
    ),
    (
        "Multi-cands",
        None,
        "ppdbench_solubility_multi_cands.json",
        "ppdbench_stability_multi_cands.json",
        MULTI_CANDS_BENCH_MO3,
    ),
]

COLORS = [
    # Nature-like muted palette with strong contrast and colorblind robustness.
    "#3C5488",  # deep blue
    "#E39B29",  # warm orange
    "#00A087",  # teal green
    "#C44E52",  # muted red
    "#8E63A9",  # muted purple
]

MARKERS = ["o", "s", "^", "P", "D"]

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.9,
    "axes.labelsize": 11,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.4,
}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def score_from_entry(entry: Any) -> Optional[float]:
    if not isinstance(entry, dict):
        return None
    v = entry.get("score")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def affinity_from_hdock_json(data: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in data.items():
        s = score_from_entry(v)
        if s is not None:
            out[str(k)] = s
    return out


def affinity_from_bench_hdock(
    bench_root: Path,
    *,
    multi_subdir: str = "multi_cands",
    scores_name: str = "cands_hdock_scores.json",
) -> Dict[str, float]:
    """<target>/<pep_XX.pdb> -> HDOCK score（与 multi_cands json 键一致）。"""
    out: Dict[str, float] = {}
    if not bench_root.is_dir():
        return out
    for d in sorted(p for p in bench_root.iterdir() if p.is_dir()):
        tid = d.name
        p = d / multi_subdir / scores_name
        if not p.is_file():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        for k, v in raw.items():
            try:
                s = float(v)
            except (TypeError, ValueError):
                continue
            name = Path(str(k)).name
            if not name.lower().endswith(".pdb"):
                continue
            out[f"{tid}/{name}"] = s
    return out


def affinity_from_top3_json(data: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for tid, blob in data.items():
        if not isinstance(blob, dict):
            continue
        for item in blob.get("top3_hdock", []):
            if not isinstance(item, dict):
                continue
            bn = item.get("peptide_basename")
            hs = item.get("hdock_score")
            if not bn or hs is None:
                continue
            try:
                out[f"{tid}/{bn}"] = float(hs)
            except (TypeError, ValueError):
                continue
    return out


def prop_dict(data: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in data.items():
        s = score_from_entry(v)
        if s is not None:
            out[str(k)] = s
    return out


def mean_per_target(
    aff: Dict[str, float],
    sol: Dict[str, float],
    stab: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每靶点：对 aff/stab/sol 齐全的 **全部** 肽，三项分别取算术平均（前四种方法通常恰为 top-3）。"""
    groups: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for key in aff:
        if key not in sol or key not in stab:
            continue
        tid = key.split("/", 1)[0]
        groups[tid].append((aff[key], stab[key], sol[key]))

    xs, ys, zs = [], [], []
    for tid in sorted(groups.keys()):
        pts = groups[tid]
        if not pts:
            continue
        xs.append(float(np.mean([p[0] for p in pts])))
        ys.append(float(np.mean([p[1] for p in pts])))
        zs.append(float(np.mean([p[2] for p in pts])))
    if not xs:
        return np.array([]), np.array([]), np.array([])
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def mean_per_target_multi_cands_three_objective_topk(
    aff: Dict[str, float],
    sol: Dict[str, float],
    stab: Dict[str, float],
    *,
    top_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    multi_cands：每靶点在候选上做靶点内 min-max，综合分 = norm(亲和力) + norm(稳定性) + norm(溶解性)
    （HDOCK 越低越好，故用 (h_max-h)/(h_max-h_min)）。取综合分最高的 top_k 条，对原始三项取平均。
    """
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
        hs = [r[1] for r in rows]
        sts = [r[2] for r in rows]
        sos = [r[3] for r in rows]
        h_min, h_max = min(hs), max(hs)
        st_min, st_max = min(sts), max(sts)
        so_min, so_max = min(sos), max(sos)

        scored: List[Tuple[float, float, float, float, str]] = []
        for key, h, st, so in rows:
            if h_max > h_min:
                nh = (h_max - h) / (h_max - h_min)
            else:
                nh = 0.5
            if st_max > st_min:
                ns = (st - st_min) / (st_max - st_min)
            else:
                ns = 0.5
            if so_max > so_min:
                no = (so - so_min) / (so_max - so_min)
            else:
                no = 0.5
            combo = nh + ns + no
            scored.append((-combo, h, st, so, key))

        scored.sort(key=lambda t: (t[0], t[1]))
        k = min(top_k, len(scored))
        pick = scored[:k]
        xs.append(float(np.mean([t[1] for t in pick])))
        ys.append(float(np.mean([t[2] for t in pick])))
        zs.append(float(np.mean([t[3] for t in pick])))

    if not xs:
        return np.array([]), np.array([]), np.array([])
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def apply_nature_rc() -> None:
    plt.rcParams.update(NATURE_RC)


def trendline_spline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int = 12,
    smooth: float = 2.2,
    clip_quantile: float = 0.05,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Robust trend line:
    - bin x, take median y per bin (less sensitive to outliers)
    - fit a smoothing spline through bin medians
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < max(20, bins * 2):
        return None

    # Sort and bin along x
    order = np.argsort(x)
    x, y = x[order], y[order]
    # Avoid spline edge oscillation by fitting on the central mass only.
    if 0.0 < clip_quantile < 0.5 and x.size >= 20:
        lo = float(np.quantile(x, clip_quantile))
        hi = float(np.quantile(x, 1.0 - clip_quantile))
        keep = (x >= lo) & (x <= hi)
        if np.count_nonzero(keep) >= max(16, bins * 2):
            x, y = x[keep], y[keep]
            order = np.argsort(x)
            x, y = x[order], y[order]
    edges = np.linspace(float(x[0]), float(x[-1]), bins + 1)

    xm: List[float] = []
    ym: List[float] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i < bins - 1:
            mask = (x >= lo) & (x < hi)
        else:
            mask = (x >= lo) & (x <= hi)
        if not np.any(mask):
            continue
        xb = x[mask]
        yb = y[mask]
        xm.append(float(np.median(xb)))
        ym.append(float(np.median(yb)))

    if len(xm) < 6:
        return None

    try:
        from scipy.interpolate import UnivariateSpline  # type: ignore
    except Exception:
        return None

    xs = np.asarray(xm, dtype=float)
    ys = np.asarray(ym, dtype=float)
    # Ensure strictly increasing x for spline
    u = np.argsort(xs)
    xs, ys = xs[u], ys[u]
    uniq = np.r_[True, np.diff(xs) > 1e-12]
    xs, ys = xs[uniq], ys[uniq]
    if xs.size < 6:
        return None

    # Smoothness scaled by variance and number of points (larger => smoother)
    s = smooth * float(xs.size) * float(np.var(ys) + 1e-12)
    # Use quadratic spline to reduce overfitting wiggles.
    spl = UnivariateSpline(xs, ys, s=s, k=2)
    x_dense = np.linspace(float(xs[0]), float(xs[-1]), 220)
    y_dense = spl(x_dense)
    return x_dense, y_dense


def plot_panel(
    methods_data: List[Tuple[str, np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_base: Path,
    dpi: int,
) -> None:
    apply_nature_rc()
    fig, ax = plt.subplots(figsize=(6.6, 4.9), layout="constrained")
    ax.set_facecolor("white")

    all_x = (
        np.concatenate([x for _, x, _ in methods_data if getattr(x, "size", 0) > 0])
        if methods_data
        else np.array([])
    )
    all_y = (
        np.concatenate([y for _, _, y in methods_data if getattr(y, "size", 0) > 0])
        if methods_data
        else np.array([])
    )

    if all_x.size > 0:
        x_pad = max(0.04 * (float(np.max(all_x)) - float(np.min(all_x))), 1.0)
        ax.set_xlim(float(np.min(all_x)) - x_pad, float(np.max(all_x)) + x_pad)
    if all_y.size > 0:
        y_pad = max(0.06 * (float(np.max(all_y)) - float(np.min(all_y))), 0.02)
        ax.set_ylim(float(np.min(all_y)) - y_pad, float(np.max(all_y)) + y_pad)

    legend_handles: List[Line2D] = []

    for i, (name, x, y) in enumerate(methods_data):
        if x.size == 0 or y.size == 0:
            continue
        c = COLORS[i % len(COLORS)]
        mk = MARKERS[i % len(MARKERS)]
        ax.scatter(
            x,
            y,
            s=30,
            alpha=0.58,
            c=c,
            marker=mk,
            edgecolors="white",
            linewidths=0.45,
            label="_nolegend_",
            rasterized=True,
            zorder=3,
        )
        # Trend line (robust smoothing), fallback to linear fit if needed
        tl = trendline_spline(x, y, bins=12, smooth=2.2, clip_quantile=0.05)
        if tl is not None:
            xs, ys = tl
            ax.plot(
                xs,
                ys,
                color=c,
                linewidth=2.2,
                alpha=0.98,
                solid_capstyle="round",
                label="_nolegend_",
                zorder=4,
            )
        elif x.size >= 2:
            a, b = np.polyfit(x.astype(float), y.astype(float), deg=1)
            x0, x1 = float(np.min(x)), float(np.max(x))
            xs = np.linspace(x0, x1, 100, dtype=float)
            ys = a * xs + b
            ax.plot(
                xs,
                ys,
                color=c,
                linewidth=2.2,
                alpha=0.98,
                solid_capstyle="round",
                label="_nolegend_",
                zorder=4,
            )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=c,
                linewidth=1.8,
                marker=mk,
                markersize=6.5,
                markerfacecolor=c,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=name,
            )
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=0.9)
    ax.tick_params(axis="both", which="minor", direction="out", length=2.5, width=0.7)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.9)
    ax.grid(axis="x", color="#EDEDED", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    savefig_png_then_pdf(fig, out_base, dpi=dpi, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA),
        help="含各 json 的目录（默认：3_Pareto_improved）",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(HERE),
        help="输出 png/pdf 目录（默认：本脚本所在 plot/）",
    )
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument(
        "--bench-root",
        type=str,
        default=str(DEFAULT_BENCH),
        help="PPDbench 根目录（multi_cands 的 cands_hdock_scores.json）",
    )
    ap.add_argument(
        "--multi-cands-top-k",
        type=int,
        default=3,
        help="multi_cands：按三目标综合分选取的肽条数，再对原始 HDOCK/stab/sol 取平均（默认 3）",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    bench_root = Path(args.bench_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    triplets: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for name, f_aff, f_sol, f_stab, f_top3 in METHODS:
        sol = prop_dict(load_json(data_dir / f_sol))
        stab = prop_dict(load_json(data_dir / f_stab))
        if f_top3 == MULTI_CANDS_BENCH_MO3:
            aff = affinity_from_bench_hdock(bench_root)
            a, st, so = mean_per_target_multi_cands_three_objective_topk(
                aff, sol, stab, top_k=args.multi_cands_top_k
            )
            print(
                f"[{name}] targets n={a.size} "
                f"(mean over top-{args.multi_cands_top_k} by 3-objective score)"
            )
        elif f_top3:
            aff = affinity_from_top3_json(load_json(data_dir / f_top3))
            a, st, so = mean_per_target(aff, sol, stab)
            print(f"[{name}] targets (mean over top-3) n={a.size}")
        else:
            assert f_aff is not None
            aff = affinity_from_hdock_json(load_json(data_dir / f_aff))
            a, st, so = mean_per_target(aff, sol, stab)
            print(f"[{name}] targets (mean over top-3) n={a.size}")
        triplets.append((name, a, st, so))

    # 1) HDOCK vs stability
    plot_panel(
        [(n, a, st) for n, a, st, _ in triplets],
        xlabel="HDOCK score (lower is better)",
        ylabel="Stability score (FoldX, higher is better)",
        title="Mean of top-3 peptides per target",
        out_base=out_dir / "ppdbench_scatter_top3_hdock_vs_stability",
        dpi=args.dpi,
    )
    # 2) HDOCK vs solubility
    plot_panel(
        [(n, a, so) for n, a, _, so in triplets],
        xlabel="HDOCK score (lower is better)",
        ylabel="Solubility (Protein-Sol scaled-sol, higher is better)",
        title="Mean of top-3 peptides per target",
        out_base=out_dir / "ppdbench_scatter_top3_hdock_vs_solubility",
        dpi=args.dpi,
    )
    # 3) stability vs solubility
    plot_panel(
        [(n, st, so) for n, _, st, so in triplets],
        xlabel="Stability score (FoldX, higher is better)",
        ylabel="Solubility (Protein-Sol scaled-sol, higher is better)",
        title="Mean of top-3 peptides per target",
        out_base=out_dir / "ppdbench_scatter_top3_stability_vs_solubility",
        dpi=args.dpi,
    )

    print(f"[FIN] wrote PNG+PDF to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired delta plots (Nature-style) to highlight method gaps.

We treat "Multi-cands" as Ours, and compute paired differences vs each baseline
per target:

- HDOCK (lower is better): Δ = baseline - ours  (so >0 means ours better)
- Stability (higher is better): Δ = ours - baseline
- Solubility (higher is better): Δ = ours - baseline

For each metric, draw violin + swarm, and annotate median with bootstrap 95% CI.
Exports high-res PNG + vector PDF (Illustrator-friendly) via plot_illustrator_export.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_ppdbench_methods_scatter as base
from plot_illustrator_export import savefig_png_then_pdf

DEFAULT_DATA = HERE.parent
DEFAULT_BENCH = Path("/root/autodl-tmp/PPDbench")


def best_hdock_per_target_dict(
    aff: Dict[str, float],
    sol: Dict[str, float],
    stab: Dict[str, float],
) -> Dict[str, Tuple[float, float, float]]:
    """Per target: pick peptide with minimal HDOCK among entries with all 3 metrics."""
    groups: Dict[str, List[Tuple[str, float, float, float]]] = defaultdict(list)
    for key in aff:
        if key not in sol or key not in stab:
            continue
        tid = key.split("/", 1)[0]
        groups[tid].append((key, aff[key], stab[key], sol[key]))

    out: Dict[str, Tuple[float, float, float]] = {}
    for tid in sorted(groups.keys()):
        rows = groups[tid]
        if not rows:
            continue
        _, a, st, so = min(rows, key=lambda r: (r[1], r[0]))
        out[tid] = (float(a), float(st), float(so))
    return out


def mean_per_target_dict(
    aff: Dict[str, float],
    sol: Dict[str, float],
    stab: Dict[str, float],
) -> Dict[str, Tuple[float, float, float]]:
    groups: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for key in aff:
        if key not in sol or key not in stab:
            continue
        tid = key.split("/", 1)[0]
        groups[tid].append((aff[key], stab[key], sol[key]))

    out: Dict[str, Tuple[float, float, float]] = {}
    for tid in sorted(groups.keys()):
        pts = groups[tid]
        if not pts:
            continue
        out[tid] = (
            float(np.mean([p[0] for p in pts])),
            float(np.mean([p[1] for p in pts])),
            float(np.mean([p[2] for p in pts])),
        )
    return out


def bootstrap_ci_median(x: np.ndarray, *, n_boot: int = 5000, seed: int = 0) -> Tuple[float, float, float]:
    """Return (median, lo, hi) with percentile bootstrap 95% CI."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    meds = np.median(x[idx], axis=1)
    med = float(np.median(x))
    lo, hi = np.quantile(meds, [0.025, 0.975])
    return med, float(lo), float(hi)


def apply_nature_rc() -> None:
    # Reuse the same RC as scatter plots for consistency.
    plt.rcParams.update(base.NATURE_RC)


def plot_delta_violin(
    deltas: List[np.ndarray],
    labels: List[str],
    colors: List[str],
    *,
    ylabel: str,
    title: str,
    subtitle: str,
    out_base: Path,
    dpi: int,
) -> None:
    apply_nature_rc()
    fig, ax = plt.subplots(figsize=(6.6, 4.0), layout="constrained")

    positions = np.arange(1, len(deltas) + 1, dtype=float)
    parts = ax.violinplot(
        deltas,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor(colors[i])
        body.set_alpha(0.22)
        body.set_linewidth(1.0)

    # Swarm (jittered points)
    rng = np.random.default_rng(1)
    for i, x in enumerate(deltas):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        jitter = rng.normal(0.0, 0.06, size=x.size)
        ax.scatter(
            positions[i] + jitter,
            x,
            s=12,
            c=colors[i],
            alpha=0.22,
            linewidths=0.0,
            rasterized=True,
            zorder=2,
        )

    # Median + 95% CI
    for i, x in enumerate(deltas):
        med, lo, hi = bootstrap_ci_median(x, n_boot=5000, seed=10 + i)
        ax.plot([positions[i] - 0.28, positions[i] + 0.28], [med, med], color=colors[i], lw=2.0, zorder=4)
        ax.plot([positions[i], positions[i]], [lo, hi], color=colors[i], lw=2.0, zorder=4)
        ax.plot([positions[i] - 0.14, positions[i] + 0.14], [lo, lo], color=colors[i], lw=2.0, zorder=4)
        ax.plot([positions[i] - 0.14, positions[i] + 0.14], [hi, hi], color=colors[i], lw=2.0, zorder=4)

    ax.axhline(0.0, color="#404040", lw=1.0, alpha=0.75, zorder=1)
    ax.text(
        0.995,
        0.0,
        "Δ=0 (ours)",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
        color="#404040",
        alpha=0.85,
    )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Baseline vs Ours", fontsize=11)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=22, ha="right")

    # Clean spines/grid (light y-grid)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title, fontsize=10, color="#4D4D4D", pad=10)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#4D4D4D",
        )

    savefig_png_then_pdf(fig, out_base, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--out-dir", type=str, default=str(HERE))
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--bench-root", type=str, default=str(DEFAULT_BENCH))
    ap.add_argument(
        "--top-k",
        type=int,
        choices=[1, 3],
        default=3,
        help="Use top-1 (single best) or top-3 (mean over 3) per target.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    bench_root = Path(args.bench_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-target metric dicts for each method
    method_dicts: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for name, f_aff, f_sol, f_stab, flag in base.METHODS:
        sol = base.prop_dict(base.load_json(data_dir / f_sol))
        stab = base.prop_dict(base.load_json(data_dir / f_stab))

        if flag == base.MULTI_CANDS_BENCH_MO3:
            aff = base.affinity_from_bench_hdock(bench_root)
            # top-k selection, then mean of selected (k=1 becomes "top-1 by 3-objective")
            a, st, so = base.mean_per_target_multi_cands_three_objective_topk(aff, sol, stab, top_k=args.top_k)
            # We need dict by target: reuse base grouping logic by re-running with dict.
            # Implement by repeating selection but storing per-target.
            # For simplicity and determinism, compute it directly here.
            groups: Dict[str, List[Tuple[str, float, float, float]]] = defaultdict(list)
            for key in aff:
                if key not in sol or key not in stab:
                    continue
                tid = key.split("/", 1)[0]
                groups[tid].append((key, aff[key], stab[key], sol[key]))
            out: Dict[str, Tuple[float, float, float]] = {}
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
                for key, h, stv, sov in rows:
                    nh = (h_max - h) / (h_max - h_min) if h_max > h_min else 0.5
                    ns = (stv - st_min) / (st_max - st_min) if st_max > st_min else 0.5
                    no = (sov - so_min) / (so_max - so_min) if so_max > so_min else 0.5
                    combo = nh + ns + no
                    scored.append((-combo, h, stv, sov, key))
                scored.sort(key=lambda t: (t[0], t[1]))
                k = min(args.top_k, len(scored))
                pick = scored[:k]
                out[tid] = (
                    float(np.mean([t[1] for t in pick])),
                    float(np.mean([t[2] for t in pick])),
                    float(np.mean([t[3] for t in pick])),
                )
            method_dicts[name] = out
            continue

        if f_aff is None:
            continue
        aff = base.affinity_from_hdock_json(base.load_json(data_dir / f_aff))

        if args.top_k == 1:
            method_dicts[name] = best_hdock_per_target_dict(aff, sol, stab)
        else:
            method_dicts[name] = mean_per_target_dict(aff, sol, stab)

    ours_name = "Multi-cands"
    if ours_name not in method_dicts:
        raise RuntimeError(f"Cannot find ours method '{ours_name}' in method_dicts.")

    ours = method_dicts[ours_name]
    baselines = [n for n in method_dicts.keys() if n != ours_name]

    # Ensure consistent order like in base.METHODS (excluding ours)
    method_order = [n for n, *_ in base.METHODS if n != ours_name]
    baselines = [n for n in method_order if n in method_dicts]

    method_color = {name: base.COLORS[i % len(base.COLORS)] for i, (name, *_rest) in enumerate(base.METHODS)}

    def paired_delta(metric_idx: int, *, lower_is_better: bool) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for b in baselines:
            bd = method_dicts[b]
            keys = sorted(set(ours.keys()) & set(bd.keys()))
            if not keys:
                out.append(np.array([]))
                continue
            o = np.array([ours[k][metric_idx] for k in keys], dtype=float)
            v = np.array([bd[k][metric_idx] for k in keys], dtype=float)
            if lower_is_better:
                out.append(v - o)  # baseline - ours
            else:
                out.append(o - v)  # ours - baseline
        return out

    colors = [method_color[b] for b in baselines]
    suffix = f"top{args.top_k}"
    title = f"Paired Δ per target ({suffix})"
    subtitle = f"Ours = {ours_name} (each dot = one target; median ± 95% CI)"

    # 1) HDOCK (lower better)
    plot_delta_violin(
        paired_delta(0, lower_is_better=True),
        labels=baselines,
        colors=colors,
        ylabel="Δ HDOCK (baseline − ours), >0 better",
        title=title,
        subtitle=subtitle,
        out_base=out_dir / f"ppdbench_paired_delta_{suffix}_hdock",
        dpi=args.dpi,
    )
    # 2) Stability (higher better)
    plot_delta_violin(
        paired_delta(1, lower_is_better=False),
        labels=baselines,
        colors=colors,
        ylabel="Δ Stability (ours − baseline), >0 better",
        title=title,
        subtitle=subtitle,
        out_base=out_dir / f"ppdbench_paired_delta_{suffix}_stability",
        dpi=args.dpi,
    )
    # 3) Solubility (higher better)
    plot_delta_violin(
        paired_delta(2, lower_is_better=False),
        labels=baselines,
        colors=colors,
        ylabel="Δ Solubility (ours − baseline), >0 better",
        title=title,
        subtitle=subtitle,
        out_base=out_dir / f"ppdbench_paired_delta_{suffix}_solubility",
        dpi=args.dpi,
    )

    print(f"[FIN] wrote paired-delta PNG+PDF to {out_dir}")


if __name__ == "__main__":
    main()


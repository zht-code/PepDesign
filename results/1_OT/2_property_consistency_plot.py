#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot publication-style figures for property consistency analysis.

This script reads the CSV/JSON outputs produced by property_consistency_analysis.py
and generates the following figures in BOTH 300 dpi PNG and editable PDF formats:

Figure A: Property distribution preservation
    A1: affinity distribution
    A2: stability distribution
    A3: solubility distribution
    + A4: JS / EMD grouped bar panel

Figure B: In-target consistency
    B1: CDF of original percentile in combined candidates
    B2: hit@1 / hit@3 / hit@5 grouped bar chart

Figure C: Global ranking preservation
    C1: top-k overlap line plot
    C2: rank-correlation grouped bar chart (Spearman / Kendall)

Figure D: Supplementary scatter plots
    D1-D3: original score vs best augmented score
    D4-D6: original score vs mean augmented score

Expected input directory structure (produced by property_consistency_analysis.py):
    parsed_original_affinity.csv
    parsed_augmented_affinity.csv
    parsed_original_solubility.csv
    parsed_augmented_solubility.csv
    parsed_original_stability.csv
    parsed_augmented_stability.csv
    per_target_affinity.csv
    per_target_solubility.csv
    per_target_stability.csv
    target_level_rank_table_affinity.csv
    target_level_rank_table_solubility.csv
    target_level_rank_table_stability.csv
    topk_overlap_affinity.csv
    topk_overlap_solubility.csv
    topk_overlap_stability.csv
    summary_metrics.json

Example
-------
python property_consistency_plotting.py \
  --input_dir /root/autodl-tmp/Peptide_3D/results/property_consistency_analysis_out \
  --output_dir /root/autodl-tmp/Peptide_3D/results/property_consistency_figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =========================================================
# Style: Nature-like, restrained and clean
# =========================================================
COLORS = {
    "affinity": "#4C78A8",   # muted blue
    "stability": "#F58518",  # muted orange
    "solubility": "#54A24B", # muted green
    "original": "#4C78A8",
    "augmented": "#E45756",
    "best": "#B279A2",
    "mean": "#72B7B2",
    "median": "#FF9DA6",
    "grid": "#D9D9D9",
    "text": "#333333",
    "spine": "#333333",
    "bg": "#FFFFFF",
    "diag": "#888888",
}

PROPERTY_ORDER = ["affinity", "stability", "solubility"]
PROPERTY_LABELS = {
    "affinity": "Affinity",
    "stability": "Stability",
    "solubility": "Solubility",
}
PROPERTY_DIRECTION = {
    "affinity": "lower_better",
    "stability": "lower_better",
    "solubility": "higher_better",
}


def set_pub_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["bg"],
        "savefig.facecolor": COLORS["bg"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": COLORS["spine"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "text.color": COLORS["text"],
        "axes.linewidth": 1.0,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,   # editable text in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def beautify_ax(ax, add_ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["spine"])
    ax.spines["bottom"].set_color(COLORS["spine"])
    if add_ygrid:
        # ax.yaxis.grid(True, linestyle="-", alpha=0.35)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# IO
# =========================================================
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# KDE utilities (implemented without scipy/seaborn)
# =========================================================
def _silverman_bandwidth(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 1.0
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    if sigma <= 0 or np.isnan(sigma):
        sigma = std if std > 0 else 1.0
    bw = 0.9 * sigma * (n ** (-1 / 5))
    return float(max(bw, 1e-6))


def kde_curve(x: np.ndarray, n_grid: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([]), np.array([])
    if len(np.unique(x)) == 1:
        center = x[0]
        grid = np.linspace(center - 1, center + 1, n_grid)
        bw = 0.2
    else:
        xmin, xmax = np.min(x), np.max(x)
        pad = 0.08 * (xmax - xmin) if xmax > xmin else 1.0
        grid = np.linspace(xmin - pad, xmax + pad, n_grid)
        bw = _silverman_bandwidth(x)
    diff = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * diff ** 2).sum(axis=1) / (len(x) * bw * np.sqrt(2 * np.pi))
    return grid, dens


# =========================================================
# Figure A: distributions + JS/EMD summary
# =========================================================
def plot_distribution_panel(ax, orig: pd.Series, aug: pd.Series, property_name: str) -> None:
    x = pd.to_numeric(orig, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(aug, errors="coerce").dropna().to_numpy(dtype=float)

    gx, dx = kde_curve(x)
    gy, dy = kde_curve(y)

    if len(gx):
        ax.plot(gx, dx, color=COLORS["original"], lw=2.0)
        ax.fill_between(gx, 0, dx, color=COLORS["original"], alpha=0.20)
    if len(gy):
        ax.plot(gy, dy, color=COLORS["augmented"], lw=2.0)
        ax.fill_between(gy, 0, dy, color=COLORS["augmented"], alpha=0.20)

    ax.set_title(PROPERTY_LABELS[property_name])
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    beautify_ax(ax)


def plot_figure_a(input_dir: Path, output_dir: Path) -> None:
    summary = load_json(input_dir / "summary_metrics.json")

    parsed = {}
    for prop in PROPERTY_ORDER:
        parsed[(prop, "original")] = load_csv(input_dir / f"parsed_original_{prop}.csv")
        parsed[(prop, "augmented")] = load_csv(input_dir / f"parsed_augmented_{prop}.csv")

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.95], height_ratios=[1.0, 0.06], wspace=0.35, hspace=0.25)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_bar = fig.add_subplot(gs[0, 3])
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")

    for ax, prop in zip(axes, PROPERTY_ORDER):
        plot_distribution_panel(
            ax,
            parsed[(prop, "original")]["score"],
            parsed[(prop, "augmented")]["score"],
            prop,
        )

    # Summary bars: JS and EMD
    js_vals = [summary[prop]["distribution_metrics"]["JS_divergence"] for prop in PROPERTY_ORDER]
    emd_vals = [summary[prop]["distribution_metrics"]["EMD_wasserstein_1d"] for prop in PROPERTY_ORDER]

    x = np.arange(len(PROPERTY_ORDER))
    width = 0.34
    ax_bar.bar(x - width / 2, js_vals, width=width, color="#7F7F7F", edgecolor="white", linewidth=0.8, label="JS divergence")
    ax_bar.bar(x + width / 2, emd_vals, width=width, color="#C7C7C7", edgecolor="white", linewidth=0.8, label="EMD")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([PROPERTY_LABELS[p] for p in PROPERTY_ORDER], rotation=20, ha="right")
    ax_bar.set_title("Distribution distance")
    ax_bar.set_ylabel("Metric value")
    beautify_ax(ax_bar)

    legend_handles = [
        Line2D([0], [0], color=COLORS["original"], lw=2.5, label="Original"),
        Line2D([0], [0], color=COLORS["augmented"], lw=2.5, label="Augmented"),
        Patch(facecolor="#7F7F7F", label="JS divergence"),
        Patch(facecolor="#C7C7C7", label="EMD"),
    ]
    ax_leg.legend(handles=legend_handles, loc="center", ncol=4)

    fig.suptitle("Figure A. Property distribution preservation", y=1.02, fontsize=13)
    save_figure(fig, output_dir / "Figure_A_distribution_preservation")


# =========================================================
# Figure B: in-target consistency
# =========================================================
def empirical_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([]), np.array([])
    x = np.sort(v)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_figure_b(input_dir: Path, output_dir: Path) -> None:
    summary = load_json(input_dir / "summary_metrics.json")
    per_target = {prop: load_csv(input_dir / f"per_target_{prop}.csv") for prop in PROPERTY_ORDER}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.2, 0.9]})
    ax_cdf, ax_hit = axes

    # B1: CDF of original percentile in combined candidates
    for prop in PROPERTY_ORDER:
        df = per_target[prop]
        x, y = empirical_cdf(df["original_percentile_in_combined"].to_numpy(dtype=float)) if len(df) else (np.array([]), np.array([]))
        if len(x):
            ax_cdf.plot(x, y, lw=2.2, color=COLORS[prop], label=PROPERTY_LABELS[prop])

    ax_cdf.set_xlabel("Original percentile in combined candidates")
    ax_cdf.set_ylabel("Cumulative fraction of targets")
    ax_cdf.set_title("CDF of within-target percentile")
    ax_cdf.set_xlim(0, 1)
    ax_cdf.set_ylim(0, 1)
    beautify_ax(ax_cdf)
    ax_cdf.legend()

    # B2: hit@k grouped bar chart
    hit_metrics = ["hit_at_1_rate", "hit_at_3_rate", "hit_at_5_rate"]
    hit_labels = ["Hit@1", "Hit@3", "Hit@5"]
    x = np.arange(len(hit_metrics))
    width = 0.22

    for i, prop in enumerate(PROPERTY_ORDER):
        vals = [summary[prop]["in_target_consistency"][m] for m in hit_metrics]
        ax_hit.bar(x + (i - 1) * width, vals, width=width, color=COLORS[prop], edgecolor="white", linewidth=0.8, label=PROPERTY_LABELS[prop])

    ax_hit.set_xticks(x)
    ax_hit.set_xticklabels(hit_labels)
    ax_hit.set_ylim(0, 1.0)
    ax_hit.set_ylabel("Rate")
    ax_hit.set_title("Within-target hit rates")
    beautify_ax(ax_hit)
    ax_hit.legend()

    fig.suptitle("Figure B. In-target consistency", y=1.02, fontsize=13)
    save_figure(fig, output_dir / "Figure_B_in_target_consistency")


# =========================================================
# Figure C: global ranking preservation
# =========================================================
def plot_figure_c(input_dir: Path, output_dir: Path) -> None:
    summary = load_json(input_dir / "summary_metrics.json")
    overlap_tables = {prop: load_csv(input_dir / f"topk_overlap_{prop}.csv") for prop in PROPERTY_ORDER}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax_line, ax_bar = axes

    # C1: top-k overlap line plot, use overlap ratio with best_aug
    for prop in PROPERTY_ORDER:
        df = overlap_tables[prop]
        if len(df) == 0:
            continue
        ax_line.plot(
            df["topk"],
            df["overlap_ratio_with_best_aug"],
            marker="o",
            ms=4,
            lw=2.0,
            color=COLORS[prop],
            label=PROPERTY_LABELS[prop],
        )

    ax_line.set_xlabel("k")
    ax_line.set_ylabel("Top-k overlap ratio")
    ax_line.set_title("Target-level top-k overlap")
    beautify_ax(ax_line)
    ax_line.legend()

    # C2: rank correlation grouped bars
    corr_names = [
        ("spearman_original_vs_best_aug", "Sp. best"),
        ("spearman_original_vs_mean_aug", "Sp. mean"),
        ("kendall_original_vs_best_aug", "Kd. best"),
        ("kendall_original_vs_mean_aug", "Kd. mean"),
    ]
    x = np.arange(len(PROPERTY_ORDER))
    width = 0.18

    palette = [COLORS["original"], COLORS["mean"], COLORS["best"], "#9C755F"]

    for j, (metric_key, metric_label) in enumerate(corr_names):
        vals = [summary[prop]["target_level_rank_and_overlap"].get(metric_key, np.nan) for prop in PROPERTY_ORDER]
        ax_bar.bar(x + (j - 1.5) * width, vals, width=width, color=palette[j], edgecolor="white", linewidth=0.8, label=metric_label)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([PROPERTY_LABELS[p] for p in PROPERTY_ORDER], rotation=20, ha="right")
    ax_bar.set_ylabel("Correlation")
    ax_bar.set_ylim(-1.0, 1.0)
    ax_bar.set_title("Rank correlation")
    beautify_ax(ax_bar)
    ax_bar.axhline(0, color=COLORS["diag"], lw=1.0, alpha=0.7)
    ax_bar.legend(ncol=2)

    fig.suptitle("Figure C. Global ranking preservation", y=1.02, fontsize=13)
    save_figure(fig, output_dir / "Figure_C_global_ranking_preservation")


# =========================================================
# Figure D: supplementary scatter plots
# =========================================================
def scatter_with_diag(ax, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, color: str) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) > 0:
        ax.scatter(x, y, s=18, alpha=0.65, color=color, edgecolors="white", linewidths=0.3)
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1.1, color=COLORS["diag"])
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    beautify_ax(ax)


def plot_figure_d(input_dir: Path, output_dir: Path) -> None:
    rank_tables = {prop: load_csv(input_dir / f"target_level_rank_table_{prop}.csv") for prop in PROPERTY_ORDER}

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for col, prop in enumerate(PROPERTY_ORDER):
        df = rank_tables[prop]
        scatter_with_diag(
            axes[0, col],
            df["original_score"].to_numpy(dtype=float),
            df["best_aug_score"].to_numpy(dtype=float),
            f"{PROPERTY_LABELS[prop]}: original vs best-aug",
            "Original score",
            "Best augmented score",
            COLORS[prop],
        )
        scatter_with_diag(
            axes[1, col],
            df["original_score"].to_numpy(dtype=float),
            df["mean_aug_score"].to_numpy(dtype=float),
            f"{PROPERTY_LABELS[prop]}: original vs mean-aug",
            "Original score",
            "Mean augmented score",
            COLORS[prop],
        )

    fig.suptitle("Figure D. Target-level score correspondence", y=1.01, fontsize=13)
    save_figure(fig, output_dir / "Figure_D_target_level_scatter")


# =========================================================
# Main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot figures for property consistency analysis.")
    parser.add_argument("--input_dir", required=True, help="Directory containing outputs from property_consistency_analysis.py")
    parser.add_argument("--output_dir", required=True, help="Directory to save figures")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_pub_style()

    plot_figure_a(input_dir, output_dir)
    plot_figure_b(input_dir, output_dir)
    plot_figure_c(input_dir, output_dir)
    plot_figure_d(input_dir, output_dir)

    print(f"[DONE] Figures saved to: {output_dir}")
    print("Saved files:")
    for stem in [
        "Figure_A_distribution_preservation",
        "Figure_B_in_target_consistency",
        "Figure_C_global_ranking_preservation",
        "Figure_D_target_level_scatter",
    ]:
        print(f"  - {output_dir / (stem + '.png')}")
        print(f"  - {output_dir / (stem + '.pdf')}")


if __name__ == "__main__":
    main()


'''

python /root/autodl-tmp/Peptide_3D/results/1_OT/2_property_consistency_plot.py \
  --input_dir /root/autodl-tmp/Peptide_3D/results/1_OT/2_property_consistency_analysis \
  --output_dir /root/autodl-tmp/Peptide_3D/results/1_OT/2_property_consistency_analysis

'''
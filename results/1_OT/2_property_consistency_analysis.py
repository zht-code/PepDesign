#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Affinity consistency analysis between original and augmented peptide datasets
Only analyze docking affinity.

Output
------
1 affinity_distribution.png/pdf
2 affinity_scatter_regression.png/pdf
3 affinity_rank_percentile.png/pdf
4 summary_metrics.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Global style (Nature-like)
# =========================================================

COLOR_ORIG = "#4C72B0"   # muted blue
COLOR_AUG = "#DD8452"    # muted orange
COLOR_REF = "#8C8C8C"    # neutral gray
COLOR_REG = "#2F5D9F"    # darker blue for regression

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================================================
# JSON parsing
# =========================================================

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_score(v):
    if isinstance(v, (float, int)):
        return float(v)

    if isinstance(v, dict):
        if "score" in v:
            return float(v["score"])

    return None


def infer_target(key):
    key = str(key)

    m = re.fullmatch(r"(.+?)_(\d+)", key)
    if m:
        return m.group(1)

    m = re.search(r"/train_data/([^/]+)/", key)
    if m:
        return m.group(1)

    m = re.search(r"/([^/]+)/cands/", key)
    if m:
        return m.group(1)

    if re.fullmatch(r"[A-Za-z0-9]+", key):
        return key

    return None


def parse_json(path, mode):
    data = read_json(path)
    rows = []

    for k, v in data.items():
        score = extract_score(v)
        if score is None:
            continue

        target = infer_target(k)
        if target is None:
            continue

        rows.append(
            {
                "target_id": target,
                "score": score,
                "source": mode,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# Metrics
# =========================================================

def wasserstein(x, y):
    x = np.sort(x)
    y = np.sort(y)

    n = max(len(x), len(y))
    q = np.linspace(0, 1, n)

    xq = np.quantile(x, q)
    yq = np.quantile(y, q)

    return np.mean(np.abs(xq - yq))


# =========================================================
# Utility
# =========================================================

def filter_for_plot(arr, min_affinity=-500):
    """Only keep values >= min_affinity for visualization."""
    arr = np.asarray(arr, dtype=float)
    return arr[arr >= min_affinity]


def clean_axis(ax):
    """Remove top/right spines for a clean journal-style figure."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_line_xy(line):
    """Extract x/y data from a matplotlib line object."""
    return line.get_xdata(), line.get_ydata()


# =========================================================
# Plot functions
# =========================================================

def plot_distribution(orig, aug, outdir, min_affinity=-500):
    """
    Plot density curves with filled area, similar to the example figure.
    """
    orig_plot = filter_for_plot(orig, min_affinity)
    aug_plot = filter_for_plot(aug, min_affinity)

    if len(orig_plot) < 2 or len(aug_plot) < 2:
        print("[Warning] Not enough points to draw KDE-like distribution.")
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    # Use pandas KDE (requires scipy in backend; usually available in scientific envs)
    line1 = pd.Series(orig_plot).plot(
        kind="kde",
        ax=ax,
        color=COLOR_ORIG,
        linewidth=2.2,
        label="Original"
    )

    line2 = pd.Series(aug_plot).plot(
        kind="kde",
        ax=ax,
        color=COLOR_AUG,
        linewidth=2.2,
        label="Augmented"
    )

    # Fill under curves
    lines = ax.get_lines()
    if len(lines) >= 2:
        x1, y1 = get_line_xy(lines[0])
        x2, y2 = get_line_xy(lines[1])

        ax.fill_between(x1, y1, color=COLOR_ORIG, alpha=0.18)
        ax.fill_between(x2, y2, color=COLOR_AUG, alpha=0.18)

    x_max = max(np.max(orig_plot), np.max(aug_plot))
    ax.set_xlim(min_affinity, x_max + 5)

    ax.set_xlabel("Docking Affinity Score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper left")

    clean_axis(ax)
    plt.tight_layout()
    plt.savefig(outdir / "affinity_distribution.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "affinity_distribution.pdf", bbox_inches="tight")
    plt.close()


def plot_scatter(orig_target, aug_target, outdir, min_affinity=-500):
    merged = pd.merge(orig_target, aug_target, on="target_id")

    merged_plot = merged[
        (merged["orig"] >= min_affinity) & (merged["aug"] >= min_affinity)
    ].copy()

    if len(merged_plot) == 0:
        print("[Warning] No scatter points remain after filtering.")
        return

    x = merged_plot["orig"].values
    y = merged_plot["aug"].values

    COLOR_POINT = "#7A8FB8"   # paired observations
    COLOR_REG = "#4C72B0"     # original-style blue
    COLOR_REF = "#DD8452"     # augmented-style orange

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    ax.scatter(
        x, y,
        s=34,
        alpha=0.55,
        color=COLOR_POINT,
        edgecolor=COLOR_POINT,
        label="Paired targets"
    )

    xy_min = min(min(x), min(y))
    xy_max = max(max(x), max(y))

    if len(merged_plot) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(xy_min, xy_max, 200)
        ax.plot(xs, m * xs + b, color=COLOR_REG, linewidth=2.2, label="Regression")

    ax.plot(
        [xy_min, xy_max], [xy_min, xy_max],
        linestyle="--",
        color=COLOR_REF,
        linewidth=2.0,
        label="y = x"
    )

    ax.set_xlabel("Original Affinity")
    ax.set_ylabel("Augmented Best Affinity")
    ax.set_xlim(min_affinity, xy_max + 5)
    ax.set_ylim(min_affinity, xy_max + 5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    plt.savefig(outdir / "affinity_scatter_regression.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "affinity_scatter_regression.pdf", bbox_inches="tight")
    plt.close()


def plot_rank_percentile(percentiles, outdir):
    percentiles = np.asarray(percentiles, dtype=float)

    if len(percentiles) == 0:
        print("[Warning] No percentiles available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    ax.hist(
        percentiles,
        bins=30,
        color=COLOR_ORIG,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6
    )

    ax.set_xlabel("Original Sample Percentile Among Augmented Candidates")
    ax.set_ylabel("Frequency")

    clean_axis(ax)
    plt.tight_layout()
    plt.savefig(outdir / "affinity_rank_percentile.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "affinity_rank_percentile.pdf", bbox_inches="tight")
    plt.close()


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_affinity", required=True)
    parser.add_argument("--aug_affinity", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument(
        "--plot_min_affinity",
        type=float,
        default=-500.0,
        help="Only show affinity >= this threshold in plots. Default: -500"
    )

    parser.add_argument(
        "--filter_metrics",
        action="store_true",
        help="If set, summary metrics are also computed after filtering by plot_min_affinity."
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_orig = parse_json(args.original_affinity, "original")
    df_aug = parse_json(args.aug_affinity, "augmented")

    if len(df_orig) == 0:
        raise ValueError("No valid original affinity scores parsed.")
    if len(df_aug) == 0:
        raise ValueError("No valid augmented affinity scores parsed.")

    orig_scores = df_orig["score"].values
    aug_scores = df_aug["score"].values

    # =========================
    # Plot: distribution
    # =========================
    plot_distribution(orig_scores, aug_scores, outdir, min_affinity=args.plot_min_affinity)

    # =========================
    # Target level aggregation
    # =========================
    # original: mean score per target
    orig_target = df_orig.groupby("target_id")["score"].mean().reset_index()
    orig_target.columns = ["target_id", "orig"]

    # augmented: best (min) score per target
    aug_target = df_aug.groupby("target_id")["score"].min().reset_index()
    aug_target.columns = ["target_id", "aug"]

    plot_scatter(orig_target, aug_target, outdir, min_affinity=args.plot_min_affinity)

    # =========================
    # Percentile
    # =========================
    percentiles = []

    for t in orig_target["target_id"]:
        o = orig_target.loc[orig_target.target_id == t, "orig"].values[0]

        if o < args.plot_min_affinity:
            continue

        a = df_aug[df_aug.target_id == t]["score"].values
        a = a[a >= args.plot_min_affinity]

        if len(a) == 0:
            continue

        combined = np.concatenate([[o], a])

        # smaller score = better rank
        rank = np.sum(combined <= o)
        percentile = rank / len(combined)

        percentiles.append(percentile)

    plot_rank_percentile(percentiles, outdir)

    # =========================
    # Metrics
    # =========================
    if args.filter_metrics:
        orig_metric_scores = orig_scores[orig_scores >= args.plot_min_affinity]
        aug_metric_scores = aug_scores[aug_scores >= args.plot_min_affinity]
    else:
        orig_metric_scores = orig_scores
        aug_metric_scores = aug_scores

    metrics = {
        "plot_min_affinity": args.plot_min_affinity,
        "filter_metrics": args.filter_metrics,
        "original_count_total": int(len(orig_scores)),
        "augmented_count_total": int(len(aug_scores)),
        "original_count_plot": int(np.sum(orig_scores >= args.plot_min_affinity)),
        "augmented_count_plot": int(np.sum(aug_scores >= args.plot_min_affinity)),
        "original_mean": float(np.mean(orig_metric_scores)),
        "augmented_mean": float(np.mean(aug_metric_scores)),
        "EMD": float(wasserstein(orig_metric_scores, aug_metric_scores)),
    }

    with open(outdir / "summary_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()


'''

# python /root/autodl-tmp/Peptide_3D/results/1_OT/2_property_consistency_analysis.py \
#   --original_affinity /root/autodl-tmp/Peptide_3D/data/hdock_scores.json \
#   --original_solubility /root/autodl-tmp/Peptide_3D/data/original_solubility_scores.json \
#   --original_stability /root/autodl-tmp/Peptide_3D/data/original_stability_scores.json \
#   --aug_affinity /root/autodl-tmp/Peptide_3D/utils/Data_augmentation/train_data_augmentation_hdock_scores_filtered.json \
#   --aug_solubility /root/autodl-tmp/Peptide_3D/data/train_data_augmentation_solubility_scores_filtered.json \
#   --aug_stability /root/autodl-tmp/Peptide_3D/data/train_data_augmentation_stability_scores.json \
#   --outdir /root/autodl-tmp/Peptide_3D/results/2_property_consistency_analysis

python /root/autodl-tmp/Peptide_3D/results/1_OT/2_property_consistency_analysis.py \
  --original_affinity /root/autodl-tmp/Peptide_3D/data/hdock_scores.json \
  --aug_affinity /root/autodl-tmp/Peptide_3D/utils/Data_augmentation/train_data_augmentation_hdock_scores_filtered.json \
  --outdir /root/autodl-tmp/Peptide_3D/results/1_OT/2_affinity_consistency \
  --plot_min_affinity -500

'''
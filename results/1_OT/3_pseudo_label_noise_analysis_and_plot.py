#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pseudo-label noise estimation and plotting from 6 JSON files.

Input:
- original affinity json
- original stability json
- original solubility json
- augmented affinity json
- augmented stability json
- augmented solubility json

What this script does:
1. Parse original and augmented scores
2. Optionally drop rows with score < --score-floor (default -500) per property, then match augmented samples to original targets
3. Compute pseudo-label noise:
   - error = pseudo_score - original_score
   - abs_error
   - consistency metrics (MAE, RMSE, Pearson, Spearman)
4. Compute noise rate under multiple thresholds
5. Generate publication-style figures
6. Save figures as:
   - 300 dpi PNG
   - editable PDF for Adobe Illustrator

Outputs:
- consistency_metrics.json
- noise_rate_table.csv
- merged_affinity.csv
- merged_stability.csv
- merged_solubility.csv
- Figure_1_error_distribution.(png/pdf)
- Figure_2_original_vs_pseudo_scatter.(png/pdf)
- Figure_3_noise_rate_curves.(png/pdf)
- Figure_4_consistency_metrics.(png/pdf)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Style
# =========================================================
COLORS = {
    "affinity": "#4C78A8",   # muted blue
    "stability": "#F58518",  # muted orange
    "solubility": "#54A24B", # muted green
    "grid": "#D9D9D9",
    "text": "#333333",
    "spine": "#333333",
    "bg": "#FFFFFF",
    "diag": "#888888",
    "hist_fill": "#A0A0A0",
    "noise": "#E45756",
}

PROPERTY_ORDER = ["affinity", "stability", "solubility"]
PROPERTY_LABELS = {
    "affinity": "Affinity",
    "stability": "Stability",
    "solubility": "Solubility",
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
        ax.yaxis.grid(True, linestyle="-", alpha=0.35)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =========================================================
# JSON parsing
# =========================================================
def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_number(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def extract_score(value):
    if is_number(value):
        return float(value)

    if isinstance(value, dict):
        if "score" in value and is_number(value["score"]):
            return float(value["score"])
        for _, v in value.items():
            if is_number(v):
                return float(v)

    return None


def infer_target_and_sample_id(key: str, mode: str) -> Tuple[Optional[str], Optional[str]]:
    """
    mode: original / augmented
    Supports:
    - bare key: 1A1M
    - augmented key: 1A1M_2
    - path style: .../train_data/1A1M/cands/peptide.pdb
    - path style: .../train_data_augmentation/1A1M_2/...
    - path style: .../1A1M/cands/pep_02.pdb
    """
    key = str(key).strip()
    key_norm = key.replace("\\", "/")

    # augmented style: 1A1M_2
    m = re.fullmatch(r"(.+?)_(\d+)", key_norm)
    if m:
        return m.group(1), key_norm

    # original path: /train_data/<target>/
    m = re.search(r"/train_data/([^/]+)/", key_norm)
    if m:
        target_id = m.group(1)
        return target_id, target_id

    # original cands path
    m = re.search(r"/([^/]+)/cands/peptide\.pdb$", key_norm)
    if m:
        target_id = m.group(1)
        return target_id, target_id

    # augmented cands path
    m = re.search(r"/([^/]+)/cands/pep_(\d+)\.pdb$", key_norm)
    if m:
        target_id = m.group(1)
        sample_id = f"{target_id}_{int(m.group(2))}"
        return target_id, sample_id

    # train_data_augmentation path
    m = re.search(r"/train_data_augmentation[^/]*/([^/]+)/", key_norm)
    if m:
        sample_id = m.group(1)
        m2 = re.fullmatch(r"(.+?)_(\d+)", sample_id)
        if m2:
            return m2.group(1), sample_id

    # bare target id
    if re.fullmatch(r"[A-Za-z0-9]+", key_norm):
        return key_norm, key_norm

    # basename fallback
    base = os.path.basename(key_norm)
    stem = os.path.splitext(base)[0]
    if mode == "augmented":
        m = re.fullmatch(r"(.+?)_(\d+)", stem)
        if m:
            return m.group(1), stem

    return None, None


def parse_score_json(json_path: str, mode: str, property_name: str) -> pd.DataFrame:
    data = read_json(json_path)
    if not isinstance(data, dict):
        raise ValueError(f"{json_path} top-level must be a JSON object.")

    rows = []
    for key, value in data.items():
        score = extract_score(value)
        if score is None:
            continue

        target_id, sample_id = infer_target_and_sample_id(key, mode)
        if target_id is None:
            continue
        if sample_id is None:
            sample_id = target_id

        rows.append({
            "target_id": target_id,
            "sample_id": sample_id,
            "score": float(score),
            "property": property_name,
            "source_type": mode,
            "raw_key": str(key),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"No valid rows parsed from {json_path}")
    return df


def filter_score_floor(df: pd.DataFrame, floor: float) -> pd.DataFrame:
    """Drop rows with score < floor (e.g. extreme low affinity) before merge/stats/plots."""
    if df.empty:
        return df
    return df.loc[df["score"] >= floor].copy()


# =========================================================
# Ranking / correlation utilities
# =========================================================
def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values):
            if abs(values[order[j + 1]] - values[order[i]]) < 1e-12:
                j += 1
            else:
                break
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xv = x - x.mean()
    yv = y - y.mean()
    denom = math.sqrt(np.sum(xv ** 2) * np.sum(yv ** 2))
    if denom == 0:
        return float("nan")
    return float(np.sum(xv * yv) / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata_average(x), rankdata_average(y))


# =========================================================
# KDE without scipy
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
# Analysis
# =========================================================
def merge_original_and_augmented(df_orig: pd.DataFrame, df_aug: pd.DataFrame, property_name: str) -> pd.DataFrame:
    # original side: collapse by target_id in case there are duplicates
    orig_target = (
        df_orig.groupby("target_id", as_index=False)["score"]
        .mean()
        .rename(columns={"score": "original_score"})
    )

    merged = pd.merge(df_aug, orig_target, on="target_id", how="inner")
    merged = merged.rename(columns={"score": "pseudo_score"})
    merged["error"] = merged["pseudo_score"] - merged["original_score"]
    merged["abs_error"] = merged["error"].abs()
    merged["property"] = property_name
    return merged


def compute_consistency_metrics(df: pd.DataFrame) -> Dict[str, float]:
    x = df["original_score"].to_numpy(dtype=float)
    y = df["pseudo_score"].to_numpy(dtype=float)
    err = df["error"].to_numpy(dtype=float)
    abs_err = df["abs_error"].to_numpy(dtype=float)

    return {
        "n_pairs": int(len(df)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias_mean_error": float(np.mean(err)),
        "pearson_r": pearson_corr(x, y),
        "spearman_r": spearman_corr(x, y),
        "original_mean": float(np.mean(x)),
        "pseudo_mean": float(np.mean(y)),
        "original_std": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "pseudo_std": float(np.std(y, ddof=1)) if len(y) > 1 else float("nan"),
    }


def build_noise_rate_table(
    merged_map: Dict[str, pd.DataFrame],
    thresholds_map: Dict[str, List[float]]
) -> pd.DataFrame:
    rows = []
    for prop, df in merged_map.items():
        abs_err = df["abs_error"].to_numpy(dtype=float)
        for thr in thresholds_map[prop]:
            rate = float(np.mean(abs_err > thr)) if len(abs_err) else float("nan")
            rows.append({
                "property": prop,
                "threshold": float(thr),
                "noise_rate": rate,
                "n_pairs": int(len(abs_err)),
            })
    return pd.DataFrame(rows)


# =========================================================
# Plotting
# =========================================================
def plot_error_distribution(merged_map: Dict[str, pd.DataFrame], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax, prop in zip(axes, PROPERTY_ORDER):
        df = merged_map[prop]
        err = df["error"].dropna().to_numpy(dtype=float)

        ax.hist(
            err,
            bins=30,
            density=True,
            color=COLORS[prop],
            alpha=0.30,
            edgecolor="white",
            linewidth=0.6,
        )

        gx, dx = kde_curve(err)
        if len(gx):
            ax.plot(gx, dx, color=COLORS[prop], lw=2.0)
            ax.fill_between(gx, 0, dx, color=COLORS[prop], alpha=0.15)

        ax.axvline(0, color=COLORS["diag"], lw=1.1, ls="--")
        ax.set_title(PROPERTY_LABELS[prop])
        ax.set_xlabel("Pseudo - original")
        ax.set_ylabel("Density")
        beautify_ax(ax)

    fig.suptitle("Figure 1. Pseudo-label error distributions", y=1.02, fontsize=13)
    save_figure(fig, outdir / "Figure_1_error_distribution")


def plot_original_vs_pseudo_scatter(merged_map: Dict[str, pd.DataFrame], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax, prop in zip(axes, PROPERTY_ORDER):
        df = merged_map[prop]
        x = df["original_score"].to_numpy(dtype=float)
        y = df["pseudo_score"].to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        ax.scatter(
            x, y,
            s=18,
            alpha=0.65,
            color=COLORS[prop],
            edgecolors="white",
            linewidths=0.3,
        )

        if len(x):
            lo = min(np.min(x), np.min(y))
            hi = max(np.max(x), np.max(y))
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            ax.plot(
                [lo - pad, hi + pad],
                [lo - pad, hi + pad],
                ls="--",
                lw=1.1,
                color=COLORS["diag"],
            )
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)

        ax.set_title(PROPERTY_LABELS[prop])
        ax.set_xlabel("Original score")
        ax.set_ylabel("Pseudo score")
        beautify_ax(ax)

    fig.suptitle("Figure 2. Original vs pseudo labels", y=1.02, fontsize=13)
    save_figure(fig, outdir / "Figure_2_original_vs_pseudo_scatter")


def plot_noise_rate_curves(noise_df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    for prop in PROPERTY_ORDER:
        sub = noise_df[noise_df["property"] == prop].sort_values("threshold")
        ax.plot(
            sub["threshold"],
            sub["noise_rate"],
            marker="o",
            ms=4.5,
            lw=2.0,
            color=COLORS[prop],
            label=PROPERTY_LABELS[prop],
        )

    ax.set_xlabel("Absolute-error threshold")
    ax.set_ylabel("Noise rate")
    ax.set_title("Figure 3. Noise rate curves")
    ax.set_ylim(0, 1.0)
    beautify_ax(ax)
    ax.legend()
    save_figure(fig, outdir / "Figure_3_noise_rate_curves")


def plot_consistency_metrics(metrics: Dict[str, Dict[str, float]], outdir: Path) -> None:
    """
    Plot MAE / RMSE / Pearson / Spearman in a 2x2 panel.
    """
    metric_keys = ["mae", "rmse", "pearson_r", "spearman_r"]
    metric_titles = {
        "mae": "MAE",
        "rmse": "RMSE",
        "pearson_r": "Pearson r",
        "spearman_r": "Spearman r",
    }

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
    axes = axes.flatten()

    x = np.arange(len(PROPERTY_ORDER))

    for ax, mkey in zip(axes, metric_keys):
        vals = [metrics[prop][mkey] for prop in PROPERTY_ORDER]
        ax.bar(
            x,
            vals,
            color=[COLORS[p] for p in PROPERTY_ORDER],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([PROPERTY_LABELS[p] for p in PROPERTY_ORDER], rotation=20, ha="right")
        ax.set_title(metric_titles[mkey])

        if mkey in {"pearson_r", "spearman_r"}:
            ax.set_ylim(-1.0, 1.0)
            ax.axhline(0, color=COLORS["diag"], lw=1.0, alpha=0.7)

        beautify_ax(ax)

    fig.suptitle("Figure 4. Consistency metrics", y=1.01, fontsize=13)
    save_figure(fig, outdir / "Figure_4_consistency_metrics")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Pseudo-label noise estimation and plotting from 6 JSON files.")
    parser.add_argument("--original_affinity", required=True)
    parser.add_argument("--original_stability", required=True)
    parser.add_argument("--original_solubility", required=True)
    parser.add_argument("--aug_affinity", required=True)
    parser.add_argument("--aug_stability", required=True)
    parser.add_argument("--aug_solubility", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--score-floor",
        type=float,
        default=-500.0,
        help="Exclude rows with score below this value before merge, metrics, and plots (default: -500)",
    )
    args = parser.parse_args()

    set_pub_style()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    original_jsons = {
        "affinity": args.original_affinity,
        "stability": args.original_stability,
        "solubility": args.original_solubility,
    }
    aug_jsons = {
        "affinity": args.aug_affinity,
        "stability": args.aug_stability,
        "solubility": args.aug_solubility,
    }

    merged_map: Dict[str, pd.DataFrame] = {}
    metrics_map: Dict[str, Dict[str, float]] = {}

    for prop in PROPERTY_ORDER:
        df_orig = filter_score_floor(
            parse_score_json(original_jsons[prop], mode="original", property_name=prop),
            args.score_floor,
        )
        df_aug = filter_score_floor(
            parse_score_json(aug_jsons[prop], mode="augmented", property_name=prop),
            args.score_floor,
        )
        if len(df_orig) == 0:
            raise RuntimeError(
                f"No original rows left for {prop} after score floor (>={args.score_floor})."
            )
        if len(df_aug) == 0:
            raise RuntimeError(
                f"No augmented rows left for {prop} after score floor (>={args.score_floor})."
            )

        merged = merge_original_and_augmented(df_orig, df_aug, prop)
        if len(merged) == 0:
            raise RuntimeError(f"No merged pairs found for {prop}. Please inspect key formats in your JSON files.")

        merged.to_csv(outdir / f"merged_{prop}.csv", index=False)
        merged_map[prop] = merged
        metrics_map[prop] = compute_consistency_metrics(merged)

    # Thresholds: you can tune these if needed
    thresholds_map = {
        "affinity": [5, 10, 20, 30, 50, 80, 100],
        "stability": [1, 2, 5, 10, 15, 20, 30],
        "solubility": [0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
    }

    noise_df = build_noise_rate_table(merged_map, thresholds_map)
    noise_df.to_csv(outdir / "noise_rate_table.csv", index=False)

    with open(outdir / "consistency_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_map, f, indent=2, ensure_ascii=False)

    # Plot
    plot_error_distribution(merged_map, outdir)
    plot_original_vs_pseudo_scatter(merged_map, outdir)
    plot_noise_rate_curves(noise_df, outdir)
    plot_consistency_metrics(metrics_map, outdir)

    print(f"[DONE] Outputs saved to: {outdir}")
    print("Generated files:")
    for name in [
        "consistency_metrics.json",
        "noise_rate_table.csv",
        "merged_affinity.csv",
        "merged_stability.csv",
        "merged_solubility.csv",
        "Figure_1_error_distribution.png",
        "Figure_1_error_distribution.pdf",
        "Figure_2_original_vs_pseudo_scatter.png",
        "Figure_2_original_vs_pseudo_scatter.pdf",
        "Figure_3_noise_rate_curves.png",
        "Figure_3_noise_rate_curves.pdf",
        "Figure_4_consistency_metrics.png",
        "Figure_4_consistency_metrics.pdf",
    ]:
        print("  -", outdir / name)

    print("\nNote:")
    print("  当前只基于 6 个属性 JSON 做伪标签噪声与一致性分析。")
    print("  真正的 confidence binning 还需要额外的 OT weight / transport confidence 文件。")


if __name__ == "__main__":
    main()




'''

python /root/autodl-tmp/Peptide_3D/results/1_OT/3_pseudo_label_noise_analysis_and_plot.py \
  --original_affinity /root/autodl-tmp/Peptide_3D/data/hdock_scores.json \
  --original_stability /root/autodl-tmp/Peptide_3D/data/original_stability_scores.json \
  --original_solubility /root/autodl-tmp/Peptide_3D/data/original_solubility_scores.json \
  --aug_affinity /root/autodl-tmp/Peptide_3D/utils/Data_augmentation/train_data_augmentation_hdock_scores_filtered.json \
  --aug_stability /root/autodl-tmp/Peptide_3D/data/train_data_augmentation_stability_scores.json \
  --aug_solubility /root/autodl-tmp/Peptide_3D/data/train_data_augmentation_solubility_scores_filtered.json \
  --outdir /root/autodl-tmp/Peptide_3D/results/1_OT/3_pseudo_label_noise_figures



'''
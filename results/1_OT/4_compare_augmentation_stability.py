#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare OT vs Random Neighbor vs Sequence Perturbation
in terms of stability (affinity-based)

Usage:
python compare_augmentation_stability.py \
  --original xxx.json \
  --ot xxx.json \
  --random xxx.json \
  --seq xxx.json \
  --outdir results_compare
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, ks_2samp, entropy

# Make PDF text editable in Illustrator.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# =========================
# JSON parsing
# =========================

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_score(v):
    if isinstance(v, (float, int)):
        return float(v)
    if isinstance(v, dict) and "score" in v:
        s = v["score"]
        if s is None:
            return None
        return float(s)
    return None


def infer_target(key):
    key = str(key)
    base = key.split("/")[-1]

    m = re.fullmatch(r"([A-Za-z0-9]+)_nb_[A-Za-z0-9]+", base)
    if m:
        return m.group(1)

    m = re.match(r"^([A-Za-z0-9]+)_aug_", base)
    if m:
        return m.group(1)

    m = re.fullmatch(r"(.+?)_(\d+)", base)
    if m:
        return m.group(1)

    m = re.search(r"/([^/]+)/cands/", key)
    if m:
        return m.group(1)

    if re.fullmatch(r"[A-Za-z0-9]+", base):
        return base

    return None


def parse_json(path):
    data = read_json(path)

    rows = []
    for k, v in data.items():
        score = extract_score(v)
        target = infer_target(k)

        if score is None or target is None:
            continue

        rows.append({"target": target, "score": score})

    return pd.DataFrame(rows, columns=["target", "score"])


def filter_score_floor(df, floor):
    """Drop rows with score < floor (extreme low affinity outliers)."""
    if df.empty:
        return df
    return df.loc[df["score"] >= floor].copy()


# =========================
# Metrics
# =========================

def wasserstein(x, y):
    x = np.sort(x)
    y = np.sort(y)

    n = max(len(x), len(y))
    q = np.linspace(0, 1, n)

    xq = np.quantile(x, q)
    yq = np.quantile(y, q)

    return np.mean(np.abs(xq - yq))


def js_divergence(x, y, bins=50):
    px, _ = np.histogram(x, bins=bins, density=True)
    py, _ = np.histogram(y, bins=bins, density=True)

    px += 1e-8
    py += 1e-8

    m = 0.5 * (px + py)
    return 0.5 * entropy(px, m) + 0.5 * entropy(py, m)


# =========================
# Target-level consistency
# =========================

def target_level(df):
    return df.groupby("target")["score"].mean().reset_index()


def best_level(df):
    return df.groupby("target")["score"].min().reset_index()


def consistency_metrics(orig, aug):
    merged = pd.merge(orig, aug, on="target")

    x = merged["score_x"]
    y = merged["score_y"]

    pearson = pearsonr(x, y)[0]
    spearman = spearmanr(x, y)[0]

    mae = np.mean(np.abs(x - y))
    rmse = np.sqrt(np.mean((x - y) ** 2))

    return pearson, spearman, mae, rmse


# =========================
# Percentile / hit@k
# =========================

def compute_percentiles(orig_df, aug_df):
    orig_target = target_level(orig_df)

    percentiles = []

    for t in orig_target["target"]:
        o = orig_target.loc[orig_target.target == t, "score"].values[0]
        a = aug_df[aug_df.target == t]["score"].values

        if len(a) == 0:
            continue

        combined = np.concatenate([[o], a])
        rank = np.sum(combined <= o)
        percentile = rank / len(combined)

        percentiles.append(percentile)

    return np.array(percentiles)


def hit_k(percentiles, k_ratio):
    return np.mean(percentiles >= (1 - k_ratio))


# =========================
# Global ranking
# =========================

def rank_correlation(orig_df, aug_df):
    o = target_level(orig_df)
    a = best_level(aug_df)

    merged = pd.merge(o, a, on="target")

    return spearmanr(merged["score_x"], merged["score_y"])[0]


# =========================
# Plot
# =========================

COLOR = {
    "orig": "#4C72B0",
    "ot": "#DD8452",
    "random": "#55A868",
    "seq": "#C44E52",
}


def plot_distribution(orig, others, outdir):
    plt.figure(figsize=(6,5))

    for name, data in others.items():
        plt.hist(data, bins=50, density=True, alpha=0.4,
                 label=name, color=COLOR[name])

    plt.hist(orig, bins=50, density=True,
             histtype="step", linewidth=2,
             label="original", color=COLOR["orig"])

    plt.legend()
    plt.xlabel("Affinity")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig(outdir / "distribution.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "distribution.pdf", format="pdf", bbox_inches="tight")
    plt.close()


def plot_scatter(orig, aug_dict, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    for ax, (name, aug) in zip(axes, aug_dict.items()):
        merged = pd.merge(orig, aug, on="target")

        x = merged["score_x"]
        y = merged["score_y"]

        ax.scatter(x, y, alpha=0.4, s=10, color=COLOR[name])

        ax.plot([x.min(), x.max()], [x.min(), x.max()], '--')

        ax.set_title(name)
        ax.set_xlabel("Original")
        ax.set_ylabel("Aug")

    plt.tight_layout()
    plt.savefig(outdir / "scatter.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "scatter.pdf", format="pdf", bbox_inches="tight")
    plt.close()
def plot_percentile(percentile_dict, outdir):
    plt.figure(figsize=(6,5))

    for name, perc in percentile_dict.items():
        plt.hist(perc, bins=30, alpha=0.5,
                 label=name, color=COLOR[name])

    plt.xlabel("Percentile of original sample in augmented set")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig(outdir / "percentile.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "percentile.pdf", format="pdf", bbox_inches="tight")
    plt.close()
def plot_ranking(results, outdir):
    methods = list(results.keys())

    rank_corr = [results[m]["RankCorr"] for m in methods]
    hit1 = [results[m]["Hit@1"] for m in methods]
    hit5 = [results[m]["Hit@5"] for m in methods]

    x = np.arange(len(methods))

    plt.figure(figsize=(8,5))

    method_colors = [COLOR[m] for m in methods]
    plt.bar(
        x - 0.2, rank_corr, width=0.2, label="RankCorr",
        color=method_colors, hatch="//", alpha=0.95,
        edgecolor="white", linewidth=0.8
    )
    plt.bar(
        x, hit1, width=0.2, label="Hit@1",
        color=method_colors, hatch="..", alpha=0.85,
        edgecolor="white", linewidth=0.8
    )
    plt.bar(
        x + 0.2, hit5, width=0.2, label="Hit@5",
        color=method_colors, hatch="xx", alpha=0.75,
        edgecolor="white", linewidth=0.8
    )

    plt.xticks(x, methods)
    plt.ylabel("Score")
    plt.title("Ranking Preservation Comparison")

    plt.legend()

    plt.tight_layout()
    plt.savefig(outdir / "ranking.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "ranking.pdf", format="pdf", bbox_inches="tight")
    plt.close()
# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--original", required=True)
    parser.add_argument("--ot", required=True)
    parser.add_argument("--random", required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--score-floor",
        type=float,
        default=-500.0,
        help="Exclude rows with score below this value from metrics and plots (default: -500)",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # load
    orig = filter_score_floor(parse_json(args.original), args.score_floor)
    ot = filter_score_floor(parse_json(args.ot), args.score_floor)
    rnd = filter_score_floor(parse_json(args.random), args.score_floor)
    seq = filter_score_floor(parse_json(args.seq), args.score_floor)

    results = {}

    for name, df in {
        "ot": ot,
        "random": rnd,
        "seq": seq
    }.items():

        results[name] = {}

        # distribution
        results[name]["EMD"] = wasserstein(orig.score, df.score)
        results[name]["JS"] = js_divergence(orig.score, df.score)
        results[name]["KS"] = ks_2samp(orig.score, df.score)[0]

        # consistency
        p, s, mae, rmse = consistency_metrics(
            target_level(orig), best_level(df)
        )
        results[name]["Pearson"] = p
        results[name]["Spearman"] = s
        results[name]["MAE"] = mae
        results[name]["RMSE"] = rmse

        # percentile
        perc = compute_percentiles(orig, df)
        results[name]["Hit@1"] = hit_k(perc, 0.01)
        results[name]["Hit@5"] = hit_k(perc, 0.05)

        # ranking
        results[name]["RankCorr"] = rank_correlation(orig, df)

    # save table
    df_res = pd.DataFrame(results).T
    df_res.to_csv(outdir / "summary.csv")

    # plots
    plot_distribution(orig.score,
                      {"ot": ot.score, "random": rnd.score, "seq": seq.score},
                      outdir)

    plot_scatter(target_level(orig),
                 {"ot": best_level(ot),
                  "random": best_level(rnd),
                  "seq": best_level(seq)},
                 outdir)

    print(df_res)
    percentile_dict = {}

    for name, df in {
        "ot": ot,
        "random": rnd,
        "seq": seq
    }.items():

        perc = compute_percentiles(orig, df)
        percentile_dict[name] = perc
    plot_percentile(percentile_dict, outdir)
    plot_ranking(results, outdir)

if __name__ == "__main__":
    main()


'''

python /root/autodl-tmp/Peptide_3D/results/1_OT/4_compare_augmentation_stability.py \
  --original /root/autodl-tmp/Peptide_3D/data/hdock_scores.json \
  --ot /root/autodl-tmp/Peptide_3D/utils/Data_augmentation/train_data_augmentation_hdock_scores_filtered.json \
  --random /root/autodl-tmp/Peptide_3D/data/train_data_augmented_random_neighbor_hdock.json \
  --seq /root/autodl-tmp/Peptide_3D/data/train_data_seq_aug_perturbation_hdock.json \
  --outdir /root/autodl-tmp/Peptide_3D/results/1_OT/4_compare_aug


'''
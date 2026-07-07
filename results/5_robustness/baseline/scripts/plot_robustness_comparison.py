#!/usr/bin/env python3
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


BASE_DIR = Path("/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline").resolve()
ROB_ROOT = Path("/root/autodl-tmp/Peptide_3D/results/5_robustness").resolve()
FIG_DIR = ROB_ROOT / "figures"
TAB_DIR = BASE_DIR / "tables"


COLORS = {
    "ours": "#1f4e79",
    "rfdiffusion": "#8c6d31",
    "proteingenerator": "#2a7f62",
    "bindcraft": "#b24a3a",
}
LABELS = {
    "ours": "Ours",
    "rfdiffusion": "RFdiffusion",
    "proteingenerator": "ProteinGenerator",
    "bindcraft": "BindCraft",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "figure.dpi": 180,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ours_agg = pd.read_csv(ROB_ROOT / "tables" / "robustness_aggregate_by_condition.csv").rename(
        columns={"perturb_type": "perturbation_type"}
    )
    ours_agg["method"] = "ours"
    ours_summary = pd.read_csv(ROB_ROOT / "tables" / "Table_5_robustness_summary.csv")
    ours_summary["method"] = "ours"

    baseline_agg_path = TAB_DIR / "baseline_robustness_aggregate_by_condition.csv"
    baseline_summary_path = TAB_DIR / "all_methods_robustness_summary_all_available.csv"
    baseline_index_path = TAB_DIR / "baseline_best_candidates.csv"
    baseline_agg = pd.read_csv(baseline_agg_path) if baseline_agg_path.is_file() else pd.DataFrame()
    baseline_summary = pd.read_csv(baseline_summary_path) if baseline_summary_path.is_file() else pd.DataFrame()
    baseline_index = pd.read_csv(baseline_index_path) if baseline_index_path.is_file() else pd.DataFrame()

    agg = pd.concat([ours_agg, baseline_agg], ignore_index=True, sort=False)
    if "method" in baseline_summary.columns:
        baseline_summary = baseline_summary[baseline_summary["method"] != "ours"]
    else:
        baseline_summary = baseline_summary.iloc[0:0].copy()
    summary = pd.concat([ours_summary, baseline_summary], ignore_index=True, sort=False)
    return agg, summary, baseline_index


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.set_title(label, loc="left", fontweight="bold", fontsize=10)


def draw_schematic(ax: plt.Axes) -> None:
    add_panel_label(ax, "a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    titles = [
        ("structure_missing", "Structure\nmissing", "#4C72B0", ["0%", "10%", "20%", "30%", "40%"]),
        ("pocket_noise", "Pocket\nnoise", "#55A868", ["0", "0.5", "1.0", "1.5", "2.0 A"]),
        ("sequence_trunc", "Sequence\ntruncation", "#C44E52", ["0%", "10%", "20%", "30%", "40%"]),
    ]
    for idx, (_, title, color, levels) in enumerate(titles):
        x0 = 0.06 + idx * 0.31
        rect = mpatches.FancyBboxPatch((x0, 0.38), 0.26, 0.32, boxstyle="round,pad=0.02", edgecolor=color, facecolor="#f7f7f7")
        ax.add_patch(rect)
        ax.text(x0 + 0.13, 0.77, title, ha="center", va="center", color=color, fontsize=8)
        for j, lvl in enumerate(levels):
            ax.scatter(x0 + 0.04 + j * 0.05, 0.50, s=18 + j * 6, color=color, alpha=0.35 + 0.12 * j)
            ax.text(x0 + 0.04 + j * 0.05, 0.29, lvl, ha="center", fontsize=6)
    ax.text(0.5, 0.12, "Target-side perturbations reused from the existing Chapter 5 robustness setup.", ha="center", fontsize=7)


def _normalize_affinity(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = sub["level_value"].astype(float).to_numpy()
    y = -sub["affinity_mean"].astype(float).to_numpy()
    y0 = y[0] if len(y) else 1.0
    return x, y / max(abs(y0), 1e-6)


def _metric_heatmap_table(agg: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    rows = []
    row_labels = []
    metrics = ["affinity_mean", "success_rate", "stability_mean", "solubility_mean"]
    methods = [m for m in ["ours", "rfdiffusion", "proteingenerator", "bindcraft"] if m in set(agg["method"])]
    for method in methods:
        for perturb in ["structure_missing", "pocket_noise", "sequence_trunc"]:
            sub = agg[(agg["method"] == method) & (agg["perturbation_type"] == perturb)].sort_values("level_value")
            if sub.empty:
                continue
            base = sub.iloc[0]
            row = []
            for metric in metrics:
                series = sub[metric].dropna()
                if metric not in sub.columns or series.empty or pd.isna(base.get(metric, np.nan)):
                    row.append(np.nan)
                    continue
                clean = float(base[metric])
                worst = float(series.iloc[-1])
                if metric == "affinity_mean":
                    clean = -clean
                    worst = -worst
                drop = (clean - worst) / max(abs(clean), 1e-6) * 100.0
                row.append(max(drop, 0.0))
            rows.append(row)
            row_labels.append(f"{LABELS.get(method, method)} | {perturb}")
    return np.array(rows, dtype=float), row_labels, ["Aff.", "Succ.", "Stab.", "Sol."]


def plot() -> None:
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    agg, summary, baseline_index = load_data()

    fig = plt.figure(figsize=(8.8, 11.5))
    gs = fig.add_gridspec(4, 2, hspace=0.48, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    draw_schematic(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "b")
    # Replace the cluttered overlay with three small multiples (one per perturbation),
    # each showing method curves only.
    ax_b.axis("off")
    perturbs = [
        ("structure_missing", "Structure missing"),
        ("pocket_noise", "Pocket noise"),
        ("sequence_trunc", "Sequence truncation"),
    ]
    methods = [m for m in ["ours", "rfdiffusion", "proteingenerator", "bindcraft"] if m in set(agg["method"])]
    sub_axes = []
    for i, (perturb, title) in enumerate(perturbs):
        # [x0, y0, w, h] in axis-relative coordinates
        inset = ax_b.inset_axes([0.02 + i * 0.325, 0.18, 0.30, 0.74])
        sub_axes.append(inset)
        inset.set_title(title, fontsize=8, pad=2)
        for method in methods:
            sub = agg[(agg["method"] == method) & (agg["perturbation_type"] == perturb)].sort_values("level_value")
            if sub.empty:
                continue
            x, y = _normalize_affinity(sub)
            inset.plot(
                x,
                y,
                marker="o",
                ms=3,
                lw=1.2,
                color=COLORS[method],
                label=LABELS[method],
            )
        inset.axhline(1.0, color="#999999", lw=0.6, ls="--")
        inset.set_xlabel("Strength", fontsize=7)
        if i == 0:
            inset.set_ylabel("Norm. affinity", fontsize=7)
        else:
            inset.set_yticklabels([])
        inset.grid(False)

    # Single shared legend (avoid duplicates).
    handles, labels = sub_axes[0].get_legend_handles_labels() if sub_axes else ([], [])
    if handles:
        ax_b.legend(handles, labels, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.02), ncol=2)

    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "c")
    arr, row_labels, col_labels = _metric_heatmap_table(agg)
    if arr.size:
        pcm = ax_c.pcolormesh(
            np.arange(arr.shape[1] + 1),
            np.arange(arr.shape[0] + 1),
            arr,
            cmap="magma",
            vmin=0,
            vmax=max(40.0, float(np.nanmax(arr))),
            shading="flat",
        )
        ax_c.set_xticks(np.arange(arr.shape[1]) + 0.5)
        ax_c.set_xticklabels(col_labels)
        ax_c.set_yticks(np.arange(arr.shape[0]) + 0.5)
        ax_c.set_yticklabels(row_labels, fontsize=6)
        plt.colorbar(pcm, ax=ax_c, fraction=0.046, pad=0.02, label="Worst-case drop (%)")
    else:
        ax_c.text(0.5, 0.5, "No baseline heatmap data", ha="center", va="center")

    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "d")
    sub = summary[summary["metric"] == "affinity_hdock"].copy()
    if not sub.empty:
        methods = [m for m in ["ours", "rfdiffusion", "proteingenerator", "bindcraft"] if m in set(sub["method"])]
        xpos = np.arange(len(methods))
        width = 0.22
        for idx, perturb in enumerate(["structure_missing", "pocket_noise", "sequence_trunc"]):
            vals = []
            for method in methods:
                row = sub[(sub["method"] == method) & (sub["perturbation_type"] == perturb)]
                vals.append(float(row["AUDC"].iloc[0]) if not row.empty else np.nan)
            ax_d.bar(xpos + (idx - 1) * width, vals, width=width, color=["#4C72B0", "#55A868", "#C44E52"][idx], label=perturb)
        ax_d.set_xticks(xpos)
        ax_d.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha="right")
        ax_d.set_ylabel("AUDC")
        ax_d.legend(frameon=False, loc="upper left")
    else:
        ax_d.text(0.5, 0.5, "No summary data", ha="center", va="center")

    for ax, perturb, label in [
        (fig.add_subplot(gs[2, 0]), "structure_missing", "e"),
        (fig.add_subplot(gs[2, 1]), "pocket_noise", "f"),
        (fig.add_subplot(gs[3, 0]), "sequence_trunc", "g"),
    ]:
        add_panel_label(ax, label)
        for method in [m for m in ["ours", "rfdiffusion", "proteingenerator", "bindcraft"] if m in set(agg["method"])]:
            sub = agg[(agg["method"] == method) & (agg["perturbation_type"] == perturb)].sort_values("level_value")
            if sub.empty:
                continue
            ax.plot(
                sub["level_value"].astype(float),
                -sub["affinity_mean"].astype(float),
                marker="o",
                ms=3,
                lw=1.2,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.set_xlabel("Perturbation strength")
        ax.set_ylabel("-HDOCK (higher better)")
        ax.legend(frameon=False, loc="best")

    ax_h = fig.add_subplot(gs[3, 1])
    add_panel_label(ax_h, "h")
    ax_h.axis("off")
    text_lines = [
        "Representative cases",
        "",
        f"Ours targets: 133",
    ]
    if not baseline_index.empty:
        for method in ["rfdiffusion", "proteingenerator", "bindcraft"]:
            sub = baseline_index[baseline_index["method"] == method]
            text_lines.append(f"{LABELS[method]} resolved targets: {int(sub['target_id'].nunique())}")
    else:
        text_lines.append("No baseline candidate index found.")
    text_lines.append("")
    text_lines.append("This panel is exported as a quantitative case summary because complete matched structure assets are not available for every baseline on the current machine.")
    ax_h.text(0.02, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=8)

    png_path = FIG_DIR / "Figure_5_robustness_comparison.png"
    pdf_path = FIG_DIR / "Figure_5_robustness_comparison.pdf"
    svg_path = FIG_DIR / "Figure_5_robustness_comparison.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)

    caption = textwrap.dedent(
        """
        Figure 5 | Robustness comparison across our method and available baselines.
        a, Schematic of the three target-side perturbation families reused from the existing Chapter 5 robustness setup: structure_missing, pocket_noise, and sequence_trunc, each evaluated across five aligned severity levels.
        b, Global normalized affinity degradation curves comparing methods across the three perturbation families.
        c, Heatmap of worst-case performance drop across affinity_hdock, success_rate, stability, and solubility, organized by method and perturbation family.
        d, Robustness summary panel showing AUDC for affinity under each perturbation family.
        e, Fine-grained structure_missing degradation curves.
        f, Fine-grained pocket_noise degradation curves.
        g, Fine-grained sequence_trunc degradation curves.
        h, Representative-case summary. On the current machine, complete matched structure assets were not available for every baseline, so this panel reports resolved-target coverage and quantitative case notes rather than a full structural montage.
        Comparison logic: our previously completed robustness outputs are reused directly from `results/5_robustness/tables`, while baseline robustness is recomputed only from already generated baseline peptide structures that could be located locally. Missing baseline assets are logged explicitly and excluded from quantitative aggregation without interrupting the rest of the pipeline.
        """
    ).strip()
    (FIG_DIR / "Figure_5_robustness_comparison_caption.txt").write_text(caption + "\n", encoding="utf-8")


if __name__ == "__main__":
    plot()

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

METHOD_ORDER = ["ours", "unconditional", "rfdiffusion", "bindcraft", "proteingenerator"]
METHOD_LABELS = {
    "ours": "Ours",
    "unconditional": "Unconditional",
    "rfdiffusion": "RFdiffusion",
    "bindcraft": "BindCraft",
    "proteingenerator": "ProteinGenerator",
}
DATASET_ORDER = ["protein_level_test", "family_level_test"]
DATASET_LABELS = {
    "protein_level_test": "Protein-level test",
    "family_level_test": "Family-level test",
}
HDOCK_SCATTER_MIN_SCORE = -500.0
HDOCK_NEAR_ZERO_THRESHOLD = 0.05
TRAIN_SIMILARITY_COLORBAR_RANGE = (0.0, 1.0)
NATURE_SOFT_MONO_PALETTE = {
    "ours": "#92B1D9",
    "unconditional": "#C1D8E9",
    "rfdiffusion": "#DBDDEF",
    "bindcraft": "#F6C8B6",
    "proteingenerator": "#D4D4D4",
}
NATURE_SOFT_EDGE = "#5F8397"


def setup_publication_style() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif", "serif"]
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["savefig.bbox"] = "tight"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def method_colors() -> Dict[str, str]:
    return dict(NATURE_SOFT_MONO_PALETTE)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> Tuple[str, str]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return str(png_path), str(pdf_path)


def finite_series(values: Iterable) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def methods_with_data(df: pd.DataFrame, dataset: str, metric: str) -> List[str]:
    present = []
    sub = df[df["dataset"] == dataset]
    for method in METHOD_ORDER:
        vals = finite_series(sub.loc[sub["method"] == method, metric])
        if len(vals):
            present.append(method)
    return present


def add_optional_trendline(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) < 3 or len(np.unique(x)) < 2:
        return
    try:
        coeff = np.polyfit(x, y, deg=1)
    except np.linalg.LinAlgError:
        return
    xp = np.linspace(np.min(x), np.max(x), 100)
    yp = coeff[0] * xp + coeff[1]
    ax.plot(xp, yp, color="black", linewidth=1.2, alpha=0.85, label="Trend")


def spread_overlapping_x_values(x: np.ndarray, width: float = 0.018) -> np.ndarray:
    """
    Add deterministic horizontal jitter only for visualization.
    This prevents identical train-similarity values (mostly 0.0) from collapsing
    into a single vertical line while keeping points close to their true x value.
    """
    if len(x) == 0:
        return x

    x_plot = x.astype(float).copy()
    rounded = np.round(x_plot, 6)
    for value in np.unique(rounded):
        idx = np.where(rounded == value)[0]
        if len(idx) <= 1:
            continue
        offsets = np.linspace(-width, width, len(idx))
        x_plot[idx] = np.clip(x_plot[idx] + offsets, 0.0, 1.0)
    return x_plot


def hdock_scatter_xlim(x_true: np.ndarray, x_plot: np.ndarray) -> Tuple[float, float]:
    """
    When most similarities collapse near zero, zoom the x-axis around zero so the
    scatter cloud becomes visually separable instead of looking like a single line.
    """
    if len(x_true) == 0:
        return -0.02, 1.02

    near_zero_mask = x_true <= HDOCK_NEAR_ZERO_THRESHOLD
    near_zero_fraction = float(np.mean(near_zero_mask))
    if near_zero_fraction >= 0.8 and np.any(near_zero_mask):
        near_zero_max = float(np.max(x_plot[near_zero_mask]))
        return -0.02, max(0.03, near_zero_max + 0.01)

    return -0.02, 1.02


def plot_hdock_scatter(
    per_candidate_df: pd.DataFrame,
    dataset: str,
    method: str,
    output_dir: Path,
    dpi: int,
) -> Tuple[bool, List[str], str]:
    sub = per_candidate_df[(per_candidate_df["dataset"] == dataset) & (per_candidate_df["method"] == method)].copy()
    sub["train_similarity"] = pd.to_numeric(sub["train_similarity"], errors="coerce")
    sub["hdock_score"] = pd.to_numeric(sub["hdock_score"], errors="coerce")
    sub = sub[
        np.isfinite(sub["train_similarity"])
        & np.isfinite(sub["hdock_score"])
        & (sub["hdock_score"] >= HDOCK_SCATTER_MIN_SCORE)
    ]
    if sub.empty:
        return False, [], f"{dataset}/{method}: no finite train_similarity-hdock pairs after filtering hdock_score >= {HDOCK_SCATTER_MIN_SCORE}"

    x_true = sub["train_similarity"].to_numpy(dtype=float)
    x_plot = spread_overlapping_x_values(x_true)
    y_plot = sub["hdock_score"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    scatter = ax.scatter(
        x_plot,
        y_plot,
        c=x_true,
        cmap="viridis",
        vmin=TRAIN_SIMILARITY_COLORBAR_RANGE[0],
        vmax=TRAIN_SIMILARITY_COLORBAR_RANGE[1],
        s=24,
        alpha=0.78,
        edgecolors="none",
    )
    add_optional_trendline(ax, x_true, y_plot)
    ax.set_xlabel("Train similarity")
    ax.set_ylabel("HDOCK score")
    ax.set_title(f"{dataset_label(dataset)} | {method_label(method)}")
    ax.set_xlim(*hdock_scatter_xlim(x_true, x_plot))
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Train similarity")
    cbar.set_ticks(np.linspace(TRAIN_SIMILARITY_COLORBAR_RANGE[0], TRAIN_SIMILARITY_COLORBAR_RANGE[1], 6))
    stem = f"hdock_scatter_{dataset}_{method}"
    paths = list(save_figure(fig, output_dir, stem, dpi))
    return True, paths, f"{dataset}/{method}: {len(sub)} points"


def plot_boxplot_by_method(
    per_candidate_df: pd.DataFrame,
    dataset: str,
    metric: str,
    ylabel: str,
    output_dir: Path,
    dpi: int,
    stem_prefix: str,
) -> Tuple[bool, List[str], str]:
    sub = per_candidate_df[per_candidate_df["dataset"] == dataset]
    data = []
    labels = []
    colors = []
    palette = method_colors()
    for method in METHOD_ORDER:
        values = finite_series(sub.loc[sub["method"] == method, metric])
        if len(values) == 0:
            continue
        data.append(values)
        labels.append(method_label(method))
        colors.append(palette[method])

    if not data:
        return False, [], f"{dataset}/{metric}: no finite data"

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    box = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.6, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(NATURE_SOFT_EDGE)
        patch.set_linewidth(1.0)
        patch.set_alpha(0.82)
    for whisker in box["whiskers"]:
        whisker.set_color(NATURE_SOFT_EDGE)
        whisker.set_linewidth(1.0)
    for cap in box["caps"]:
        cap.set_color(NATURE_SOFT_EDGE)
        cap.set_linewidth(1.0)
    for median in box["medians"]:
        median.set_color("#3E5E70")
        median.set_linewidth(1.2)
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.set_title(dataset_label(dataset))
    ax.tick_params(axis="x", rotation=25)
    stem = f"{stem_prefix}_{dataset}"
    paths = list(save_figure(fig, output_dir, stem, dpi))
    return True, paths, f"{dataset}/{metric}: {len(data)} methods"


def zoomed_bar_ylim(metric: str, means: Sequence[float], stds: Sequence[float]) -> Tuple[float, float] | None:
    """Return a data-driven truncated y-axis for tightly clustered headline metrics.

    Limits include the full mean ± SD range. Fraction metrics retain a small margin
    above 1.0 so capped error bars remain visible.
    """
    if metric not in {"plddt", "ramachandran_compliance", "novelty"}:
        return None

    mean_arr = np.asarray(means, dtype=float)
    std_arr = np.asarray(stds, dtype=float)
    lower_data = float(np.min(mean_arr - std_arr))
    upper_data = float(np.max(mean_arr + std_arr))

    if metric == "plddt":
        lower = max(0.0, np.floor(lower_data / 5.0) * 5.0)
        upper = min(100.0, np.ceil(upper_data / 5.0) * 5.0)
        return float(lower), float(upper)

    lower = max(0.0, np.floor((lower_data - 0.01) / 0.05) * 0.05)
    upper = np.ceil((upper_data + 0.01) / 0.05) * 0.05
    return float(lower), float(upper)


def add_truncated_axis_marker(ax: plt.Axes) -> None:
    """Mark a non-zero y-axis baseline with a compact diagonal break symbol."""
    kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, linewidth=1.0)
    ax.plot((-0.018, 0.018), (-0.012, 0.012), **kwargs)
    ax.plot((-0.018, 0.018), (0.008, 0.032), **kwargs)


def plot_bar_from_aggregate(
    aggregate_df: pd.DataFrame,
    dataset: str,
    metric: str,
    ylabel: str,
    output_dir: Path,
    dpi: int,
    stem_prefix: str,
    skip_if_all_nan: bool = False,
) -> Tuple[bool, List[str], str]:
    sub = aggregate_df[aggregate_df["dataset"] == dataset]
    means = []
    stds = []
    labels = []
    colors = []
    palette = method_colors()

    # Error bars use mean ± std consistently across all bar charts.
    for method in METHOD_ORDER:
        method_sub = sub[sub["method"] == method]
        if method_sub.empty:
            continue
        mean_value = pd.to_numeric(method_sub[f"{metric}_mean"], errors="coerce").iloc[0]
        std_value = pd.to_numeric(method_sub[f"{metric}_std"], errors="coerce").iloc[0]
        count_value = pd.to_numeric(method_sub[f"{metric}_count"], errors="coerce").iloc[0]
        if not np.isfinite(mean_value) or count_value <= 0:
            continue
        means.append(float(mean_value))
        stds.append(float(std_value) if np.isfinite(std_value) else 0.0)
        labels.append(method_label(method))
        colors.append(palette[method])

    if skip_if_all_nan and not means:
        return False, [], f"{dataset}/{metric}: no aggregate values"
    if not means:
        return False, [], f"{dataset}/{metric}: no aggregate values"

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor=NATURE_SOFT_EDGE,
        ecolor=NATURE_SOFT_EDGE,
        linewidth=0.9,
        alpha=0.92,
        width=0.68,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.set_title(dataset_label(dataset))

    y_limits = zoomed_bar_ylim(metric, means, stds)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
        add_truncated_axis_marker(ax)

    stem = f"{stem_prefix}_{dataset}"
    paths = list(save_figure(fig, output_dir, stem, dpi))
    return True, paths, f"{dataset}/{metric}: {len(labels)} methods"


def summarize_methods_by_dataset(per_candidate_df: pd.DataFrame, metrics: Sequence[str]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for dataset in DATASET_ORDER:
        summary[dataset] = {}
        for metric in metrics:
            summary[dataset][metric] = len(methods_with_data(per_candidate_df, dataset, metric))
    return summary

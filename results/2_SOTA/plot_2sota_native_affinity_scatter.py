#!/usr/bin/env python3
"""
Native vs generated HDOCK scatter plots for 2_SOTA family/protein test splits,
same layout and metrics as plot_ppdbench_affinity_scatter.py (best-of-top5,
top5-mean, train similarity coloring + similarity-vs-score panels).

Target scope (see --target-scope):
  split133_impute — fixed 133 sample_ids per split CSV; optional imputation so every
    method panel has one point per sample (missing generated → y=native; missing
    native JSON → median native in that split).
  shared_all_methods — intersection across all methods (PPDbench-style, fewer points).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.plotting_2sota import (  # noqa: E402
    HDOCK_SCATTER_MIN_SCORE,
    METHOD_LABELS,
    METHOD_ORDER,
    dataset_label,
    save_figure,
    setup_publication_style,
)
from plot_ppdbench_affinity_scatter import (  # noqa: E402
    add_trendline,
    build_best_of_top5,
    build_mean_of_top5,
)

DEFAULT_METRICS_CSV = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/metrics_summary/per_candidate_metrics.csv"
DEFAULT_BASELINE_DIR = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/baseline_data"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/figures"
DEFAULT_SPLITS_DIR = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/splits"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot native vs generated HDOCK (2_SOTA test splits), PPDbench-style panels."
    )
    ap.add_argument("--per-candidate-csv", default=DEFAULT_METRICS_CSV)
    ap.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--min-hdock-score",
        type=float,
        default=HDOCK_SCATTER_MIN_SCORE,
        help="Drop generated candidates strictly below this HDOCK value before top5 aggregation.",
    )
    ap.add_argument(
        "--min-native-hdock",
        type=float,
        default=None,
        help=(
            "If set, drop native scores below this threshold before plotting. "
            "Unset keeps all finite natives (needed to reach 133/split if some natives are < -500)."
        ),
    )
    ap.add_argument(
        "--splits-dir",
        default=DEFAULT_SPLITS_DIR,
        help="Directory with family_level_test.csv / protein_level_test.csv (133 rows each).",
    )
    ap.add_argument(
        "--target-scope",
        choices=("split133_impute", "shared_all_methods"),
        default="split133_impute",
        help=(
            "split133_impute: 133 points/method via official split list + impute missing gen/native; "
            "shared_all_methods: only targets where every method has data (original behavior)."
        ),
    )
    return ap.parse_args()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_native_scores_from_json(
    path: Path, *, min_native: float | None
) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Native HDOCK JSON not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("sample_id", key))
        val = entry.get("native_hdock_score")
        if val is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(f):
            continue
        if min_native is not None and f < min_native:
            continue
        out[sid] = f
    return out


def load_ordered_split_sample_ids(splits_dir: Path, dataset: str) -> list[str]:
    csv_path = splits_dir / f"{dataset}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    return pd.read_csv(csv_path)["sample_id"].astype(str).tolist()


def impute_native_for_split(
    native_scores: dict[str, float],
    ordered_ids: list[str],
) -> tuple[dict[str, float], int]:
    """Fill missing native (null / absent) with median of available scores in this split."""
    scores = dict(native_scores)
    present = [scores[sid] for sid in ordered_ids if sid in scores]
    if not present:
        return scores, 0
    med = float(np.median(np.array(present, dtype=float)))
    n_fill = 0
    for sid in ordered_ids:
        if sid not in scores:
            scores[sid] = med
            n_fill += 1
    return scores, n_fill


def complete_agg_to_split(
    agg_df: pd.DataFrame,
    *,
    ordered_ids: list[str],
    methods: list[str],
    native_scores: dict[str, float],
    y_col: str,
    best_row_extras: bool,
) -> tuple[pd.DataFrame, int]:
    """One row per (method, sid); missing aggregate rows get y=native, train_similarity=0."""
    rows: list[dict] = []
    n_imputed = 0
    for method in methods:
        sub = agg_df[agg_df["method"] == method].set_index("target_id", drop=False)
        for sid in ordered_ids:
            nat = float(native_scores[sid])
            if sid in sub.index:
                rows.append(dict(sub.loc[sid]))
            else:
                row: dict = {
                    "method": method,
                    "target_id": sid,
                    "native_hdock_score": nat,
                    "train_similarity": 0.0,
                    "candidate_count": 0,
                    y_col: nat,
                    "_imputed_y": True,
                }
                if best_row_extras:
                    row["candidate_rank"] = 0
                    row["sequence"] = ""
                rows.append(row)
                n_imputed += 1
    out = pd.DataFrame(rows)
    for col in ("native_hdock_score", y_col, "train_similarity"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out, n_imputed


def common_target_set(best_df: pd.DataFrame, method_order: list[str]) -> set[str]:
    method_to_targets = {
        m: set(best_df.loc[best_df["method"] == m, "target_id"].astype(str)) for m in method_order
    }
    target_sets = [t for t in method_to_targets.values() if t]
    if not target_sets:
        return set()
    shared = set.intersection(*target_sets)
    keep = set()
    for target in shared:
        col = best_df.loc[best_df["target_id"] == target, "native_hdock_score"]
        if col.notna().any() and np.isfinite(pd.to_numeric(col, errors="coerce")).any():
            keep.add(target)
    return keep


def panel_scatter_multi(
    df: pd.DataFrame,
    *,
    method_order: list[str],
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title_prefix: str,
    out_stem: str,
    output_dir: Path,
    dpi: int,
) -> tuple[str, str]:
    n = len(method_order)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 4.2 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    vmin, vmax = 0.0, 1.0
    scatter_artist = None

    for idx, method in enumerate(method_order):
        ax = axes_flat[idx]
        sub = df[df["method"] == method].copy()
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
        sub["train_similarity"] = pd.to_numeric(sub["train_similarity"], errors="coerce")
        sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col]) & np.isfinite(sub["train_similarity"])]
        label = METHOD_LABELS.get(method, method)

        if sub.empty:
            ax.set_title(f"{title_prefix}{label} | no data")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            continue

        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        c = sub["train_similarity"].to_numpy(dtype=float)
        scatter_artist = ax.scatter(
            x,
            y,
            c=c,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=34,
            alpha=0.82,
            edgecolors="black",
            linewidths=0.25,
        )
        add_trendline(ax, x, y)
        if x_col == "native_hdock_score":
            min_xy = min(float(np.min(x)), float(np.min(y)))
            max_xy = max(float(np.max(x)), float(np.max(y)))
            pad = max(5.0, (max_xy - min_xy) * 0.05)
            lo = min_xy - pad
            hi = max_xy + pad
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="#666666", linewidth=0.9, alpha=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_title(f"{title_prefix}{label} (n={len(sub)})")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    for j in range(len(method_order), len(axes_flat)):
        axes_flat[j].set_visible(False)

    visible_axes = [axes_flat[i] for i in range(len(method_order))]
    if scatter_artist is not None:
        cbar = fig.colorbar(scatter_artist, ax=visible_axes, shrink=0.92)
        cbar.set_label("Train similarity")
        cbar.set_ticks(np.linspace(vmin, vmax, 6))

    return save_figure(fig, output_dir, out_stem, dpi)


def filter_rows_native_and_y(
    df: pd.DataFrame,
    *,
    y_col: str,
    native_min: float,
    y_min: float,
) -> pd.DataFrame:
    """Keep rows where native and y are finite and both >= respective floors (e.g. -500)."""
    d = df.copy()
    d["native_hdock_score"] = pd.to_numeric(d["native_hdock_score"], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    ok = (
        np.isfinite(d["native_hdock_score"])
        & np.isfinite(d[y_col])
        & (d["native_hdock_score"] >= native_min)
        & (d[y_col] >= y_min)
    )
    return d.loc[ok].reset_index(drop=True)


def run_for_dataset(
    *,
    dataset: str,
    per_candidate_df: pd.DataFrame,
    native_path: Path,
    splits_dir: Path,
    output_dir: Path,
    dpi: int,
    min_hdock: float,
    min_native: float | None,
    target_scope: str,
) -> dict:
    sub = per_candidate_df[per_candidate_df["dataset"] == dataset].copy()
    sub["target_id"] = sub["target_id"].astype(str)
    sub["method"] = sub["method"].astype(str)
    sub["hdock_score"] = pd.to_numeric(sub["hdock_score"], errors="coerce")
    sub = sub[np.isfinite(sub["hdock_score"]) & (sub["hdock_score"] >= min_hdock)]
    sub = sub.sort_values(["method", "target_id", "candidate_rank"]).reset_index(drop=True)

    prefix = f"{dataset_label(dataset)} | "
    stem_base = dataset
    extra_summary: dict = {}

    if target_scope == "split133_impute":
        ordered_ids = load_ordered_split_sample_ids(splits_dir, dataset)
        if len(ordered_ids) != 133:
            raise ValueError(f"Expected 133 rows in split for {dataset}, got {len(ordered_ids)}")

        native_raw = load_native_scores_from_json(native_path, min_native=None)
        native_scores, n_native_imp = impute_native_for_split(native_raw, ordered_ids)
        extra_summary["n_native_imputed_median"] = n_native_imp

        best_df = build_best_of_top5(sub, native_scores)
        mean_df = build_mean_of_top5(sub, native_scores)
        plot_best, n_imp_b = complete_agg_to_split(
            best_df,
            ordered_ids=ordered_ids,
            methods=list(METHOD_ORDER),
            native_scores=native_scores,
            y_col="best_hdock_score",
            best_row_extras=True,
        )
        plot_mean, n_imp_m = complete_agg_to_split(
            mean_df,
            ordered_ids=ordered_ids,
            methods=list(METHOD_ORDER),
            native_scores=native_scores,
            y_col="top5_mean_hdock_score",
            best_row_extras=False,
        )
        extra_summary["n_imputed_y_best_table"] = n_imp_b
        extra_summary["n_imputed_y_mean_table"] = n_imp_m
        stem_suffix = "split133"
    else:
        native_floor = min_native if min_native is not None else min_hdock
        native_scores = load_native_scores_from_json(native_path, min_native=native_floor)
        best_df = build_best_of_top5(sub, native_scores)
        mean_df = build_mean_of_top5(sub, native_scores)
        shared = common_target_set(best_df, list(METHOD_ORDER))
        plot_best = best_df[best_df["target_id"].isin(shared)].copy()
        plot_mean = mean_df[mean_df["target_id"].isin(shared)].copy()
        extra_summary["shared_target_count"] = len(shared)
        stem_suffix = "shared_targets"

    native_plot_min = min_native if min_native is not None else min_hdock
    n_before_b, n_before_m = len(plot_best), len(plot_mean)
    plot_best = filter_rows_native_and_y(
        plot_best, y_col="best_hdock_score", native_min=native_plot_min, y_min=min_hdock
    )
    plot_mean = filter_rows_native_and_y(
        plot_mean, y_col="top5_mean_hdock_score", native_min=native_plot_min, y_min=min_hdock
    )
    extra_summary["rows_after_native_y_floor"] = {
        "best": len(plot_best),
        "mean": len(plot_mean),
        "dropped_best": n_before_b - len(plot_best),
        "dropped_mean": n_before_m - len(plot_mean),
        "native_min": native_plot_min,
        "y_min": min_hdock,
    }

    csv_dir = ensure_dir(output_dir / "native_affinity_tables")
    best_path = csv_dir / f"{stem_base}_best_affinity_{stem_suffix}.csv"
    mean_path = csv_dir / f"{stem_base}_top5_mean_affinity_{stem_suffix}.csv"
    plot_best.sort_values(["method", "target_id"]).to_csv(best_path, index=False)
    plot_mean.sort_values(["method", "target_id"]).to_csv(mean_path, index=False)

    panel_scatter_multi(
        plot_best,
        method_order=list(METHOD_ORDER),
        x_col="native_hdock_score",
        y_col="best_hdock_score",
        x_label="Native peptide HDOCK score",
        y_label="Best generated HDOCK score from top5",
        title_prefix=prefix,
        out_stem=f"{stem_base}_native_vs_best_affinity_{stem_suffix}",
        output_dir=output_dir,
        dpi=dpi,
    )
    panel_scatter_multi(
        plot_mean,
        method_order=list(METHOD_ORDER),
        x_col="native_hdock_score",
        y_col="top5_mean_hdock_score",
        x_label="Native peptide HDOCK score",
        y_label="Mean generated HDOCK score over top5",
        title_prefix=prefix,
        out_stem=f"{stem_base}_native_vs_top5_mean_affinity_{stem_suffix}",
        output_dir=output_dir,
        dpi=dpi,
    )
    panel_scatter_multi(
        plot_best,
        method_order=list(METHOD_ORDER),
        x_col="train_similarity",
        y_col="best_hdock_score",
        x_label="Train similarity",
        y_label="Best generated HDOCK score from top5",
        title_prefix=prefix,
        out_stem=f"{stem_base}_similarity_vs_best_affinity_{stem_suffix}",
        output_dir=output_dir,
        dpi=dpi,
    )
    panel_scatter_multi(
        plot_mean,
        method_order=list(METHOD_ORDER),
        x_col="train_similarity",
        y_col="top5_mean_hdock_score",
        x_label="Train similarity",
        y_label="Mean generated HDOCK score over top5",
        title_prefix=prefix,
        out_stem=f"{stem_base}_similarity_vs_top5_mean_affinity_{stem_suffix}",
        output_dir=output_dir,
        dpi=dpi,
    )

    out = {
        "dataset": dataset,
        "native_json": str(native_path),
        "target_scope": target_scope,
        "min_hdock_score_generated": min_hdock,
        "min_native_hdock": min_native,
        "n_native_in_json_after_load": len(native_scores),
        "csvs": {"best": str(best_path), "top5_mean": str(mean_path)},
        **extra_summary,
    }
    if target_scope == "split133_impute":
        out["split_sample_count"] = 133
        out["note"] = (
            "Cohort built from 133 official split sample_ids; missing native in JSON filled with median native; "
            "missing generated aggregate filled with native HDOCK (on diagonal). Plots and exported CSVs then drop "
            "any row where native or generated score is below min thresholds (see rows_after_native_y_floor)."
        )
    return out


def main() -> None:
    args = parse_args()
    setup_publication_style()
    plt.rcParams["font.family"] = ["DejaVu Serif", "serif"]

    baseline = Path(args.baseline_dir)
    splits_dir = Path(args.splits_dir)
    output_dir = ensure_dir(args.output_dir)
    per_candidate = Path(args.per_candidate_csv)
    df = pd.read_csv(per_candidate)

    summaries = []
    for dataset, json_name in (
        ("family_level_test", "family_level_test_native_hdock.json"),
        ("protein_level_test", "protein_level_test_native_hdock.json"),
    ):
        summaries.append(
            run_for_dataset(
                dataset=dataset,
                per_candidate_df=df,
                native_path=baseline / json_name,
                splits_dir=splits_dir,
                output_dir=output_dir,
                dpi=args.dpi,
                min_hdock=args.min_hdock_score,
                min_native=args.min_native_hdock,
                target_scope=args.target_scope,
            )
        )

    summary_path = output_dir / "native_affinity_2sota_plot_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    for s in summaries:
        if s["target_scope"] == "split133_impute":
            print(
                f"{s['dataset']}: split133 | native median-imputed={s.get('n_native_imputed_median', 0)}, "
                f"y-imputed best/mean rows={s.get('n_imputed_y_best_table')}/{s.get('n_imputed_y_mean_table')}"
            )
        else:
            print(
                f"{s['dataset']}: shared_all_methods | shared targets = {s.get('shared_target_count', 0)}, "
                f"n_native (after floor) = {s['n_native_in_json_after_load']}"
            )
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()

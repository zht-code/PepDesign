#!/usr/bin/env python3
"""
Figure 5 robustness comparison (updated): 8-panel figure a–h for main text.

Outputs (under baseline/ only):
  figures/Figure_5_robustness_comparison_updated.png|.pdf
  figures/Figure_5_robustness_comparison_updated_caption.txt
  tables/figure5_panel_E_representation_conditions.csv  (audit: strengths used in panel e)
  logs/figure5_robustness_updated_run.log

Primary quantitative inputs:
  tables/all_methods_condition_curves_intersection.csv  (condition-wise means, intersection targets)
  tables/all_methods_robustness_summary_intersection.csv (AUDC / slopes / drops per metric)

Sample-level inputs (panel g, optional panel h):
  raw_results/{method}/samples_pocket_noise_lvl1p0_r0.csv
  tables/samples_pocket_noise_lvl0p0_r0.csv  (ours; under 5_robustness/tables, read-only)

Ours sample CSV path is resolved relative to ROB_ROOT when not present under baseline/raw_results/ours/.
"""
from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

BASE_DIR = Path("/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline").resolve()
ROB_ROOT = Path("/root/autodl-tmp/Peptide_3D/results/5_robustness").resolve()
FIG_DIR = BASE_DIR / "figures"
TAB_DIR = BASE_DIR / "tables"
LOG_DIR = BASE_DIR / "logs"
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
METHOD_ORDER = ["ours", "rfdiffusion", "proteingenerator", "bindcraft"]
PERTURB_ORDER = ["structure_missing", "pocket_noise", "sequence_trunc"]
PERTURB_COLORS = {"structure_missing": "#4C72B0", "pocket_noise": "#55A868", "sequence_trunc": "#C44E52"}


def _log(msg: str, log_lines: list[str]) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    log_lines.append(line)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "figure.dpi": 180,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.set_title(label, loc="left", fontweight="bold", fontsize=10)


def load_agg_intersection(log: list[str]) -> pd.DataFrame:
    path = TAB_DIR / "all_methods_condition_curves_intersection.csv"
    if not path.is_file():
        _log(f"MISSING {path}", log)
        return pd.DataFrame()
    df = pd.read_csv(path)
    _log(f"Loaded agg intersection: {path} rows={len(df)}", log)
    return df


def load_summary_intersection(log: list[str]) -> pd.DataFrame:
    path = TAB_DIR / "all_methods_robustness_summary_intersection.csv"
    if not path.is_file():
        _log(f"MISSING {path}", log)
        return pd.DataFrame()
    df = pd.read_csv(path)
    _log(f"Loaded summary intersection: {path} rows={len(df)}", log)
    return df


def load_intersection_targets(log: list[str]) -> list[str]:
    path = TAB_DIR / "intersection_targets.csv"
    if not path.is_file():
        _log(f"MISSING {path}; panel g/h use all samples per method.", log)
        return []
    t = pd.read_csv(path)["target_id"].astype(str).tolist()
    _log(f"Intersection targets: n={len(t)}", log)
    return t


def draw_schematic(ax: plt.Axes) -> None:
    add_panel_label(ax, "a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    titles = [
        ("structure_missing", "Structure\nmissing", "#4C72B0", ["0%", "10%", "20%", "30%", "40%"]),
        ("pocket_noise", "Pocket\nnoise", "#55A868", ["0", "0.5", "1.0", "1.5", "2.0 Å"]),
        ("sequence_trunc", "Sequence\ntruncation", "#C44E52", ["0%", "10%", "20%", "30%", "40%"]),
    ]
    for idx, (_, title, color, levels) in enumerate(titles):
        x0 = 0.06 + idx * 0.31
        rect = mpatches.FancyBboxPatch(
            (x0, 0.38), 0.26, 0.32, boxstyle="round,pad=0.02", edgecolor=color, facecolor="#f7f7f7"
        )
        ax.add_patch(rect)
        ax.text(x0 + 0.13, 0.77, title, ha="center", va="center", color=color, fontsize=8)
        for j, lvl in enumerate(levels):
            ax.scatter(x0 + 0.04 + j * 0.05, 0.50, s=18 + j * 6, color=color, alpha=0.35 + 0.12 * j)
            ax.text(x0 + 0.04 + j * 0.05, 0.29, lvl, ha="center", fontsize=6)
    ax.text(
        0.5,
        0.12,
        "Target-side perturbations: structure_missing, pocket_noise (Å RMSD), sequence_trunc (% removed).",
        ha="center",
        fontsize=7,
    )


def _normalize_affinity(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = sub["level_value"].astype(float).to_numpy()
    y = -sub["affinity_mean"].astype(float).to_numpy()
    y0 = y[0] if len(y) else 1.0
    return x, y / max(abs(y0), 1e-6)


def _normalize_success(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = sub["level_value"].astype(float).to_numpy()
    y = sub["success_rate"].astype(float).to_numpy()
    y0 = y[0] if len(y) else 1.0
    return x, y / max(abs(y0), 1e-6)


def _affinity_relative_drop_pct(clean_raw: float, pert_raw: float) -> float:
    """Higher is better: use negated HDOCK. Positive drop = performance loss (%)."""
    c = -float(clean_raw)
    p = -float(pert_raw)
    return float(max(0.0, (c - p) / max(abs(c), 1e-6) * 100.0))


def plot_panel_b(ax: plt.Axes, agg: pd.DataFrame) -> None:
    add_panel_label(ax, "b")
    ax.axis("off")
    methods = [m for m in METHOD_ORDER if m in set(agg["method"])]
    perturbs = [
        ("structure_missing", "Structure missing"),
        ("pocket_noise", "Pocket noise"),
        ("sequence_trunc", "Sequence truncation"),
    ]
    sub_axes = []
    for i, (perturb, title) in enumerate(perturbs):
        inset = ax.inset_axes([0.02 + i * 0.325, 0.18, 0.30, 0.74])
        sub_axes.append(inset)
        inset.set_title(title, fontsize=8, pad=2)
        for method in methods:
            sub = agg[(agg["method"] == method) & (agg["perturbation_type"] == perturb)].sort_values("level_value")
            if sub.empty:
                continue
            x, y = _normalize_affinity(sub)
            inset.plot(
                x, y, marker="o", ms=3, lw=1.2, color=COLORS[method], label=LABELS[method]
            )
        inset.axhline(1.0, color="#999999", lw=0.6, ls="--")
        inset.set_xlabel("Strength (native units)", fontsize=7)
        if i == 0:
            inset.set_ylabel("Normalized affinity", fontsize=7)
        else:
            inset.set_yticklabels([])
        inset.grid(False)
    handles, labels = sub_axes[0].get_legend_handles_labels() if sub_axes else ([], [])
    if handles:
        ax.legend(handles, labels, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.02), ncol=2)


def plot_panel_c(ax: plt.Axes, agg: pd.DataFrame) -> None:
    add_panel_label(ax, "c")
    rows = []
    row_labels = []
    metrics = ["affinity_mean", "success_rate", "stability_mean", "solubility_mean"]
    methods = [m for m in METHOD_ORDER if m in set(agg["method"])]
    for method in methods:
        for perturb in PERTURB_ORDER:
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
    arr = np.array(rows, dtype=float) if rows else np.zeros((0, 0))
    if arr.size:
        pcm = ax.pcolormesh(
            np.arange(arr.shape[1] + 1),
            np.arange(arr.shape[0] + 1),
            arr,
            cmap="magma",
            vmin=0,
            vmax=max(40.0, float(np.nanmax(arr))),
            shading="flat",
        )
        ax.set_xticks(np.arange(arr.shape[1]) + 0.5)
        ax.set_xticklabels(["Aff.", "Succ.", "Stab.", "Sol."])
        ax.set_yticks(np.arange(arr.shape[0]) + 0.5)
        ax.set_yticklabels(row_labels, fontsize=6)
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02, label="Worst-case drop (%)")
    else:
        ax.text(0.5, 0.5, "No heatmap data", ha="center", va="center")


def _series_for_method_violin(vals: list[float]) -> np.ndarray | None:
    """At least two finite points for matplotlib violin KDE; duplicate singletons. None if no data."""
    arr = np.array([float(v) for v in vals if pd.notna(v) and np.isfinite(v)], dtype=float)
    if arr.size >= 2:
        return arr
    if arr.size == 1:
        return np.array([arr[0], arr[0]], dtype=float)
    return None


def plot_panel_d(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """AUDC (affinity) + sensitivity_slope (affinity) + max_drop as violin small multiples."""
    add_panel_label(ax, "d")
    ax.axis("off")
    sub = summary[summary["metric"] == "affinity_hdock"].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No summary data", ha="center", va="center", transform=ax.transAxes)
        return
    methods = [m for m in METHOD_ORDER if m in set(sub["method"])]
    metrics_plot = [
        ("AUDC", "AUDC"),
        ("sensitivity_slope", "Sensitivity (slope)"),
        ("max_drop", "Max drop (%)"),
    ]
    for k, (col, title) in enumerate(metrics_plot):
        inset = ax.inset_axes([0.06, 0.72 - k * 0.30, 0.90, 0.26])
        xpos = np.arange(len(methods), dtype=float)
        series_per_method: list[np.ndarray | None] = []
        for method in methods:
            vals: list[float] = []
            for perturb in PERTURB_ORDER:
                row = sub[(sub["method"] == method) & (sub["perturbation_type"] == perturb)]
                if row.empty or col not in row.columns:
                    vals.append(float("nan"))
                else:
                    v = row[col].iloc[0]
                    vals.append(float(v) if pd.notna(v) else float("nan"))
            series_per_method.append(_series_for_method_violin(vals))
        for i, method in enumerate(methods):
            s = series_per_method[i]
            if s is None:
                continue
            parts = inset.violinplot(
                [s],
                positions=[xpos[i]],
                widths=0.58,
                showmeans=False,
                showmedians=True,
                showextrema=True,
            )
            body = parts["bodies"][0]
            body.set_facecolor(COLORS[method])
            body.set_edgecolor("#222222")
            body.set_linewidth(0.6)
            body.set_alpha(0.78)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                coll = parts.get(key)
                if coll is None:
                    continue
                coll.set_color("#222222")
                coll.set_linewidth(0.9 if key == "cmedians" else 0.55)
        inset.set_xticks(xpos)
        inset.set_xticklabels([LABELS[m] for m in methods], rotation=12, ha="right", fontsize=6)
        yl = inset.set_ylabel(title, fontsize=7, labelpad=6)
        yl.set_clip_on(False)
        inset.grid(True, axis="y", linewidth=0.4, alpha=0.35)
        if k == 0:
            handles = [
                mpatches.Patch(facecolor=COLORS[m], edgecolor="#222222", linewidth=0.5, label=LABELS[m])
                for m in methods
            ]
            inset.legend(handles=handles, frameon=False, fontsize=5.5, loc="upper right", ncol=2)
    ax.text(
        0.5,
        0.02,
        "Violins: per method, distribution of the three perturbation-family values (intersection summary, affinity_hdock).",
        ha="center",
        fontsize=7,
    )


def _row_at_level(sub: pd.DataFrame, level: float) -> pd.Series | None:
    m = sub[np.isclose(sub["level_value"].astype(float), float(level))]
    if m.empty:
        return None
    return m.iloc[0]


def build_panel_e_table(agg: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    """Representative strengths for panel e (documented)."""
    rows = []
    specs = [
        ("structure_missing", 20.0, "20% missing"),
        ("structure_missing", 40.0, "40% missing"),
        ("pocket_noise", 1.0, "1.0 Å"),
        ("pocket_noise", 2.0, "2.0 Å"),
        ("sequence_trunc", 20.0, "20% trunc"),
        ("sequence_trunc", 40.0, "40% trunc"),
    ]
    methods = [m for m in METHOD_ORDER if m in set(agg["method"])]
    for perturb, lvl, tag in specs:
        for method in methods:
            sub = agg[(agg["method"] == method) & (agg["perturbation_type"] == perturb)].sort_values("level_value")
            if sub.empty:
                continue
            clean_row = _row_at_level(sub, 0.0)
            pert_row = _row_at_level(sub, lvl)
            if clean_row is None or pert_row is None:
                continue
            drop = _affinity_relative_drop_pct(clean_row["affinity_mean"], pert_row["affinity_mean"])
            rows.append(
                {
                    "method": method,
                    "perturbation_type": perturb,
                    "level_value": lvl,
                    "condition_label": tag,
                    "relative_drop_affinity_pct": drop,
                }
            )
    df = pd.DataFrame(rows)
    out = TAB_DIR / "figure5_panel_E_representation_conditions.csv"
    df.to_csv(out, index=False)
    _log(f"Wrote panel E audit table: {out}", log)
    return df


def plot_panel_e(ax: plt.Axes, e_df: pd.DataFrame) -> None:
    add_panel_label(ax, "e")
    if e_df.empty:
        ax.text(0.5, 0.5, "No data for panel e", ha="center", va="center", transform=ax.transAxes)
        return
    methods = [m for m in METHOD_ORDER if m in set(e_df["method"])]
    xlabs: list[str] = []
    order = [
        ("structure_missing", 20.0),
        ("structure_missing", 40.0),
        ("pocket_noise", 1.0),
        ("pocket_noise", 2.0),
        ("sequence_trunc", 20.0),
        ("sequence_trunc", 40.0),
    ]
    short = {
        "structure_missing": "Struct. miss.",
        "pocket_noise": "Pocket noise",
        "sequence_trunc": "Seq. trunc.",
    }
    for perturb, lvl in order:
        suf = " Å" if perturb == "pocket_noise" else "%"
        xlabs.append(f"{short[perturb]}\n{lvl:g}{suf}")

    n_x = len(order)
    x0 = np.arange(n_x)
    bw = 0.18
    offsets = np.linspace(-(len(methods) - 1) * bw / 2, (len(methods) - 1) * bw / 2, len(methods))
    for mi, method in enumerate(methods):
        heights = []
        for perturb, lvl in order:
            r = e_df[
                (e_df["method"] == method)
                & (e_df["perturbation_type"] == perturb)
                & np.isclose(e_df["level_value"].astype(float), float(lvl))
            ]
            heights.append(float(r["relative_drop_affinity_pct"].iloc[0]) if not r.empty else 0.0)
        ax.bar(x0 + offsets[mi], heights, width=bw * 0.92, color=COLORS[method], label=LABELS[method], edgecolor="white", linewidth=0.3)
    ax.set_xticks(x0)
    ax.set_xticklabels(xlabs, fontsize=6, rotation=0)
    ax.set_ylabel("Relative drop in affinity (%)")
    ax.legend(frameon=False, ncol=2, fontsize=6.5, loc="upper left")
    ax.axhline(0.0, color="#888888", lw=0.5)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)


def plot_panel_f(ax: plt.Axes, agg: pd.DataFrame) -> None:
    """Same layout as b but success_rate normalized to clean (panel f)."""
    add_panel_label(ax, "f")
    ax.axis("off")
    methods = [m for m in METHOD_ORDER if m in set(agg["method"])]
    perturbs = [
        ("structure_missing", "Structure missing"),
        ("pocket_noise", "Pocket noise"),
        ("sequence_trunc", "Sequence truncation"),
    ]
    sub_axes = []
    for i, (perturb, title) in enumerate(perturbs):
        inset = ax.inset_axes([0.02 + i * 0.325, 0.18, 0.30, 0.74])
        sub_axes.append(inset)
        inset.set_title(title, fontsize=8, pad=2)
        for method in methods:
            sub = agg[(agg["method"] == method) & (agg["perturbation_type"] == perturb)].sort_values("level_value")
            if sub.empty or "success_rate" not in sub.columns:
                continue
            x, y = _normalize_success(sub)
            inset.plot(x, y, marker="s", ms=3, lw=1.2, color=COLORS[method], label=LABELS[method])
        inset.axhline(1.0, color="#999999", lw=0.6, ls="--")
        inset.set_xlabel("Strength (native units)", fontsize=7)
        if i == 0:
            inset.set_ylabel("Normalized success rate", fontsize=7)
        else:
            inset.set_yticklabels([])
        inset.set_ylim(bottom=0.0)
        inset.grid(False)
    h, lab = sub_axes[0].get_legend_handles_labels() if sub_axes else ([], [])
    if h:
        ax.legend(h, lab, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.02), ncol=2)


def _load_pocket_noise_samples(method: str, log: list[str]) -> pd.DataFrame:
    if method == "ours":
        p = ROB_ROOT / "tables" / "samples_pocket_noise_lvl1p0_r0.csv"
    else:
        p = BASE_DIR / "raw_results" / method / "samples_pocket_noise_lvl1p0_r0.csv"
    if not p.is_file():
        _log(f"Missing sample file for {method}: {p}", log)
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["method"] = method
    if "affinity_hdock" not in df.columns and "affinity" in df.columns:
        df = df.rename(columns={"affinity": "affinity_hdock"})
    return df


def plot_panel_g(ax: plt.Axes, targets: list[str], log: list[str]) -> None:
    add_panel_label(ax, "g")
    parts = []
    for method in METHOD_ORDER:
        df = _load_pocket_noise_samples(method, log)
        if df.empty:
            continue
        if targets:
            df = df[df["target_id"].astype(str).isin(targets)].copy()
        s = pd.to_numeric(df["affinity_hdock"], errors="coerce").dropna()
        if s.empty:
            continue
        parts.append((LABELS[method], s.to_numpy(), COLORS[method]))
    if not parts:
        ax.text(0.5, 0.5, "No sample-level affinity for pocket noise 1.0 Å", ha="center", va="center", transform=ax.transAxes)
        return
    positions = np.arange(1, len(parts) + 1)
    vp = ax.violinplot([p[1] for p in parts], positions=positions, showmeans=False, showmedians=True, widths=0.65)
    for i, (_, _, color) in enumerate(parts):
        vp["bodies"][i].set_facecolor(color)
        vp["bodies"][i].set_alpha(0.38)
        vp["bodies"][i].set_edgecolor(color)
    ax.set_xticks(positions)
    ax.set_xticklabels([p[0] for p in parts], rotation=15, ha="right")
    ax.set_ylabel("HDOCK affinity (raw score)")
    ax.axhline(0, color="#cccccc", lw=0.5)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    note = "Pocket noise 1.0 Å; intersection targets" if targets else "Pocket noise 1.0 Å; all available samples"
    ax.text(0.5, 1.02, note, transform=ax.transAxes, ha="center", fontsize=7, style="italic")


def plot_panel_h(ax: plt.Axes, agg: pd.DataFrame, targets: list[str], log: list[str]) -> None:
    add_panel_label(ax, "h")
    ax.axis("off")
    # Mean affinity clean vs pocket 1.0 per method (aggregate), as compact bars
    methods = [m for m in METHOD_ORDER if m in set(agg["method"])]
    clean_v, p1_v = [], []
    for method in methods:
        c = agg[
            (agg["method"] == method)
            & (agg["perturbation_type"] == "pocket_noise")
            & np.isclose(agg["level_value"].astype(float), 0.0)
        ]
        p = agg[
            (agg["method"] == method)
            & (agg["perturbation_type"] == "pocket_noise")
            & np.isclose(agg["level_value"].astype(float), 1.0)
        ]
        clean_v.append(float(c["affinity_mean"].iloc[0]) if not c.empty else np.nan)
        p1_v.append(float(p["affinity_mean"].iloc[0]) if not p.empty else np.nan)
    inset = ax.inset_axes([0.08, 0.22, 0.88, 0.62])
    x = np.arange(len(methods))
    w = 0.35
    inset.bar(x - w / 2, clean_v, width=w, label="Clean (0 Å)", color="#c5d5e5", edgecolor="#666666", linewidth=0.4)
    inset.bar(x + w / 2, p1_v, width=w, label="Pocket noise 1.0 Å", color="#f6c8b6", edgecolor="#666666", linewidth=0.4)
    inset.set_xticks(x)
    inset.set_xticklabels([LABELS[m] for m in methods], rotation=12, ha="right")
    inset.set_ylabel("Mean HDOCK affinity\n(intersection agg.)", fontsize=8)
    inset.legend(frameon=False, fontsize=7, loc="lower right")
    inset.axhline(0, color="#bbbbbb", lw=0.5)
    lines = [
        "Representative comparison (aggregate means, intersection cohort).",
        f"Intersection targets: {len(targets) if targets else 'n/a'}.",
        "Lower (more negative) HDOCK is better; bars show cohort means from condition curves.",
    ]
    ax.text(0.5, 0.08, "\n".join(lines), ha="center", va="center", fontsize=7, transform=ax.transAxes)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    _log("=== Figure 5 updated run ===", log_lines)
    meta = {
        "primary_tables": [
            str(TAB_DIR / "all_methods_condition_curves_intersection.csv"),
            str(TAB_DIR / "all_methods_robustness_summary_intersection.csv"),
        ],
        "dataset": "intersection_only",
        "sample_reads_for_g": "raw_results/<method>/samples_pocket_noise_lvl1p0_r0.csv + ours from 5_robustness/tables/",
    }
    _log("Meta: " + json.dumps(meta), log_lines)

    style()
    agg = load_agg_intersection(log_lines)
    summary = load_summary_intersection(log_lines)
    targets = load_intersection_targets(log_lines)

    e_df = build_panel_e_table(agg, log_lines) if not agg.empty else pd.DataFrame()

    fig = plt.figure(figsize=(9.0, 12.0))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.34)

    draw_schematic(fig.add_subplot(gs[0, 0]))
    plot_panel_b(fig.add_subplot(gs[0, 1]), agg)
    plot_panel_c(fig.add_subplot(gs[1, 0]), agg)
    plot_panel_d(fig.add_subplot(gs[1, 1]), summary)
    plot_panel_e(fig.add_subplot(gs[2, 0]), e_df)
    plot_panel_f(fig.add_subplot(gs[2, 1]), agg)
    plot_panel_g(fig.add_subplot(gs[3, 0]), targets, log_lines)
    plot_panel_h(fig.add_subplot(gs[3, 1]), agg, targets, log_lines)

    png_path = FIG_DIR / "Figure_5_robustness_comparison_updated.png"
    pdf_path = FIG_DIR / "Figure_5_robustness_comparison_updated.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    _log(f"Saved {png_path}", log_lines)
    _log(f"Saved {pdf_path}", log_lines)

    caption = textwrap.dedent(
        """
        Figure 5 (updated) | Robustness comparison across Ours, RFdiffusion, ProteinGenerator, and BindCraft (intersection cohort).

        a, Schematic of three target-side perturbation families: structure_missing (fraction of missing receptor atoms),
        pocket_noise (Gaussian pocket coordinate noise, Å RMSD), and sequence_trunc (C-terminal truncation percentage).

        b, Normalized affinity degradation curves (each method normalized to its clean-condition mean; higher is better after sign flip).
        Three small multiples share the same method color legend. Strengths use native units (% or Å) aligned across methods.

        c, Heatmap of worst-case relative performance drop (%) across affinity (HDOCK mean), success rate, stability, and solubility;
        rows are method × perturbation family.

        d, Robustness summary from intersection tables for affinity_hdock: AUDC, sensitivity slope, and max drop as three stacked
        strip panels; each violin is one method, summarizing the three perturbation-family values (structure_missing, pocket_noise, sequence_trunc).

        e, Representative-strength relative drop in affinity (%): structure_missing and sequence_trunc at 20% and 40%;
        pocket_noise at 1.0 Å (moderate) and 2.0 Å (stronger). Drops compare each condition to the clean (level 0) mean using the
        same negated-HDOCK convention as panel c.

        f, Success-rate degradation curves mirroring panel b: normalized success rate (relative to clean) versus perturbation strength,
        same layout and method colors for direct comparison with affinity trends.

        g, Affinity (HDOCK) distribution under pocket noise 1.0 Å: violin bodies with median lines; per-method sample CSVs.
        When intersection_targets.csv is present, distributions are restricted to that target list for comparable support.

        h, Representative aggregate comparison: mean HDOCK affinity for clean vs pocket noise 1.0 Å per method from the same
        intersection condition curves used in b–f (cohort means, not single-structure montage).

        Data: quantitative panels use `tables/all_methods_condition_curves_intersection.csv` and
        `tables/all_methods_robustness_summary_intersection.csv` (intersection-only alignment). Sample-level panel g additionally
        reads per-method `samples_pocket_noise_lvl1p0_r0.csv` under baseline/raw_results, and Ours samples from
        `results/5_robustness/tables/samples_pocket_noise_lvl1p0_r0.csv`.
        """
    ).strip()
    cap_path = FIG_DIR / "Figure_5_robustness_comparison_updated_caption.txt"
    cap_path.write_text(caption + "\n", encoding="utf-8")
    _log(f"Wrote caption {cap_path}", log_lines)

    log_path = LOG_DIR / "figure5_robustness_updated_run.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

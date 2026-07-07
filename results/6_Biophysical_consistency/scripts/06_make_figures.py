#!/usr/bin/env python3
"""
06 — 论文风格附图（Fig7a–f + Supplementary）

基于 ``tables/`` 汇总表与 Table_S8，生成 PNG/PDF，并写 ``figures/figure_manifest.md``。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cohort_comparison import (
    METHOD_PLOT_ORDER,
    attach_method_filtered,
    cohort_for_cross_method_plots,
    methods_in_order,
)
from utils.logging_utils import setup_run_logger
from utils.nature_style import (
    ACCENT_MARK,
    MUTED_LINE,
    NATURE_DIVERGING,
    NATURE_SEQUENTIAL,
    NPG,
    apply_nature_style,
)
from utils.paths import ProjectPaths, load_config


def _method_color(method: str) -> str:
    """与 METHOD_PLOT_ORDER 对齐的离散色（NPG 子集）。"""
    try:
        idx = list(METHOD_PLOT_ORDER).index(method)
    except ValueError:
        idx = 0
    return NPG[idx % len(NPG)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--tables-dir", type=Path, default=None)
    p.add_argument("--figures-dir", type=Path, default=None)
    p.add_argument("--log-dir", type=Path, default=None)
    return p.parse_args()


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _append_build_log(logger: logging.Logger) -> None:
    p = ROOT / "logs" / "06_make_figures.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(p, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def fig7a_foldability(_s4: pd.DataFrame, s11: pd.DataFrame, out_dir: Path) -> None:
    # FCS 来自 S11；保留 S4 形参与 manifest 中「同键对齐」说明一致
    if s11.empty or "method" not in s11.columns:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    fcs_all = pd.to_numeric(s11["FCS"], errors="coerce").dropna()
    if fcs_all.empty:
        return
    bins = np.linspace(float(fcs_all.min()), float(fcs_all.max()), 30)
    for met in methods_in_order(s11):
        sub = s11.loc[s11["method"] == met, "FCS"]
        v = pd.to_numeric(sub, errors="coerce").dropna()
        if len(v) == 0:
            continue
        ax.hist(
            v,
            bins=bins,
            alpha=0.52,
            label=f"{met} (n={len(v)})",
            density=True,
            color=_method_color(met),
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xlabel("FCS (rank-mean foldability)")
    ax.set_ylabel("Density")
    ax.set_title("Fig 7a — Foldability (FCS) by generation method")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _save(fig, out_dir, "Fig7a_foldability_comparison")


def fig7b_clash_hbond(s4: pd.DataFrame, out_dir: Path) -> None:
    if s4.empty or "method" not in s4.columns:
        return
    ok = s4.get("s2_analysis_status", "").astype(str) == "ok"
    d = s4.loc[ok].copy()
    if d.empty:
        return
    clash = pd.to_numeric(d["s2_clash_count"], errors="coerce")
    hb = pd.to_numeric(d["s2_intrapeptide_hbond_count"], errors="coerce")
    hyd = pd.to_numeric(d["s2_hydrophobic_cohesion_score"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for met in methods_in_order(d):
        sel = d["method"] == met
        if not sel.any():
            continue
        c = _method_color(met)
        axes[0].scatter(clash[sel], hb[sel], s=12, alpha=0.4, c=c, edgecolors="none", label=met)
        axes[1].scatter(clash[sel], hyd[sel], s=12, alpha=0.4, c=c, edgecolors="none", label=met)
        axes[2].scatter(hb[sel], hyd[sel], s=12, alpha=0.4, c=c, edgecolors="none", label=met)
    axes[0].set_xlabel("Clash count")
    axes[0].set_ylabel("Intrapeptide H-bond count")
    axes[1].set_xlabel("Clash count")
    axes[1].set_ylabel("Hydrophobic cohesion (heuristic)")
    axes[2].set_xlabel("Intrapeptide H-bond count")
    axes[2].set_ylabel("Hydrophobic cohesion")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Fig 7b — Clash / H-bond / hydrophobic cohesion (by method)", y=1.12)
    plt.tight_layout()
    _save(fig, out_dir, "Fig7b_clash_hbond_hydrophobicity")


def fig7c_solubility_hotspot(s11: pd.DataFrame, s7_path: Path, out_dir: Path) -> None:
    if s11.empty or "method" not in s11.columns:
        return
    s7 = pd.read_csv(s7_path)
    keys = ["target_id", "peptide_id", "group"]
    m = s11[keys + ["SCS", "method"]].merge(
        s7[keys + ["hotspot_burden", "aggregation_liability_index"]],
        on=keys,
        how="inner",
    )
    if m.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    for met in methods_in_order(m):
        sub = m.loc[m["method"] == met]
        if sub.empty:
            continue
        c = _method_color(met)
        axes[0].scatter(
            sub["SCS"],
            sub["hotspot_burden"],
            s=14,
            alpha=0.45,
            label=met,
            c=c,
            edgecolors="none",
        )
    axes[0].set_xlabel("SCS (solubility compatibility)")
    axes[0].set_ylabel("Hotspot burden")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    for met in methods_in_order(m):
        sub = m.loc[m["method"] == met]
        if sub.empty:
            continue
        c = _method_color(met)
        axes[1].scatter(
            sub["SCS"],
            sub["aggregation_liability_index"],
            s=14,
            alpha=0.45,
            label=met,
            c=c,
            edgecolors="none",
        )
    axes[1].set_xlabel("SCS")
    axes[1].set_ylabel("ALI (aggregation liability)")
    fig.suptitle("Fig 7c — Solubility vs aggregation proxies (by method)", y=1.06)
    plt.tight_layout()
    _save(fig, out_dir, "Fig7c_solubility_hotspot_comparison")


def fig7d_tradeoff(s11: pd.DataFrame, out_dir: Path) -> None:
    if "method" not in s11.columns:
        return
    sub = s11.dropna(subset=["ICS", "SCS"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for met in methods_in_order(sub):
        d = sub.loc[sub["method"] == met]
        if d.empty:
            continue
        ax.scatter(
            d["SCS"],
            d["ICS"],
            s=18,
            alpha=0.48,
            label=met,
            c=_method_color(met),
            edgecolors="none",
        )
    ax.set_xlabel("SCS")
    ax.set_ylabel("ICS (interface complementarity)")
    ax.set_title("Fig 7d — Interface vs solubility (by method)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_dir, "Fig7d_interface_vs_solubility_tradeoff")


def fig7e_ics(s11: pd.DataFrame, tabs: Path, out_dir: Path) -> None:
    """
    多面板：按生成方法的 ICS 分布、方法箱线、肽长度分层、高样本靶标、ICS 分项（按方法分组条带）。
    """
    if s11.empty or "method" not in s11.columns:
        return
    s1_path = tabs / "Table_S1_master_sequence_table.csv"
    if s1_path.exists():
        s1 = pd.read_csv(s1_path, usecols=["target_id", "peptide_id", "group", "length"])
        m = s11.merge(s1, on=["target_id", "peptide_id", "group"], how="left")
    else:
        m = s11.copy()
        m["length"] = np.nan

    iface = m.dropna(subset=["ICS"]).copy()
    if iface.empty:
        return

    fig = plt.figure(figsize=(11.4, 8.2), layout="constrained")
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.15, 1.0, 1.05], hspace=0.32, wspace=0.28)

    ax_hist = fig.add_subplot(gs[0, :])
    ics_all = pd.to_numeric(iface["ICS"], errors="coerce").dropna()
    if ics_all.empty:
        return
    bins = np.linspace(float(ics_all.min()), float(ics_all.max()), 30)
    methods_ord = methods_in_order(iface)
    for met in methods_ord:
        vals = pd.to_numeric(iface.loc[iface["method"] == met, "ICS"], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        ax_hist.hist(
            vals,
            bins=bins,
            density=True,
            alpha=0.48,
            label=f"{met} (n={len(vals)})",
            color=_method_color(met),
            edgecolor="white",
            linewidth=0.45,
        )
        if len(vals) >= 8:
            kde_x = np.linspace(float(np.min(vals)), float(np.max(vals)), 160)
            try:
                kde = stats.gaussian_kde(vals)
                ax_hist.plot(kde_x, kde(kde_x), color=_method_color(met), lw=1.35, ls="-", alpha=0.85)
            except (np.linalg.LinAlgError, ValueError):
                pass
    pool = ics_all.values
    for q, ls in ((0.25, ":"), (0.75, ":")):
        ax_hist.axvline(np.quantile(pool, q), color=MUTED_LINE, ls=ls, lw=0.9, alpha=0.55)
    ax_hist.axvline(
        np.median(pool),
        color=ACCENT_MARK,
        ls="--",
        lw=1.1,
        alpha=0.75,
        label="Pooled median",
    )
    ax_hist.set_xlabel("ICS (interface complementarity, 0–1)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"Fig 7e — ICS by method (n={len(iface)} with complex metrics)")
    ax_hist.legend(frameon=False, loc="upper left", fontsize=7)

    ax_grp = fig.add_subplot(gs[1, 0])
    gdata, glabels, methods_used = [], [], []
    for met in methods_ord:
        sub = iface.loc[iface["method"] == met]
        v = sub["ICS"].dropna().values
        if len(v):
            gdata.append(v)
            glabels.append(f"{met}\n(n={len(v)})")
            methods_used.append(met)
    if gdata:
        try:
            bp = ax_grp.boxplot(gdata, tick_labels=glabels, patch_artist=True, widths=0.55)
        except TypeError:
            bp = ax_grp.boxplot(gdata, labels=glabels, patch_artist=True, widths=0.55)
        for j, patch in enumerate(bp["boxes"]):
            met = methods_used[j]
            patch.set(
                facecolor=_method_color(met),
                edgecolor=MUTED_LINE,
                linewidth=0.85,
                alpha=0.88,
            )
    ax_grp.set_ylabel("ICS")
    ax_grp.set_title("By generation method")
    ax_grp.tick_params(axis="x", labelsize=8)

    ax_len = fig.add_subplot(gs[1, 1])
    len_ser = pd.to_numeric(iface["length"], errors="coerce")
    iface = iface.assign(_length=len_ser)
    iface_ln = iface.dropna(subset=["_length"])
    if len(iface_ln["_length"].unique()) >= 3:
        try:
            iface_ln = iface_ln.assign(
                len_bin=pd.qcut(iface_ln["_length"], q=4, duplicates="drop", labels=False)
            )
        except (ValueError, TypeError):
            iface_ln = iface_ln.assign(len_bin=pd.cut(iface_ln["_length"], bins=4, labels=False))
        ldata, llabels = [], []
        for b in sorted(iface_ln["len_bin"].dropna().unique()):
            sub = iface_ln.loc[iface_ln["len_bin"] == b]
            v = sub["ICS"].values
            if len(v) == 0:
                continue
            ldata.append(v)
            lo, hi = sub["_length"].min(), sub["_length"].max()
            llabels.append(f"Len bin {int(b)+1}\n[{int(lo)}–{int(hi)}]\nn={len(v)}")
        if ldata:
            try:
                bp2 = ax_len.boxplot(ldata, tick_labels=llabels, patch_artist=True, widths=0.65)
            except TypeError:
                bp2 = ax_len.boxplot(ldata, labels=llabels, patch_artist=True, widths=0.65)
            for j, patch in enumerate(bp2["boxes"]):
                patch.set(
                    facecolor=NPG[(j + 2) % len(NPG)],
                    edgecolor=MUTED_LINE,
                    linewidth=0.85,
                    alpha=0.88,
                )
        else:
            ax_len.text(0.5, 0.5, "Length bins empty", ha="center", va="center", transform=ax_len.transAxes)
    else:
        ax_len.text(0.5, 0.5, "Length unavailable", ha="center", va="center", transform=ax_len.transAxes)
    ax_len.set_ylabel("ICS")
    ax_len.set_title("By peptide length quartile")
    ax_len.tick_params(axis="x", labelsize=7)

    ax_tgt = fig.add_subplot(gs[1, 2])
    cnt = iface.groupby("target_id").size().sort_values(ascending=False)
    top_ids = [tid for tid in cnt.index if cnt[tid] >= 3][:10]
    if not top_ids:
        top_ids = list(cnt.head(8).index)
    tdata, tlabels = [], []
    for tid in top_ids:
        v = iface.loc[iface["target_id"] == tid, "ICS"].values
        if len(v):
            tdata.append(v)
            tlabels.append(f"{tid}\n(n={len(v)})")
    if tdata:
        try:
            bp3 = ax_tgt.boxplot(tdata, tick_labels=tlabels, patch_artist=True, widths=0.6)
        except TypeError:
            bp3 = ax_tgt.boxplot(tdata, labels=tlabels, patch_artist=True, widths=0.6)
        for j, patch in enumerate(bp3["boxes"]):
            patch.set(
                facecolor=NPG[(j + 4) % len(NPG)],
                edgecolor=MUTED_LINE,
                linewidth=0.85,
                alpha=0.88,
            )
        ax_tgt.tick_params(axis="x", rotation=35, labelsize=7)
    ax_tgt.set_ylabel("ICS")
    ax_tgt.set_title("Top targets (≥3 peptides with ICS, else top counts)")

    ax_bar = fig.add_subplot(gs[2, :])
    comp_cols = [
        ("ICS_r_contacts", "Contacts (rank)"),
        ("ICS_r_hbond", "H-bond / iface res."),
        ("ICS_r_elec_comp", "Elec. complement."),
        ("ICS_r_hyd_match", "Hydrophobic match"),
        ("ICS_r_patch_overlap", "Hydrophobic patch"),
        ("ICS_r_packing", "Packing density"),
    ]
    labels_short: list[str] = []
    matrix: list[list[float]] = []
    for col, lab in comp_cols:
        if col not in iface.columns:
            continue
        row = []
        for met in methods_ord:
            iface_m = iface.loc[iface["method"] == met]
            x = pd.to_numeric(iface_m[col], errors="coerce").dropna()
            row.append(float(np.mean(x)) if len(x) else float("nan"))
        labels_short.append(lab)
        matrix.append(row)
    if matrix:
        mat = np.asarray(matrix, dtype=float)
        n_rows, n_met = mat.shape
        ypos = np.arange(n_rows)
        bar_h = 0.72 / max(n_met, 1)
        for mi, met in enumerate(methods_ord[:n_met]):
            offs = (mi - (n_met - 1) / 2.0) * bar_h
            ax_bar.barh(
                ypos + offs,
                mat[:, mi],
                height=bar_h * 0.92,
                label=met,
                color=_method_color(met),
                edgecolor=MUTED_LINE,
                linewidth=0.55,
                alpha=0.9,
            )
        ax_bar.set_yticks(ypos)
        ax_bar.set_yticklabels(labels_short, fontsize=9)
        ax_bar.set_xlabel("Mean rank-percentile sub-score (within method, peptides with ICS)")
        ax_bar.set_xlim(0, 1.05)
        ax_bar.axvline(0.5, color=MUTED_LINE, ls=":", lw=1, alpha=0.65)
        ax_bar.set_title("ICS decomposition — mean sub-scores by method")
        ax_bar.legend(frameon=False, ncol=min(4, n_met), loc="lower right", fontsize=8)
    _save(fig, out_dir, "Fig7e_interface_complementarity_comparison")


def fig7f_contact_heatmap(s8: pd.DataFrame, out_dir: Path, top_n: int = 36) -> None:
    if s8.empty or "source" not in s8.columns:
        return
    s8 = cohort_for_cross_method_plots(s8)
    if s8.empty or "method" not in s8.columns:
        return
    methods_ord = methods_in_order(s8)
    if not methods_ord:
        return
    metrics = ["residue_contact_count", "atomic_contact_count", "interface_hbond_count"]
    for c in metrics:
        if c not in s8.columns:
            return
    by_m = {m: set(s8.loc[s8["method"] == m, "target_id"].astype(str)) for m in methods_ord}
    common = set.intersection(*[by_m[m] for m in methods_ord]) if len(methods_ord) > 1 else by_m[methods_ord[0]]
    if len(common) < 3:
        common = set.union(*[by_m[m] for m in methods_ord])
    s8c = s8[s8["target_id"].astype(str).isin(common)].copy()
    pooled = (
        s8c.groupby("target_id")[metrics[0]]
        .mean()
        .sort_values(ascending=False)
    )
    tops = list(pooled.head(min(top_n, len(pooled))).index)
    if not tops:
        return
    n_m = len(methods_ord)
    fig, axes = plt.subplots(
        1, n_m, figsize=(max(10.0, 3.4 * n_m), 4.0), squeeze=False, layout="constrained"
    )
    axes = axes.ravel()
    im_last = None
    for ax, met in zip(axes, methods_ord):
        sub = s8c.loc[s8c["method"] == met].groupby("target_id")[metrics].mean()
        sub = sub.reindex(tops).apply(pd.to_numeric, errors="coerce")
        if sub.isna().all().all():
            ax.set_axis_off()
            continue
        zz = sub.T.astype(float)
        arr = zz.to_numpy(dtype=float)
        mu = np.nanmean(arr, axis=1, keepdims=True)
        sig = np.nanstd(arr, axis=1, keepdims=True)
        zn = (arr - mu) / (sig + 1e-9)
        zn = np.where(np.isfinite(zn), zn, 0.0)
        im_last = ax.imshow(zn, aspect="auto", cmap=NATURE_DIVERGING, vmin=-2, vmax=2)
        ax.set_yticks(range(zz.shape[0]))
        ax.set_yticklabels(["contacts", "atom pairs", "iface H-bonds"], fontsize=8)
        ax.set_xticks(range(zz.shape[1]))
        ax.set_xticklabels(list(zz.columns), rotation=55, ha="right", fontsize=6)
        ax.set_title(met, fontsize=10)
    fig.suptitle("Fig 7f — Interface contact metrics (z-score within method)", y=1.05)
    if im_last is not None:
        fig.colorbar(im_last, ax=axes.tolist(), fraction=0.02, pad=0.04, label="Row z-score")
    _save(fig, out_dir, "Fig7f_contact_enrichment_heatmap")


def fig_sup_method_target_scores(s11: pd.DataFrame, out_dir: Path, top_n: int = 40) -> None:
    """多方法时：靶标 ×（分数_方法）宽表，列方向 z-score（与旧版 supplementary 文件名兼容）。"""
    if s11.empty or "method" not in s11.columns:
        return
    methods = methods_in_order(s11)
    if len(methods) < 2:
        return
    scores = [c for c in ("FCS", "SCS", "ICS", "ALI", "OBCS") if c in s11.columns]
    if len(scores) < 2:
        return
    n_t = s11.groupby("target_id").size().sort_values(ascending=False)
    tops = list(n_t.head(min(top_n, len(n_t))).index)
    parts: list[pd.DataFrame] = []
    for sc in scores:
        g = s11.groupby(["target_id", "method"])[sc].mean().unstack("method")
        g = g.reindex(tops)
        g.columns = [f"{sc}_{c}" for c in g.columns]
        parts.append(g)
    wide = pd.concat(parts, axis=1)
    num = wide.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if len(num) < 2:
        return
    z = (num - num.mean()) / (num.std(ddof=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(max(10.5, 0.32 * z.shape[1]), max(5.8, 0.17 * len(z))))
    im = ax.imshow(z.values, aspect="auto", cmap=NATURE_DIVERGING, vmin=-2, vmax=2)
    ax.set_yticks(range(len(z.index)))
    ax.set_yticklabels(list(z.index), fontsize=7)
    ax.set_xticks(range(len(z.columns)))
    ax.set_xticklabels(list(z.columns), rotation=70, ha="right", fontsize=6)
    ax.set_title("Supplementary — Target-level scores by method (column z-score)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    _save(fig, out_dir, "Supplementary_target_level_heatmap")


def fig_sup_target_heatmap(s12: pd.DataFrame, out_dir: Path, top_n: int = 40) -> None:
    if "target_id" not in s12.columns:
        return
    s12 = s12.sort_values("n_peptides", ascending=False) if "n_peptides" in s12.columns else s12
    s12 = s12.head(min(top_n, len(s12)))
    cols = [c for c in s12.columns if c in ("mean_FCS", "mean_SCS", "mean_ICS", "mean_ALI", "mean_OBCS")]
    if len(cols) < 2:
        cols = [c for c in s12.columns if c.startswith("mean_") and "_generated" not in c and "_reference" not in c][:8]
    num = s12[cols].apply(pd.to_numeric, errors="coerce")
    num.index = s12["target_id"].values
    num = num.dropna(how="all")
    if len(num) == 0:
        return
    z = (num - num.mean()) / (num.std(ddof=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(7.5, max(6.0, len(z) * 0.18)))
    im = ax.imshow(z.values, aspect="auto", cmap=NATURE_DIVERGING, vmin=-2, vmax=2)
    ax.set_yticks(range(len(z.index)))
    ax.set_yticklabels(list(z.index), fontsize=7)
    ax.set_xticks(range(len(z.columns)))
    ax.set_xticklabels([c.replace("mean_", "") for c in z.columns], rotation=35, ha="right")
    ax.set_title("Supplementary — Target-level score heatmap (z-scored)")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    plt.tight_layout()
    _save(fig, out_dir, "Supplementary_target_level_heatmap")


def write_manifest(path: Path, entries: list[tuple[str, str]]) -> None:
    lines = ["# Figure manifest", "", "所有图由 `06_make_figures.py` 从下列数据表生成。", ""]
    for stem, desc in entries:
        lines.append(f"## `{stem}.png` / `.pdf`")
        lines.append("")
        lines.append(desc)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_dirs()
    tabs = args.tables_dir or paths.tables
    figs = args.figures_dir or paths.figures
    figs.mkdir(parents=True, exist_ok=True)
    apply_nature_style()

    log = setup_run_logger(paths.logs, "06_make_figures")
    _append_build_log(log)

    s4 = pd.read_csv(Path(tabs) / "Table_S4_foldability_summary.csv")
    s11 = pd.read_csv(Path(tabs) / "Table_S11_biophysical_summary_scores.csv")
    s12 = pd.read_csv(Path(tabs) / "Table_S12_target_level_summary.csv")
    s8_path = Path(tabs) / "Table_S8_interface_metrics.csv"
    s8 = pd.read_csv(s8_path) if s8_path.exists() else pd.DataFrame()
    s7_path = Path(tabs) / "Table_S7_aggregation_hotspot_summary.csv"

    s4_plot, s11_plot = attach_method_filtered(s4, s11)
    if s11_plot.empty:
        log.warning("Cohort for cross-method plots is empty; check Table_S11 `source` and all_samples rows.")
    n_methods = len(methods_in_order(s11_plot)) if not s11_plot.empty else 0
    log.info("Cross-method cohort: n_peptides=%s, methods=%s", len(s11_plot), n_methods)

    manifest: list[tuple[str, str]] = []

    fig7a_foldability(s4_plot, s11_plot, figs)
    manifest.append(
        (
            "Fig7a_foldability_comparison",
            "- **Table_S11**（`FCS`）与 **Table_S4** 同键对齐。\n"
            "- 由 ``source`` 解析 **Ours / RFdiffusion / ProteinGenerator / BindCraft**；"
            "SOTA 仅使用 ``all_samples:`` 全量指标行，避免 ``baseline_input_index:`` 重复与缺失界面。\n"
            "- 按方法着色的 FCS 密度直方图。",
        )
    )

    fig7b_clash_hbond(s4_plot, figs)
    manifest.append(
        (
            "Fig7b_clash_hbond_hydrophobicity",
            "- **Table_S4**：`s2_clash_count`, `s2_intrapeptide_hbond_count`, `s2_hydrophobic_cohesion_score`（`s2_analysis_status==ok`），按方法着色。",
        )
    )

    fig7c_solubility_hotspot(s11_plot, s7_path, figs)
    manifest.append(
        (
            "Fig7c_solubility_hotspot_comparison",
            "- **Table_S11**（`SCS`）与 **Table_S7**（`hotspot_burden`, `aggregation_liability_index`），按方法分色散点。",
        )
    )

    fig7d_tradeoff(s11_plot, figs)
    manifest.append(
        (
            "Fig7d_interface_vs_solubility_tradeoff",
            "- **Table_S11**：`ICS` vs `SCS`，按方法分色。",
        )
    )

    fig7e_ics(s11_plot, Path(tabs), figs)
    manifest.append(
        (
            "Fig7e_interface_complementarity_comparison",
            "- **Table_S11**（`ICS` 及 `ICS_r_*`）与 **Table_S1**（`length`）。\n"
            "- 面板：按方法的 ICS 直方图+KDE；方法箱线；肽长度四分位；高样本靶标；"
            "ICS 六项子指标 **分组横向条（并排按方法）**。",
        )
    )

    fig7f_contact_heatmap(s8, figs)
    manifest.append(
        (
            "Fig7f_contact_enrichment_heatmap",
            "- **Table_S8**：按方法分面板（含 **Ours** 与三种 SOTA），"
            "各方法共有靶标子集上行内 z-score；指标为 residue / atomic contacts 与界面 H-bond。",
        )
    )

    if n_methods >= 2:
        fig_sup_method_target_scores(s11_plot, figs)
        manifest.append(
            (
                "Supplementary_target_level_heatmap",
                "- **Table_S11** 按 `target_id`×`method` 聚合 `FCS/SCS/ICS/ALI/OBCS`，宽列 `分数_方法`，列 z-score。",
            )
        )
    else:
        fig_sup_target_heatmap(s12, figs)
        manifest.append(
            (
                "Supplementary_target_level_heatmap",
                "- **Table_S12_target_level_summary.csv**：靶标级 `mean_*` 分数子集（单方法回退），行 z-score。",
            )
        )

    write_manifest(figs / "figure_manifest.md", manifest)
    log.info("Figures written to %s", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

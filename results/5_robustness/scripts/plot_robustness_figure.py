#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature-style 8-panel figure from aggregated robustness tables.
Reads: tables/robustness_aggregate_by_condition.csv, tables/Table_5_robustness_summary.csv,
       cases/selected_cases.json (optional).
Writes: figures/Figure_5_robustness_main.pdf|.png, figures/Figure_5_robustness_caption.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_ROB = _SCRIPTS.parent
_FIG = _ROB / "figures"
_TAB = _ROB / "tables"
_CASE = _ROB / "cases"
_MET = _ROB / "metrics"


def _style():
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
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            # Illustrator：TrueType 嵌入为文字对象（非 Type3 轮廓）
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # 可选 SVG 导出时保留 <text>，便于直接进 AI 编辑
            "svg.fonttype": "none",
            # 减少过度简化路径，避免在 AI 里打散成奇怪子路径
            "path.simplify": False,
        }
    )


def _norm_x(pert: str, lv: float) -> float:
    if pert == "pocket_noise":
        return lv / 2.0 if lv <= 2.0 else lv / 2.0
    return lv / 40.0


def _affinity_higher(v: float) -> float:
    return -float(v)


def _affinity_drop_row(ss: pd.DataFrame, lv: float, c: str) -> float:
    ss = ss.sort_values("level_value")
    clean = _affinity_higher(float(ss.iloc[0][c]))
    row = ss[np.isclose(ss["level_value"].astype(float), lv)]
    if row.empty:
        return 0.0
    v = _affinity_higher(float(row.iloc[0][c]))
    return (clean - v) / max(abs(clean), 1e-6) * 100.0


def _drop_simple(ss: pd.DataFrame, lv: float, c: str) -> float:
    ss = ss.sort_values("level_value")
    clean = float(ss.iloc[0][c])
    row = ss[np.isclose(ss["level_value"].astype(float), lv)]
    if row.empty or not np.isfinite(clean):
        return 0.0
    v = float(row.iloc[0][c])
    return (clean - v) / max(abs(clean), 1e-6) * 100.0


def panel_schematic(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("a", loc="left", fontweight="bold", fontsize=10)
    ax.text(0.5, 0.92, "Target perturbation design", ha="center", fontsize=9)
    y0 = 0.55
    w, h = 0.22, 0.28
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    titles = ["Structure dropout", "Pocket noise (Å)", "Sequence crop"]
    for i, (c, t) in enumerate(zip(colors, titles)):
        x = 0.08 + i * 0.3
        rect = mpatches.FancyBboxPatch(
            (x, y0), w, h, boxstyle="round,pad=0.01", linewidth=1.0, edgecolor=c, facecolor="#f5f5f5"
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y0 + h + 0.03, t, ha="center", fontsize=8, color=c)
        for k, lab in enumerate(["L0", "L1", "L2", "L3", "L4"]):
            ax.plot([x + 0.04 + k * 0.04], [y0 + 0.08], "o", ms=3, color=c, alpha=0.3 + 0.15 * k)
    ax.text(0.5, 0.12, "Five severity levels per family (0% → 40% or 0–2 Å)", ha="center", fontsize=7, color="#333")


def main() -> int:
    _style()
    agg_path = _TAB / "robustness_aggregate_by_condition.csv"
    sum_path = _TAB / "Table_5_robustness_summary.csv"
    if not agg_path.is_file():
        print(f"[plot] missing {agg_path}; run pipeline aggregate first.", file=sys.stderr)
        return 1
    agg = pd.read_csv(agg_path)
    summary = pd.read_csv(sum_path) if sum_path.is_file() else pd.DataFrame()

    fig = plt.figure(figsize=(7.2, 10.0))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_schematic(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title("b", loc="left", fontweight="bold", fontsize=10)
    cols = {"structure_missing": "#4C72B0", "pocket_noise": "#55A868", "sequence_trunc": "#C44E52"}
    for pert, col in cols.items():
        sub = agg[agg["perturb_type"] == pert].sort_values("level_value")
        if sub.empty:
            continue
        xn = [_norm_x(pert, v) for v in sub["level_value"].astype(float).values]
        y = [_affinity_higher(v) for v in sub["affinity_mean"].astype(float).values]
        y0 = y[0]
        yn = [(v - y0) / max(abs(y0), 1e-6) for v in y]
        ax_b.plot(xn, yn, "-o", ms=3, lw=1.2, color=col, label=pert.replace("_", " "))
    ax_b.axhline(0, color="#999", lw=0.6, ls="--")
    ax_b.set_xlabel("Normalized perturbation intensity")
    ax_b.set_ylabel("Normalized affinity (↑ better)")
    ax_b.legend(frameon=False, loc="lower left")

    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title("c", loc="left", fontweight="bold", fontsize=10)
    rows = []
    row_labels = []
    for pert in ["structure_missing", "pocket_noise", "sequence_trunc"]:
        sub_sorted = agg[agg["perturb_type"] == pert].sort_values("level_value")
        if sub_sorted.empty:
            continue
        for _, r in sub_sorted.iterrows():
            lv = float(r["level_value"])
            row_labels.append(f"{pert[:3]} {lv:g}")
            rows.append(
                [
                    _affinity_drop_row(sub_sorted, lv, "affinity_mean"),
                    _drop_simple(sub_sorted, lv, "success_rate"),
                    _drop_simple(sub_sorted, lv, "stability_mean"),
                    _drop_simple(sub_sorted, lv, "solubility_mean"),
                ]
            )
    if rows:
        arr = np.array(rows, dtype=float)
        ny, nx = arr.shape
        # 用 pcolormesh 输出矢量色块；imshow 在 PDF 中常为嵌入栅格，AI 难编辑
        xc = np.arange(nx + 1)
        yc = np.arange(ny + 1)
        im = ax_c.pcolormesh(
            xc,
            yc,
            arr,
            shading="flat",
            cmap="magma",
            vmin=0,
            vmax=80,
            rasterized=False,
        )
        ax_c.set_xlim(xc[0], xc[-1])
        ax_c.set_ylim(yc[0], yc[-1])
        ax_c.set_xticks(np.arange(nx) + 0.5)
        ax_c.set_xticklabels(["Aff.", "Succ.", "Stab.", "Sol."])
        ax_c.set_yticks(np.arange(ny) + 0.5)
        ax_c.set_yticklabels(row_labels, fontsize=5)
        ax_c.set_aspect("auto")
        plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.02, label="Relative drop %")
    ax_c.set_xlabel("Metric")

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title("d", loc="left", fontweight="bold", fontsize=10)
    if not summary.empty:
        sub = summary[summary["metric"] == "affinity_hdock"]
        if not sub.empty:
            x = np.arange(len(sub))
            ax_d.bar(x, sub["AUDC"].fillna(0), color="#4C72B0", width=0.55, label="AUDC (affinity)")
            ax2 = ax_d.twinx()
            ax2.plot(
                x,
                sub["sensitivity_slope"].fillna(0),
                "o-",
                color="#C44E52",
                ms=4,
                lw=1,
                label="Sensitivity slope",
            )
            ax_d.set_xticks(x)
            ax_d.set_xticklabels([str(s)[:4] for s in sub["perturbation_type"]], rotation=15, ha="right")
            ax_d.set_ylabel("AUDC (%·norm)")
            ax2.set_ylabel("Slope vs. level")
            h1, l1 = ax_d.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax_d.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6, loc="upper left")
            txt = []
            for _, r in sub.iterrows():
                t10 = r["drop_10_threshold"] if "drop_10_threshold" in r.index else float("nan")
                t20 = r["drop_20_threshold"] if "drop_20_threshold" in r.index else float("nan")
                s10 = f"{t10:.3g}" if pd.notna(t10) else "NA"
                s20 = f"{t20:.3g}" if pd.notna(t20) else "NA"
                txt.append(f"{str(r['perturbation_type'])[:4]}: τ10={s10}, τ20={s20}")
            ax_d.text(0.02, 0.02, "\n".join(txt), transform=ax_d.transAxes, fontsize=5, va="bottom")

    def fine_curve(ax, pert: str, letter: str):
        ax.set_title(f"{letter}", loc="left", fontweight="bold", fontsize=10)
        sub = agg[agg["perturb_type"] == pert].sort_values("level_value")
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center")
            return
        x = sub["level_value"].astype(float).values
        ax.plot(x, sub["affinity_mean"], "-o", color="#4C72B0", ms=3, label="HDOCK (raw)")
        ax2 = ax.twinx()
        ax2.plot(x, sub["success_rate"], "-s", color="#C44E52", ms=3, label="Success rate")
        ax.set_xlabel("Perturbation level")
        ax.set_ylabel("HDOCK (lower better)", color="#4C72B0")
        ax2.set_ylabel("Success rate", color="#C44E52")
        ax.legend(loc="upper left", frameon=False, fontsize=6)
        ax2.legend(loc="upper right", frameon=False, fontsize=6)

    ax_e = fig.add_subplot(gs[2, 0])
    fine_curve(ax_e, "structure_missing", "e")
    ax_f = fig.add_subplot(gs[2, 1])
    fine_curve(ax_f, "pocket_noise", "f")
    ax_g = fig.add_subplot(gs[3, 0])
    fine_curve(ax_g, "sequence_trunc", "g")

    ax_h = fig.add_subplot(gs[3, 1])
    ax_h.set_title("h", loc="left", fontweight="bold", fontsize=10)
    case_path = _CASE / "selected_cases.json"
    if case_path.is_file():
        with open(case_path, "r", encoding="utf-8") as f:
            cases = json.load(f)
        if cases:
            tid = str(cases[0].get("target_id", "target"))
            clean_a = float(cases[0].get("affinity_hdock_clean", 0))
            pert_a = float(cases[0].get("affinity_hdock_pert", 0))
            ax_h.bar(
                [0, 1],
                [-clean_a, -pert_a],
                color=["#4C72B0", "#C44E52"],
                width=0.5,
                label="−HDOCK (↑)",
            )
            ax_h.set_xticks([0, 1])
            ax_h.set_xticklabels(["clean", "perturbed"])
            ax_h.set_ylabel("Negated HDOCK")
            ax_h.set_title(f"Representative: {tid}", fontsize=8)
    else:
        ax_h.text(0.5, 0.5, "Run pipeline to export cases/selected_cases.json", ha="center", va="center")

    fig.suptitle(
        "Degradation landscape of the final model under target perturbations",
        fontsize=11,
        y=0.995,
    )
    _FIG.mkdir(parents=True, exist_ok=True)
    out_pdf = _FIG / "Figure_5_robustness_main.pdf"
    out_png = _FIG / "Figure_5_robustness_main.png"
    out_svg = _FIG / "Figure_5_robustness_main.svg"
    # 显式关闭栅格化；PDF 用 TrueType 文字（fonttype 42）
    fig.savefig(
        out_pdf,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"Creator": "matplotlib (Illustrator-friendly)"},
    )
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    cap = _FIG / "Figure_5_robustness_caption.txt"
    cap.write_text(
        """
Figure 5. Robustness of the final peptide design model to imperfect target information.

(a) Schematic of three perturbation families applied only at test time: random residue-level structure dropout in the receptor encoder input, Gaussian noise on pocket backbone atoms (within 10 Å of the peptide in the crystal), and random contiguous cropping of the receptor sequence with aligned coordinates.

(b) Global affinity degradation trajectories after min–max normalization of raw perturbation strengths (structure/sequence: 0–40%; pocket: 0–2 Å). Curves show change in negated HDOCK score relative to the clean condition per family.

(c) Heatmap of relative performance drop (%) for affinity (negated HDOCK), composite success rate, stability, and solubility; rows enumerate perturbation family and severity.

(d) Summary bars from the analytical robustness table: area-under-degradation-curve (AUDC) for affinity and interpolated perturbation strengths at which relative drop exceeds 10% and 20% (see metrics definitions in README).

(e–g) Fine-grained mean HDOCK (left axis) and success rate (right axis) versus physical perturbation level for each family.

(h) Representative target panel: clean versus strongly perturbed structure-missing condition for the top-1 candidate, highlighting loss of predicted binding quality.

Metrics: HDOCK affinity (lower is better), FoldX-derived stability proxy, Protein-Sol solubility; success rate requires all three to pass configured thresholds.
""".strip()
        + "\n",
        encoding="utf-8",
    )
    print(f"[plot] wrote {out_pdf} {out_png} {out_svg} {cap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

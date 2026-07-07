"""图形导出：PDF + PNG。Nature 风格淡色配色；肽相关图可限定 Top-N（与 pipeline 一致）。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_hex
from scipy.stats import ranksums

log = logging.getLogger(__name__)

# --- Nature 期刊常见的低饱和度配色（与 scanpy 散点清淡分类色一致）---
PALETTE_SOFT: list[str] = [
    "#8EB6D9",  # powder blue — 柱状图主色，与 UMAP 主分类一致
    "#9DC9B8",  # seafoam
    "#E5B4B8",  # dusty rose
    "#C4B5D8",  # soft lavender
    "#D9C68A",  # muted gold
    "#A8BE9A",  # sage
    "#D4A89A",  # blush terracotta
    "#B8CDE8",  # sky tint
]

BAR_PRIMARY = PALETTE_SOFT[0]
BAR_SECONDARY = PALETTE_SOFT[1]
BAR_ACCENT = PALETTE_SOFT[2]

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "nature_soft_seq", ["#FAFCFE", "#E8F0F7", "#C5DBEB", "#9BBFDB", "#7BA3C9"]
)

# 发表用热图：行顺序（列名 → y 轴显示名）
HEATMAP_SCORE_ROW_ORDER: list[tuple[str, str]] = [
    ("blockade_similarity_score", "blockade_similarity"),
    ("T_cell_activation_prediction", "T_cell_activation"),
    ("IFNG_prediction", "IFNG"),
    ("cytotoxicity_prediction", "cytotoxicity"),
    ("tumor_suppression_prediction", "tumor_suppression"),
    ("exhaustion_down_prediction", "exhaustion_down"),
]

# pathway boxplot：逻辑顺序 PD1/PDL1 → TCR → NFAT → IFNG → exhaustion（与 pathway_scoring 列一致）
PATHWAY_BOXPLOT_ORDER: list[tuple[str, str]] = [
    ("score_PD1_PDL1_core", "PD1_PDL1"),
    ("score_TCR_activation", "TCR"),
    ("score_NFAT", "NFAT"),
    ("score_IFNG_response", "IFNG"),
    ("score_exhaustion", "exhaustion"),
]

# risk boxplot：toxicity → proliferation → inflammation → EMT → stemness
RISK_BOXPLOT_ORDER: list[tuple[str, str]] = [
    ("risk_toxicity_dna", "Toxicity"),
    ("risk_proliferation", "Proliferation"),
    ("risk_inflammatory", "Inflammation"),
    ("risk_emt", "EMT"),
    ("risk_stemness", "Stemness"),
]

# 发表图：PBMC / TIL 对照色（Illustrator 友好）
COND_COLOR_PBMC = "#0072B2"
COND_COLOR_TIL = "#009E73"  # Wong 绿，色盲友好

RECOMMENDATION_COLORS: dict[str, str] = {
    "Strong candidate": PALETTE_SOFT[0],
    "Moderate candidate": PALETTE_SOFT[1],
    "Not recommended": PALETTE_SOFT[2],
}

# 肽排名图（lollipop）：中性茎 + 端点色；Top3 为同系深浅橙（Wong 友好）
_RANK_LOLLIPOP_STEM = "#D8DEE9"
_RANK_MARKER_OTHER = "#3C5A78"
_RANK_MARKER_TOP = ("#9C3D00", "#D55E00", "#E89C6A")  # 第1–3名（由上至下为图中末三行）

# 发表用 UMAP：色盲友好、饱和度较高（与浅 pastel 区分）
UMAP_PUBLICATION_PALETTE: list[str] = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


def _try_set_cjk_font() -> None:
    """尽量使用可显示中文的字体，避免分组标题缺字。"""
    names = {f.name for f in fm.fontManager.ttflist}
    for pref in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
    ):
        if any(pref in n for n in names):
            matplotlib.rcParams["font.sans-serif"] = [pref, "DejaVu Sans", "Arial", "sans-serif"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


def _nature_rc() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4A4A4A",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def _palette_n_categories(n: int) -> list[str]:
    if n <= len(PALETTE_SOFT):
        return PALETTE_SOFT[:n]
    return [to_hex(c) for c in sns.color_palette("pastel", n_colors=n, desat=0.72)]


def _filter_peptides(df: pd.DataFrame, peptide_ids: Sequence[str] | None) -> pd.DataFrame:
    if not peptide_ids:
        return df
    ids = list(peptide_ids)
    sub = df[df["peptide_id"].isin(ids)].copy()
    if sub.empty:
        log.warning("peptide_ids 过滤后无行，退回全表")
        return df
    order = {pid: i for i, pid in enumerate(ids)}
    sub["_plot_order"] = sub["peptide_id"].map(lambda x: order.get(x, 9999))
    sub = sub.sort_values("_plot_order").drop(columns=["_plot_order"])
    return sub


def _save_both(fig: plt.Figure, path: Path) -> None:
    path_no_suffix = path
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_suffix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_publication_figure(fig: plt.Figure, path: Path, *, png_dpi: int = 300) -> None:
    """PDF 使用 TrueType 轮廓（fonttype 42），便于 Adobe Illustrator 编辑文字。"""
    path_no_suffix = path
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig.savefig(
            path_no_suffix.with_suffix(".pdf"),
            bbox_inches="tight",
            format="pdf",
        )
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)


def _save_volcano_publication(fig: plt.Figure, path: Path, *, png_dpi: int = 300) -> None:
    """火山图发表用保存（与 _save_publication_figure 相同）。"""
    _save_publication_figure(fig, path, png_dpi=png_dpi)


def _umap_display_label_map(config: dict[str, Any] | None, col: str) -> dict[str, str]:
    """config['umap_display_labels'][col] → {原始类别: 图上显示名}。"""
    if not config:
        return {}
    root = config.get("umap_display_labels")
    if not isinstance(root, dict):
        return {}
    block = root.get(col)
    if not isinstance(block, dict):
        return {}
    return {str(k): str(v) for k, v in block.items()}


def _umap_default_display_name(cat: str) -> str:
    return str(cat).replace("_", " ")


def _umap_publication_palette(n: int) -> list[str]:
    base = list(UMAP_PUBLICATION_PALETTE)
    if n <= len(base):
        return base[:n]
    extra = [to_hex(c) for c in sns.color_palette("tab10", n_colors=n - len(base))]
    return base + extra


def _plot_umap_publication(
    adata,
    color_col: str,
    path: Path,
    *,
    title: str,
    config: dict[str, Any] | None = None,
    point_size: float = 3.0,
) -> None:
    """
    发表风格 UMAP：小点、高对比色、类群质心文字标签、无刻度、轴名 UMAP1/2、图例外置；
    可选密度等高线（config['umap_show_density']）；PDF fonttype 42。
    """
    if "X_umap" not in adata.obsm:
        log.warning("adata.obsm 无 X_umap，跳过 UMAP 图")
        return
    _try_set_cjk_font()
    _nature_rc()

    xy = np.asarray(adata.obsm["X_umap"], dtype=float)
    labels = np.asarray(adata.obs[color_col].astype(str))
    ok = np.isfinite(xy).all(axis=1)
    xy, labels = xy[ok], labels[ok]
    if len(xy) == 0:
        return

    label_map = _umap_display_label_map(config, str(color_col))
    show_density = bool(config.get("umap_show_density")) if config else False

    uniq, counts = np.unique(labels, return_counts=True)
    # 先画细胞数多的类（底层），少的在上层
    order = uniq[np.argsort(-counts)]

    colors = _umap_publication_palette(len(order))
    color_by_cat = {c: colors[i] for i, c in enumerate(order)}

    fig_w, fig_h = 6.4, 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if show_density and len(xy) > 50:
        try:
            rng = np.random.default_rng(int(config.get("random_seed", 42)) if config else 42)
            n_cap = min(10000, len(xy))
            idx = rng.choice(len(xy), n_cap, replace=False) if len(xy) > n_cap else np.arange(len(xy))
            sxy = xy[idx]
            sns.kdeplot(
                x=sxy[:, 0],
                y=sxy[:, 1],
                ax=ax,
                levels=8,
                colors="#555555",
                linewidths=0.35,
                alpha=0.32,
                fill=False,
                zorder=0,
            )
        except Exception as exc:
            log.debug("UMAP 密度等高线跳过: %s", exc)

    for cat in order[::-1]:
        m = labels == cat
        if not m.any():
            continue
        c = color_by_cat[cat]
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=point_size,
            c=c,
            label=label_map.get(cat, _umap_default_display_name(cat)),
            edgecolors="white",
            linewidths=0.08,
            rasterized=True,
            zorder=2,
            alpha=0.92,
        )

    for cat in order:
        m = labels == cat
        if m.sum() < 5:
            continue
        cx = float(np.median(xy[m, 0]))
        cy = float(np.median(xy[m, 1]))
        txt = label_map.get(cat, _umap_default_display_name(cat))
        t = ax.text(
            cx,
            cy,
            txt,
            fontsize=8.5,
            fontweight="semibold",
            color="#111111",
            ha="center",
            va="center",
            zorder=10,
        )
        t.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0, labelleft=False, labelbottom=False)
    ax.set_xlabel("UMAP1", fontsize=10)
    ax.set_ylabel("UMAP2", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=6.5,
            markerfacecolor=color_by_cat[c],
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=label_map.get(c, _umap_default_display_name(c)),
        )
        for c in order
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fancybox=False,
        edgecolor="#BBBBBB",
        fontsize=8.5,
        handletextpad=0.6,
    )

    pad = 0.04 * max(xy[:, 0].max() - xy[:, 0].min(), xy[:, 1].max() - xy[:, 1].min(), 1e-9)
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)

    fig.subplots_adjust(left=0.10, right=0.74, top=0.94, bottom=0.10)
    _save_publication_figure(fig, path, png_dpi=300)


def plot_umap_condition(
    adata, cond_col: str, path: Path, *, config: dict[str, Any] | None = None
) -> None:
    _plot_umap_publication(
        adata,
        cond_col,
        path,
        title="UMAP by condition",
        config=config,
    )


def plot_umap_celltype(adata, ct_col: str, path: Path, *, config: dict[str, Any] | None = None) -> None:
    _plot_umap_publication(
        adata,
        ct_col,
        path,
        title="UMAP by cell type",
        config=config,
    )


def _pvalue_to_significance_stars(p: float) -> str:
    """标准星号：* p<0.05, ** p<0.01, *** p<0.001；否则返回空串。"""
    if not np.isfinite(p) or p >= 0.05:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def _pathpatch_xcenter_data(ax: plt.Axes, patch: PathPatch) -> float:
    bb = patch.get_extents()
    inv = ax.transData.inverted()
    x0 = inv.transform((bb.x0, bb.y0))[0]
    x1 = inv.transform((bb.x1, bb.y0))[0]
    return float(0.5 * (x0 + x1))


def plot_pathway_boxplot(adata, config: dict[str, Any], path: Path) -> None:
    """
    发表风格：通路按 PD1_PDL1→TCR→NFAT→IFNG→exhaustion；
    PBMC 蓝 / TIL 绿、细线、无离群点；Wilcoxon rank-sum（ranksums）星号括号；
    PDF fonttype 42 便于 Illustrator。
    """
    cond = str(config["condition_column"])
    ctrl = str(config["control_label"])
    treat = str(config["treatment_label"])
    row_spec = [(c, lab) for c, lab in PATHWAY_BOXPLOT_ORDER if c in adata.obs.columns]
    if not row_spec:
        log.warning("无 pathway score 列，跳过 boxplot")
        return
    used_cols = [c for c, _ in row_spec]
    short_labels = [lab for _, lab in row_spec]
    rename = dict(row_spec)

    _nature_rc()
    df = adata.obs[[cond] + used_cols].melt(id_vars=[cond], var_name="pathway", value_name="score")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df["pathway"] = df["pathway"].astype(str).map(rename)

    order = [lab for lab in short_labels if lab in set(df["pathway"].unique())]
    if not order:
        return

    hue_order = [ctrl, treat]
    palette = {ctrl: COND_COLOR_PBMC, treat: COND_COLOR_TIL}

    fig_w = max(9.4, 1.42 * len(order) + 4.2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.25))

    # 显式 boxprops：仅描边，勿设 facecolor；否则会把 hue×palette 的蓝/绿填充全部盖成单色（看起来像「没有颜色」）。
    sns.boxplot(
        data=df,
        x="pathway",
        y="score",
        hue=cond,
        ax=ax,
        order=order,
        hue_order=hue_order,
        palette=palette,
        linewidth=0.38,
        showfliers=False,
        boxprops={"edgecolor": "#333333", "linewidth": 0.38},
        whiskerprops={"linewidth": 0.38, "color": "#333333"},
        capprops={"linewidth": 0.38, "color": "#333333"},
        medianprops={"linewidth": 0.75, "color": "#1A1A1A"},
        flierprops={"marker": ""},
    )

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=28, ha="right", fontsize=9.5)
    ax.tick_params(axis="x", pad=12)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Pathway score", fontsize=10)
    ax.set_title("Pathway scores by condition", fontsize=11, pad=8)
    ax.legend(title="", frameon=False, fontsize=9)

    n_pw = len(order)
    paths = [c for c in ax.get_children() if isinstance(c, PathPatch)]
    if len(paths) < 2 * n_pw:
        log.warning("boxplot PathPatch 数量异常: %s (预期 %d)", len(paths), 2 * n_pw)
    else:
        y_min, y_max = ax.get_ylim()
        y_span = max(y_max - y_min, 1e-9)
        y_annot_max = y_max

        for pw_idx, short in enumerate(order):
            sub_c = df[(df["pathway"] == short) & (df[cond] == ctrl)]["score"].values
            sub_t = df[(df["pathway"] == short) & (df[cond] == treat)]["score"].values
            if len(sub_c) < 3 or len(sub_t) < 3:
                continue
            _, p = ranksums(sub_t, sub_c)
            stars = _pvalue_to_significance_stars(p)
            if not stars:
                continue

            # seaborn：先按 hue 再按 x — hue_order 为 [ctrl, treat]
            k0 = 0 * n_pw + pw_idx
            k1 = 1 * n_pw + pw_idx
            if k0 >= len(paths) or k1 >= len(paths):
                continue
            x1 = _pathpatch_xcenter_data(ax, paths[k0])
            x2 = _pathpatch_xcenter_data(ax, paths[k1])
            x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)

            local_top = float(np.nanpercentile(np.r_[sub_c, sub_t], 99))
            y_base = local_top + 0.04 * y_span
            h = 0.028 * y_span
            ax.plot([x1, x1, x2, x2], [y_base, y_base + h, y_base + h, y_base], color="#222222", lw=0.7, clip_on=False)
            ax.text(
                0.5 * (x1 + x2),
                y_base + h + 0.012 * y_span,
                stars,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="semibold",
                color="#111111",
                clip_on=False,
            )
            y_annot_max = max(y_annot_max, y_base + h + 0.05 * y_span)

        ax.set_ylim(y_min, max(y_max, y_annot_max + 0.02 * y_span))

    fig.subplots_adjust(bottom=0.30, left=0.10, right=0.98)
    fig.text(
        0.5,
        0.02,
        f"Wilcoxon rank-sum ({treat} vs {ctrl}):  *  p<0.05,  **  p<0.01,  ***  p<0.001",
        ha="center",
        fontsize=7.5,
        color="#333333",
        transform=fig.transFigure,
    )
    _save_publication_figure(fig, path, png_dpi=300)


# 发表用火山图：未在 config 指定时仅标注这些基因（色盲友好色见 plot_volcano）
VOLCANO_PUBLICATION_DEFAULT_GENES: tuple[str, ...] = (
    "PDCD1",
    "CD274",  # PD-L1；强 filter_genes 下常需经 config 补回后才出现在 DEG 表
    "PDCD1LG2",  # PD-L2
    "HAVCR2",
    "LAG3",
    "STAT1",
    "IFNG",
    "GZMB",
    "PRF1",
)
VOLCANO_COLOR_NS = "#B0B0B0"
VOLCANO_COLOR_UP = "#D55E00"
VOLCANO_COLOR_DOWN = "#0072B2"

# 火山图点名：用户常用名 -> AnnData / DEG 中可能出现的符号顺序
_VOLCANO_GENE_SYMBOL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "PDL1": ("CD274", "PDL1"),
    "PD-L1": ("CD274", "PDL1"),
    "PDL-1": ("CD274", "PDL1"),
}


def _volcano_resolve_gene_row(d: pd.DataFrame, symbol: str) -> tuple[str, pd.Series] | None:
    """在 DEG 表 d['gene'] 中解析符号（含 PD-L1 / PDL1 -> CD274）。"""
    sym = str(symbol).strip()
    if not sym:
        return None
    tried: list[str] = []
    pool = (sym,)
    u = sym.upper().replace(" ", "")
    if u in _VOLCANO_GENE_SYMBOL_CANDIDATES:
        pool = _VOLCANO_GENE_SYMBOL_CANDIDATES[u] + (sym,)
    for cand in pool:
        if not cand or cand in tried:
            continue
        tried.append(cand)
        sub = d[d["gene"] == cand]
        if not sub.empty:
            return cand, sub.iloc[0]
    return None


def plot_volcano(
    deg: pd.DataFrame,
    path: Path,
    padj_col: str = "pvals_adj",
    fc_col: str = "logfoldchanges",
    *,
    padj_thr: float = 0.05,
    abs_lfc_thr: float = 0.5,
    padj_thr_vline: float | None = None,
    abs_lfc_thr_vline: float | None = None,
    ymax: float = 100.0,
    xmax_cap: float = 4.2,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    label_genes: Sequence[str] | None = None,
    label_groups: list[dict[str, Any]] | None = None,
    gene_display: dict[str, str] | None = None,
    prior_down_genes: Sequence[str] | None = None,
    prior_down_caption: str | None = None,
    use_adjust_text: bool = True,
) -> None:
    """
    发表风格火山图：x 为 log2FC，y 为 -log10(adjusted p-value)，y 截断至 ymax。

    - 非显著：默认用 **hexbin 密度底图**（连续灰区，避免上万离散圆点显「碎」）；点数过少时回退为无描边小散点；
      上调 #D55E00；下调 #0072B2（Wong 色盲友好）；显著点保留细白描边便于辨识。
    - 着色：由 ``padj_thr``、``abs_lfc_thr`` 决定蓝/橙/灰（与第三层签名一致时常呈中间灰、两翼有色）。
    - 虚线：可用 ``padj_thr_vline`` / ``abs_lfc_thr_vline`` 单独指定（默认与着色阈值相同）；
      若仅放宽着色而虚线仍要 ±0.5 / padj=0.05 的发表版式，请把二者分开配置。
    - x 轴：若传入 ``xlim=(xmin, xmax)`` 则固定范围；否则按数据 99.5% 分位估计且不超过 ``xmax_cap``。
    - 画布：``figsize=(w,h)`` 英寸；不设时默认较窄（约 4.25×4.45），避免横向过宽。
    - 图例：右上、白底灰框（与常见 Nature 风格火山图一致）。
    - 基因标签：若已安装 ``adjusttext`` 且 ``use_adjust_text`` 为 True，则自动避让；否则沿用偏移标注。
    - 未传 label_groups / label_genes 时，仅标注 VOLCANO_PUBLICATION_DEFAULT_GENES。
    - PDF 以 pdf.fonttype=42 保存，便于 Illustrator 编辑文字。

    label_groups: 可选分组标题 + 基因列表（非空时启用分组逻辑）；
    gene_display: 符号 -> 覆盖显示文本。
    prior_down_genes: 可选；对解析到的基因若 log₂FC<0 且非上调，则与「下调」同色绘制（
      不改变坐标或 p 值；用于区室共线等导致 FDR 不显著时的生物学参照轴展示）。
    prior_down_caption: 非空时在图下方添加说明脚注（强烈建议与 prior_down_genes 同用）。
    """
    if deg.empty or fc_col not in deg.columns:
        log.warning("DEG 为空，跳过火山图")
        return
    _try_set_cjk_font()
    gene_display = dict(gene_display or {})
    if (not label_groups) and (not label_genes):
        label_genes = list(VOLCANO_PUBLICATION_DEFAULT_GENES)

    _nature_rc()
    d = deg.copy()
    d["gene"] = d["gene"].astype(str)
    d["logFC"] = pd.to_numeric(d[fc_col], errors="coerce")
    d = d[np.isfinite(d["logFC"])].copy()

    padj_num = pd.to_numeric(d[padj_col], errors="coerce").fillna(1.0)
    padj_safe = np.clip(padj_num.values, 1e-300, None)
    d["neglog10padj"] = np.clip(-np.log10(padj_safe), 0.0, float(ymax))

    lfc = d["logFC"]
    padj = padj_num
    sig_up = (padj < padj_thr) & (lfc > abs_lfc_thr)
    sig_down_stat = (padj < padj_thr) & (lfc < -abs_lfc_thr)
    prior_syms: set[str] = set()
    for g in prior_down_genes or []:
        g = str(g).strip()
        if not g:
            continue
        hit = _volcano_resolve_gene_row(d, g)
        if hit:
            prior_syms.add(hit[0])
    prior_mask = d["gene"].isin(prior_syms) if prior_syms else pd.Series(False, index=d.index)
    # 仅当真实 logFC<0 且非「统计上调」时，才把 prior 基因并入下调着色（不伪造统计显著）
    sig_down = sig_down_stat | (prior_mask & (lfc < 0) & ~sig_up)
    not_sig = ~(sig_up | sig_down)

    pvline = float(padj_thr if padj_thr_vline is None else padj_thr_vline)
    lvline = float(abs_lfc_thr if abs_lfc_thr_vline is None else abs_lfc_thr_vline)
    pvline = float(np.clip(pvline, 1e-300, 1.0 - 1e-12))
    y_thr = float(-np.log10(pvline))

    if xlim is not None and len(xlim) == 2:
        xmin, xmax = float(xlim[0]), float(xlim[1])
    else:
        data_max = float(np.nanpercentile(np.abs(d["logFC"].values), 99.5))
        xmax_abs = float(np.clip(max(data_max * 1.08, abs_lfc_thr * 1.5), abs_lfc_thr * 1.2, float(xmax_cap)))
        xmin, xmax = -xmax_abs, xmax_abs
    y_max_plot = float(ymax)

    _fw, _fh = (figsize if figsize is not None else (4.25, 4.45))
    fig, ax = plt.subplots(figsize=(_fw, _fh))
    # 非显著：hexbin 密度云（连续灰，不像离散圆点「散」）；显著点略大、白边
    pt_sig = 10
    _edge_w_sig = 0.28
    use_hexbin_ns = bool(not_sig.any()) and int(not_sig.sum()) >= 150
    x_ns = d.loc[not_sig, "logFC"].values
    y_ns = d.loc[not_sig, "neglog10padj"].values

    if use_hexbin_ns:
        nbin = int(np.clip(35 + np.sqrt(len(x_ns)) / 4.0, 42, 90))
        hb = ax.hexbin(
            x_ns,
            y_ns,
            gridsize=nbin,
            cmap="Greys",
            mincnt=1,
            alpha=0.82,
            linewidths=0,
            edgecolors="none",
            extent=(xmin, xmax, 0.0, y_max_plot),
            zorder=1,
            rasterized=True,
        )
        arr = hb.get_array()
        if arr is not None and len(arr) > 0:
            pos = arr[arr > 0]
            if len(pos):
                vmax = float(np.percentile(pos, 97))
                hb.set_clim(0.0, max(vmax, 1.0))
    else:
        if len(x_ns) > 0:
            ax.scatter(
                x_ns,
                y_ns,
                c=VOLCANO_COLOR_NS,
                s=5,
                alpha=0.42,
                edgecolors="none",
                linewidths=0,
                label="Not significant",
                rasterized=True,
                zorder=1,
            )
    ax.scatter(
        d.loc[sig_down, "logFC"],
        d.loc[sig_down, "neglog10padj"],
        c=VOLCANO_COLOR_DOWN,
        s=pt_sig,
        alpha=0.95,
        edgecolors="white",
        linewidths=_edge_w_sig,
        label="Downregulated",
        rasterized=True,
        zorder=3,
    )
    ax.scatter(
        d.loc[sig_up, "logFC"],
        d.loc[sig_up, "neglog10padj"],
        c=VOLCANO_COLOR_UP,
        s=pt_sig,
        alpha=0.95,
        edgecolors="white",
        linewidths=_edge_w_sig,
        label="Upregulated",
        rasterized=True,
        zorder=4,
    )

    ax.axvline(lvline, color="#333333", lw=0.9, linestyle="--", zorder=2, alpha=0.85)
    ax.axvline(-lvline, color="#333333", lw=0.9, linestyle="--", zorder=2, alpha=0.85)
    ax.axhline(y_thr, color="#333333", lw=0.9, linestyle="--", zorder=2, alpha=0.85)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, float(ymax))
    ax.set_xlabel(r"log$_2$ fold change")
    ymax_int = int(round(float(ymax)))
    ax.set_ylabel(
        r"$-$log$_{10}$(adjusted $p$-value)" + "\n" + rf"(capped at {ymax_int})",
        fontsize=10,
    )
    ax.set_title("Volcano plot", fontsize=11, pad=6)

    def _stagger_textoffset(
        stag_i: int,
        x_data: float,
        y_data: float,
        *,
        for_title: bool = False,
        group_i: int = 0,
    ) -> tuple[tuple[float, float], str, str]:
        if for_title:
            presets = [
                (0, 22),
                (-44, 30),
                (48, 26),
                (-32, 36),
                (36, 34),
            ]
            dx, dy = presets[group_i % len(presets)]
            return (dx, dy), "center", "bottom"

        side = 1.0 if x_data >= 0 else -1.0
        ring = (stag_i % 6) + 1
        ang = (stag_i * 0.85) % (2 * np.pi)
        # 略放大径向初值，减轻多标签同 y（如均顶到 ymax）时 adjust_text 起调过挤
        base_r = 16.0 + ring * 7.0
        dx = side * abs(np.cos(ang) * base_r) + (stag_i % 3) * 6 * side
        dy = 10.0 + np.sin(ang) * (8.0 + (stag_i % 4) * 6.0) + (stag_i // 3) * 8.0
        ha = "left" if side > 0 else "right"
        va = "bottom"
        return (dx, dy), ha, va

    volcano_label_artists: list[Any] = []

    def _annotate_gene(
        symbol: str,
        text_override: str | None,
        stag_i: int,
    ) -> tuple[float, float] | None:
        sym = str(symbol).strip()
        if not sym:
            return None
        resolved = _volcano_resolve_gene_row(d, sym)
        if resolved is None:
            log.warning("火山图标注：基因 %s 不在 DEG 表中", sym)
            return None
        resolved_sym, row = resolved
        x_, y_ = float(row["logFC"]), float(row["neglog10padj"])
        label_txt = (
            text_override
            or gene_display.get(resolved_sym)
            or gene_display.get(sym)
            or sym
        ).strip()
        (dx, dy), ha, va = _stagger_textoffset(stag_i, x_, y_, for_title=False)
        ann = ax.annotate(
            label_txt,
            (x_, y_),
            fontsize=7.5,
            color="#222222",
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="#BBBBBB",
                linewidth=0.45,
                alpha=0.92,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#888888",
                lw=0.5,
                shrinkA=0,
                shrinkB=1,
                connectionstyle="arc3,rad=0.12",
            ),
            zorder=8 + stag_i % 3,
        )
        volcano_label_artists.append(ann)
        return x_, y_

    stag_counter = 0
    if label_groups is not None and len(label_groups) > 0:
        for gi, grp in enumerate(label_groups):
            if not isinstance(grp, dict):
                continue
            title = str(grp.get("title", "") or "").strip()
            raw_genes = grp.get("genes") or []
            xs: list[float] = []
            ys: list[float] = []
            for item in raw_genes:
                sym = ""
                ovr: str | None = None
                if isinstance(item, str):
                    sym = item.strip()
                elif isinstance(item, dict):
                    sym = str(item.get("symbol", "")).strip()
                    tx = item.get("text")
                    if isinstance(tx, str) and tx.strip():
                        ovr = tx.strip()
                if not sym:
                    continue
                pt = _annotate_gene(sym, ovr, stag_counter)
                stag_counter += 1
                if pt is not None:
                    xs.append(pt[0])
                    ys.append(pt[1])
            if title and xs and ys:
                cx = float(np.mean(xs))
                cy = float(np.max(ys))
                (tdx, tdy), tha, tva = _stagger_textoffset(0, cx, cy, for_title=True, group_i=gi)
                grp_ann = ax.annotate(
                    title,
                    xy=(cx, cy),
                    xytext=(tdx, tdy),
                    textcoords="offset points",
                    ha=tha,
                    va=tva,
                    fontsize=8,
                    color="#2C2C2C",
                    bbox=dict(
                        boxstyle="round,pad=0.28",
                        facecolor="#FAFAFA",
                        edgecolor="#CCCCCC",
                        linewidth=0.6,
                        alpha=0.96,
                    ),
                    arrowprops=dict(
                        arrowstyle="-",
                        color="#888888",
                        lw=0.55,
                        connectionstyle="arc3,rad=0.08",
                    ),
                    zorder=5 + gi,
                )
                volcano_label_artists.append(grp_ann)
    else:
        allowed_pub = set(VOLCANO_PUBLICATION_DEFAULT_GENES)
        label_set = [str(g).strip() for g in (label_genes or []) if str(g).strip()]
        # 发表：仅标注预定义关键基因，避免 config 混入其它符号造成版面拥挤
        label_set = [g for g in label_set if g in allowed_pub]

        def _gene_sort_y(sym: str) -> float:
            hit = _volcano_resolve_gene_row(d, sym)
            return float(hit[1]["neglog10padj"]) if hit else -1.0

        label_set = sorted(label_set, key=_gene_sort_y, reverse=True)
        for g in label_set:
            _annotate_gene(g, None, stag_counter)
            stag_counter += 1

    lbl_ns = "Not significant"
    lbl_down = "Downregulated"
    lbl_up = "Upregulated"
    if prior_syms and (prior_mask & (lfc < 0) & ~sig_up & ~sig_down_stat).any():
        lbl_down = "Downregulated (incl. axis highlight)"
    if use_hexbin_ns:
        handle_ns: Any = Patch(facecolor="#A8A8A8", edgecolor="#CCCCCC", linewidth=0.35, alpha=0.9, label=lbl_ns)
    else:
        handle_ns = Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markerfacecolor=VOLCANO_COLOR_NS,
            markeredgecolor="none",
            markersize=5.5,
            alpha=0.65,
            label=lbl_ns,
        )
    handle_down = Line2D(
        [0],
        [0],
        linestyle="None",
        marker="o",
        markerfacecolor=VOLCANO_COLOR_DOWN,
        markeredgecolor="white",
        markeredgewidth=0.55,
        markersize=7,
        label=lbl_down,
    )
    handle_up = Line2D(
        [0],
        [0],
        linestyle="None",
        marker="o",
        markerfacecolor=VOLCANO_COLOR_UP,
        markeredgecolor="white",
        markeredgewidth=0.55,
        markersize=7,
        label=lbl_up,
    )
    ax.legend(
        handles=[handle_ns, handle_down, handle_up],
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#999999",
        framealpha=1.0,
        markerscale=0.85,
        handletextpad=0.35,
        borderaxespad=0.2,
    )

    if use_adjust_text and volcano_label_artists:
        try:
            from adjustText import adjust_text

            # adjusttext>=1.x API（与 0.7 的 expand_points/lim 不同）
            adjust_text(
                volcano_label_artists,
                ax=ax,
                expand=(1.55, 1.72),
                force_text=(0.18, 0.36),
                iter_lim=1200,
                arrowprops=dict(
                    arrowstyle="-",
                    color="#888888",
                    lw=0.55,
                    shrinkA=3,
                    shrinkB=3,
                ),
            )
        except Exception as exc:
            log.debug("adjust_text 未应用（可 pip install adjusttext）: %s", exc)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cap = (prior_down_caption or "").strip()
    if cap:
        fig.tight_layout(rect=(0, 0.12, 1, 0.98))
        fig.text(
            0.5,
            0.02,
            cap,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#333333",
            wrap=True,
            transform=fig.transFigure,
        )
    else:
        fig.tight_layout()
    _save_volcano_publication(fig, path)


def _ranking_top_n_from_config(config: dict[str, Any] | None) -> int:
    if not config:
        return 12
    n = int(config.get("plot_ranking_top_n", 12))
    return int(np.clip(n, 10, 15))


def _ranking_use_lollipop_from_config(config: dict[str, Any] | None) -> bool:
    if not config:
        return True
    return bool(config.get("plot_ranking_lollipop", True))


def _prepare_peptide_ranking_df(
    df: pd.DataFrame,
    score_col: str,
    peptide_ids: Sequence[str] | None,
    *,
    top_n: int,
) -> pd.DataFrame:
    if df.empty or "peptide_id" not in df.columns or score_col not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=[score_col]).copy()
    d = _filter_peptides(d, peptide_ids)
    d = d.sort_values(score_col, ascending=False).head(max(1, int(top_n)))
    if d.empty:
        return d
    # 作图顺序：低分在 y 轴下方，高分在上方（横向 lollipop/barh）
    return d.sort_values(score_col, ascending=True).reset_index(drop=True)


def _format_ranking_score(x: float) -> str:
    ax_ = abs(float(x))
    if ax_ >= 100:
        return f"{x:.1f}"
    if ax_ >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _plot_peptide_ranking_publication(
    df: pd.DataFrame,
    score_col: str,
    path: Path,
    *,
    title: str,
    y_axis_label: str,
    peptide_ids: Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
    recommendation_column: str | None = None,
) -> None:
    """
    发表用肽排名：Top 10–15、横向 lollipop（或 barh）；中性灰茎 + 端点着色；
    Top3 橙系渐变；其余钢蓝或由 recommendation 着色；浅斑马纹行底；带名次前缀；
    PDF fonttype 42。
    """
    top_n = _ranking_top_n_from_config(config)
    use_lollipop = _ranking_use_lollipop_from_config(config)
    d = _prepare_peptide_ranking_df(df, score_col, peptide_ids, top_n=top_n)
    if d.empty:
        log.warning("肽排名图无数据，跳过: %s", path)
        return

    _nature_rc()
    n = len(d)
    y_pos = np.arange(n, dtype=float)
    scores = pd.to_numeric(d[score_col], errors="coerce").values.astype(float)
    pids = d["peptide_id"].astype(str).tolist()

    fig_h = max(4.6, 0.46 * n + 1.85)
    fig_w = 7.35
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FDFDFE")

    if recommendation_column and recommendation_column in d.columns:
        rec_colors = [RECOMMENDATION_COLORS.get(str(r), PALETTE_SOFT[3]) for r in d[recommendation_column].values]
    else:
        rec_colors = None

    xmax = float(np.nanmax(scores)) if len(scores) else 1.0
    xmin = float(min(0.0, np.nanmin(scores))) if len(scores) else 0.0
    if not np.isfinite(xmax) or not np.isfinite(xmin):
        xmin, xmax = 0.0, 1.0
    if xmax <= xmin:
        xmax = xmin + 1e-6
    x_pad = (xmax - xmin) * 0.18 + 0.03

    dot_colors: list[str] = []
    marker_sizes: list[float] = []
    for i in range(n):
        is_top3 = i >= n - 3
        if is_top3:
            tier = i - (n - 3)
            dot_colors.append(_RANK_MARKER_TOP[2 - tier])
            marker_sizes.append(78.0 - tier * 8.0)
        elif rec_colors is not None:
            dot_colors.append(rec_colors[i])
            marker_sizes.append(44.0)
        else:
            dot_colors.append(_RANK_MARKER_OTHER)
            marker_sizes.append(44.0)

    # 浅斑马纹（y=0 为最低分在底部）
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, facecolor="#F1F4F8", edgecolor="none", zorder=0, alpha=0.65)

    if use_lollipop:
        for i in range(n):
            ax.plot(
                [xmin, scores[i]],
                [y_pos[i], y_pos[i]],
                color=_RANK_LOLLIPOP_STEM,
                lw=1.15,
                solid_capstyle="round",
                zorder=1,
                clip_on=False,
            )
        ax.scatter(
            scores,
            y_pos,
            s=marker_sizes,
            c=dot_colors,
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
            clip_on=False,
        )
    else:
        ax.barh(
            y_pos,
            scores - xmin,
            left=xmin,
            height=0.52,
            color=dot_colors,
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
            alpha=0.92,
        )

    for i in range(n):
        is_top3 = i >= n - 3
        ax.text(
            scores[i] + x_pad * 0.06,
            y_pos[i],
            _format_ranking_score(scores[i]),
            va="center",
            ha="left",
            fontsize=7.75 if is_top3 else 7.25,
            fontweight="semibold" if is_top3 else "normal",
            color="#2A2A2A" if is_top3 else "#5C5C5C",
            clip_on=False,
            zorder=4,
        )

    y_labels = [f"{n - i:>2}.  {pids[i]}" for i in range(n)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8.75, color="#333333")
    for i, t in enumerate(ax.get_yticklabels()):
        if i >= n - 3:
            t.set_fontweight("semibold")

    ax.set_xlabel(y_axis_label, fontsize=10, color="#222222", labelpad=8)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=11.5, pad=12, fontweight="semibold", color="#1A1A1A", loc="left")
    ax.set_xlim(xmin, xmax + x_pad)
    ax.set_ylim(-0.62, n - 1 + 0.62)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", labelsize=9, colors="#444444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C5C9D1")
    ax.spines["bottom"].set_color("#C5C9D1")
    ax.grid(axis="x", linestyle=(0, (1, 3)), linewidth=0.55, alpha=0.38, color="#B8BEC8", zorder=0)

    if recommendation_column and recommendation_column in d.columns:
        seen: set[str] = set()
        handles = []
        for lab in d[recommendation_column].astype(str).unique():
            if lab in seen:
                continue
            seen.add(lab)
            col = RECOMMENDATION_COLORS.get(lab, PALETTE_SOFT[3])
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    markersize=6.5,
                    markerfacecolor=col,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    label=lab,
                )
            )
        if handles:
            ax.legend(
                handles=handles,
                loc="lower right",
                frameon=False,
                fontsize=8,
                borderaxespad=0.5,
            )

    fig.subplots_adjust(left=0.22, right=0.94, top=0.90, bottom=0.11)
    _save_publication_figure(fig, path, png_dpi=300)


def plot_peptide_binding_rank(
    layer1: pd.DataFrame,
    path: Path,
    *,
    peptide_ids: Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    _plot_peptide_ranking_publication(
        layer1,
        "binding_score",
        path,
        title="Peptide binding score (top candidates)",
        y_axis_label="Binding score",
        peptide_ids=peptide_ids,
        config=config,
    )


def plot_similarity_heatmap(
    layer4: pd.DataFrame,
    path: Path,
    *,
    peptide_ids: Sequence[str] | None = None,
    cmap: str = "viridis",
) -> None:
    """
    发表风格：仅颜色编码（默认 viridis，可选 Blues 等）、无格内数字；
    分数若在 [0,1] 内则色标固定 0–1 便于跨行一致；行顺序见 HEATMAP_SCORE_ROW_ORDER；
    x 每 5 个肽标签、45°；Top5 三角 + 加粗；colorbar 标 Score；PDF fonttype 42。
    """
    row_spec = [(c, lab) for c, lab in HEATMAP_SCORE_ROW_ORDER if c in layer4.columns]
    if not row_spec:
        return
    ordered_cols = [c for c, _ in row_spec]
    row_labels = [lab for _, lab in row_spec]

    d = layer4.sort_values("blockade_similarity_score", ascending=False)
    d = _filter_peptides(d, peptide_ids)
    if d.empty or "peptide_id" not in d.columns:
        return
    # 列顺序：按 blockade 相似度降序，便于与第一行语义一致
    d = d.sort_values("blockade_similarity_score", ascending=False)

    mat = d.set_index("peptide_id")[ordered_cols].T.astype(float)
    mat.index = pd.Index(row_labels, name="")

    _nature_rc()
    n_pep, n_row = int(mat.shape[1]), int(mat.shape[0])
    w = max(10.0, min(30.0, 0.24 * n_pep + 5.5))
    h = max(4.0, 0.42 * n_row + 3.2)
    fig, ax = plt.subplots(figsize=(w, h))

    vals = mat.values.astype(float)
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        vmin, vmax = 0.0, 1.0
    elif lo >= -1e-5 and hi <= 1.0 + 1e-5:
        # 归一化分数：全面板统一 0–1 色标
        vmin, vmax = 0.0, 1.0
    elif lo >= vmax:
        vmax = lo + 1e-6
        vmin = lo
    else:
        vmin, vmax = lo, hi

    sns.heatmap(
        mat,
        annot=False,
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"shrink": 0.78, "label": "Score", "aspect": 24, "pad": 0.02},
        square=False,
        xticklabels=True,
        yticklabels=True,
    )

    ids = [str(x) for x in mat.columns]
    xlab = [ids[i] if i % 5 == 0 else "" for i in range(n_pep)]
    ax.set_xticklabels(xlab, rotation=45, ha="right", fontsize=8.5)
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=4)
    for t in ax.get_yticklabels():
        t.set_fontsize(9)

    top5_set = set(
        d.nlargest(5, "blockade_similarity_score")["peptide_id"].astype(str).tolist()
    )
    for i, lab in enumerate(xlab):
        if lab and lab in top5_set:
            ax.get_xticklabels()[i].set_fontweight("bold")

    trans_top = blended_transform_factory(ax.transData, ax.transAxes)
    for j, pid in enumerate(ids):
        if pid not in top5_set:
            continue
        ax.scatter(
            j + 0.5,
            1.01,
            s=42,
            marker="^",
            c="#D55E00",
            edgecolors="white",
            linewidths=0.4,
            transform=trans_top,
            clip_on=False,
            zorder=30,
        )

    ax.set_xlabel("")
    ax.set_title("Peptide vs blockade-like predictions", fontsize=11, pad=12)
    fig.subplots_adjust(bottom=0.24, left=0.20, right=0.88, top=0.90)
    _save_publication_figure(fig, path, png_dpi=300)


def plot_risk_boxplot(adata, config: dict[str, Any], path: Path) -> None:
    """
    发表风格：风险基因集按 Toxicity→Proliferation→Inflammation→EMT→Stemness；
    PBMC #0072B2 / TIL #009E73；无离群点；Wilcoxon rank-sum 星号；组间加宽图幅。
    """
    cond = str(config["condition_column"])
    ctrl = str(config["control_label"])
    treat = str(config["treatment_label"])
    row_spec = [(c, lab) for c, lab in RISK_BOXPLOT_ORDER if c in adata.obs.columns]
    if not row_spec:
        log.warning("无 risk 列（或未命中预定顺序列），跳过风险箱线图")
        return
    used_cols = [c for c, _ in row_spec]
    short_labels = [lab for _, lab in row_spec]
    rename = dict(row_spec)

    _nature_rc()
    df = adata.obs[[cond] + used_cols].melt(id_vars=[cond], var_name="risk", value_name="score")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df["risk"] = df["risk"].astype(str).map(rename)

    order = [lab for lab in short_labels if lab in set(df["risk"].unique())]
    if not order:
        return

    hue_order = [ctrl, treat]
    palette = {ctrl: COND_COLOR_PBMC, treat: COND_COLOR_TIL}

    fig_w = max(11.0, 1.62 * len(order) + 5.2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.4))

    bp_kw: dict[str, Any] = dict(
        data=df,
        x="risk",
        y="score",
        hue=cond,
        ax=ax,
        order=order,
        hue_order=hue_order,
        palette=palette,
        linewidth=0.38,
        showfliers=False,
        whiskerprops={"linewidth": 0.38, "color": "#333333"},
        capprops={"linewidth": 0.38, "color": "#333333"},
        medianprops={"linewidth": 0.75, "color": "#1A1A1A"},
    )
    try:
        sns.boxplot(**bp_kw, width=0.55, dodge=True)
    except TypeError:
        sns.boxplot(**bp_kw)

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=28, ha="right", fontsize=9.5)
    ax.tick_params(axis="x", pad=14)
    ax.tick_params(axis="y", labelsize=9)
    ax.margins(x=0.09)
    ax.set_xlabel("")
    ax.set_ylabel("Risk gene-set score", fontsize=10)
    ax.set_title("Risk scores by condition", fontsize=11, pad=8)
    ax.legend(
        title="",
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    n_pw = len(order)
    paths = [c for c in ax.get_children() if isinstance(c, PathPatch)]
    if len(paths) < 2 * n_pw:
        log.warning("risk boxplot PathPatch 数量异常: %s (预期 %d)", len(paths), 2 * n_pw)
    else:
        y_min, y_max = ax.get_ylim()
        y_span = max(y_max - y_min, 1e-9)
        y_annot_max = y_max

        for pw_idx, short in enumerate(order):
            sub_c = df[(df["risk"] == short) & (df[cond] == ctrl)]["score"].values
            sub_t = df[(df["risk"] == short) & (df[cond] == treat)]["score"].values
            if len(sub_c) < 3 or len(sub_t) < 3:
                continue
            _, p = ranksums(sub_t, sub_c)
            stars = _pvalue_to_significance_stars(p)
            if not stars:
                continue

            k0 = 0 * n_pw + pw_idx
            k1 = 1 * n_pw + pw_idx
            if k0 >= len(paths) or k1 >= len(paths):
                continue
            x1 = _pathpatch_xcenter_data(ax, paths[k0])
            x2 = _pathpatch_xcenter_data(ax, paths[k1])
            x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)

            local_top = float(np.nanpercentile(np.r_[sub_c, sub_t], 99))
            y_base = local_top + 0.04 * y_span
            h = 0.028 * y_span
            ax.plot([x1, x1, x2, x2], [y_base, y_base + h, y_base + h, y_base], color="#222222", lw=0.7, clip_on=False)
            ax.text(
                0.5 * (x1 + x2),
                y_base + h + 0.012 * y_span,
                stars,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="semibold",
                color="#111111",
                clip_on=False,
            )
            y_annot_max = max(y_annot_max, y_base + h + 0.05 * y_span)

        ax.set_ylim(y_min, max(y_max, y_annot_max + 0.02 * y_span))

    fig.subplots_adjust(bottom=0.30, left=0.10, right=0.76, top=0.93)
    fig.text(
        0.5,
        0.02,
        f"Wilcoxon rank-sum ({treat} vs {ctrl}):  *  p<0.05,  **  p<0.01,  ***  p<0.001",
        ha="center",
        fontsize=7.5,
        color="#333333",
        transform=fig.transFigure,
    )
    _save_publication_figure(fig, path, png_dpi=300)


def plot_peptide_safety(
    layer5: pd.DataFrame,
    path: Path,
    *,
    peptide_ids: Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    if layer5.empty:
        return
    _plot_peptide_ranking_publication(
        layer5,
        "safety_score",
        path,
        title="Peptide safety score (higher is safer)",
        y_axis_label="Safety score",
        peptide_ids=peptide_ids,
        config=config,
    )


def plot_blockade_similarity_rank(
    layer4: pd.DataFrame,
    path: Path,
    *,
    peptide_ids: Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    _plot_peptide_ranking_publication(
        layer4,
        "blockade_similarity_score",
        path,
        title="Peptide blockade similarity (top candidates)",
        y_axis_label="Blockade similarity",
        peptide_ids=peptide_ids,
        config=config,
    )


def plot_final_ranking(
    final_df: pd.DataFrame,
    path: Path,
    *,
    peptide_ids: Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    rec_col = "recommendation" if "recommendation" in final_df.columns else None
    _plot_peptide_ranking_publication(
        final_df,
        "final_score",
        path,
        title="Final candidate ranking",
        y_axis_label="Final score",
        peptide_ids=peptide_ids,
        config=config,
        recommendation_column=rec_col,
    )

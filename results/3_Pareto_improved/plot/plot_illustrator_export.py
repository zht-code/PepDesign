# -*- coding: utf-8 -*-
"""
Adobe Illustrator 友好导出：
- pdf.fonttype = 42：文字为 TrueType/可编辑，避免 Type 3 字体问题
- 先保存高 dpi PNG（可保留 scatter rasterized 以控制体积），再关闭全图 rasterized 后存 PDF，
  使散点为矢量路径，便于在 AI 中改色、拆分组。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure


def apply_illustrator_pdf_rc() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def set_figure_rasterized(fig: matplotlib.figure.Figure, rasterized: bool) -> None:
    for artist in fig.findobj(include_self=False):
        if hasattr(artist, "set_rasterized"):
            try:
                artist.set_rasterized(rasterized)
            except Exception:
                pass


def savefig_png_then_pdf(
    fig: matplotlib.figure.Figure,
    out_base: Path,
    *,
    dpi: int = 300,
    facecolor: str = "white",
) -> None:
    """先写 PNG（dpi），再写矢量 PDF（无栅格化散点）。"""
    apply_illustrator_pdf_rc()
    out_base = Path(out_base)
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
    set_figure_rasterized(fig, False)
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor=facecolor,
        format="pdf",
    )

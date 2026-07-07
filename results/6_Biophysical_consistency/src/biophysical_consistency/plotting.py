from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _savefig_dual(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    # Type3 字库在 Illustrator 中常无法编辑文字；42 = TrueType 嵌入
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_histograms(
    df: pd.DataFrame,
    columns: Iterable[str],
    out_stem: Path,
) -> None:
    cols = [c for c in columns if c in df.columns]

    def usable_numeric(col: str) -> bool:
        s = pd.to_numeric(df[col], errors="coerce")
        return bool(s.notna().sum() > 0)

    cols = [c for c in cols if usable_numeric(c)]
    if not cols:
        return
    n = len(cols)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(7, 2.4 * n))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            ax.set_title(f"{col} (empty)")
            continue
        ax.hist(series, bins=30, color="#4C72B0", alpha=0.85)
        ax.set_title(col)
        ax.set_ylabel("count")
    fig.suptitle("Biophysical metrics distributions", y=1.0)
    fig.tight_layout()
    _savefig_dual(fig, out_stem)

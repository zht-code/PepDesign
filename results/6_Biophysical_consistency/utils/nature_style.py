"""
Nature 期刊常见视觉风格（参考 NPG / 柔和对比、偏色盲友好）。

不依赖 seaborn；提供 rcParams、离散色、以及用于 imshow 的 LinearSegmentedColormap。
"""

from __future__ import annotations

from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# Nature Publishing Group (ggsci npg) 风格离散色 — 饱和度略压低以适配印刷
NPG = [
    "#D56F4C",  # coral
    "#4C9A8E",  # teal
    "#3D5A80",  # slate blue
    "#7A8BA3",  # gray-blue
    "#E8A87C",  # sand
    "#5B8BA0",  # steel
    "#9BB89C",  # sage
    "#C44E52",  # muted red
]

TEXT = "#1A1A1A"
MUTED_LINE = "#5C5C5C"
GRID = "#E6E6E6"
FILL_LIGHT = "#D9E2EF"  # 直方图填充
ACCENT_LINE = "#2E4A6F"  # KDE / 主曲线
ACCENT_MARK = "#A6514B"  # 中位数等强调线

# 发散色图：蓝 — 浅灰 — 砖红（替代高饱和 RdBu）
NATURE_DIVERGING = LinearSegmentedColormap.from_list(
    "nature_diverging",
    ["#3A5A92", "#D4DCE8", "#F5F5F5", "#E8C5C0", "#B54745"],
    N=256,
)

# 顺序色图：近白 — 青蓝（替代 viridis / YlOrRd 高亮）
NATURE_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "nature_sequential",
    ["#FFFDF8", "#C8DFF0", "#6FA8D6", "#2A5A87"],
    N=256,
)

NATURE_SEQUENTIAL_WARM = LinearSegmentedColormap.from_list(
    "nature_sequential_warm",
    ["#FFFDF5", "#F5D9B8", "#E8A87C", "#C26E4A"],
    N=256,
)


def mpl_rcparams_illustrator_friendly_pdf() -> dict:
    """TrueType 嵌入 PDF，便于 Adobe Illustrator 编辑文字与改色（避免 Type3 字库）。"""
    return {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def mpl_rcparams_nature() -> dict:
    return {
        **mpl_rcparams_illustrator_friendly_pdf(),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED_LINE,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "grid.color": GRID,
        "grid.alpha": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "axes.prop_cycle": cycler(color=NPG),
    }


def apply_nature_style() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams.update(mpl_rcparams_nature())
    mpl.rcParams["axes.prop_cycle"] = cycler(color=NPG)

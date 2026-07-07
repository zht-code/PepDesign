#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五种方法在 (HDOCK↓, 稳定性↑, 溶解性↑) 三目标下的 Hypervolume 与成对 Pareto 支配率。

- DPO×3 + SFT：top3 模式 = 每靶点对齐肽三项分别平均；top1 = 每靶点 HDOCK 最优单肽。
- multi_cands（Ours）：始终用靶点内三目标 min-max 综合分选 top-k（k=3 或 1）再平均 / 单点。

对每个模式 (top1 / top3) 输出：
  - CSV 表格：Method, Hypervolume, Dominance rate（支配率为该行对其它方法平均支配比例）
  - 支配矩阵 CSV：行 A、列 B = 同靶点上 A 支配 B 的比例
  - Hypervolume 柱状图：柱高为 **bootstrap 重采样 HV 的中位数**，误差线为 2.5%–97.5% 分位
    （全样本 plug-in HV 常高于 bootstrap 云的上分位，若以它为柱高则上误差为 0；中位数保证上下 whisker 通常都可见）。
  - CSV / 表格图 中的 Hypervolume 仍为 **全样本 plug-in** 估计。
  - 支配率热图（矩阵）

依赖：pymoo（Hypervolume）、pandas、matplotlib、numpy。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from pymoo.indicators.hv import HV
except ImportError as e:
    raise SystemExit(
        "需要 pymoo：pip install pymoo\n" + str(e)
    ) from e

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_ppdbench_methods_scatter as base
from plot_illustrator_export import savefig_png_then_pdf

DEFAULT_DATA = HERE.parent
DEFAULT_BENCH = Path("/root/autodl-tmp/PPDbench")

# 与 METHODS 顺序一致，用于表格/图
SHORT_LABELS = [
    "Affinity-only",
    "Stability-only",
    "Weighted-sum",
    "SFT",
    "Ours",
]

COLORS = base.COLORS


def _dict_mean_per_target(
    aff: Dict[str, float],
    sol: Dict[str, float],
    stab: Dict[str, float],
) -> Dict[str, Tuple[float, float, float]]:
    groups: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for key in aff:
        if key not in sol or key not in stab:
            continue
        tid = key.split("/", 1)[0]
        groups[tid].append((aff[key], stab[key], sol[key]))
    out: Dict[str, Tuple[float, float, float]] = {}
    for tid, pts in groups.items():
        out[tid] = (
            float(np.mean([p[0] for p in pts])),
            float(np.mean([p[1] for p in pts])),
            float(np.mean([p[2] for p in pts])),
        )
    return out


def _dict_best_hdock(
    aff: Dict[str, float],
    sol: Dict[str, float],
    stab: Dict[str, float],
) -> Dict[str, Tuple[float, float, float]]:
    groups: Dict[str, List[Tuple[str, float, float, float]]] = defaultdict(list)
    for key in aff:
        if key not in sol or key not in stab:
            continue
        tid = key.split("/", 1)[0]
        groups[tid].append((key, aff[key], stab[key], sol[key]))
    out: Dict[str, Tuple[float, float, float]] = {}
    for tid, rows in groups.items():
        _, h, st, so = min(rows, key=lambda r: (r[1], r[0]))
        out[tid] = (float(h), float(st), float(so))
    return out


def _dict_multi_mo_topk(
    aff: Dict[str, float],
    sol: Dict[str, float],
    stab: Dict[str, float],
    top_k: int,
) -> Dict[str, Tuple[float, float, float]]:
    groups: Dict[str, List[Tuple[str, float, float, float]]] = defaultdict(list)
    for key in aff:
        if key not in sol or key not in stab:
            continue
        tid = key.split("/", 1)[0]
        groups[tid].append((key, aff[key], stab[key], sol[key]))

    out: Dict[str, Tuple[float, float, float]] = {}
    for tid in sorted(groups.keys()):
        rows = groups[tid]
        if not rows:
            continue
        hs = [r[1] for r in rows]
        sts = [r[2] for r in rows]
        sos = [r[3] for r in rows]
        h_min, h_max = min(hs), max(hs)
        st_min, st_max = min(sts), max(sts)
        so_min, so_max = min(sos), max(sos)

        scored: List[Tuple[float, float, float, float, str]] = []
        for key, h, st, so in rows:
            if h_max > h_min:
                nh = (h_max - h) / (h_max - h_min)
            else:
                nh = 0.5
            if st_max > st_min:
                ns = (st - st_min) / (st_max - st_min)
            else:
                ns = 0.5
            if so_max > so_min:
                no = (so - so_min) / (so_max - so_min)
            else:
                no = 0.5
            combo = nh + ns + no
            scored.append((-combo, h, st, so, key))

        scored.sort(key=lambda t: (t[0], t[1]))
        k = min(top_k, len(scored))
        pick = scored[:k]
        out[tid] = (
            float(np.mean([t[1] for t in pick])),
            float(np.mean([t[2] for t in pick])),
            float(np.mean([t[3] for t in pick])),
        )
    return out


def collect_method_dicts(
    data_dir: Path,
    bench_root: Path,
    mode: str,
) -> List[Dict[str, Tuple[float, float, float]]]:
    assert mode in ("top1", "top3")
    k = 1 if mode == "top1" else 3
    dicts: List[Dict[str, Tuple[float, float, float]]] = []
    for _name, f_aff, f_sol, f_stab, f_flag in base.METHODS:
        sol = base.prop_dict(base.load_json(data_dir / f_sol))
        stab = base.prop_dict(base.load_json(data_dir / f_stab))
        if f_flag == base.MULTI_CANDS_BENCH_MO3:
            aff = base.affinity_from_bench_hdock(bench_root)
            d = _dict_multi_mo_topk(aff, sol, stab, top_k=k)
        elif f_flag:
            aff = base.affinity_from_top3_json(base.load_json(data_dir / f_flag))
            d = _dict_best_hdock(aff, sol, stab) if mode == "top1" else _dict_mean_per_target(aff, sol, stab)
        else:
            assert f_aff is not None
            aff = base.affinity_from_hdock_json(base.load_json(data_dir / f_aff))
            d = _dict_best_hdock(aff, sol, stab) if mode == "top1" else _dict_mean_per_target(aff, sol, stab)
        dicts.append(d)
    return dicts


def align_points(
    dicts: List[Dict[str, Tuple[float, float, float]]],
) -> Tuple[List[str], np.ndarray]:
    """返回公共靶点 id 列表与 P.shape == (5, n, 3)，列为 HDOCK, stab, sol。"""
    common: Optional[set] = None
    for d in dicts:
        s = set(d.keys())
        common = s if common is None else common & s
    if not common:
        raise RuntimeError("无公共靶点")
    tids = sorted(common)
    n = len(tids)
    p = np.zeros((len(dicts), n, 3), dtype=np.float64)
    for i, d in enumerate(dicts):
        for j, tid in enumerate(tids):
            p[i, j, :] = d[tid]
    return tids, p


def global_normalize_minimize(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将原生 (HDOCK↓, stab↑, sol↑) 单调变换为三维 **最小化** 目标，且各维约 ∈ [0,1]，
    便于 Hypervolume 数量级与文献表格可比。变换对全体方法所有靶点共用同一 min/max。
    f1=(h-h_min)/r_h, f2=(st_max-st)/r_st, f3=(so_max-so)/r_so
    """
    flat = P.reshape(-1, 3)
    h_min, h_max = flat[:, 0].min(), flat[:, 0].max()
    st_min, st_max = flat[:, 1].min(), flat[:, 1].max()
    so_min, so_max = flat[:, 2].min(), flat[:, 2].max()
    rh = h_max - h_min + 1e-12
    rst = st_max - st_min + 1e-12
    rso = so_max - so_min + 1e-12
    f = np.zeros_like(flat)
    f[:, 0] = (flat[:, 0] - h_min) / rh
    f[:, 1] = (st_max - flat[:, 1]) / rst
    f[:, 2] = (so_max - flat[:, 2]) / rso
    F = f.reshape(P.shape)
    ref = np.max(f, axis=0) + 0.05
    return F, ref


def hypervolume_minimize(F: np.ndarray, ref: np.ndarray) -> float:
    hv = HV(ref_point=ref)
    return float(hv.do(F))


def bootstrap_hypervolume(
    F: np.ndarray,
    ref: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    """
    返回 (全样本 plug-in HV, bootstrap 中位数, 2.5% 分位, 97.5% 分位)。
    柱状图应以中位数为柱高、[lo, hi] 为误差线，使上下 whisker 通常均大于 0。
    """
    rng = np.random.default_rng(seed)
    point = hypervolume_minimize(F, ref)
    vals = []
    n = F.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(hypervolume_minimize(F[idx], ref))
    arr = np.asarray(vals, dtype=np.float64)
    med = float(np.median(arr))
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return point, med, float(lo), float(hi)


def dominates_native(a: np.ndarray, b: np.ndarray) -> bool:
    """a,b 形状 (3,) = HDOCK, stab, sol。HDOCK 越小越好，后两者越大越好。"""
    h_ok = a[0] <= b[0]
    st_ok = a[1] >= b[1]
    so_ok = a[2] >= b[2]
    strict = (a[0] < b[0]) or (a[1] > b[1]) or (a[2] > b[2])
    return bool(h_ok and st_ok and so_ok and strict)


def dominance_matrix(P: np.ndarray) -> np.ndarray:
    """P: (5, n, 3) 原生空间。M[i,j] = 靶点中方法 i 支配方法 j 的比例。"""
    m = P.shape[0]
    n = P.shape[1]
    M = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                M[i, j] = np.nan
                continue
            cnt = 0
            for t in range(n):
                if dominates_native(P[i, t], P[j, t]):
                    cnt += 1
            M[i, j] = cnt / n
    return M


def row_dominance_rate(M: np.ndarray) -> np.ndarray:
    """每行对非对角元素的平均支配率。"""
    m = M.shape[0]
    out = np.zeros(m)
    for i in range(m):
        vals = [M[i, j] for j in range(m) if i != j and not np.isnan(M[i, j])]
        out[i] = float(np.mean(vals)) if vals else 0.0
    return out


def plot_hv_bars(
    labels: List[str],
    hv_median: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    title: str,
    out_base: Path,
    dpi: int,
) -> None:
    # Keep consistent styling/colors with other plots
    base.apply_nature_rc()
    x = np.arange(len(labels))
    err_lo = np.clip(hv_median - lo, 0.0, None)
    err_hi = np.clip(hi - hv_median, 0.0, None)
    fig, ax = plt.subplots(figsize=(6.6, 4.0), layout="constrained")
    bars = ax.bar(
        x,
        hv_median,
        yerr=[err_lo, err_hi],
        capsize=4,
        color=COLORS[: len(labels)],
        edgecolor="none",
        alpha=0.90,
    )
    _ = bars
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Hypervolume ↑ (bootstrap median)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    savefig_png_then_pdf(fig, out_base, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_dominance_heatmap(
    labels: List[str],
    M: np.ndarray,
    title: str,
    out_base: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), layout="constrained")
    plot_data = np.ma.masked_invalid(M.copy())
    vmax = float(np.nanmax(M))
    im = ax.imshow(plot_data, cmap="viridis", vmin=0.0, vmax=max(vmax, 1e-6))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Method B (column)")
    ax.set_ylabel("Method A (row)")
    ax.set_title(title + "\ncell = fraction of targets where A Pareto-dominates B")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dominance rate")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="white", fontsize=11)
            else:
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="w", fontsize=9)
    savefig_png_then_pdf(fig, out_base, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_metrics_table_fig(
    df: pd.DataFrame,
    hv_best_idx: int,
    dr_best_idx: int,
    title: str,
    out_base: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.2), layout="constrained")
    ax.axis("off")
    cells = []
    for i, row in df.iterrows():
        r = [row["Method"], f"{row['Hypervolume']:.3f}", f"{row['Dominance rate']:.3f}"]
        cells.append(r)
    tbl = ax.table(
        cellText=cells,
        colLabels=["Method", "Hypervolume ↑", "Dominance rate ↑"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.15, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            continue
        if col == 0:
            if row - 1 == hv_best_idx or row - 1 == dr_best_idx:
                cell.set_text_props(weight="bold")
        if col == 1 and row - 1 == hv_best_idx:
            cell.set_text_props(weight="bold")
        if col == 2 and row - 1 == dr_best_idx:
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=12, pad=12)
    savefig_png_then_pdf(fig, out_base, dpi=dpi, facecolor="white")
    plt.close(fig)


def run_mode(
    data_dir: Path,
    bench_root: Path,
    out_dir: Path,
    mode: str,
    *,
    n_boot: int,
    seed: int,
    dpi: int,
) -> None:
    dicts = collect_method_dicts(data_dir, bench_root, mode)
    _tids, P = align_points(dicts)
    F_norm, ref = global_normalize_minimize(P)

    m = P.shape[0]
    hv_point = np.zeros(m)
    hv_median = np.zeros(m)
    hv_lo = np.zeros(m)
    hv_hi = np.zeros(m)
    for i in range(m):
        Fi = F_norm[i]
        pt, med, lo, hi = bootstrap_hypervolume(Fi, ref, n_boot=n_boot, seed=seed + i)
        hv_point[i] = pt
        hv_median[i] = med
        hv_lo[i] = lo
        hv_hi[i] = hi

    M = dominance_matrix(P)
    dr = row_dominance_rate(M)

    df = pd.DataFrame(
        {
            "Method": SHORT_LABELS,
            "Hypervolume": hv_point,
            "Dominance rate": dr,
        }
    )
    prefix = out_dir / f"ppdbench_pareto_metrics_{mode}"
    df.to_csv(prefix.with_suffix(".csv"), index=False)

    mat_df = pd.DataFrame(M, index=SHORT_LABELS, columns=SHORT_LABELS)
    mat_df.to_csv(f"{prefix}_dominance_matrix.csv")

    hv_best = int(np.argmax(hv_point))
    dr_best = int(np.argmax(dr))

    plot_metrics_table_fig(
        df,
        hv_best_idx=hv_best,
        dr_best_idx=dr_best,
        title=f"PPDbench Pareto metrics ({mode})",
        out_base=Path(str(prefix) + "_table"),
        dpi=dpi,
    )
    plot_hv_bars(
        SHORT_LABELS,
        hv_median,
        hv_lo,
        hv_hi,
        title=f"Hypervolume ({mode})\n95% CI from bootstrap; bar = median (table/CSV = plug-in HV)",
        out_base=Path(str(prefix) + "_hypervolume_bar"),
        dpi=dpi,
    )
    plot_dominance_heatmap(
        SHORT_LABELS,
        M,
        title=f"Paired Pareto dominance ({mode})",
        out_base=Path(str(prefix) + "_dominance_heatmap"),
        dpi=dpi,
    )

    print(f"[{mode}] n_targets={P.shape[1]} HV_ref={ref}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--out-dir", type=str, default=str(HERE))
    ap.add_argument("--bench-root", type=str, default=str(DEFAULT_BENCH))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--bootstrap", type=int, default=800, help="Hypervolume bootstrap 次数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", type=str, default="top1,top3", help="逗号分隔：top1,top3")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    bench_root = Path(args.bench_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
        run_mode(
            data_dir,
            bench_root,
            out_dir,
            mode,
            n_boot=args.bootstrap,
            seed=args.seed,
            dpi=args.dpi,
        )
    print(f"[FIN] outputs in {out_dir}")


if __name__ == "__main__":
    main()

"""
跨方法对比：从 Table_S1/S11 等的 ``source`` 列推断生成方法（Ours vs SOTA）。

SOTA 行若来自 ``baseline_input_index:`` 往往缺少界面等指标，与 ``all_samples:`` 全量行重复；
默认作图队列对 SOTA **仅保留** ``all_samples:`` 前缀行；本方法（无 baseline 路径）保留全部。
"""

from __future__ import annotations

import pandas as pd

# 图例与作图顺序（本方法优先，其后 SOTA）
METHOD_PLOT_ORDER: tuple[str, ...] = ("Ours", "RFdiffusion", "ProteinGenerator", "BindCraft")


def infer_method(source: pd.Series) -> pd.Series:
    """根据 ``source``（及路径子串）推断方法标签。

    先匹配 SOTA 关键字，再将 ``ours:`` 前缀或 ``PPDbench`` 路径 **强制** 标为 **Ours**，
    避免误归类（并兼容 ``…/PPDbench/…/multi_cands/…`` 本方法输出目录）。
    """
    s = source.astype(str).str.lower()
    out = pd.Series("Ours", index=source.index, dtype=object)
    out[s.str.contains("rfdiffusion", na=False)] = "RFdiffusion"
    out[s.str.contains("proteingenerator", na=False)] = "ProteinGenerator"
    out[s.str.contains("bindcraft", na=False)] = "BindCraft"
    ours_force = s.str.startswith("ours:") | s.str.contains("ppdbench", na=False)
    out[ours_force] = "Ours"
    return out


def cohort_for_cross_method_plots(df: pd.DataFrame, *, source_col: str = "source") -> pd.DataFrame:
    """
    返回带 ``method`` 列的子表，用于 Fig7 等同台对比。

    - **SOTA**（RFdiffusion / ProteinGenerator / BindCraft）：仅保留 ``source`` 以
      ``all_samples:`` 开头的行，与全量结构/界面指标一致，避免 ``baseline_input_index:``
      元数据行与缺失界面混排。
    - **Ours**：保留所有非 SOTA 路径行（不要求 ``all_samples:`` 前缀）。
    """
    if source_col not in df.columns:
        return df.assign(method=infer_method(pd.Series("", index=df.index)))
    m = infer_method(df[source_col])
    src = df[source_col].astype(str)
    is_sota = m != "Ours"
    keep = (~is_sota) | src.str.startswith("all_samples:")
    out = df.loc[keep].copy()
    out["method"] = infer_method(out[source_col])
    return out


def methods_in_order(df: pd.DataFrame, *, method_col: str = "method") -> list[str]:
    if df is None or len(df) == 0 or method_col not in df.columns:
        return []
    present = set(df[method_col].astype(str).unique())
    return [m for m in METHOD_PLOT_ORDER if m in present]


def attach_method_filtered(
    s4: pd.DataFrame,
    s11: pd.DataFrame,
    *,
    keys: tuple[str, ...] = ("target_id", "peptide_id", "group"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对 S11 作 cohort 过滤并附 ``method``；S4 与过滤后的 S11 按 keys 内连接。"""
    s11p = cohort_for_cross_method_plots(s11)
    if s11p.empty:
        return s4.iloc[0:0].copy(), s11p
    meta = s11p[list(keys) + ["method"]].drop_duplicates()
    s4p = s4.merge(meta, on=list(keys), how="inner")
    return s4p, s11p

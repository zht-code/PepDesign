"""Aggregate sample tables, robustness derivatives (AUDC, drop thresholds, slopes)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics_eval import to_higher_better


def relative_drop_pct(clean: float, pert: float, eps: float = 1e-6) -> float:
    """(clean - pert) / max(|clean|, eps) * 100; clean, pert are higher-is-better."""
    denom = max(abs(clean), eps)
    return float((clean - pert) / denom * 100.0)


def audc_trapz(levels_norm: np.ndarray, drops_pct: np.ndarray) -> float:
    """Area under degradation (relative drop %) vs normalized perturbation level in [0,1]."""
    order = np.argsort(levels_norm)
    x = levels_norm[order]
    y = np.clip(drops_pct[order], 0.0, None)
    if len(x) < 2:
        return float(y[0]) if len(y) else 0.0
    return float(np.trapz(y, x))


def first_threshold_level(levels: np.ndarray, drops: np.ndarray, target_drop: float) -> float | None:
    """First perturbation level (physical units) where drop >= target_drop (linear interp)."""
    if len(levels) < 2:
        return None
    order = np.argsort(levels)
    L = levels[order].astype(float)
    D = drops[order].astype(float)
    for i in range(len(L)):
        if D[i] >= target_drop:
            if i == 0:
                return float(L[i])
            # linear interp between (i-1) and i
            L0, L1 = L[i - 1], L[i]
            D0, D1 = D[i - 1], D[i]
            if D1 == D0:
                return float(L1)
            t = (target_drop - D0) / (D1 - D0)
            return float(L0 + t * (L1 - L0))
    return None


def sensitivity_slope(levels: np.ndarray, values: np.ndarray) -> float:
    """Linear slope of metric (higher-better) vs physical perturbation level."""
    mask = np.isfinite(levels) & np.isfinite(values)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(levels[mask].astype(float), values[mask].astype(float), 1)[0])


def summarize_condition(df: pd.DataFrame, metric_col: str, level_col: str = "level_value") -> dict[str, Any]:
    """Per (perturb_type, repeat): aggregate mean top-1 metrics already in rows."""
    g = df.groupby(level_col, sort=True)[metric_col].agg(["mean", "std", "median", "count"])
    return g.reset_index().to_dict(orient="records")


def build_curve_table(
    df_agg: pd.DataFrame,
    *,
    perturb_type: str,
    metric: str,
    level_col: str,
    clean_level: float,
    normalize_levels: str,
) -> pd.DataFrame:
    """
    df_agg columns: perturb_type, level_value, mean (higher-better), ...
    normalize_levels: 'minmax' maps physical levels to [0,1] for this perturb type.
    """
    sub = df_agg[(df_agg["perturb_type"] == perturb_type)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values(level_col)
    clean_row = sub[np.isclose(sub[level_col].astype(float), float(clean_level))]
    if clean_row.empty:
        clean_val = float(sub[sub[level_col] == sub[level_col].min()]["mean"].iloc[0])
    else:
        clean_val = float(clean_row["mean"].iloc[0])

    levels = sub[level_col].astype(float).values
    vals = sub["mean"].astype(float).values
    drops = np.array([relative_drop_pct(clean_val, v) for v in vals])
    if normalize_levels == "minmax" and levels.max() > levels.min():
        xn = (levels - levels.min()) / (levels.max() - levels.min())
    else:
        xn = levels

    rows = []
    for i in range(len(sub)):
        rows.append(
            {
                "perturb_type": perturb_type,
                "metric": metric,
                "level": float(levels[i]),
                "level_norm": float(xn[i]),
                "mean_higher_better": float(vals[i]),
                "relative_drop_pct": float(drops[i]),
                "clean_reference": clean_val,
            }
        )
    return pd.DataFrame(rows)


def robustness_summary_row(
    perturb_type: str,
    metric: str,
    levels: np.ndarray,
    clean_val: float,
    pert_vals: np.ndarray,
    level_norm: np.ndarray,
) -> dict[str, Any]:
    drops = np.array(
        [
            relative_drop_pct(clean_val, float(v)) if np.isfinite(v) else float("nan")
            for v in pert_vals
        ]
    )
    max_drop = float(np.nanmax(drops)) if np.any(np.isfinite(drops)) else float("nan")
    mask = np.isfinite(drops) & np.isfinite(level_norm)
    audc = (
        audc_trapz(level_norm[mask].astype(float), drops[mask])
        if mask.sum() >= 2
        else float(np.nanmean(drops[mask]) if mask.any() else 0.0)
    )
    d10 = first_threshold_level(levels.astype(float), drops, 10.0)
    d20 = first_threshold_level(levels.astype(float), drops, 20.0)
    slope = sensitivity_slope(levels.astype(float), pert_vals.astype(float))
    return {
        "perturbation_type": perturb_type,
        "metric": metric,
        "AUDC": audc,
        "drop_10_threshold": d10,
        "drop_20_threshold": d20,
        "sensitivity_slope": slope,
        "clean_mean": float(clean_val),
        "max_drop": max_drop,
        "notes": "",
    }


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

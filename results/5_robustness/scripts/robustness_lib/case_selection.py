"""Pick representative targets: strong clean performance + interpretable degradation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics_eval import to_higher_better


def select_representative_targets(
    df: pd.DataFrame,
    *,
    clean_label: str = "clean",
    max_cases: int = 2,
    min_clean_affinity_higher: float = 0.0,
) -> list[dict[str, Any]]:
    """
    df rows: target_id, perturb_type, level_value, repeat_id, affinity_hdock, stability, solubility
    Uses rows with highest perturb level per type to score degradation vs clean (level 0).
    """
    out: list[dict[str, Any]] = []
    if df.empty:
        return out

    for ptype in df["perturb_type"].unique():
        sub = df[df["perturb_type"] == ptype]
        levels = sorted(sub["level_value"].unique())
        if len(levels) < 2:
            continue
        clean_lv, max_lv = float(levels[0]), float(levels[-1])
        d0 = sub[np.isclose(sub["level_value"].astype(float), clean_lv)]
        d1 = sub[np.isclose(sub["level_value"].astype(float), max_lv)]
        if d0.empty or d1.empty:
            continue

        scores = []
        for tid in d0["target_id"].unique():
            r0 = d0[d0["target_id"] == tid].iloc[0]
            r1m = d1[d1["target_id"] == tid]
            if r1m.empty:
                continue
            r1 = r1m.groupby("repeat_id").mean(numeric_only=True).mean()
            aff0 = to_higher_better("affinity_hdock", float(r0["affinity_hdock"]))
            aff1 = to_higher_better("affinity_hdock", float(r1["affinity_hdock"]) if np.isfinite(r1.get("affinity_hdock", np.nan)) else None)
            if aff0 is None or aff1 is None:
                continue
            if aff0 < min_clean_affinity_higher:
                continue
            drop = aff0 - aff1
            scores.append((drop * aff0, tid, float(aff0), float(aff1), ptype, max_lv))

        scores.sort(reverse=True, key=lambda x: x[0])
        for row in scores[:max_cases]:
            _, tid, a0, a1, pt, ml = row
            out.append(
                {
                    "target_id": tid,
                    "perturb_type": pt,
                    "max_level": ml,
                    "clean_neg_hdock_proxy": a0,
                    "perturbed_neg_hdock_proxy": a1,
                }
            )
    # Deduplicate target_id keeping first
    seen = set()
    uniq = []
    for c in out:
        if c["target_id"] in seen:
            continue
        seen.add(c["target_id"])
        uniq.append(c)
        if len(uniq) >= max_cases:
            break
    return uniq[:max_cases]


def save_case_panel_data(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

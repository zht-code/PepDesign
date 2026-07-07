from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_method_manifest(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [
        "dataset", "split_name", "target_id", "method", "candidate_rank",
        "receptor_pdb", "reference_peptide_pdb", "generated_peptide_pdb", "generated_sequence"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    return df


def select_topk_by_hdock(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    # Lower HDOCK score is typically better; change sort if your setup differs.
    out = []
    for (dataset, split_name, target_id, method), sub in df.groupby(["dataset", "split_name", "target_id", "method"]):
        ranked = sub.sort_values("hdock_score", ascending=True, na_position="last").head(k)
        out.append(ranked)
    return pd.concat(out, ignore_index=True) if out else df.iloc[:0].copy()


def summarize_top5_affinity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, split_name, method), sub in df.groupby(["dataset", "split_name", "method"]):
        rows.append({
            "dataset": dataset,
            "split_name": split_name,
            "method": method,
            "n_targets": int(sub["target_id"].nunique()),
            "top5_hdock_mean": float(sub["hdock_score"].mean()),
            "top5_hdock_median": float(sub["hdock_score"].median()),
            "best_hdock_mean": float(sub.groupby("target_id")["hdock_score"].min().mean()),
        })
    return pd.DataFrame(rows)

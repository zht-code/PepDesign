from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from baseline_adapters import load_method_manifest, select_topk_by_hdock, summarize_top5_affinity
from metrics_affinity import contact_map_consistency, parse_hdock_score
from metrics_generation import PerplexityScorer, add_generation_metrics
from metrics_structure import clash_score, mean_plddt_from_pdb, ramachandran_compliance
from utils_io import ensure_dir


def load_train_sequences(train_metadata_csv: str) -> List[str]:
    df = pd.read_csv(train_metadata_csv)
    col = "peptide_sequence" if "peptide_sequence" in df.columns else "generated_sequence"
    return df[col].dropna().astype(str).tolist()


def add_affinity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hdock_score"] = df["hdock_result"].apply(parse_hdock_score) if "hdock_result" in df.columns else float("nan")

    cmc = []
    for _, row in df.iterrows():
        native_complex = row.get("native_complex_pdb", None)
        pred_complex = row.get("pred_complex_pdb", None)
        if pd.isna(native_complex) or pd.isna(pred_complex):
            cmc.append(float("nan"))
        else:
            try:
                cmc.append(contact_map_consistency(str(native_complex), str(pred_complex)))
            except Exception:
                cmc.append(float("nan"))
    df["contact_map_consistency"] = cmc
    return df


def add_structure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    plddt, rama, clash = [], [], []
    for _, row in df.iterrows():
        pdb = row["generated_peptide_pdb"]
        try:
            plddt.append(mean_plddt_from_pdb(pdb))
        except Exception:
            plddt.append(float("nan"))
        try:
            rama.append(ramachandran_compliance(pdb))
        except Exception:
            rama.append(float("nan"))
        try:
            clash.append(clash_score(pdb))
        except Exception:
            clash.append(float("nan"))
    df["pLDDT"] = plddt
    df["ramachandran_compliance"] = rama
    df["clash_score"] = clash
    return df


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "hdock_score", "contact_map_consistency", "pLDDT", "ramachandran_compliance",
        "clash_score", "perplexity", "repetition_rate", "max_train_similarity", "is_novel"
    ]
    rows = []
    for (dataset, split_name, method), sub in df.groupby(["dataset", "split_name", "method"]):
        row = {"dataset": dataset, "split_name": split_name, "method": method, "n_candidates": len(sub)}
        for col in numeric_cols:
            if col in sub.columns:
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_median"] = float(sub[col].median())
        row["novelty_ratio"] = float(sub["is_novel"].mean()) if "is_novel" in sub.columns else float("nan")
        row["n_targets"] = int(sub["target_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--train-metadata", required=True, help="training metadata csv for novelty comparison")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--novelty-threshold", type=float, default=0.8)
    ap.add_argument("--perplexity-model", default=None, help="optional model path for perplexity scorer adapter")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    figdir = ensure_dir(outdir / "figures")

    df = load_method_manifest(args.manifest)
    train_sequences = load_train_sequences(args.train_metadata)

    ppl_scorer = PerplexityScorer(args.perplexity_model) if args.perplexity_model else None

    df = add_affinity_metrics(df)
    df = add_structure_metrics(df)
    df = add_generation_metrics(
        df,
        train_sequences=train_sequences,
        novelty_threshold=args.novelty_threshold,
        perplexity_scorer=ppl_scorer,
    )

    df.to_csv(outdir / "per_candidate_metrics.csv", index=False)

    agg = aggregate_metrics(df)
    agg.to_csv(outdir / "aggregate_metrics.csv", index=False)

    top5 = select_topk_by_hdock(df[df["method"].isin(["RFdiffusion", "bindcraft", "proteingenerator", "protein_generator"])], k=5)
    top5.to_csv(outdir / "top5_candidates_external_methods.csv", index=False)

    top5_summary = summarize_top5_affinity(top5) if len(top5) else pd.DataFrame()
    top5_summary.to_csv(outdir / "top5_affinity_summary.csv", index=False)

    print("Saved:")
    print(outdir / "per_candidate_metrics.csv")
    print(outdir / "aggregate_metrics.csv")
    print(outdir / "top5_affinity_summary.csv")


if __name__ == "__main__":
    main()

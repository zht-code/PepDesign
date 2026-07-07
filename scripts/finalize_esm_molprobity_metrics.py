from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_2_sota_metrics import aggregate_metrics, build_top5_summary, write_metric_definition  # noqa: E402
from utils.structure_metrics import _sequence_key  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize metrics tables from ESMFold and MolProbity backend outputs.")
    parser.add_argument("--metrics-dir", default="/root/autodl-tmp/Peptide_3D/results/2_SOTA/metrics_summary")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    backend_dir = metrics_dir / "_structure_backend"

    per_candidate_path = metrics_dir / "per_candidate_metrics.csv"
    esmfold_output_path = backend_dir / "esmfold_output.csv"
    molprobity_output_path = backend_dir / "molprobity_output.csv"

    per_candidate_df = pd.read_csv(per_candidate_path)
    per_candidate_df["query_id"] = per_candidate_df.apply(
        lambda row: f"{row['dataset']}|{row['method']}|{row['target_id']}|{int(row['candidate_rank']):02d}",
        axis=1,
    )
    per_candidate_df["sequence_id"] = per_candidate_df["sequence"].fillna("").astype(str).map(_sequence_key)

    esmfold_df = pd.read_csv(esmfold_output_path)
    molprobity_df = pd.read_csv(molprobity_output_path)

    seq_to_plddt = dict(zip(esmfold_df["sequence_id"].astype(str), pd.to_numeric(esmfold_df["plddt"], errors="coerce")))
    per_candidate_df["plddt"] = per_candidate_df["sequence_id"].map(seq_to_plddt)

    molprobity_df = molprobity_df.rename(columns={"peptide_only_pdb": "peptide_only_pdb_backend"})
    per_candidate_df = per_candidate_df.drop(columns=["ramachandran_compliance", "clash_score", "peptide_only_pdb"], errors="ignore")
    per_candidate_df = per_candidate_df.merge(
        molprobity_df[["query_id", "ramachandran_compliance", "clash_score", "peptide_only_pdb_backend"]],
        on="query_id",
        how="left",
    )
    per_candidate_df = per_candidate_df.rename(columns={"peptide_only_pdb_backend": "peptide_only_pdb"})

    ordered_columns = [
        "dataset",
        "method",
        "target_id",
        "candidate_rank",
        "hdock_score",
        "contact_consistency",
        "plddt",
        "ramachandran_compliance",
        "clash_score",
        "perplexity",
        "repetition_rate",
        "train_similarity",
        "novelty",
        "sequence",
        "pdb_path",
        "sequence_path",
        "json_path",
        "receptor_pdb",
        "reference_peptide_pdb",
        "pred_complex_pdb",
        "peptide_only_pdb",
        "candidate_name",
        "protein_id",
    ]
    per_candidate_df = per_candidate_df[ordered_columns].sort_values(
        ["dataset", "method", "target_id", "candidate_rank"]
    ).reset_index(drop=True)

    per_candidate_df.to_csv(per_candidate_path, index=False)
    aggregate_metrics(per_candidate_df).to_csv(metrics_dir / "aggregate_metrics.csv", index=False)
    build_top5_summary(per_candidate_df).to_csv(metrics_dir / "top5_summary.csv", index=False)
    write_metric_definition(metrics_dir / "metric_definition.md")


if __name__ == "__main__":
    main()

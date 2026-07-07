from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.affinity_parser import normalize_existing_path  # noqa: E402
from utils.generation_metrics import PerplexityScorer, repetition_rate, resolve_candidate_sequence  # noqa: E402
from utils.mmseqs_similarity import compute_train_similarity, novelty_from_similarity, prepare_train_fastas  # noqa: E402
from utils.result_indexer import build_results_index, resolve_input_paths  # noqa: E402
from utils.structure_metrics import batch_compute_structure_metrics, compute_contact_consistency_for_row  # noqa: E402


LOGGER = logging.getLogger("eval_2_sota_metrics")
NUMERIC_METRICS = [
    "hdock_score",
    "contact_consistency",
    "plddt",
    "ramachandran_compliance",
    "clash_score",
    "perplexity",
    "repetition_rate",
    "train_similarity",
    "novelty",
]


def _resolve_output_dir(path_str: str) -> Path:
    resolved = normalize_existing_path(path_str)
    if resolved:
        return Path(resolved)
    path = Path(path_str).expanduser()
    if path_str.startswith("/autodl-tmp/"):
        path = Path("/root") / path_str.lstrip("/")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _metric_stats(series: pd.Series) -> Dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    return {
        "mean": float(numeric.mean()) if not numeric.dropna().empty else float("nan"),
        "median": float(numeric.median()) if not numeric.dropna().empty else float("nan"),
        "std": float(numeric.std(ddof=0)) if not numeric.dropna().empty else float("nan"),
        "count": int(numeric.notna().sum()),
    }


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (dataset, method), sub in df.groupby(["dataset", "method"], dropna=False):
        row = {
            "dataset": dataset,
            "method": method,
            "candidate_count": int(len(sub)),
            "target_count": int(sub["target_id"].nunique()),
            "novelty_ratio": float(pd.to_numeric(sub["novelty"], errors="coerce").mean()),
        }
        for metric in NUMERIC_METRICS:
            stats = _metric_stats(sub[metric])
            for suffix, value in stats.items():
                row[f"{metric}_{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)


def build_top5_summary(df: pd.DataFrame) -> pd.DataFrame:
    target_level = (
        df.groupby(["dataset", "method", "target_id"], dropna=False)[NUMERIC_METRICS]
        .mean(numeric_only=True)
        .reset_index()
    )
    rows: List[dict] = []
    for (dataset, method), sub in target_level.groupby(["dataset", "method"], dropna=False):
        row = {
            "dataset": dataset,
            "method": method,
            "target_count": int(sub["target_id"].nunique()),
        }
        for metric in NUMERIC_METRICS:
            stats = _metric_stats(sub[metric])
            for suffix, value in stats.items():
                row[f"{metric}_top5_{suffix}"] = value
        row["novelty_ratio_top5_mean"] = float(pd.to_numeric(sub["novelty"], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)


def write_metric_definition(output_path: Path) -> None:
    text = """# 2_SOTA Unified Metric Definitions

- `hdock_score`: parsed from existing HDOCK result files / json payloads. Lower is better.
- `contact_consistency`: Jaccard overlap of receptor-peptide interface residue contacts between native complex and predicted top1 docked complex, reusing project `contact_map_consistency()`.
- `plddt`: recomputed with ESMFold from the candidate peptide sequence. The reported value is the mean pLDDT read from the ESMFold output PDB B-factors.
- `ramachandran_compliance`: recomputed with MolProbity Ramachandran analysis (`mmtbx.validation.ramalyze`) on the peptide-only PDB, reported as the favored-residue fraction.
- `clash_score`: recomputed with MolProbity clashscore on the peptide-only PDB after hydrogen addition with Reduce, reported as clashes per 1000 atoms.
- `perplexity`: placeholder column. The current repository does not expose a ready-to-call peptide LM scorer, so values remain `NaN` unless a scorer is implemented later.
- `repetition_rate`: repeated 3-gram ratio over the peptide sequence, i.e. fraction of overlapping 3-mers belonging to a 3-mer type observed more than once.
- `train_similarity`: nearest-neighbor sequence identity against the training set, computed with MMseqs2 (`easy-search`, best hit `pident` / 100). By default the training pool is built from `/autodl-tmp/train_data` peptide files. If exhaustive MMseqs2 search still finds no detectable hit, the similarity is recorded as `0.0`.
- `novelty`: binary flag derived from `train_similarity < novelty_threshold`.
- `novelty_ratio`: mean of `novelty` during aggregation.

## Implementation Notes

- Native complexes are cached under `_native_complex_cache` by merging receptor and native peptide PDBs.
- For multi-chain candidate PDBs, peptide-only structure metrics and sequence extraction are computed on the shortest chain, or the chain whose length is closest to the native peptide length when available.
- Existing project modules were reused where available for indexing, sequence extraction, peptide-chain selection, affinity contact consistency, and native complex merging.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Unified 2_SOTA metric evaluation.")
    parser.add_argument("--baseline-dir", default="/autodl-tmp/Peptide_3D/results/2_SOTA/baseline_data")
    parser.add_argument("--unconditional-dir", default="/autodl-tmp/Peptide_3D/results/2_SOTA/unconditional")
    parser.add_argument("--ours-family-dir", default="/autodl-tmp/Peptide_3D/results/2_SOTA/family_level_test")
    parser.add_argument("--ours-protein-dir", default="/autodl-tmp/Peptide_3D/results/2_SOTA/protein_level_test")
    parser.add_argument("--output-dir", default="/autodl-tmp/Peptide_3D/results/2_SOTA/metrics_summary")
    parser.add_argument("--train-fasta", default=None)
    parser.add_argument("--train-root", default="/autodl-tmp/train_data")
    parser.add_argument("--mmseqs", default="/root/autodl-fs/mmseqs-linux-gpu/mmseqs/bin/mmseqs")
    parser.add_argument("--novelty-threshold", type=float, default=0.8)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--esmfold-python", default="/root/venvs/esmfold/bin/python")
    parser.add_argument("--molprobity-python", default="/root/miniconda3/envs/sota_cctbx/bin/python")
    parser.add_argument("--esmfold-torch-home", default="/root/autodl-tmp/torch_cache")
    parser.add_argument("--esmfold-chunk-size", type=int, default=128)
    args = parser.parse_args()

    paths = resolve_input_paths(
        project_root=PROJECT_ROOT,
        baseline_dir=args.baseline_dir,
        unconditional_dir=args.unconditional_dir,
        ours_family_dir=args.ours_family_dir,
        ours_protein_dir=args.ours_protein_dir,
    )
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Building cross-method results index ...")
    index_df, split_files = build_results_index(
        project_root=paths["project_root"],
        baseline_dir=paths["baseline_dir"],
        unconditional_dir=paths["unconditional_dir"],
        ours_family_dir=paths["ours_family_dir"],
        ours_protein_dir=paths["ours_protein_dir"],
        topk=args.topk,
    )
    index_csv = output_dir / "results_index.csv"
    index_df.to_csv(index_csv, index=False)
    LOGGER.info("Saved results index to %s", index_csv)

    perplexity_scorer = PerplexityScorer()
    if not perplexity_scorer.available:
        LOGGER.warning("Perplexity scorer is unavailable in the current repository; `perplexity` will be NaN.")

    native_complex_cache_dir = output_dir / "_native_complex_cache"
    records: List[dict] = []
    success_count = 0
    failure_count = 0

    LOGGER.info("Computing per-candidate structure and generation metrics for %d candidates ...", len(index_df))
    for row in index_df.to_dict(orient="records"):
        record = dict(row)
        try:
            sequence = resolve_candidate_sequence(
                pdb_path=record.get("pdb_path"),
                reference_peptide_pdb=record.get("reference_peptide_pdb"),
            )
            record["sequence"] = sequence
            record["repetition_rate"] = repetition_rate(sequence)
            record["perplexity"] = perplexity_scorer.perplexity(sequence)
            record["plddt"] = float("nan")
            record["ramachandran_compliance"] = float("nan")
            record["clash_score"] = float("nan")
            try:
                record["contact_consistency"] = compute_contact_consistency_for_row(
                    record,
                    native_complex_cache_dir=native_complex_cache_dir,
                )
            except Exception as exc:
                LOGGER.warning("Failed contact consistency for %s: %s", record.get("pdb_path"), exc)
                record["contact_consistency"] = float("nan")

            metric_values = [record.get(metric) for metric in ("contact_consistency",)]
            any_metric = any(value is not None and not (isinstance(value, float) and math.isnan(value)) for value in metric_values)
            if sequence or any_metric:
                success_count += 1
            else:
                failure_count += 1
            records.append(record)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "Failed to evaluate row %s/%s/%s/%s: %s",
                record.get("dataset"),
                record.get("method"),
                record.get("target_id"),
                record.get("candidate_rank"),
                exc,
            )
            record["sequence"] = ""
            record["repetition_rate"] = float("nan")
            record["perplexity"] = float("nan")
            record["plddt"] = float("nan")
            record["ramachandran_compliance"] = float("nan")
            record["clash_score"] = float("nan")
            record["contact_consistency"] = float("nan")
            failure_count += 1
            records.append(record)

    per_candidate_df = pd.DataFrame(records)
    per_candidate_df["query_id"] = per_candidate_df.apply(
        lambda row: f"{row['dataset']}|{row['method']}|{row['target_id']}|{int(row['candidate_rank']):02d}",
        axis=1,
    )

    LOGGER.info("Recomputing pLDDT with ESMFold and structural validation with MolProbity ...")
    structure_backend_df = batch_compute_structure_metrics(
        per_candidate_df=per_candidate_df,
        output_dir=output_dir,
        esmfold_python=args.esmfold_python,
        molprobity_python=args.molprobity_python,
        esmfold_torch_home=args.esmfold_torch_home,
        esmfold_chunk_size=args.esmfold_chunk_size,
    )
    per_candidate_df = per_candidate_df.drop(columns=["plddt", "ramachandran_compliance", "clash_score"], errors="ignore")
    per_candidate_df = per_candidate_df.merge(structure_backend_df, on="query_id", how="left")

    train_fastas = prepare_train_fastas(
        split_files=split_files,
        output_dir=output_dir,
        train_fasta=args.train_fasta,
        train_root=args.train_root,
    )
    dataset_to_queries = {
        dataset: list(
            per_candidate_df.loc[per_candidate_df["dataset"] == dataset, ["query_id", "sequence"]].itertuples(index=False, name=None)
        )
        for dataset in per_candidate_df["dataset"].dropna().unique()
    }
    similarity_map = compute_train_similarity(
        dataset_to_queries=dataset_to_queries,
        dataset_to_train_fasta=train_fastas,
        mmseqs=args.mmseqs,
        output_dir=output_dir,
    )

    per_candidate_df["train_similarity"] = per_candidate_df.apply(
        lambda row: similarity_map.get(row["dataset"], {}).get(row["query_id"], float("nan")),
        axis=1,
    )
    per_candidate_df["novelty"] = per_candidate_df["train_similarity"].apply(
        lambda value: novelty_from_similarity(value, args.novelty_threshold)
    )

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
    for column in ordered_columns:
        if column not in per_candidate_df.columns:
            per_candidate_df[column] = None
    per_candidate_df = per_candidate_df[ordered_columns].sort_values(["dataset", "method", "target_id", "candidate_rank"]).reset_index(drop=True)

    per_candidate_path = output_dir / "per_candidate_metrics.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    top5_path = output_dir / "top5_summary.csv"
    metric_def_path = output_dir / "metric_definition.md"

    per_candidate_df.to_csv(per_candidate_path, index=False)
    aggregate_metrics(per_candidate_df).to_csv(aggregate_path, index=False)
    build_top5_summary(per_candidate_df).to_csv(top5_path, index=False)
    write_metric_definition(metric_def_path)

    counts = per_candidate_df.groupby(["dataset", "method"]).size().reset_index(name="candidate_count")
    LOGGER.info("Finished unified evaluation.")
    LOGGER.info("Total samples: %d", len(per_candidate_df))
    LOGGER.info("Successful parses: %d", success_count)
    LOGGER.info("Failures: %d", failure_count)
    LOGGER.info("Included candidate counts by dataset x method:\n%s", counts.to_string(index=False))
    LOGGER.info("Saved per-candidate metrics: %s", per_candidate_path)
    LOGGER.info("Saved aggregate metrics: %s", aggregate_path)
    LOGGER.info("Saved top5 summary: %s", top5_path)
    LOGGER.info("Saved metric definition: %s", metric_def_path)


if __name__ == "__main__":
    main()



'''

python scripts/eval_2_sota_metrics.py \
  --baseline-dir /autodl-tmp/Peptide_3D/results/2_SOTA/baseline_data \
  --unconditional-dir /autodl-tmp/Peptide_3D/results/2_SOTA/unconditional \
  --ours-family-dir /autodl-tmp/Peptide_3D/results/2_SOTA/family_level_test \
  --ours-protein-dir /autodl-tmp/Peptide_3D/results/2_SOTA/protein_level_test \
  --output-dir /autodl-tmp/Peptide_3D/results/2_SOTA/metrics_summary \
  --mmseqs /root/autodl-fs/mmseqs-linux-gpu/mmseqs/bin/mmseqs \
  --novelty-threshold 0.8 \
  --topk 5

'''
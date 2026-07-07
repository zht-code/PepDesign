from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from utils.affinity_parser import (
    derive_complex_path,
    iter_bindcraft_results,
    iter_proteingenerator_results,
    iter_rf_results,
    normalize_existing_path,
    parse_ours_hdock_json,
    read_unconditional_progress_rows,
    resolve_rf_candidate_path,
    rf_candidate_rank_from_name,
    safe_float,
)


LOGGER = logging.getLogger(__name__)
DATASETS = ("protein_level_test", "family_level_test")
METHODS = ("ours", "unconditional", "rfdiffusion", "bindcraft", "proteingenerator")


def _normalize_user_dir(path_str: str) -> Path:
    resolved = normalize_existing_path(path_str)
    if resolved is None:
        raw = Path(path_str).expanduser()
        if raw.exists():
            return raw.resolve()
        if path_str.startswith("/autodl-tmp/"):
            alt = Path("/root") / path_str.lstrip("/")
            if alt.exists():
                return alt.resolve()
        raise FileNotFoundError(f"Path does not exist: {path_str}")
    return Path(resolved)


def _load_split_metadata(project_root: Path) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, Path]]:
    split_dir = project_root / "results" / "2_SOTA" / "splits"
    metadata: Dict[str, Dict[str, dict]] = {}
    train_split_files: Dict[str, Path] = {}
    for dataset in DATASETS:
        split_path = split_dir / f"{dataset}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split metadata: {split_path}")
        df = pd.read_csv(split_path)
        key = "sample_id" if "sample_id" in df.columns else "target_id"
        metadata[dataset] = {str(row[key]): row.to_dict() for _, row in df.iterrows()}
        train_split_path = split_dir / f"{dataset.replace('_test', '_train')}.csv"
        if not train_split_path.is_file():
            raise FileNotFoundError(f"Missing train split metadata: {train_split_path}")
        train_split_files[dataset] = train_split_path
    return metadata, train_split_files


def _target_meta(dataset_meta: Dict[str, dict], dataset: str, target_id: str) -> dict:
    row = dataset_meta.get(target_id, {})
    return {
        "dataset": dataset,
        "target_id": target_id,
        "protein_id": row.get("protein_id"),
        "receptor_pdb": normalize_existing_path(row.get("receptor_pdb")),
        "reference_peptide_pdb": normalize_existing_path(row.get("peptide_pdb") or row.get("reference_peptide_pdb")),
    }


def _candidate_rank_from_filename(filename: str) -> Optional[int]:
    match = re.search(r"_(\d+)\.pdb$", filename)
    if match:
        return int(match.group(1))
    return None


def _records_from_ours(dataset: str, dataset_root: Path, dataset_meta: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for target_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        target_id = target_dir.name
        meta = _target_meta(dataset_meta, dataset, target_id)
        cands_dir = target_dir / "multi_cands1"
        if not cands_dir.is_dir():
            LOGGER.warning("Missing multi_cands1 for ours target %s/%s", dataset, target_id)
            continue

        score_map = parse_ours_hdock_json(cands_dir / "cands_hdock_scores.json")
        for pdb_path in sorted(cands_dir.glob("pep_*.pdb")):
            rank = _candidate_rank_from_filename(pdb_path.name)
            pred_complex = normalize_existing_path(
                str(Path("/root/autodl-tmp/tmp_hdock_2sota_multi") / dataset / target_id / pdb_path.stem / "model_1.pdb")
            )
            rows.append(
                {
                    **meta,
                    "method": "ours",
                    "candidate_rank": rank,
                    "pdb_path": str(pdb_path.resolve()),
                    "sequence_path": None,
                    "json_path": str((cands_dir / "cands_hdock_scores.json").resolve()) if (cands_dir / "cands_hdock_scores.json").exists() else None,
                    "hdock_score": score_map.get(pdb_path.name, score_map.get(str(pdb_path.resolve()), float("nan"))),
                    "pred_complex_pdb": pred_complex,
                    "candidate_name": pdb_path.name,
                }
            )
    return rows


def _records_from_unconditional(unconditional_dir: Path) -> List[dict]:
    progress_json = unconditional_dir / "all_test_sets_affinity_progress.json"
    if not progress_json.is_file():
        raise FileNotFoundError(f"Missing unconditional progress json: {progress_json}")

    rows: List[dict] = []
    for row in read_unconditional_progress_rows(progress_json):
        dataset = str(row.get("split_name"))
        if dataset not in DATASETS:
            continue
        rows.append(
            {
                "dataset": dataset,
                "method": "unconditional",
                "target_id": str(row.get("sample_id")),
                "protein_id": row.get("protein_id"),
                "candidate_rank": int(row.get("candidate_rank")),
                "pdb_path": normalize_existing_path(row.get("generated_peptide_pdb")),
                "sequence_path": normalize_existing_path(
                    str(unconditional_dir / dataset / str(row.get("sample_id")) / "generated_sequences.fasta")
                ),
                "json_path": str(progress_json.resolve()),
                "hdock_score": safe_float(row.get("hdock_score")),
                "receptor_pdb": normalize_existing_path(row.get("receptor_pdb")),
                "reference_peptide_pdb": normalize_existing_path(row.get("reference_peptide_pdb")),
                "pred_complex_pdb": normalize_existing_path(
                    str(
                        Path("/root/autodl-tmp/hdock_unconditional_work")
                        / dataset
                        / str(row.get("sample_id"))
                        / f"cand_{int(row.get('candidate_rank')):02d}"
                        / "model_1.pdb"
                    )
                ),
                "candidate_name": Path(str(row.get("generated_peptide_pdb"))).name,
            }
        )
    return rows


def _records_from_rfdiffusion(baseline_dir: Path, dataset: str, dataset_meta: Dict[str, dict]) -> List[dict]:
    json_name = "RFdiffusion_hdock_family_cands_affinity.json" if dataset == "family_level_test" else "RFdiffusion_hdock_protein_cands_affinity.json"
    json_path = baseline_dir / json_name
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing RFdiffusion json: {json_path}")

    rows: List[dict] = []
    for result in iter_rf_results(json_path):
        target_id = str(result.get("target_id"))
        meta = _target_meta(dataset_meta, dataset, target_id)
        cand_name = str(result.get("cand_pdb"))
        pdb_path = resolve_rf_candidate_path(baseline_dir, dataset, target_id, cand_name, result.get("cand_pdb_path"))
        rows.append(
            {
                **meta,
                "method": "rfdiffusion",
                "candidate_rank": rf_candidate_rank_from_name(cand_name),
                "pdb_path": pdb_path,
                "sequence_path": None,
                "json_path": str(json_path.resolve()),
                "hdock_score": safe_float(result.get("hdock_score_top1")),
                "pred_complex_pdb": derive_complex_path(result) or pdb_path,
                "candidate_name": cand_name,
            }
        )
    return rows


def _records_from_bindcraft(baseline_dir: Path, dataset: str, dataset_meta: Dict[str, dict]) -> List[dict]:
    json_name = "hdock_bindcraft_family_affinity.json" if dataset == "family_level_test" else "hdock_bindcraft_protein_affinity.json"
    json_path = baseline_dir / json_name
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing BindCraft json: {json_path}")

    rows: List[dict] = []
    for result in iter_bindcraft_results(json_path):
        target_id = str(result.get("target_id"))
        meta = _target_meta(dataset_meta, dataset, target_id)
        pdb_path = normalize_existing_path(
            result.get("bindcraft_pdb"),
            extra_candidates=[
                baseline_dir
                / ("bindcraft_family_level_test_data" if dataset == "family_level_test" else "bindcraft_protein_level_test_data")
                / target_id
                / str(result.get("cand_pdb")),
            ],
        )
        rows.append(
            {
                **meta,
                "method": "bindcraft",
                "candidate_rank": None,
                "pdb_path": pdb_path,
                "sequence_path": None,
                "json_path": str(json_path.resolve()),
                "hdock_score": safe_float(result.get("hdock_score_top1")),
                "pred_complex_pdb": derive_complex_path(result),
                "candidate_name": str(result.get("cand_pdb")),
            }
        )
    return rows


def _records_from_proteingenerator(baseline_dir: Path, dataset: str, dataset_meta: Dict[str, dict]) -> List[dict]:
    json_name = f"proteingenerator_{dataset}_docking_results.json"
    json_path = baseline_dir / json_name
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing ProteinGenerator json: {json_path}")

    rows: List[dict] = []
    folder_name = "proteingenerator_family" if dataset == "family_level_test" else "proteingenerator_protein"
    for result in iter_proteingenerator_results(json_path):
        target_id = str(result.get("target_id"))
        meta = _target_meta(dataset_meta, dataset, target_id)
        candidate_name = Path(str(result.get("peptide_pdb"))).name
        pdb_path = normalize_existing_path(
            result.get("peptide_pdb"),
            extra_candidates=[baseline_dir / folder_name / target_id / candidate_name],
        )
        rows.append(
            {
                **meta,
                "method": "proteingenerator",
                "candidate_rank": None,
                "pdb_path": pdb_path,
                "sequence_path": None,
                "json_path": str(json_path.resolve()),
                "hdock_score": safe_float(result.get("score")),
                "pred_complex_pdb": derive_complex_path(result) or pdb_path,
                "candidate_name": candidate_name,
            }
        )
    return rows


def _clip_and_rank(records: Iterable[dict], topk: int) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    output_frames: List[pd.DataFrame] = []
    for (dataset, method, target_id), sub in df.groupby(["dataset", "method", "target_id"], dropna=False):
        original_count = len(sub)
        work = sub.copy()

        if method in {"ours", "unconditional", "rfdiffusion"}:
            work["candidate_rank"] = pd.to_numeric(work["candidate_rank"], errors="coerce")
            work = work.sort_values(["candidate_rank", "hdock_score"], na_position="last")
        else:
            work = work.sort_values("hdock_score", ascending=True, na_position="last").reset_index(drop=True)
            work["candidate_rank"] = range(1, len(work) + 1)

        if original_count < topk:
            LOGGER.warning("Method %s on %s/%s has only %d candidates (< topk=%d)", method, dataset, target_id, original_count, topk)

        work = work.head(topk).copy()
        work["candidate_rank"] = range(1, len(work) + 1)
        output_frames.append(work)

    out = pd.concat(output_frames, ignore_index=True)
    out["candidate_rank"] = out["candidate_rank"].astype(int)
    return out


def build_results_index(
    project_root: Path,
    baseline_dir: Path,
    unconditional_dir: Path,
    ours_family_dir: Path,
    ours_protein_dir: Path,
    topk: int,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    split_metadata, split_files = _load_split_metadata(project_root)

    all_rows: List[dict] = []
    all_rows.extend(_records_from_ours("family_level_test", ours_family_dir, split_metadata["family_level_test"]))
    all_rows.extend(_records_from_ours("protein_level_test", ours_protein_dir, split_metadata["protein_level_test"]))
    all_rows.extend(_records_from_unconditional(unconditional_dir))
    all_rows.extend(_records_from_rfdiffusion(baseline_dir, "family_level_test", split_metadata["family_level_test"]))
    all_rows.extend(_records_from_rfdiffusion(baseline_dir, "protein_level_test", split_metadata["protein_level_test"]))
    all_rows.extend(_records_from_bindcraft(baseline_dir, "family_level_test", split_metadata["family_level_test"]))
    all_rows.extend(_records_from_bindcraft(baseline_dir, "protein_level_test", split_metadata["protein_level_test"]))
    all_rows.extend(_records_from_proteingenerator(baseline_dir, "family_level_test", split_metadata["family_level_test"]))
    all_rows.extend(_records_from_proteingenerator(baseline_dir, "protein_level_test", split_metadata["protein_level_test"]))

    df = _clip_and_rank(all_rows, topk=topk)
    required_columns = [
        "dataset",
        "method",
        "target_id",
        "candidate_rank",
        "pdb_path",
        "sequence_path",
        "json_path",
        "receptor_pdb",
        "reference_peptide_pdb",
        "pred_complex_pdb",
        "hdock_score",
        "candidate_name",
        "protein_id",
    ]
    for column in required_columns:
        if column not in df.columns:
            df[column] = None

    df = df[required_columns]
    return df.sort_values(["dataset", "method", "target_id", "candidate_rank"]).reset_index(drop=True), split_files


def resolve_input_paths(
    project_root: Path,
    baseline_dir: str,
    unconditional_dir: str,
    ours_family_dir: str,
    ours_protein_dir: str,
) -> Dict[str, Path]:
    return {
        "project_root": project_root.resolve(),
        "baseline_dir": _normalize_user_dir(baseline_dir),
        "unconditional_dir": _normalize_user_dir(unconditional_dir),
        "ours_family_dir": _normalize_user_dir(ours_family_dir),
        "ours_protein_dir": _normalize_user_dir(ours_protein_dir),
    }

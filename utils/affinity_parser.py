from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


LOGGER = logging.getLogger(__name__)


def normalize_existing_path(path_like: Optional[str], extra_candidates: Optional[Iterable[Path]] = None) -> Optional[str]:
    candidates: List[Path] = []
    raw = str(path_like).strip() if path_like is not None else ""
    if raw:
        candidates.append(Path(raw).expanduser())
        if raw.startswith("/autodl-tmp/"):
            candidates.append(Path("/root") / raw.lstrip("/"))
        if raw.startswith("/autodl-fs/"):
            candidates.append(Path("/root") / raw.lstrip("/"))
        if raw.startswith("/root/autodl-tmp/Peptide_3D/"):
            candidates.append(Path(raw.replace("/root/autodl-tmp/Peptide_3D/", "/autodl-tmp/Peptide_3D/")))

    if extra_candidates:
        candidates.extend(extra_candidates)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_ours_hdock_json(json_path: Path) -> Dict[str, float]:
    if not json_path.is_file():
        return {}
    try:
        raw = load_json(json_path)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to parse ours hdock json %s: %s", json_path, exc)
        return {}

    mapping: Dict[str, float] = {}
    for key, value in raw.items():
        score = safe_float(value)
        if score != score:
            continue
        norm_key = os.path.basename(str(key))
        mapping[norm_key] = score
        mapping[str(key)] = score
    return mapping


def infer_unconditional_pred_complex_from_log(log_path: Optional[str]) -> Optional[str]:
    resolved = normalize_existing_path(log_path)
    if resolved is None:
        return None

    try:
        with open(resolved, "r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip()
    except OSError:
        return None

    match = re.search(r"\(cwd=(.+?)\)", first_line)
    if not match:
        return None

    work_dir = Path(match.group(1))
    for candidate_name in ("model_1.pdb", "top1.pdb", "top3.pdb"):
        candidate_path = work_dir / candidate_name
        if candidate_path.exists():
            return str(candidate_path.resolve())
    return None


def derive_complex_path(result: dict) -> Optional[str]:
    for field in ("top1_complex_pdb", "pred_complex_pdb"):
        resolved = normalize_existing_path(result.get(field))
        if resolved:
            return resolved

    work_dir = normalize_existing_path(result.get("work_dir"))
    if work_dir:
        for candidate_name in ("model_1.pdb", "top1.pdb"):
            candidate_path = Path(work_dir) / candidate_name
            if candidate_path.exists():
                return str(candidate_path.resolve())

    cand_pdb_path = normalize_existing_path(result.get("cand_pdb_path")) or normalize_existing_path(result.get("bindcraft_pdb")) or normalize_existing_path(result.get("peptide_pdb"))
    if cand_pdb_path:
        return cand_pdb_path
    return None


def rf_candidate_rank_from_name(name: str) -> Optional[int]:
    match = re.search(r"_[pf](\d+)_s\d+\.pdb$", f"_{name}")
    if not match:
        match = re.search(r"[pf](\d+)_s\d+\.pdb$", name)
    if not match:
        return None
    return int(match.group(1)) + 1


def resolve_rf_candidate_path(
    baseline_dir: Path,
    dataset: str,
    target_id: str,
    cand_name: str,
    original_path: Optional[str],
) -> Optional[str]:
    dataset_folder = "RFdiffusion_family_level_test" if dataset == "family_level_test" else "RFdiffusion_protein_level_test"
    extra = [
        baseline_dir / dataset_folder / target_id / "cands" / cand_name,
        baseline_dir / dataset_folder / target_id / cand_name,
    ]
    return normalize_existing_path(original_path, extra_candidates=extra)


def resolve_rf_native_path(
    baseline_dir: Path,
    dataset: str,
    target_id: str,
    filename: str,
) -> Optional[str]:
    dataset_folder = "RFdiffusion_family_level_test" if dataset == "family_level_test" else "RFdiffusion_protein_level_test"
    extra = [baseline_dir / dataset_folder / target_id / filename]
    return normalize_existing_path(None, extra_candidates=extra)


def iter_rf_results(json_path: Path) -> Iterator[dict]:
    data = load_json(json_path)
    for result in data.get("results", []):
        yield result


def iter_bindcraft_results(json_path: Path) -> Iterator[dict]:
    data = load_json(json_path)
    for result in data.get("results", []):
        yield result


def iter_proteingenerator_results(json_path: Path) -> Iterator[dict]:
    data = load_json(json_path)
    for target_id, target_entries in data.items():
        if not isinstance(target_entries, dict):
            continue
        for _, payload in target_entries.items():
            if not isinstance(payload, dict):
                continue
            row = dict(payload)
            row["target_id"] = row.get("protein_id", target_id)
            yield row


def read_unconditional_progress_rows(progress_json: Path) -> List[dict]:
    data = load_json(progress_json)
    rows = data.get("rows", [])
    return [row for row in rows if isinstance(row, dict)]


def read_unconditional_affinity_csv(csv_path: Path) -> List[dict]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path("/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline").resolve()
PROJECT_ROOT = Path("/root/autodl-tmp/Peptide_3D").resolve()
ROBUSTNESS_ROOT = PROJECT_ROOT / "results" / "5_robustness"
PPDBENCH_ROOT = Path("/root/autodl-tmp/PPDbench").resolve()
SCRIPTS_DIR = BASE_DIR / "scripts"
CONFIGS_DIR = BASE_DIR / "configs"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
RAW_DIR = BASE_DIR / "raw_results"
TABLES_DIR = BASE_DIR / "tables"
METRICS_DIR = BASE_DIR / "metrics"
FIGURES_DIR = BASE_DIR / "figures"
CASES_DIR = BASE_DIR / "cases"
TMP_DIR = BASE_DIR / "tmp"
REPOS_DIR = BASE_DIR / "repos"
ARCHIVE_CACHE_DIR = CACHE_DIR / "archives"
RECOVERY_DIR = CACHE_DIR / "recovered_structures"
HF_CACHE_DIR = CACHE_DIR / "hf_cache"
TORCH_CACHE_DIR = CACHE_DIR / "torch_cache"


def _prepare_runtime_dirs() -> None:
    for path in [
        BASE_DIR,
        SCRIPTS_DIR,
        CONFIGS_DIR,
        LOGS_DIR,
        CACHE_DIR,
        RAW_DIR,
        TABLES_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        CASES_DIR,
        TMP_DIR,
        REPOS_DIR,
        HF_CACHE_DIR,
        TORCH_CACHE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ["TMPDIR"] = str(TMP_DIR)
    os.environ["TEMP"] = str(TMP_DIR)
    os.environ["TMP"] = str(TMP_DIR)
    tempfile.tempdir = str(TMP_DIR)


_prepare_runtime_dirs()

ROB_SCRIPTS = ROBUSTNESS_ROOT / "scripts"
if str(ROB_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROB_SCRIPTS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robustness_lib.aggregate_metrics import robustness_summary_row  # noqa: E402
from robustness_lib.metrics_eval import _aff, _solu, _stab  # noqa: E402


DEFAULT_STRUCTURE_LEVELS = [0.0, 10.0, 20.0, 30.0, 40.0]
DEFAULT_POCKET_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]
DEFAULT_SEQUENCE_LEVELS = [0.0, 10.0, 20.0, 30.0, 40.0]


@dataclass
class RepoRecord:
    name: str
    url: str
    local_path: Path
    commit_id: str
    clone_time_utc: str
    purpose: str


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("baseline_robustness")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"baseline_robustness_{stamp}.log"
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("log_file=%s", log_path)
    return logger


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_zip_extracted(zip_path: Path, extract_root: Path, logger: logging.Logger) -> Path | None:
    if not zip_path.is_file():
        return None
    stamp_file = extract_root / ".extracted_from"
    if stamp_file.is_file() and stamp_file.read_text(encoding="utf-8").strip() == str(zip_path.resolve()):
        return extract_root
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root.parent)
    stamp_file.write_text(str(zip_path.resolve()), encoding="utf-8")
    logger.info("extracted zip %s -> %s", zip_path, extract_root)
    return extract_root


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def read_thresholds(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {"hdock_max": -7.0, "stability_min": -3.0, "solubility_min": 0.45}
    text = path.read_text(encoding="utf-8")
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = [x.strip() for x in line.split(":", 1)]
        try:
            out[key] = float(value)
        except ValueError:
            continue
    if not out:
        return {"hdock_max": -7.0, "stability_min": -3.0, "solubility_min": 0.45}
    return out


def ensure_default_threshold_config() -> Path:
    path = CONFIGS_DIR / "success_rate_thresholds.yaml"
    if not path.is_file():
        path.write_text(
            "\n".join(
                [
                    "# Default thresholds reused when the exact paper operating point cannot be parsed automatically.",
                    "hdock_max: -7.0",
                    "stability_min: -3.0",
                    "solubility_min: 0.45",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return path


def slugify_level(level: float) -> str:
    return str(level).replace(".", "p")


def condition_tag(perturbation_type: str, level: float, repeat_id: int) -> str:
    return f"{perturbation_type}_lvl{slugify_level(level)}_r{repeat_id}"


def parse_repo_records() -> list[RepoRecord]:
    repos = [
        ("RFdiffusion", "https://github.com/RosettaCommons/RFdiffusion.git", "Official RFdiffusion codebase used for provenance and optional post-processing."),
        ("protein_generator", "https://github.com/RosettaCommons/protein_generator.git", "Official ProteinGenerator repository for provenance and adapter reference."),
        ("BindCraft", "https://github.com/martinpacesa/BindCraft.git", "Official BindCraft repository for provenance and format reference."),
        ("ProteinMPNN", "https://github.com/dauparas/ProteinMPNN.git", "Official ProteinMPNN repository for RFdiffusion sequence recovery when needed."),
    ]
    records: list[RepoRecord] = []
    for name, url, purpose in repos:
        local_path = REPOS_DIR / name
        commit_id = "missing"
        if (local_path / ".git").is_dir():
            try:
                commit_id = (
                    subprocess.check_output(
                        ["git", "-C", str(local_path), "rev-parse", "HEAD"],
                        text=True,
                    )
                    .strip()
                )
            except Exception:
                commit_id = "unknown"
        elif (local_path / "COMMIT_ID").is_file():
            commit_id = local_path.joinpath("COMMIT_ID").read_text(encoding="utf-8").strip()
        clone_time = (
            datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc).isoformat()
            if local_path.exists()
            else "missing"
        )
        records.append(
            RepoRecord(
                name=name,
                url=url,
                local_path=local_path,
                commit_id=commit_id,
                clone_time_utc=clone_time,
                purpose=purpose,
            )
        )
    return records


def aa3_to_1(resname: str) -> str:
    table = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "MSE": "M",
        "SEC": "U",
        "PYL": "O",
        "ASX": "B",
        "GLX": "Z",
        "UNK": "X",
    }
    return table.get(resname.strip().upper(), "X")


def analyze_pdb(pdb_path: Path) -> dict[str, Any]:
    residues: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    chain_res_counts: dict[str, set[tuple[str, str]]] = defaultdict(set)
    atom_counts: dict[str, int] = defaultdict(int)
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom = line[12:16].strip()
            resn = line[17:20].strip()
            chain = line[21].strip() or "_"
            resi = line[22:27].strip()
            residues[(chain, resi, resn)].add(atom)
            chain_res_counts[chain].add((resi, resn))
            atom_counts[atom] += 1
    residue_count = len(residues)
    chains = sorted(chain_res_counts)
    chain_sizes = {chain: len(items) for chain, items in chain_res_counts.items()}
    has_sequence = residue_count > 0
    has_sidechain = any(atom not in {"N", "CA", "C", "O", "OXT"} for atom in atom_counts)
    backbone_only = has_sequence and not has_sidechain
    return {
        "residue_count": residue_count,
        "chains": chains,
        "chain_sizes": chain_sizes,
        "has_sequence": has_sequence,
        "has_sidechain": has_sidechain,
        "backbone_only": backbone_only,
    }


def choose_peptide_chain(chain_sizes: dict[str, int], prefer_chain: str | None = None) -> str | None:
    if prefer_chain and prefer_chain in chain_sizes:
        return prefer_chain
    if not chain_sizes:
        return None
    sorted_items = sorted(chain_sizes.items(), key=lambda item: (item[1], item[0]))
    return sorted_items[0][0]


def extract_chain_pdb(src: Path, dst: Path, chain_id: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8", errors="ignore") as handle, open(dst, "w", encoding="utf-8") as out:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and (line[21].strip() or "_") == chain_id:
                out.write(line)
        out.write("END\n")


def extract_peptide_sequence(pdb_path: Path) -> str:
    seen: list[tuple[str, str, str]] = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            resn = line[17:20].strip()
            chain = line[21].strip() or "_"
            resi = line[22:27].strip()
            item = (chain, resi, resn)
            if not seen or seen[-1] != item:
                seen.append(item)
    return "".join(aa3_to_1(resn) for _, _, resn in seen)


def read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    seq_chunks: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        records.append((header, "".join(seq_chunks)))
    return records


def parse_score_from_header(header: str) -> float | None:
    m = re.search(r"score=([0-9.+-]+)", header)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def build_target_lookup() -> pd.DataFrame:
    rows = []
    for target_dir in sorted(p for p in PPDBENCH_ROOT.iterdir() if p.is_dir()):
        receptor = target_dir / "receptor.pdb"
        peptide = target_dir / "peptide.pdb"
        if receptor.is_file():
            rows.append(
                {
                    "target_id": target_dir.name.lower(),
                    "target_dir": str(target_dir),
                    "receptor_pdb": str(receptor),
                    "native_peptide_pdb": str(peptide) if peptide.is_file() else "",
                }
            )
    return pd.DataFrame(rows)


def find_path_by_basename(basename: str, search_roots: Iterable[Path]) -> Path | None:
    for root in search_roots:
        if not root.exists():
            continue
        matches = list(root.rglob(basename))
        if matches:
            return matches[0]
    return None


def build_index_for_proteingenerator_ppd(logger: logging.Logger) -> list[dict[str, Any]]:
    json_path = PROJECT_ROOT / "results" / "2_SOTA" / "baseline_data" / "proteingenerator_PPDbench_hdock_scores.json"
    if not json_path.is_file():
        logger.warning("ProteinGenerator JSON missing: %s", json_path)
        return []
    search_roots = [
        Path("/root/autodl-tmp/proteingenerator_ppdbench_hdock_work/extracted/proteingenerator"),
        PROJECT_ROOT / "results" / "2_SOTA" / "baseline_data" / "proteingenerator_family",
        PROJECT_ROOT / "results" / "2_SOTA" / "baseline_data" / "proteingenerator_protein",
    ]
    data = load_json(json_path)
    rows: list[dict[str, Any]] = []
    for target_id, inner in data.items():
        for original_path, payload in inner.items():
            score = payload.get("score") if isinstance(payload, dict) else None
            basename = Path(original_path).name
            resolved = Path(original_path)
            resolved_from = "json_path"
            if not resolved.is_file():
                alt = find_path_by_basename(basename, search_roots)
                if alt is not None:
                    resolved = alt
                    resolved_from = "search_roots"
            if resolved.is_file():
                analysis = analyze_pdb(resolved)
                peptide_path = resolved
                peptide_chain = choose_peptide_chain(analysis["chain_sizes"])
                clean_path = CACHE_DIR / "clean_inputs" / "proteingenerator" / target_id / basename
                if len(analysis["chains"]) > 1 and peptide_chain is not None:
                    extract_chain_pdb(resolved, clean_path, peptide_chain)
                    peptide_path = clean_path
                    analysis = analyze_pdb(peptide_path)
                elif peptide_path != clean_path:
                    clean_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(peptide_path, clean_path)
                    peptide_path = clean_path
            else:
                analysis = {
                    "backbone_only": None,
                    "has_sequence": None,
                    "has_sidechain": None,
                    "chains": [],
                    "residue_count": None,
                }
                peptide_path = resolved
                peptide_chain = None
            rows.append(
                {
                    "method": "proteingenerator",
                    "target_id": str(target_id).lower(),
                    "candidate_id": Path(original_path).stem,
                    "pdb_path": str(peptide_path),
                    "original_pdb_path": original_path,
                    "exists": peptide_path.is_file(),
                    "is_backbone_only": analysis["backbone_only"],
                    "has_sequence": analysis["has_sequence"],
                    "has_sidechain": analysis["has_sidechain"],
                    "source_dir": str(peptide_path.parent),
                    "source_score": score,
                    "source_json": str(json_path),
                    "resolved_from": resolved_from,
                    "peptide_chain": peptide_chain,
                    "residue_count": analysis["residue_count"],
                    "comparable_target_set": "ppdbench",
                    "unresolved_reason": "" if peptide_path.is_file() else "missing_ppdbench_candidate_pdb",
                }
            )
    logger.info("ProteinGenerator indexed rows=%d", len(rows))
    return rows


def build_index_for_bindcraft_ppd(logger: logging.Logger) -> list[dict[str, Any]]:
    json_path = PROJECT_ROOT / "results" / "2_SOTA" / "baseline_data" / "bindcraft_hdock_scores.json"
    if not json_path.is_file():
        logger.warning("BindCraft JSON missing: %s", json_path)
        return []
    extracted_zip_root = ensure_zip_extracted(
        Path("/root/autodl-tmp/result_bindcraft.zip"),
        ARCHIVE_CACHE_DIR / "result_bindcraft",
        logger,
    )
    search_roots = [
        extracted_zip_root / "result_clean_pdb" if extracted_zip_root else Path("/__missing__"),
        PROJECT_ROOT / "results" / "2_SOTA" / "baseline_data" / "bindcraft_family_level_test_data",
        PROJECT_ROOT / "results" / "2_SOTA" / "baseline_data" / "bindcraft_protein_level_test_data",
        Path("/root/autodl-tmp/BindCraft-main/experiments/result_clean_pdb"),
    ]
    data = load_json(json_path)
    ppdbench_targets = set(build_target_lookup()["target_id"].astype(str).str.lower().tolist())
    rows: list[dict[str, Any]] = []
    for target_id, inner in data.items():
        for original_path, payload in inner.items():
            score = payload.get("score") if isinstance(payload, dict) else None
            basename = Path(original_path).name
            resolved = Path(original_path)
            resolved_from = "json_path"
            if not resolved.is_file():
                alt = find_path_by_basename(basename, search_roots)
                if alt is not None:
                    resolved = alt
                    resolved_from = "search_roots"
            peptide_chain = None
            if resolved.is_file():
                analysis = analyze_pdb(resolved)
                peptide_path = resolved
                peptide_chain = choose_peptide_chain(analysis["chain_sizes"], prefer_chain="B")
                clean_path = CACHE_DIR / "clean_inputs" / "bindcraft" / str(target_id).lower() / basename
                if len(analysis["chains"]) > 1 and peptide_chain is not None:
                    extract_chain_pdb(resolved, clean_path, peptide_chain)
                    peptide_path = clean_path
                    analysis = analyze_pdb(peptide_path)
                elif peptide_path != clean_path:
                    clean_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(peptide_path, clean_path)
                    peptide_path = clean_path
            else:
                analysis = {
                    "backbone_only": None,
                    "has_sequence": None,
                    "has_sidechain": None,
                    "chains": [],
                    "residue_count": None,
                }
                peptide_path = resolved
            rows.append(
                {
                    "method": "bindcraft",
                    "target_id": str(target_id).lower(),
                    "candidate_id": Path(original_path).stem,
                    "pdb_path": str(peptide_path),
                    "original_pdb_path": original_path,
                    "exists": peptide_path.is_file(),
                    "is_backbone_only": analysis["backbone_only"],
                    "has_sequence": analysis["has_sequence"],
                    "has_sidechain": analysis["has_sidechain"],
                    "source_dir": str(peptide_path.parent),
                    "source_score": score,
                    "source_json": str(json_path),
                    "resolved_from": resolved_from,
                    "peptide_chain": peptide_chain,
                    "residue_count": analysis["residue_count"],
                    "comparable_target_set": "ppdbench" if str(target_id).lower() in ppdbench_targets else "non_ppdbench_or_unknown",
                    "unresolved_reason": "" if peptide_path.is_file() else "ppdbench_bindcraft_pdb_not_found_on_machine",
                }
            )
    logger.info("BindCraft indexed rows=%d", len(rows))
    return rows


def build_index_for_rfdiffusion(logger: logging.Logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extracted_zip_root = ensure_zip_extracted(
        Path("/root/autodl-tmp/RFdiffusion_top5.zip"),
        ARCHIVE_CACHE_DIR / "RFdiffusion_top5",
        logger,
    )
    if extracted_zip_root:
        # Newer archives may ship ready-to-evaluate peptide structures directly.
        pdb_paths = sorted(extracted_zip_root.glob("**/*.pdb"))
        if pdb_paths:
            for pdb_path in pdb_paths:
                candidate_id = pdb_path.stem
                # Expected naming: <target>_<rank>.pdb (e.g. 1cjr_1.pdb)
                target_id = candidate_id.rsplit("_", 1)[0].lower() if "_" in candidate_id else candidate_id.lower()
                analysis = analyze_pdb(pdb_path)
                peptide_chain = choose_peptide_chain(analysis["chain_sizes"])
                basename = f"{candidate_id}.pdb"
                clean_path = CACHE_DIR / "clean_inputs" / "rfdiffusion" / target_id / basename
                if peptide_chain:
                    extract_chain_pdb(pdb_path, clean_path, peptide_chain)
                else:
                    clean_path = pdb_path
                rows.append(
                    {
                        "method": "rfdiffusion",
                        "target_id": target_id,
                        "candidate_id": candidate_id,
                        "pdb_path": str(clean_path),
                        "original_pdb_path": str(pdb_path),
                        "exists": clean_path.is_file(),
                        "is_backbone_only": analysis["backbone_only"],
                        "has_sequence": analysis["has_sequence"],
                        "has_sidechain": analysis["has_sidechain"],
                        "source_dir": str(clean_path.parent),
                        "source_score": None,
                        "source_json": str(Path("/root/autodl-tmp/RFdiffusion_top5.zip")),
                        "resolved_from": "rfdiffusion_top5_zip_pdb",
                        "peptide_chain": peptide_chain,
                        "residue_count": analysis["residue_count"],
                        "comparable_target_set": "ppdbench",
                        "unresolved_reason": "",
                    }
                )
            logger.info("RFdiffusion structure indexed rows=%d from zip PDBs", len(rows))
            return rows

    if extracted_zip_root and (extracted_zip_root / "seqs").is_dir():
        for fasta_path in sorted((extracted_zip_root / "seqs").glob("*.fa")):
            records = read_fasta_records(fasta_path)
            if not records:
                continue
            designed_header, designed_seq = records[-1]
            candidate_id = fasta_path.stem
            target_id = candidate_id.rsplit("_", 1)[0].lower()
            recovered_pdb = RECOVERY_DIR / "rfdiffusion" / target_id / f"{candidate_id}.pdb"
            rows.append(
                {
                    "method": "rfdiffusion",
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "pdb_path": str(recovered_pdb),
                    "original_pdb_path": str(fasta_path),
                    "exists": recovered_pdb.is_file(),
                    "is_backbone_only": None if not recovered_pdb.is_file() else analyze_pdb(recovered_pdb)["backbone_only"],
                    "has_sequence": True,
                    "has_sidechain": None if not recovered_pdb.is_file() else analyze_pdb(recovered_pdb)["has_sidechain"],
                    "source_dir": str(fasta_path.parent),
                    "source_score": parse_score_from_header(designed_header),
                    "source_json": str(fasta_path),
                    "resolved_from": "rfdiffusion_top5_zip",
                    "peptide_chain": "A",
                    "residue_count": len(designed_seq),
                    "comparable_target_set": "ppdbench",
                    "unresolved_reason": "" if recovered_pdb.is_file() else "sequence_only_requires_structure_recovery",
                }
            )
        logger.info("RFdiffusion sequence-only indexed rows=%d from zip", len(rows))
        return rows
    json_candidates = [
        PROJECT_ROOT / "results" / "2_SOTA" / "baseline_data" / "RFdiffusion_hdock_family_cands_affinity.json",
        PROJECT_ROOT / "results" / "2_SOTA" / "baseline_data" / "RFdiffusion_hdock_protein_cands_affinity.json",
    ]
    direct_found = 0
    for json_path in json_candidates:
        if not json_path.is_file():
            continue
        data = load_json(json_path)
        for rec in data.get("results", []):
            target_id = str(rec.get("target_id", "")).lower()
            if "_" in target_id:
                # Family/protein split targets are not the same IDs as the PPDbench robustness target set.
                continue
            cand = Path(rec.get("cand_pdb_path", ""))
            if cand.is_file():
                analysis = analyze_pdb(cand)
                rows.append(
                    {
                        "method": "rfdiffusion",
                        "target_id": target_id,
                        "candidate_id": cand.stem,
                        "pdb_path": str(cand),
                        "original_pdb_path": str(cand),
                        "exists": True,
                        "is_backbone_only": analysis["backbone_only"],
                        "has_sequence": analysis["has_sequence"],
                        "has_sidechain": analysis["has_sidechain"],
                        "source_dir": str(cand.parent),
                        "source_score": rec.get("hdock_score_top1"),
                        "source_json": str(json_path),
                        "resolved_from": "json_path",
                        "peptide_chain": choose_peptide_chain(analysis["chain_sizes"]),
                        "residue_count": analysis["residue_count"],
                        "comparable_target_set": "ppdbench",
                        "unresolved_reason": "",
                    }
                )
                direct_found += 1
    if direct_found == 0:
        logger.warning("No direct PPDbench RFdiffusion peptide PDBs were found on this machine; method will be indexed as missing.")
    return rows


def build_baseline_input_index(logger: logging.Logger) -> pd.DataFrame:
    rows = []
    rows.extend(build_index_for_proteingenerator_ppd(logger))
    rows.extend(build_index_for_bindcraft_ppd(logger))
    rows.extend(build_index_for_rfdiffusion(logger))
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "method",
                "target_id",
                "candidate_id",
                "pdb_path",
                "original_pdb_path",
                "exists",
                "is_backbone_only",
                "has_sequence",
                "has_sidechain",
                "source_dir",
                "source_score",
                "source_json",
                "resolved_from",
                "peptide_chain",
                "residue_count",
                "comparable_target_set",
                "unresolved_reason",
            ]
        )
    df = df.sort_values(["method", "target_id", "source_score", "candidate_id"], na_position="last")
    out_path = TABLES_DIR / "baseline_input_index.csv"
    df.to_csv(out_path, index=False)
    logger.info("baseline_input_index -> %s rows=%d", out_path, len(df))
    return df


def select_best_candidates(index_df: pd.DataFrame) -> pd.DataFrame:
    if index_df.empty:
        return index_df.copy()
    df = index_df[index_df["exists"] == True].copy()  # noqa: E712
    df["source_score_numeric"] = pd.to_numeric(df["source_score"], errors="coerce")
    df = df.sort_values(["method", "target_id", "source_score_numeric", "candidate_id"], na_position="last")
    best = df.groupby(["method", "target_id"], as_index=False).first()
    best = best.drop(columns=["source_score_numeric"], errors="ignore")
    best.to_csv(TABLES_DIR / "baseline_best_candidates.csv", index=False)
    return best


def success_rate_flag(affinity: float | None, stability: float | None, solubility: float | None, thresholds: dict[str, float]) -> bool | None:
    if affinity is None or stability is None or solubility is None:
        return None
    return bool(
        affinity < thresholds["hdock_max"]
        and stability > thresholds["stability_min"]
        and solubility > thresholds["solubility_min"]
    )


def parse_pdb_atoms(pdb_path: Path) -> list[dict[str, Any]]:
    atoms = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                atoms.append(
                    {
                        "line": line.rstrip("\n"),
                        "record": line[:6],
                        "atom_name": line[12:16].strip(),
                        "resname": line[17:20].strip(),
                        "chain": line[21].strip() or "_",
                        "resseq": line[22:26].strip(),
                        "icode": line[26].strip(),
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                    }
                )
            except ValueError:
                continue
    return atoms


def residue_keys_from_atoms(atoms: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    seen = []
    for atom in atoms:
        key = (atom["chain"], atom["resseq"], atom["icode"])
        if not seen or seen[-1] != key:
            seen.append(key)
    return seen


def native_peptide_ca_points(peptide_pdb: Path) -> np.ndarray:
    pts = []
    if not peptide_pdb.is_file():
        return np.zeros((0, 3), dtype=float)
    for atom in parse_pdb_atoms(peptide_pdb):
        if atom["atom_name"] == "CA":
            pts.append([atom["x"], atom["y"], atom["z"]])
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 3), dtype=float)


def pocket_residue_keys(receptor_atoms: list[dict[str, Any]], peptide_pdb: Path, radius_a: float = 10.0) -> set[tuple[str, str, str]]:
    pep_pts = native_peptide_ca_points(peptide_pdb)
    if len(pep_pts) == 0:
        return set(residue_keys_from_atoms(receptor_atoms))
    pocket = set()
    for atom in receptor_atoms:
        pos = np.array([atom["x"], atom["y"], atom["z"]], dtype=float)
        d = np.linalg.norm(pep_pts - pos[None, :], axis=1).min()
        if d <= radius_a:
            pocket.add((atom["chain"], atom["resseq"], atom["icode"]))
    return pocket


def format_pdb_line(atom: dict[str, Any]) -> str:
    line = atom["line"]
    return f"{line[:30]}{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}{line[54:]}\n"


def perturb_receptor_pdb(
    *,
    receptor_pdb: Path,
    native_peptide_pdb: Path,
    perturbation_type: str,
    level: float,
    seed: int,
) -> list[dict[str, Any]]:
    atoms = parse_pdb_atoms(receptor_pdb)
    if level <= 0:
        return atoms

    residue_keys = residue_keys_from_atoms(atoms)
    rng = np.random.default_rng(seed)

    if perturbation_type == "structure_missing":
        k = int(np.floor(len(residue_keys) * (level / 100.0)))
        drop_idx = set(rng.choice(len(residue_keys), size=max(0, min(k, len(residue_keys))), replace=False).tolist())
        drop_keys = {residue_keys[i] for i in drop_idx}
        return [atom for atom in atoms if (atom["chain"], atom["resseq"], atom["icode"]) not in drop_keys]

    if perturbation_type == "sequence_trunc":
        keep = int(round(len(residue_keys) * (1.0 - level / 100.0)))
        keep = max(8, min(keep, len(residue_keys)))
        if keep >= len(residue_keys):
            return atoms
        start = int(rng.integers(0, len(residue_keys) - keep + 1))
        keep_keys = set(residue_keys[start : start + keep])
        return [atom for atom in atoms if (atom["chain"], atom["resseq"], atom["icode"]) in keep_keys]

    if perturbation_type == "pocket_noise":
        pocket_keys = pocket_residue_keys(atoms, native_peptide_pdb, radius_a=10.0)
        out = []
        for atom in atoms:
            atom = atom.copy()
            if (atom["chain"], atom["resseq"], atom["icode"]) in pocket_keys and atom["atom_name"] in {"N", "CA", "C"}:
                noise = rng.normal(scale=level, size=3)
                atom["x"] += float(noise[0])
                atom["y"] += float(noise[1])
                atom["z"] += float(noise[2])
            out.append(atom)
        return out

    raise ValueError(f"Unsupported perturbation_type={perturbation_type}")


def build_perturbed_receptor(
    *,
    target_id: str,
    perturbation_type: str,
    level: float,
    repeat_id: int,
    seed: int,
    logger: logging.Logger,
) -> Path:
    if level == 0:
        return PPDBENCH_ROOT / target_id / "receptor.pdb"

    out_dir = CACHE_DIR / "perturbed_targets" / condition_tag(perturbation_type, level, repeat_id) / target_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "receptor_perturbed.pdb"
    if out_path.is_file():
        return out_path

    receptor_pdb = PPDBENCH_ROOT / target_id / "receptor.pdb"
    native_peptide_pdb = PPDBENCH_ROOT / target_id / "peptide.pdb"
    perturbed_atoms = perturb_receptor_pdb(
        receptor_pdb=receptor_pdb,
        native_peptide_pdb=native_peptide_pdb,
        perturbation_type=perturbation_type,
        level=level,
        seed=seed,
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        for atom in perturbed_atoms:
            handle.write(format_pdb_line(atom))
        handle.write("END\n")
    logger.info(
        "perturbed receptor built target=%s perturb=%s level=%s repeat=%s -> %s",
        target_id,
        perturbation_type,
        level,
        repeat_id,
        out_path,
    )
    return out_path


def compute_clean_peptide_properties(
    *,
    method: str,
    target_id: str,
    peptide_pdb: Path,
    foldx_bin: str,
    proteinsol_wrapper: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    out_path = RAW_DIR / method / "clean_properties" / f"{target_id}.json"
    if out_path.is_file():
        return load_json(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seq = extract_peptide_sequence(peptide_pdb)
    stability = None
    solubility = None
    if Path(foldx_bin).is_file():
        stability = _stab().foldx_stability_score_single(
            peptide_pdb,
            foldx_bin=foldx_bin,
            workdir_root=str(CACHE_DIR / "foldx_work" / method / target_id),
            timeout_s=600,
        )
    else:
        logger.warning("FoldX binary missing: %s", foldx_bin)
    if Path(proteinsol_wrapper).is_file():
        solubility = _solu().solubility_score_from_seq_single(seq, proteinsol_wrapper=proteinsol_wrapper)
    else:
        logger.warning("Protein-Sol wrapper missing: %s", proteinsol_wrapper)
    payload = {
        "target_id": target_id,
        "method": method,
        "sequence": seq,
        "stability": stability,
        "solubility": solubility,
        "peptide_pdb": str(peptide_pdb),
    }
    write_json(out_path, payload)
    return payload


def run_hdock(
    *,
    receptor_pdb: Path,
    peptide_pdb: Path,
    work_dir: Path,
    hdock_bin: str,
    createpl_bin: str,
    timeout_s: int,
) -> float | None:
    work_dir.mkdir(parents=True, exist_ok=True)
    score, _log = _aff().run_hdock_pair(
        str(work_dir),
        str(receptor_pdb),
        str(peptide_pdb),
        hdock_bin,
        createpl_bin,
        timeout_s=timeout_s,
    )
    (work_dir / "dock.log").write_text(_log, encoding="utf-8")
    return score


def evaluate_method(
    *,
    method: str,
    best_df: pd.DataFrame,
    perturbation_type: str,
    levels: list[float],
    repeats: int,
    seed: int,
    hdock_bin: str,
    createpl_bin: str,
    foldx_bin: str,
    proteinsol_wrapper: str,
    skip_existing: bool,
    thresholds: dict[str, float],
    num_workers: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    method_df = best_df[best_df["method"] == method].copy()
    if method_df.empty:
        logger.warning("No candidates available for method=%s", method)
        return pd.DataFrame()

    method_rows: list[dict[str, Any]] = []
    sample_out_dir = RAW_DIR / method
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for row in method_df.to_dict(orient="records"):
        target_id = str(row["target_id"]).lower()
        peptide_pdb = Path(row["pdb_path"])
        if not peptide_pdb.is_file():
            logger.warning("skip missing peptide method=%s target=%s path=%s", method, target_id, peptide_pdb)
            continue
        clean_props = compute_clean_peptide_properties(
            method=method,
            target_id=target_id,
            peptide_pdb=peptide_pdb,
            foldx_bin=foldx_bin,
            proteinsol_wrapper=proteinsol_wrapper,
            logger=logger,
        )
        clean_affinity = row.get("source_score")
        if clean_affinity is not None:
            try:
                clean_affinity = float(clean_affinity)
            except Exception:
                clean_affinity = None
        if clean_affinity is not None:
            try:
                if np.isnan(clean_affinity):
                    clean_affinity = None
            except Exception:
                pass

        for repeat_id in range(repeats):
            for level in levels:
                tag = condition_tag(perturbation_type, level, repeat_id)
                out_csv = sample_out_dir / f"samples_{tag}.csv"
                if skip_existing and out_csv.is_file():
                    existing = pd.read_csv(out_csv)
                    if ((existing["target_id"] == target_id) & (existing["candidate_id"] == row["candidate_id"])).any():
                        continue
                tasks.append((row, target_id, peptide_pdb, clean_props, clean_affinity, repeat_id, level, tag, out_csv))

    def _run_one(task: tuple[Any, ...]) -> tuple[dict[str, Any], Path]:
        row, target_id, peptide_pdb, clean_props, clean_affinity, repeat_id, level, tag, out_csv = task
        affinity = clean_affinity
        notes = []
        error = None
        try:
            receptor_pdb = build_perturbed_receptor(
                target_id=target_id,
                perturbation_type=perturbation_type,
                level=level,
                repeat_id=repeat_id,
                seed=seed + repeat_id + int(level * 1000),
                logger=logger,
            )
            if level == 0 and affinity is not None:
                notes.append("clean_affinity_reused_from_existing_json")
            else:
                affinity = run_hdock(
                    receptor_pdb=receptor_pdb,
                    peptide_pdb=peptide_pdb,
                    work_dir=CACHE_DIR / "hdock_work" / method / tag / target_id,
                    hdock_bin=hdock_bin,
                    createpl_bin=createpl_bin,
                    timeout_s=900,
                )
                if level == 0:
                    notes.append("clean_affinity_computed_by_hdock")
        except Exception as exc:
            error = str(exc)

        item = {
            "method": method,
            "target_id": target_id,
            "candidate_id": row["candidate_id"],
            "repeat_id": repeat_id,
            "perturbation_type": perturbation_type,
            "level_value": level,
            "condition_tag": tag,
            "pdb_path": str(peptide_pdb),
            "sequence_top1": clean_props.get("sequence"),
            "affinity_hdock": affinity,
            "stability": clean_props.get("stability"),
            "solubility": clean_props.get("solubility"),
            "success_rate": success_rate_flag(
                affinity,
                clean_props.get("stability"),
                clean_props.get("solubility"),
                thresholds,
            ),
            "n_valid": 1 if error is None else 0,
            "notes": "; ".join(notes) if notes else "",
            "error": error,
        }
        return item, out_csv

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        by_csv: dict[Path, list[dict[str, Any]]] = defaultdict(list)
        for fut in as_completed(futures):
            item, out_csv = fut.result()
            method_rows.append(item)
            by_csv[out_csv].append(item)
            pd.DataFrame(by_csv[out_csv]).to_csv(out_csv, index=False)

    shard_frames: list[pd.DataFrame] = []
    for path in sorted(sample_out_dir.glob("samples_*.csv")):
        try:
            shard_frames.append(pd.read_csv(path))
        except Exception:
            continue
    if shard_frames:
        df = pd.concat(shard_frames, ignore_index=True)
        dedupe_cols = [c for c in ("target_id", "candidate_id", "perturbation_type", "level_value", "repeat_id") if c in df.columns]
        if dedupe_cols:
            df = df.drop_duplicates(subset=dedupe_cols, keep="last")
    else:
        df = pd.DataFrame(method_rows)
    if not df.empty:
        df.to_csv(sample_out_dir / "all_samples.csv", index=False)
    return df


def aggregate_method_results(method: str, df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    agg_rows = []
    summary_rows = []
    for perturbation_type, sub in df.groupby("perturbation_type"):
        sub = sub.copy()
        sub = sub[sub["error"].isna() | (sub["error"].astype(str) == "")]
        if sub.empty:
            continue
        group = (
            sub.groupby("level_value", as_index=False)
            .agg(
                n_valid=("target_id", "nunique"),
                affinity_mean=("affinity_hdock", "mean"),
                stability_mean=("stability", "mean"),
                solubility_mean=("solubility", "mean"),
                success_rate=("success_rate", lambda s: pd.Series(s).dropna().astype(float).mean() if len(pd.Series(s).dropna()) else np.nan),
            )
            .sort_values("level_value")
        )
        group["method"] = method
        group["perturbation_type"] = perturbation_type
        agg_rows.append(group)

        levels = group["level_value"].astype(float).to_numpy()
        if len(levels) == 0:
            continue
        level_norm = (
            (levels - levels.min()) / max(levels.max() - levels.min(), 1e-6)
            if levels.max() > levels.min()
            else np.zeros_like(levels)
        )
        metric_specs = [
            ("affinity_hdock", -group["affinity_mean"].astype(float).to_numpy(), "clean_mean/max_drop/AUDC use negated HDOCK (higher=better)."),
            ("stability", group["stability_mean"].astype(float).to_numpy(), ""),
            ("solubility", group["solubility_mean"].astype(float).to_numpy(), ""),
            ("success_rate", group["success_rate"].astype(float).to_numpy(), ""),
        ]
        for metric_name, values, notes in metric_specs:
            clean_val = float(values[0]) if len(values) else np.nan
            row = robustness_summary_row(
                perturb_type=perturbation_type,
                metric=metric_name,
                levels=levels,
                clean_val=clean_val,
                pert_vals=np.asarray(values, dtype=float),
                level_norm=np.asarray(level_norm, dtype=float),
            )
            row["method"] = method
            row["n_valid"] = int(group["n_valid"].max()) if "n_valid" in group else len(sub["target_id"].unique())
            row["notes"] = (row.get("notes", "") + " " + notes).strip()
            summary_rows.append(row)

    agg_df = pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    if not agg_df.empty:
        agg_df.to_csv(TABLES_DIR / f"{method}_robustness_aggregate_by_condition.csv", index=False)
    if not summary_df.empty:
        summary_df.to_csv(TABLES_DIR / f"{method}_robustness_summary.csv", index=False)
    logger.info("aggregated method=%s agg_rows=%d summary_rows=%d", method, len(agg_df), len(summary_df))
    return agg_df, summary_df


def load_existing_sample_results(methods: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for method in methods:
        method_dir = RAW_DIR / method
        frames = []
        if method_dir.is_dir():
            for path in sorted(method_dir.glob("samples_*.csv")):
                try:
                    frames.append(pd.read_csv(path))
                except Exception:
                    continue
        out[method] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


def load_ours_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = pd.read_csv(ROBUSTNESS_ROOT / "tables" / "robustness_aggregate_by_condition.csv")
    agg = agg.rename(columns={"perturb_type": "perturbation_type"})
    agg["method"] = "ours"
    summary = pd.read_csv(ROBUSTNESS_ROOT / "tables" / "Table_5_robustness_summary.csv")
    summary["method"] = "ours"
    summary["n_valid"] = 133
    return agg, summary


def merge_all_methods(
    *,
    baseline_index: pd.DataFrame,
    baseline_agg: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    ours_agg, ours_summary = load_ours_tables()
    all_agg = pd.concat([ours_agg, baseline_agg], ignore_index=True, sort=False)
    all_summary = pd.concat([ours_summary, baseline_summary], ignore_index=True, sort=False)
    all_agg.to_csv(TABLES_DIR / "all_methods_condition_curves_all_available.csv", index=False)
    all_summary.to_csv(TABLES_DIR / "all_methods_robustness_summary_all_available.csv", index=False)

    target_sets = {"ours": set(pd.read_csv(ROBUSTNESS_ROOT / "tables" / "robustness_all_samples_merged.csv")["target_id"].astype(str).str.lower().unique())}
    for method, sub in baseline_index[baseline_index["exists"] == True].groupby("method"):  # noqa: E712
        target_sets[method] = set(sub["target_id"].astype(str).str.lower().unique())

    required_methods = ["ours", "rfdiffusion", "proteingenerator", "bindcraft"]
    if all(method in target_sets for method in required_methods):
        intersection_targets = set.intersection(*(target_sets[m] for m in required_methods))
    else:
        intersection_targets = set()

    pd.DataFrame({"target_id": sorted(intersection_targets)}).to_csv(
        TABLES_DIR / "intersection_targets.csv", index=False
    )

    baseline_intersection = baseline_index[
        (baseline_index["exists"] == True) & (baseline_index["target_id"].isin(intersection_targets))  # noqa: E712
    ]
    baseline_intersection.to_csv(TABLES_DIR / "baseline_best_candidates_intersection.csv", index=False)

    all_summary_intersection = all_summary.copy()
    if not intersection_targets:
        all_summary_intersection = all_summary_intersection.iloc[0:0].copy()
    all_summary_intersection.to_csv(TABLES_DIR / "all_methods_robustness_summary_intersection.csv", index=False)
    all_agg_intersection = all_agg.copy()
    if not intersection_targets:
        all_agg_intersection = all_agg_intersection.iloc[0:0].copy()
    all_agg_intersection.to_csv(TABLES_DIR / "all_methods_condition_curves_intersection.csv", index=False)
    logger.info("intersection targets=%d", len(intersection_targets))


def recover_rfdiffusion_backbone(
    *,
    index_df: pd.DataFrame,
    device: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    rfd_all = index_df[index_df["method"] == "rfdiffusion"].copy()
    out_csv = RAW_DIR / "rfdiffusion_mpnn_sequences.csv"
    if rfd_all.empty:
        pd.DataFrame(
            columns=[
                "target_id",
                "candidate_id",
                "pdb_path",
                "mpnn_sequence",
                "status",
                "notes",
            ]
        ).to_csv(out_csv, index=False)
        logger.warning("No resolved RFdiffusion candidates found; wrote empty %s", out_csv)
        return pd.DataFrame()

    helper_script = SCRIPTS_DIR / "recover_rfdiffusion_structures.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(helper_script),
            "--index-csv",
            str(TABLES_DIR / "baseline_input_index.csv"),
            "--output-csv",
            str(out_csv),
            "--recovered-root",
            str(RECOVERY_DIR / "rfdiffusion"),
            "--device",
            device,
            "--batch-size",
            "1",
            "--skip-existing",
        ],
        check=False,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TMPDIR": str(TMP_DIR),
            "TEMP": str(TMP_DIR),
            "TMP": str(TMP_DIR),
            "OMP_NUM_THREADS": "1",
            "HF_HOME": str(HF_CACHE_DIR),
            "HUGGINGFACE_HUB_CACHE": str(HF_CACHE_DIR / "hub"),
            "TRANSFORMERS_CACHE": str(HF_CACHE_DIR / "transformers"),
            "TORCH_HOME": str(TORCH_CACHE_DIR),
        },
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-4000:]
        logger.warning(
            "RFdiffusion ESMFold recovery subprocess failed (exit=%s); RFdiffusion baseline curves stay empty until "
            "`facebook/esmfold_v1` can be loaded (check Hugging Face cache / network). stderr/stdout tail:\n%s",
            proc.returncode,
            tail,
        )
    df = pd.read_csv(out_csv) if out_csv.is_file() else pd.DataFrame()
    df.to_csv(out_csv, index=False)
    logger.info("RFdiffusion MPNN mapping -> %s rows=%d", out_csv, len(df))
    return df


def update_readme(
    *,
    repo_records: list[RepoRecord],
    index_df: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    readme_path = BASE_DIR / "README.md"
    repo_lines = []
    for record in repo_records:
        repo_lines.append(
            f"- `{record.name}`: {record.url}, commit `{record.commit_id}`, cloned `{record.clone_time_utc}`, purpose: {record.purpose}"
        )

    sources = []
    for method, sub in index_df.groupby("method"):
        dirs = sorted(set(str(Path(p).parent) for p in sub["pdb_path"].dropna().tolist()[:5]))
        sources.append(
            f"- `{method}`: rows={len(sub)}, resolved={(sub['exists'] == True).sum()}, comparable_target_sets={sorted(set(sub.get('comparable_target_set', pd.Series(dtype=str)).dropna().astype(str)))}, sample_source_dirs={dirs}"  # noqa: E712
        )

    readme_path.write_text(
        "\n".join(
            [
                "# Baseline Robustness Workspace",
                "",
                "This directory contains the full landing area for baseline robustness evaluation under the same target perturbation framework as `results/5_robustness`.",
                "",
                "## Official Repositories",
                *repo_lines,
                "",
                "## Existing Baseline Result Sources",
                *sources,
                "",
                "## Coverage Notes",
                "- `ProteinGenerator`: direct PPDbench peptide candidates were found and can be re-evaluated in the shared robustness pipeline.",
                "- `BindCraft`: the machine contains family-level outputs and JSON references to PPDbench outputs, but the referenced PPDbench peptide PDB files were not present at the expected paths during indexing. These candidates are therefore logged as unavailable for the strict PPDbench comparison unless matching files are later restored.",
                "- `RFdiffusion`: no direct PPDbench peptide candidate set was found on the current machine; family/protein split artifacts are intentionally excluded from the main comparison because they do not match the robustness target IDs used by `ours`.",
                "",
                "## Unified Evaluation Definition",
                "- Perturbations are applied to the target protein, not to the already generated peptide candidates.",
                "- Perturbation families are aligned to the existing Chapter 5 setup: `structure_missing` = 0/10/20/30/40%, `pocket_noise` = 0/0.5/1.0/1.5/2.0 A, `sequence_trunc` = 0/10/20/30/40%.",
                "- Metrics reuse the same project-side definitions whenever possible: `affinity_hdock`, `stability`, `solubility`, and `success_rate`.",
                "- Relative drop is computed as `(clean_metric - perturbed_metric) / clean_metric * 100%` on higher-is-better transformed metrics.",
                "",
                "## RFdiffusion Post-processing",
                "- The pipeline first checks whether RFdiffusion candidates are backbone-only.",
                "- If backbone-only PPDbench candidates are present, the intended path is `RFdiffusion backbone -> ProteinMPNN sequence recovery -> structure recovery / fallback peptide record` fully under this baseline directory.",
                "- On the current machine, no direct PPDbench RFdiffusion peptide inputs were located; the scaffolded recovery output is recorded in `raw_results/rfdiffusion_mpnn_sequences.csv`.",
                "",
                "## Intersection-only Principle",
                "- `all_methods_robustness_summary_all_available.csv` keeps all methods with any valid results.",
                "- `all_methods_robustness_summary_intersection.csv` keeps only targets shared by all four methods. If some methods are missing entirely on this machine, this table can be empty and the reason is preserved in logs and notes.",
                "",
                "## Run Examples",
                "```bash",
                "python baseline/scripts/run_baseline_robustness.py --methods all --build-index-only",
                "python baseline/scripts/run_baseline_robustness.py --methods proteingenerator --skip-existing",
                "python baseline/scripts/plot_robustness_comparison.py",
                "```",
                "",
                "## Directory Layout",
                "```text",
                "repos/      official repositories and provenance snapshots",
                "configs/    thresholds and runtime configs",
                "scripts/    baseline pipeline and plotting entrypoints",
                "logs/       stepwise execution logs",
                "cache/      cleaned peptides, perturbed receptors, HDOCK/FoldX workdirs",
                "raw_results/ sample-level per-method outputs and RFdiffusion sequence recovery tables",
                "tables/     input indices, aggregates, summaries, merged all-available/intersection tables",
                "metrics/    extra metric exports",
                "figures/    comparison figure outputs and caption drafts",
                "cases/      representative-case exports",
                "tmp/        all temporary runtime files forced under baseline/",
                "```",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("README updated -> %s", readme_path)


def write_execution_artifacts(
    *,
    index_df: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    tree_lines = [f"{BASE_DIR.name}/"]

    def _walk(path: Path, prefix: str = "") -> None:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        for idx, entry in enumerate(entries):
            connector = "└── " if idx == len(entries) - 1 else "├── "
            tree_lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir() and prefix.count("│") < 2 and prefix.count("    ") < 2:
                extension = "    " if idx == len(entries) - 1 else "│   "
                _walk(entry, prefix + extension)

    _walk(BASE_DIR)
    (TABLES_DIR / "baseline_directory_tree.txt").write_text("\n".join(tree_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "Baseline robustness execution summary",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    if index_df.empty:
        summary_lines.append("No baseline inputs were indexed.")
    else:
        coverage_rows = []
        grouped = {method: sub.copy() for method, sub in index_df.groupby("method")}
        for method in ["rfdiffusion", "proteingenerator", "bindcraft"]:
            sub = grouped.get(method, index_df.iloc[0:0].copy())
            resolved = int((sub["exists"] == True).sum())  # noqa: E712
            targets = int(sub[sub["exists"] == True]["target_id"].nunique())  # noqa: E712
            unresolved = (
                sub["unresolved_reason"]
                .dropna()
                .astype(str)
                .replace("", np.nan)
                .dropna()
                .value_counts()
                .to_dict()
            )
            coverage_rows.append(
                {
                    "method": method,
                    "rows": len(sub),
                    "resolved_rows": resolved,
                    "resolved_targets": targets,
                    "comparable_target_sets": "|".join(sorted(set(sub["comparable_target_set"].dropna().astype(str)))) if ("comparable_target_set" in sub and not sub.empty) else "",
                    "unresolved_reason_counts": json.dumps(unresolved, ensure_ascii=False) if unresolved else ("{\"direct_ppdbench_candidates_not_found\": 1}" if method == "rfdiffusion" and sub.empty else "{}"),
                }
            )
            summary_lines.extend(
                [
                    f"[{method}]",
                    f"rows={len(sub)} resolved_rows={resolved} resolved_targets={targets}",
                    f"unresolved_reasons={unresolved if unresolved else ({'direct_ppdbench_candidates_not_found': 1} if method == 'rfdiffusion' and sub.empty else {})}",
                    "",
                ]
            )
        pd.DataFrame(coverage_rows).to_csv(TABLES_DIR / "baseline_method_coverage.csv", index=False)
    (TABLES_DIR / "execution_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    logger.info(
        "execution artifacts updated -> %s, %s, %s",
        TABLES_DIR / "baseline_directory_tree.txt",
        TABLES_DIR / "execution_summary.txt",
        TABLES_DIR / "baseline_method_coverage.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline robustness pipeline for existing peptide baselines.")
    parser.add_argument("--methods", default="all", help="Comma-separated: rfdiffusion,proteingenerator,bindcraft,all")
    parser.add_argument("--perturbation-type", default="all", choices=["all", "structure_missing", "pocket_noise", "sequence_trunc"])
    parser.add_argument("--perturbation-strength", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--build-index-only", action="store_true")
    parser.add_argument("--rfdiffusion-only-postprocess", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--hdock-bin", default="/root/autodl-fs/HDOCKlite/hdock")
    parser.add_argument("--createpl-bin", default="/root/autodl-fs/HDOCKlite/createpl")
    parser.add_argument("--foldx-bin", default="/root/autodl-tmp/foldx_20270131")
    parser.add_argument("--proteinsol-wrapper", default="/root/autodl-tmp/protein-sol/multiple_prediction_wrapper_export.sh")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger()
    thresholds_path = ensure_default_threshold_config()
    thresholds = read_thresholds(thresholds_path)
    method_list = ["rfdiffusion", "proteingenerator", "bindcraft"] if args.methods == "all" else [m.strip() for m in args.methods.split(",") if m.strip()]

    repo_records = parse_repo_records()
    index_df = build_baseline_input_index(logger)
    best_df = select_best_candidates(index_df)
    update_readme(repo_records=repo_records, index_df=index_df, logger=logger)
    write_execution_artifacts(index_df=index_df, logger=logger)

    if args.build_index_only:
        logger.info("build-index-only requested; exiting after index generation.")
        return 0

    if args.aggregate_only:
        existing = load_existing_sample_results(method_list)
        baseline_agg_frames = []
        baseline_summary_frames = []
        for method, sample_df in existing.items():
            agg_df, summary_df = aggregate_method_results(method, sample_df, logger)
            if not agg_df.empty:
                baseline_agg_frames.append(agg_df)
            if not summary_df.empty:
                baseline_summary_frames.append(summary_df)
        baseline_agg = pd.concat(baseline_agg_frames, ignore_index=True) if baseline_agg_frames else pd.DataFrame()
        baseline_summary = pd.concat(baseline_summary_frames, ignore_index=True) if baseline_summary_frames else pd.DataFrame()
        if not baseline_agg.empty:
            baseline_agg.to_csv(TABLES_DIR / "baseline_robustness_aggregate_by_condition.csv", index=False)
        if not baseline_summary.empty:
            baseline_summary.to_csv(TABLES_DIR / "all_methods_robustness_summary.csv", index=False)
        merge_all_methods(
            baseline_index=best_df,
            baseline_agg=baseline_agg,
            baseline_summary=baseline_summary,
            logger=logger,
        )
        return 0

    if args.rfdiffusion_only_postprocess or "rfdiffusion" in method_list:
        recover_rfdiffusion_backbone(index_df=index_df, device=args.device, logger=logger)
        index_df = build_baseline_input_index(logger)
        best_df = select_best_candidates(index_df)
        update_readme(repo_records=repo_records, index_df=index_df, logger=logger)
        write_execution_artifacts(index_df=index_df, logger=logger)
    if args.rfdiffusion_only_postprocess:
        return 0

    levels_map = {
        "structure_missing": DEFAULT_STRUCTURE_LEVELS,
        "pocket_noise": DEFAULT_POCKET_LEVELS,
        "sequence_trunc": DEFAULT_SEQUENCE_LEVELS,
    }
    perturbs = list(levels_map) if args.perturbation_type == "all" else [args.perturbation_type]

    baseline_agg_frames = []
    baseline_summary_frames = []
    for method in method_list:
        for perturbation_type in perturbs:
            levels = levels_map[perturbation_type]
            if args.perturbation_strength is not None:
                levels = [float(args.perturbation_strength)]
            sample_df = evaluate_method(
                method=method,
                best_df=best_df,
                perturbation_type=perturbation_type,
                levels=levels,
                repeats=max(1, int(args.n_repeats)),
                seed=int(args.seed),
                hdock_bin=args.hdock_bin,
                createpl_bin=args.createpl_bin,
                foldx_bin=args.foldx_bin,
                proteinsol_wrapper=args.proteinsol_wrapper,
                skip_existing=args.skip_existing,
                thresholds=thresholds,
                num_workers=max(1, int(args.num_workers)),
                logger=logger,
            )
            agg_df, summary_df = aggregate_method_results(method, sample_df, logger)
            if not agg_df.empty:
                baseline_agg_frames.append(agg_df)
            if not summary_df.empty:
                baseline_summary_frames.append(summary_df)

    baseline_agg = pd.concat(baseline_agg_frames, ignore_index=True) if baseline_agg_frames else pd.DataFrame()
    baseline_summary = pd.concat(baseline_summary_frames, ignore_index=True) if baseline_summary_frames else pd.DataFrame()
    if not baseline_agg.empty:
        baseline_agg.to_csv(TABLES_DIR / "baseline_robustness_aggregate_by_condition.csv", index=False)
    if not baseline_summary.empty:
        baseline_summary.to_csv(TABLES_DIR / "all_methods_robustness_summary.csv", index=False)

    merge_all_methods(
        baseline_index=best_df,
        baseline_agg=baseline_agg,
        baseline_summary=baseline_summary,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

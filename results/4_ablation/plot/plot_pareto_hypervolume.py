#!/usr/bin/env python3
"""
Plot Pareto hypervolume for multi-objective optimization across ablation methods.

========================
Core idea (IMPORTANT)
========================
Pareto hypervolume (HV) MUST be computed from a *set of candidate points* per (method, target, top-k),
NOT from "3 averaged metrics per method". For each target, we build candidate vectors:

    x_i = (affinity_value, structure_quality, developability?)   where "larger is better" for each objective

Then for each method and target:
  1) sort candidates by affinity_value and select top-k
  2) take the non-dominated subset (Pareto front)
  3) compute hypervolume w.r.t. a reference point
  4) repeat for all targets, and report mean/std/sem across targets (main result)

========================
Default objectives (3 objectives; auto fallback to 2)
========================
We default to 3 objectives for hypervolume:
1) Affinity (larger is better)
   - Prefer HDOCK / docking score. Since original HDOCK score is usually "smaller is better",
     we convert to: affinity_value = -hdock_score

2) Structure quality (larger is better)
   Priority:
   - mean pLDDT from PDB B-factor (if present and non-trivial)
   - TM-score (if found from existing csv/json; optional)
   - fallback: -RMSD to reference peptide (Kabsch aligned), with clear logging

3) Developability (larger is better)
   Priority:
   - developability / solubility / stability scores from discovered json/csv/tsv files
   - If not found, we DO NOT fabricate values; we automatically fallback to 2 objectives:
       affinity + structure_quality
     and print:
       "Developability data not found, fallback to 2-objective Pareto hypervolume"

========================
Normalization (GLOBAL min-max)
========================
Before HV, all objectives are normalized globally across *all methods, all candidates*:
    z = (x - xmin) / (xmax - xmin + eps), eps=1e-8
Normalization is NOT per-method.

Reference point for HV after normalization:
    reference_point = [0, 0, ...]

Hypervolume interpretation:
  HV is larger when the Pareto set covers more of the desirable objective space, implying better trade-offs.

========================
Outputs
========================
Images (300 dpi PNG + vector PDF):
  - pareto_hv_top3_by_method.png / pdf
  - pareto_hv_top5_by_method.png / pdf
  - pareto_hv_top10_by_method.png / pdf
  - pareto_hv_vs_topk.png / pdf

CSVs:
  - candidate_objectives_merged.csv
  - candidate_objectives_normalized.csv
  - per_target_hypervolume.csv
  - hypervolume_summary.csv
  - data_source_log.csv

Run:
  python plot_pareto_hypervolume.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# Global configuration defaults
# -----------------------------

EPS: float = 1e-8
TOPK_LIST: List[int] = [1, 3, 5, 10]

METHOD_ORDER: List[str] = ["Base", "Base+OT", "Base+DPO", "Full"]


@dataclass
class SourceLogRow:
    method: str
    data_type: str
    source_path: str
    status: str
    notes: str


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _stem_id(s: str) -> str:
    """Normalize an identifier for fuzzy matching."""
    s = s.strip()
    s = s.replace("\\", "/")
    s = s.split("/")[-1]
    s = re.sub(r"\.(pdb|cif|json|csv|tsv)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return s.lower()


def parse_affinity_scores(method: str, json_path: Path, source_log: List[SourceLogRow]) -> pd.DataFrame:
    """
    Parse docking / HDOCK scores with robust field handling.

    Expected output columns:
      method, target_id, candidate_id, hdock_score, affinity_value, pdb_path
    """
    if not json_path.exists():
        source_log.append(SourceLogRow(method, "affinity_json", str(json_path), "missing", "File not found"))
        return pd.DataFrame(columns=["method", "target_id", "candidate_id", "hdock_score", "affinity_value", "pdb_path"])

    data = load_json(json_path)
    rows: List[Dict[str, Any]] = []

    # Case A: mapping of pdb_path -> score (Full file observed)
    if isinstance(data, dict) and all(isinstance(k, str) for k in data.keys()) and all(
        isinstance(v, (int, float)) for v in data.values()
    ):
        for k, v in data.items():
            pdb_path = str(k)
            hdock_score = float(v)
            affinity_value = -hdock_score
            cand = Path(pdb_path).name
            candidate_id = cand
            # infer target_id from path if possible: .../PPDbench/<target>/...
            target_id = "unknown"
            m = re.search(r"/PPDbench/([^/]+)/", pdb_path)
            if m:
                target_id = m.group(1)
            rows.append(
                {
                    "method": method,
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "hdock_score": hdock_score,
                    "affinity_value": affinity_value,
                    "pdb_path": pdb_path,
                }
            )
        source_log.append(SourceLogRow(method, "affinity_json", str(json_path), "ok", "Parsed as pdb_path->score map"))
        return pd.DataFrame(rows)

    # Case B: dict of records keyed by something like "target/pep_01.pdb" (Base/OT/DPO observed)
    if isinstance(data, dict):
        for key, rec in data.items():
            if not isinstance(rec, dict):
                continue

            # target id
            target_id = rec.get("target_id") or rec.get("target") or rec.get("protein") or rec.get("receptor")
            if target_id is None and isinstance(key, str) and "/" in key:
                target_id = key.split("/")[0]
            target_id = str(target_id) if target_id is not None else "unknown"

            # candidate id
            candidate_id = (
                rec.get("peptide_basename")
                or rec.get("candidate")
                or rec.get("name")
                or rec.get("pdb")
                or rec.get("peptide")
                or rec.get("sample_id")
            )
            if candidate_id is None and isinstance(key, str) and "/" in key:
                candidate_id = key.split("/")[-1]
            candidate_id = str(candidate_id) if candidate_id is not None else str(key)

            # score
            score_fields = ["score", "hdock_score", "docking_score", "hdock", "affinity"]
            hdock_score = None
            for sf in score_fields:
                if sf in rec:
                    hdock_score = _safe_float(rec.get(sf))
                    if hdock_score is not None:
                        break
            if hdock_score is None:
                # some records might store in "value"
                hdock_score = _safe_float(rec.get("value"))

            if hdock_score is None:
                # Skip but log once per method at the end
                continue

            # pdb path (optional)
            pdb_path = rec.get("peptide_pdb") or rec.get("pdb_path") or rec.get("pdb") or rec.get("peptide")
            pdb_path = str(pdb_path) if pdb_path is not None else ""

            affinity_value = -float(hdock_score)
            rows.append(
                {
                    "method": method,
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "hdock_score": float(hdock_score),
                    "affinity_value": affinity_value,
                    "pdb_path": pdb_path,
                }
            )

        source_log.append(
            SourceLogRow(
                method,
                "affinity_json",
                str(json_path),
                "ok",
                "Parsed as dict-of-records (fields: target_id, peptide_basename, score, peptide_pdb)",
            )
        )
        df = pd.DataFrame(rows)
        if df.empty:
            source_log.append(SourceLogRow(method, "affinity_json", str(json_path), "warning", "No valid score records parsed"))
        return df

    source_log.append(SourceLogRow(method, "affinity_json", str(json_path), "error", f"Unsupported JSON type: {type(data)}"))
    return pd.DataFrame(columns=["method", "target_id", "candidate_id", "hdock_score", "affinity_value", "pdb_path"])


def scan_pdb_files(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []
    return sorted([p for p in root_dir.rglob("*.pdb") if p.is_file()])


def extract_plddt_from_pdb(pdb_path: Path) -> Optional[float]:
    """
    Extract mean pLDDT from PDB B-factor column (ATOM/HETATM lines).

    Notes:
      In some pipelines, pLDDT is stored in B-factor. However, many generated PDBs may have B-factor=0.00.
      If B-factors are all 0 (or all identical), we treat it as "not present" and return None.
    """
    if not pdb_path.exists():
        return None

    b_factors: List[float] = []
    try:
        with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # PDB B-factor: columns 61-66 (1-indexed), 60:66 (0-indexed)
                    if len(line) >= 66:
                        bf = _safe_float(line[60:66])
                        if bf is not None:
                            b_factors.append(float(bf))
    except Exception:
        return None

    if len(b_factors) == 0:
        return None
    arr = np.array(b_factors, dtype=float)
    if not np.isfinite(arr).any():
        return None

    # If B-factors are all ~0 or constant, very likely not pLDDT.
    if float(np.nanstd(arr)) < 1e-6:
        if abs(float(np.nanmean(arr))) < 1e-6:
            return None
        # constant but non-zero: still suspicious; treat as missing
        return None

    return float(np.nanmean(arr))


def _parse_pdb_coords(pdb_path: Path, atom_selector: str = "CA") -> Tuple[np.ndarray, List[Tuple[str, int, str]]]:
    """
    Parse coordinates for selected atoms.
    Returns (coords: Nx3, keys: list of (chain, resseq, atom_name)).

    This is a lightweight parser, sufficient for RMSD alignment on peptides.
    """
    coords: List[List[float]] = []
    keys: List[Tuple[str, int, str]] = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 54:
                continue
            atom_name = line[12:16].strip()
            if atom_selector and atom_name != atom_selector:
                continue
            chain_id = line[21].strip() or "?"
            try:
                resseq = int(line[22:26])
            except Exception:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            coords.append([x, y, z])
            keys.append((chain_id, resseq, atom_name))
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=float), []
    return np.asarray(coords, dtype=float), keys


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> Optional[float]:
    """
    Compute RMSD after optimal rotation (Kabsch). P and Q: Nx3.
    """
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 2:
        return None
    P0 = P - P.mean(axis=0, keepdims=True)
    Q0 = Q - Q.mean(axis=0, keepdims=True)
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    P_rot = P0 @ R
    diff = P_rot - Q0
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return rmsd


def _compute_rmsd_to_reference(candidate_pdb: Path, reference_pdb: Path) -> Optional[float]:
    """
    Compute CA RMSD between candidate and reference, with residue-order matching (by line order of CA atoms).

    If CA atoms count mismatch, fall back to first min(Ncand, Nref).
    """
    if not candidate_pdb.exists() or not reference_pdb.exists():
        return None
    P, _ = _parse_pdb_coords(candidate_pdb, atom_selector="CA")
    Q, _ = _parse_pdb_coords(reference_pdb, atom_selector="CA")
    if P.shape[0] == 0 or Q.shape[0] == 0:
        return None
    n = min(P.shape[0], Q.shape[0])
    if n < 2:
        return None
    return _kabsch_rmsd(P[:n], Q[:n])


def _infer_target_id_from_path(p: Path) -> Optional[str]:
    """
    Try to infer target_id from a path like .../PPDbench/<target_id>/...
    """
    parts = list(p.as_posix().split("/"))
    for i in range(len(parts) - 1):
        if parts[i] == "PPDbench" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def load_structure_quality(
    method: str,
    affinity_df: pd.DataFrame,
    pdb_search_roots: List[Path],
    source_log: List[SourceLogRow],
) -> pd.DataFrame:
    """
    Build per-candidate structure_quality using:
      1) mean pLDDT from candidate PDB B-factors (if valid)
      2) fallback to -RMSD vs reference peptide (/root/autodl-tmp/PPDbench/<target>/peptide.pdb)
    """
    # Gather candidate pdb paths from affinity_df when possible
    cand_records: List[Dict[str, Any]] = []

    # Build a quick lookup from candidate_id stem to discovered pdb paths (for fuzzy matching)
    discovered_pdbs: List[Path] = []
    for r in pdb_search_roots:
        discovered_pdbs.extend(scan_pdb_files(r))
    pdb_by_stem: Dict[str, List[Path]] = {}
    for p in discovered_pdbs:
        pdb_by_stem.setdefault(_stem_id(p.name), []).append(p)

    used_plddt = 0
    used_rmsd = 0
    missing_pdb = 0

    for _, row in affinity_df.iterrows():
        target_id = str(row.get("target_id", "unknown"))
        candidate_id = str(row.get("candidate_id", ""))
        pdb_path_str = str(row.get("pdb_path", "") or "")

        cand_pdb: Optional[Path] = None
        if pdb_path_str and Path(pdb_path_str).exists():
            cand_pdb = Path(pdb_path_str)
        else:
            # fuzzy match by candidate filename (stem)
            stem = _stem_id(candidate_id)
            if stem in pdb_by_stem and len(pdb_by_stem[stem]) > 0:
                # prefer matching target_id in path if possible
                candidates = pdb_by_stem[stem]
                hit = None
                for c in candidates:
                    tid = _infer_target_id_from_path(c)
                    if tid is not None and tid == target_id:
                        hit = c
                        break
                cand_pdb = hit or candidates[0]

        plddt = None
        if cand_pdb is not None:
            plddt = extract_plddt_from_pdb(cand_pdb)

        structure_quality = None
        quality_source = ""
        reference_pdb = None

        if plddt is not None:
            structure_quality = plddt
            quality_source = "plddt_bfactor"
            used_plddt += 1
        else:
            # Fallback: -RMSD to reference peptide
            # Note: This is allowed only as fallback, and we must be explicit in logs.
            # We attempt to locate reference peptide by target id.
            # Common convention: /root/autodl-tmp/PPDbench/<target_id>/peptide.pdb
            reference_pdb = Path("/root/autodl-tmp/PPDbench") / target_id / "peptide.pdb"
            if cand_pdb is not None and reference_pdb.exists():
                rmsd = _compute_rmsd_to_reference(cand_pdb, reference_pdb)
                if rmsd is not None and math.isfinite(rmsd):
                    structure_quality = -float(rmsd)
                    quality_source = "neg_rmsd_to_reference"
                    used_rmsd += 1
                else:
                    structure_quality = None
                    quality_source = "missing"
            else:
                structure_quality = None
                quality_source = "missing"

        if cand_pdb is None:
            missing_pdb += 1

        cand_records.append(
            {
                "method": method,
                "target_id": target_id,
                "candidate_id": candidate_id,
                "structure_quality": structure_quality,
                "structure_quality_source": quality_source,
                "pdb_path": str(cand_pdb) if cand_pdb is not None else "",
                "reference_pdb": str(reference_pdb) if reference_pdb is not None else "",
            }
        )

    df = pd.DataFrame(cand_records)
    source_log.append(
        SourceLogRow(
            method,
            "structure_quality",
            ";".join(str(p) for p in pdb_search_roots),
            "ok",
            f"Candidates={len(df)}; pLDDT_used={used_plddt}; -RMSD_used={used_rmsd}; missing_pdb={missing_pdb}",
        )
    )
    return df


def search_developability_files(search_roots: List[Path]) -> List[Path]:
    """
    Search possible developability-related files under given roots.
    """
    keywords = ["solubility", "stability", "developability", "score"]
    exts = {".json", ".csv", ".tsv"}
    hits: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            name = p.name.lower()
            if any(k in name for k in keywords):
                hits.append(p)
    # prefer shorter paths (more "canonical") then stable sort
    hits = sorted(set(hits), key=lambda x: (len(str(x)), str(x)))
    return hits


def _try_load_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".json":
            obj = load_json(path)
            # allow list[dict] or dict-of-records
            if isinstance(obj, list):
                if len(obj) == 0:
                    return pd.DataFrame()
                if isinstance(obj[0], dict):
                    return pd.DataFrame(obj)
            if isinstance(obj, dict):
                # dict-of-dicts
                if len(obj) == 0:
                    return pd.DataFrame()
                if all(isinstance(v, dict) for v in obj.values()):
                    df = pd.DataFrame(list(obj.values()))
                    # keep key if helpful
                    df["_key"] = list(obj.keys())
                    return df
        elif path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            return pd.read_csv(path, sep=sep)
    except Exception:
        return None
    return None


def load_developability(
    method: str,
    candidate_df: pd.DataFrame,
    search_roots: List[Path],
    source_log: List[SourceLogRow],
) -> pd.DataFrame:
    """
    Attempt to load developability scores and map them to (target_id, candidate_id).

    Returns DataFrame columns:
      method,target_id,candidate_id,developability,developability_source
    """
    hits = search_developability_files(search_roots)
    if len(hits) == 0:
        source_log.append(
            SourceLogRow(method, "developability_search", ";".join(str(p) for p in search_roots), "missing", "No files found")
        )
        return pd.DataFrame(columns=["method", "target_id", "candidate_id", "developability", "developability_source"])

    # Candidate keys to match
    cand_keys = set((_stem_id(str(t)), _stem_id(str(c))) for t, c in zip(candidate_df["target_id"], candidate_df["candidate_id"]))

    # heuristics: common columns
    key_cols = ["target", "target_id", "protein", "receptor"]
    cand_cols = ["candidate", "candidate_id", "name", "pdb", "peptide", "sample_id", "_key"]
    score_cols = ["developability", "solubility", "stability", "score", "camsol"]

    mapped_rows: List[Dict[str, Any]] = []
    used_file = None
    used_col = None

    # Try files in order; stop once we obtain reasonable matches
    for fp in hits[:50]:
        df0 = _try_load_table(fp)
        if df0 is None or df0.empty:
            continue
        cols = {c.lower(): c for c in df0.columns}

        # find score col
        score_col = None
        for s in score_cols:
            if s in cols:
                score_col = cols[s]
                break
        if score_col is None:
            continue

        # find key columns
        target_col = None
        for t in key_cols:
            if t in cols:
                target_col = cols[t]
                break

        cand_col = None
        for c in cand_cols:
            if c in cols:
                cand_col = cols[c]
                break

        # if no explicit target/candidate columns, attempt to parse from a combined key column
        if target_col is None and cand_col is None:
            continue

        # Map rows
        for _, r in df0.iterrows():
            tval = r.get(target_col) if target_col is not None else None
            cval = r.get(cand_col) if cand_col is not None else None

            # combined key parsing
            if (tval is None or str(tval).strip() == "") and isinstance(cval, str) and "/" in cval:
                parts = cval.split("/")
                tval = parts[0]
                cval = parts[-1]

            if tval is None or cval is None:
                continue

            key = (_stem_id(str(tval)), _stem_id(str(cval)))
            if key not in cand_keys:
                continue
            sval = _safe_float(r.get(score_col))
            if sval is None:
                continue
            mapped_rows.append(
                {
                    "method": method,
                    "target_id": str(tval),
                    "candidate_id": str(cval),
                    "developability": float(sval),
                    "developability_source": f"{fp.name}:{score_col}",
                }
            )

        if len(mapped_rows) > 10:
            used_file = fp
            used_col = score_col
            break

    if len(mapped_rows) == 0:
        source_log.append(
            SourceLogRow(method, "developability_search", ";".join(str(p) for p in search_roots), "warning", "Files found but no matches to candidates")
        )
        return pd.DataFrame(columns=["method", "target_id", "candidate_id", "developability", "developability_source"])

    source_log.append(
        SourceLogRow(
            method,
            "developability",
            str(used_file) if used_file is not None else "multiple",
            "ok",
            f"Mapped {len(mapped_rows)} rows; score_col={used_col}",
        )
    )
    return pd.DataFrame(mapped_rows).drop_duplicates(subset=["method", "target_id", "candidate_id"])


def merge_candidate_objectives(
    affinity_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    develop_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge to candidate-level unified table:
      method,target_id,candidate_id,affinity_value,structure_quality,developability,pdb_path
    """
    base = affinity_df.copy()
    base["method"] = base["method"].astype(str)
    base["target_id"] = base["target_id"].astype(str)
    base["candidate_id"] = base["candidate_id"].astype(str)

    # Clean candidate_id to basename if it looks like a path
    base["candidate_id"] = base["candidate_id"].map(lambda x: Path(str(x)).name)

    st = structure_df[["method", "target_id", "candidate_id", "structure_quality", "pdb_path"]].copy()
    st["candidate_id"] = st["candidate_id"].map(lambda x: Path(str(x)).name)

    merged = pd.merge(
        base,
        st,
        on=["method", "target_id", "candidate_id"],
        how="left",
        suffixes=("", "_st"),
    )

    # prefer PDB path from structure extraction when available
    merged["pdb_path"] = merged["pdb_path_st"].where(merged["pdb_path_st"].notna() & (merged["pdb_path_st"] != ""), merged["pdb_path"])
    merged = merged.drop(columns=["pdb_path_st"], errors="ignore")

    if develop_df is not None and not develop_df.empty:
        dv = develop_df[["method", "target_id", "candidate_id", "developability"]].copy()
        dv["candidate_id"] = dv["candidate_id"].map(lambda x: Path(str(x)).name)
        merged = pd.merge(merged, dv, on=["method", "target_id", "candidate_id"], how="left")
    else:
        merged["developability"] = np.nan

    # Keep only candidates with required objectives (affinity + structure quality)
    merged["affinity_value"] = pd.to_numeric(merged["affinity_value"], errors="coerce")
    merged["structure_quality"] = pd.to_numeric(merged["structure_quality"], errors="coerce")
    merged["developability"] = pd.to_numeric(merged["developability"], errors="coerce")

    return merged[
        ["method", "target_id", "candidate_id", "affinity_value", "structure_quality", "developability", "pdb_path"]
    ].copy()


def normalize_objectives(df: pd.DataFrame, eps: float = EPS) -> Tuple[pd.DataFrame, List[str]]:
    """
    Global min-max normalization across all methods/candidates.
    Returns normalized df and list of objectives actually used.
    """
    out = df.copy()
    obj_cols = [("affinity_value", "affinity_norm"), ("structure_quality", "structure_norm"), ("developability", "developability_norm")]
    used: List[str] = []

    for raw, norm in obj_cols:
        vals = pd.to_numeric(out[raw], errors="coerce")
        finite = vals[np.isfinite(vals)]
        if finite.shape[0] == 0:
            out[norm] = np.nan
            continue
        vmin = float(finite.min())
        vmax = float(finite.max())
        if abs(vmax - vmin) < 1e-12:
            out[norm] = np.nan
            continue
        out[norm] = (vals - vmin) / (vmax - vmin + eps)
        used.append(norm)

    return out, used


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    True if point a dominates point b (all >= and at least one >).
    Assumes maximization and finite arrays.
    """
    return np.all(a >= b) and np.any(a > b)


def get_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Return non-dominated subset of points. points: NxD
    """
    if points.size == 0:
        return points
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if dominates(points[j], points[i]):
                keep[i] = False
                break
    return points[keep]


def _hv_2d(points: np.ndarray, ref: Tuple[float, float] = (0.0, 0.0)) -> float:
    """
    Hypervolume in 2D for maximization, reference at ref.
    points are assumed within [0,1] and non-dominated is recommended but not required.
    Computes area of union of rectangles [ref, p].
    """
    if points.size == 0:
        return 0.0
    pts = points.copy()
    # clamp
    pts = np.clip(pts, 0.0, 1.0)
    # keep only points above ref
    pts = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])]
    if pts.shape[0] == 0:
        return 0.0
    # sort by x descending
    pts = pts[np.argsort(-pts[:, 0])]
    hv = 0.0
    y_max = ref[1]
    for x, y in pts:
        if y <= y_max:
            continue
        hv += (x - ref[0]) * (y - y_max)
        y_max = y
    return float(hv)


def _hv_3d(points: np.ndarray, ref: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> float:
    """
    Hypervolume in 3D for maximization using a stable slicing approach.

    For small N (top-k <= 10), O(N^2 log N) is fine and robust.
    """
    if points.size == 0:
        return 0.0
    pts = np.clip(points.copy(), 0.0, 1.0)
    pts = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1]) & (pts[:, 2] > ref[2])]
    if pts.shape[0] == 0:
        return 0.0

    # Sort unique x levels descending; use incremental slabs.
    xs = sorted(set(float(x) for x in pts[:, 0]), reverse=True)
    xs.append(ref[0])
    hv = 0.0

    for i in range(len(xs) - 1):
        x_hi = xs[i]
        x_lo = xs[i + 1]
        if x_hi <= x_lo:
            continue
        slab_pts = pts[pts[:, 0] >= x_hi - 1e-12]  # include points with x >= current threshold
        if slab_pts.shape[0] == 0:
            continue
        yz = slab_pts[:, 1:3]
        yz_front = get_pareto_front(yz)
        area = _hv_2d(yz_front, ref=(ref[1], ref[2]))
        hv += (x_hi - x_lo) * area

    return float(hv)


def compute_hypervolume(points: np.ndarray, reference_point: Sequence[float]) -> float:
    """
    Compute hypervolume for maximization points in [0,1]^D.
    Preference: pymoo if available; fallback to pure python 2D/3D implementation.
    """
    if points.size == 0:
        return 0.0
    D = points.shape[1]
    ref = np.asarray(reference_point, dtype=float).reshape(-1)
    if ref.shape[0] != D:
        raise ValueError(f"Reference point dim mismatch: ref={ref.shape[0]}, points={D}")

    # Try pymoo
    try:
        from pymoo.indicators.hv import HV  # type: ignore

        # pymoo HV expects minimization by default; we can convert to minimization by negating,
        # but since our ref is 0 and points are in [0,1] for maximization, easiest is:
        # transform to minimization: f = 1 - x, ref_min = 1 - ref
        F = 1.0 - np.clip(points, 0.0, 1.0)
        ref_min = 1.0 - np.clip(ref, 0.0, 1.0)
        hv = float(HV(ref_point=ref_min)(F))
        return hv
    except Exception:
        pass

    # Fallback pure python
    if D == 2:
        front = get_pareto_front(points)
        return _hv_2d(front, ref=(float(ref[0]), float(ref[1])))
    if D == 3:
        front = get_pareto_front(points)
        return _hv_3d(front, ref=(float(ref[0]), float(ref[1]), float(ref[2])))
    raise NotImplementedError(f"Fallback HV supports only 2D/3D, got D={D}")


def compute_per_target_hv(
    norm_df: pd.DataFrame,
    topk_list: Sequence[int],
    objective_norm_cols: List[str],
    source_log: List[SourceLogRow],
) -> pd.DataFrame:
    """
    Per (method, target, topk) compute HV on top-k by affinity (raw affinity_value).
    """
    rows: List[Dict[str, Any]] = []
    ref = [0.0 for _ in objective_norm_cols]

    for method in METHOD_ORDER:
        df_m = norm_df[norm_df["method"] == method].copy()
        if df_m.empty:
            continue
        for target_id, df_t in df_m.groupby("target_id"):
            # sort by raw affinity_value descending (larger better)
            df_t = df_t.sort_values("affinity_value", ascending=False)
            for k in topk_list:
                df_k = df_t.head(int(k)).copy()
                # require at least affinity+structure_norm finite (developability optional)
                mat = df_k[objective_norm_cols].to_numpy(dtype=float)
                mask = np.all(np.isfinite(mat), axis=1)
                mat = mat[mask]
                n_used = int(mat.shape[0])
                if n_used == 0:
                    hv = np.nan
                    n_obj = len(objective_norm_cols)
                else:
                    front = get_pareto_front(mat)
                    try:
                        hv = compute_hypervolume(front, ref)
                    except Exception as e:
                        hv = np.nan
                        source_log.append(SourceLogRow(method, "hypervolume", "internal", "error", f"{target_id} top{k}: {e}"))
                    n_obj = int(mat.shape[1])
                rows.append(
                    {
                        "method": method,
                        "target_id": target_id,
                        "topk": int(k),
                        "n_candidates_used": n_used,
                        "n_objectives": n_obj,
                        "hypervolume": hv,
                    }
                )

    return pd.DataFrame(rows)


def summarize_hv(per_target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize HV across targets per (method, topk): mean/std/sem.
    """
    rows: List[Dict[str, Any]] = []
    for (method, topk), g in per_target_df.groupby(["method", "topk"]):
        vals = pd.to_numeric(g["hypervolume"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        n = int(vals.shape[0])
        if n == 0:
            mean_hv = np.nan
            std_hv = np.nan
            sem_hv = np.nan
        else:
            mean_hv = float(vals.mean())
            std_hv = float(vals.std(ddof=1)) if n >= 2 else 0.0
            sem_hv = float(std_hv / math.sqrt(n)) if n >= 2 else 0.0
        rows.append(
            {
                "method": str(method),
                "topk": int(topk),
                "n_targets": n,
                "mean_hv": mean_hv,
                "std_hv": std_hv,
                "sem_hv": sem_hv,
            }
        )
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    out = out.sort_values(["topk", "method"])
    return out


def _set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "savefig.bbox": "tight",
        }
    )


def plot_hv_by_method(
    summary_df: pd.DataFrame,
    topk: int,
    out_dir: Path,
    error_mode: str = "sem",
) -> Tuple[Path, Path]:
    """
    A. Line plot across methods for a fixed top-k.
    """
    df = summary_df[summary_df["topk"] == int(topk)].copy()
    df = df.set_index("method").reindex(METHOD_ORDER).reset_index()

    y = df["mean_hv"].to_numpy(dtype=float)
    if error_mode.lower() == "std":
        yerr = df["std_hv"].to_numpy(dtype=float)
    else:
        yerr = df["sem_hv"].to_numpy(dtype=float)

    x = np.arange(len(METHOD_ORDER))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(x, y, marker="o", linewidth=2.0, color="black")
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=0)
    ax.set_ylabel("Mean Pareto hypervolume")
    ax.set_title(f"Pareto hypervolume across methods (Top-{topk})")
    ax.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.8)
    fig.tight_layout()

    png_path = out_dir / f"pareto_hv_top{topk}_by_method.png"
    pdf_path = out_dir / f"pareto_hv_top{topk}_by_method.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)  # vector PDF
    plt.close(fig)
    return png_path, pdf_path


def plot_hv_vs_topk(summary_df: pd.DataFrame, topk_list: Sequence[int], out_dir: Path) -> Tuple[Path, Path]:
    """
    B. HV vs Top-k plot, one line per method.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.array([int(k) for k in topk_list], dtype=int)

    colors = {
        "Base": "#4C72B0",
        "Base+OT": "#55A868",
        "Base+DPO": "#C44E52",
        "Full": "#8172B2",
    }

    for method in METHOD_ORDER:
        df_m = summary_df[summary_df["method"] == method].copy()
        df_m = df_m.set_index("topk").reindex(x).reset_index()
        y = df_m["mean_hv"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, label=method, color=colors.get(method, None))

    ax.set_xticks(x)
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Mean Pareto hypervolume")
    ax.set_title("Pareto hypervolume vs Top-k")
    ax.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    png_path = out_dir / "pareto_hv_vs_topk.png"
    pdf_path = out_dir / "pareto_hv_vs_topk.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _try_install_pymoo() -> bool:
    """
    Try to install pymoo. Return True if import works after install attempt.
    """
    try:
        import pymoo  # type: ignore  # noqa: F401

        return True
    except Exception:
        pass

    print("[INFO] pymoo not found. Trying to install pymoo via pip...", flush=True)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymoo"])
    except Exception as e:
        print(f"[WARN] Failed to install pymoo: {e}", flush=True)
        return False

    try:
        import pymoo  # type: ignore  # noqa: F401

        print("[INFO] pymoo installed successfully.", flush=True)
        return True
    except Exception:
        return False


def main() -> None:
    # -----------------------------
    # Paths (centralized config)
    # -----------------------------
    out_dir = Path("/root/autodl-tmp/Peptide_3D/results/4_ablation/plot")
    ensure_dir(out_dir)

    affinity_paths = {
        "Base": Path("/root/autodl-tmp/Peptide_3D/results/4_ablation/ppdbench_hdock_ablation_base.json"),
        "Base+OT": Path("/root/autodl-tmp/Peptide_3D/results/4_ablation/ppdbench_hdock_ablation_base_ot.json"),
        "Base+DPO": Path("/root/autodl-tmp/Peptide_3D/results/4_ablation/ppdbench_hdock_ablation_base_dpo.json"),
        "Full": Path("/root/autodl-tmp/PPDbench/1cjr/multi_cands/cands_hdock_scores.json"),
    }

    pdb_roots = {
        "Base": [Path("/root/autodl-tmp/PPDbench")],
        "Base+OT": [Path("/root/autodl-tmp/PPDbench")],
        "Base+DPO": [Path("/root/autodl-tmp/PPDbench")],
        "Full": [Path("/root/autodl-tmp/PPDbench/1cjr/multi_cands")],
    }

    develop_search_roots = [
        Path("/root/autodl-tmp/Peptide_3D/results/4_ablation"),
        Path("/root/autodl-tmp/PPDbench"),
    ]

    # -----------------------------
    # Prepare logs + style
    # -----------------------------
    source_log: List[SourceLogRow] = []
    _set_paper_style()

    # -----------------------------
    # Optional: use pymoo indicator if possible
    # -----------------------------
    pymoo_ok = _try_install_pymoo()
    if pymoo_ok:
        print("[INFO] Hypervolume backend: pymoo (preferred).", flush=True)
    else:
        print("[INFO] Hypervolume backend: pure-python fallback (2D/3D).", flush=True)

    # -----------------------------
    # Load affinity per method
    # -----------------------------
    affinity_dfs: Dict[str, pd.DataFrame] = {}
    for method in METHOD_ORDER:
        df = parse_affinity_scores(method, affinity_paths[method], source_log)
        affinity_dfs[method] = df
        print(f"[INFO] {method}: parsed affinity candidates = {len(df)}", flush=True)

    # -----------------------------
    # Load structure quality per method
    # -----------------------------
    struct_dfs: Dict[str, pd.DataFrame] = {}
    for method in METHOD_ORDER:
        df_aff = affinity_dfs[method]
        df_st = load_structure_quality(method, df_aff, pdb_roots[method], source_log)
        struct_dfs[method] = df_st
        n_st = int(pd.to_numeric(df_st["structure_quality"], errors="coerce").notna().sum()) if not df_st.empty else 0
        print(f"[INFO] {method}: parsed structure-quality candidates = {n_st} / {len(df_st)}", flush=True)

    # -----------------------------
    # Merge candidate-level objectives (affinity + structure + developability)
    # -----------------------------
    merged_all: List[pd.DataFrame] = []
    dev_any = False

    # first, create a per-method merged (affinity + structure)
    premerge_dfs: Dict[str, pd.DataFrame] = {}
    for method in METHOD_ORDER:
        pre = merge_candidate_objectives(affinity_dfs[method], struct_dfs[method], develop_df=None)
        premerge_dfs[method] = pre

    # attempt to load developability using all candidates across methods (better matching)
    concat_pre = pd.concat(list(premerge_dfs.values()), ignore_index=True) if len(premerge_dfs) else pd.DataFrame()
    dev_df = load_developability("ALL_METHODS", concat_pre, develop_search_roots, source_log)
    if not dev_df.empty:
        dev_any = True

    for method in METHOD_ORDER:
        pre = premerge_dfs[method]
        if dev_any:
            dv_m = dev_df.copy()
            dv_m["method"] = method  # apply method-wise merge by (target,candidate) only
            merged = merge_candidate_objectives(affinity_dfs[method], struct_dfs[method], develop_df=dv_m)
        else:
            merged = pre
        merged_all.append(merged)
        print(f"[INFO] {method}: merged candidates = {len(merged)}", flush=True)

    merged_df = pd.concat(merged_all, ignore_index=True)

    # enforce requirement: candidate must have affinity + structure_quality
    before = len(merged_df)
    merged_df = merged_df[np.isfinite(merged_df["affinity_value"].to_numpy(dtype=float)) & np.isfinite(merged_df["structure_quality"].to_numpy(dtype=float))]
    after = len(merged_df)
    if after < before:
        print(f"[INFO] Filtered candidates missing affinity/structure: {before} -> {after}", flush=True)

    # developability availability
    dev_present = np.isfinite(pd.to_numeric(merged_df["developability"], errors="coerce")).any()
    if not dev_present:
        print("Developability data not found, fallback to 2-objective Pareto hypervolume", flush=True)

    # Save merged candidates CSV
    merged_csv = out_dir / "candidate_objectives_merged.csv"
    save_csv(merged_df, merged_csv)
    print(f"[INFO] Saved merged candidate table: {merged_csv}", flush=True)

    # -----------------------------
    # Normalize objectives globally
    # -----------------------------
    norm_df, used_norm_cols = normalize_objectives(merged_df, eps=EPS)

    # Determine objectives to use for HV:
    # Always require affinity_norm + structure_norm; developability_norm included only if available (non-all-NaN).
    required = ["affinity_norm", "structure_norm"]
    for r in required:
        if r not in norm_df.columns or not np.isfinite(pd.to_numeric(norm_df[r], errors="coerce")).any():
            raise RuntimeError(f"Required objective '{r}' has no valid values. Cannot compute HV.")
    objective_cols = required.copy()
    if dev_present and "developability_norm" in norm_df.columns and np.isfinite(pd.to_numeric(norm_df["developability_norm"], errors="coerce")).any():
        objective_cols.append("developability_norm")
    print(f"[INFO] Using objectives dimension = {len(objective_cols)} ({'3D' if len(objective_cols)==3 else '2D'})", flush=True)

    norm_csv = out_dir / "candidate_objectives_normalized.csv"
    save_csv(
        norm_df[["method", "target_id", "candidate_id"] + objective_cols].rename(columns={"affinity_norm": "affinity_norm", "structure_norm": "structure_norm"}),
        norm_csv,
    )
    print(f"[INFO] Saved normalized objectives: {norm_csv}", flush=True)

    # -----------------------------
    # Compute per-target HV
    # -----------------------------
    per_target_df = compute_per_target_hv(norm_df, TOPK_LIST, objective_cols, source_log)
    per_target_csv = out_dir / "per_target_hypervolume.csv"
    save_csv(per_target_df, per_target_csv)
    print(f"[INFO] Saved per-target HV: {per_target_csv}", flush=True)

    # Print: per top-k per method effective targets
    for k in TOPK_LIST:
        for method in METHOD_ORDER:
            g = per_target_df[(per_target_df["topk"] == int(k)) & (per_target_df["method"] == method)]
            n_eff = int(pd.to_numeric(g["hypervolume"], errors="coerce").notna().sum())
            print(f"[INFO] top{k} {method}: effective targets = {n_eff}", flush=True)

    # -----------------------------
    # Summarize HV across targets
    # -----------------------------
    summary_df = summarize_hv(per_target_df)
    summary_csv = out_dir / "hypervolume_summary.csv"
    save_csv(summary_df, summary_csv)
    print(f"[INFO] Saved HV summary: {summary_csv}", flush=True)

    # Print: per method mean HV at each top-k
    for method in METHOD_ORDER:
        df_m = summary_df[summary_df["method"] == method].sort_values("topk")
        msg = ", ".join([f"top{k}: {v:.4f}" if np.isfinite(v) else f"top{k}: nan" for k, v in zip(df_m["topk"], df_m["mean_hv"])])
        print(f"[INFO] {method}: mean hypervolume = {msg}", flush=True)

    # -----------------------------
    # Plot figures
    # -----------------------------
    # A. across methods for top3/top5/top10
    for k in [3, 5, 10]:
        if k not in TOPK_LIST:
            continue
        png, pdf = plot_hv_by_method(summary_df, topk=k, out_dir=out_dir, error_mode="sem")
        print(f"[INFO] Saved figure: {png}", flush=True)
        print(f"[INFO] Saved figure: {pdf}", flush=True)

    # B. HV vs top-k
    png, pdf = plot_hv_vs_topk(summary_df, TOPK_LIST, out_dir=out_dir)
    print(f"[INFO] Saved figure: {png}", flush=True)
    print(f"[INFO] Saved figure: {pdf}", flush=True)

    # -----------------------------
    # Save data_source_log.csv
    # -----------------------------
    log_csv = out_dir / "data_source_log.csv"
    with log_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "data_type", "source_path", "status", "notes"])
        for r in source_log:
            w.writerow([r.method, r.data_type, r.source_path, r.status, r.notes])
    print(f"[INFO] Saved data source log: {log_csv}", flush=True)


if __name__ == "__main__":
    main()


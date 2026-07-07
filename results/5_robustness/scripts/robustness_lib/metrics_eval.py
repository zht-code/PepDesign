"""Affinity (HDOCK), stability (FoldX), solubility (Protein-Sol) — reuse 3_Pareto_improved scripts."""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from typing import Any, Optional

from .paths import PROJECT_ROOT

_PARETO = PROJECT_ROOT / "results" / "3_Pareto_improved"
_SOL_LOCK = threading.Lock()


def _load(name: str, rel: str):
    path = _PARETO / rel
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_aff_mod = None
_stab_mod = None
_solu_mod = None


def _aff():
    global _aff_mod
    if _aff_mod is None:
        _aff_mod = _load("ppb_aff", "compute_ppdbench_generated_affinity.py")
    return _aff_mod


def _stab():
    global _stab_mod
    if _stab_mod is None:
        _stab_mod = _load("ppb_stab", "compute_ppdbench_generated_stability.py")
    return _stab_mod


def _solu():
    global _solu_mod
    if _solu_mod is None:
        _solu_mod = _load("ppb_solu", "compute_ppdbench_generated_solubility.py")
    return _solu_mod


def load_thresholds(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_peptide(
    *,
    receptor_pdb: Path,
    peptide_pdb: Path,
    hdock_bin: str,
    createpl_bin: str,
    foldx_bin: str,
    proteinsol_wrapper: str,
    hdock_work_root: Path,
    foldx_work_root: Path,
    hdock_timeout: int,
) -> dict[str, Any]:
    """Run external tools on one peptide; missing binaries yield None scores."""
    out: dict[str, Any] = {
        "affinity_hdock": None,
        "stability": None,
        "solubility": None,
        "logs": [],
    }
    work_hdock = hdock_work_root / peptide_pdb.stem
    work_hdock.mkdir(parents=True, exist_ok=True)

    if Path(hdock_bin).is_file():
        try:
            score, log = _aff().run_hdock_pair(
                str(work_hdock),
                str(receptor_pdb),
                str(peptide_pdb),
                hdock_bin,
                createpl_bin,
                timeout_s=hdock_timeout,
            )
            out["affinity_hdock"] = score
            out["logs"].append(log[:2000])
        except Exception as e:
            out["logs"].append(f"[hdock error] {e}")
    else:
        out["logs"].append(f"[skip] hdock bin missing: {hdock_bin}")

    if Path(foldx_bin).is_file():
        try:
            out["stability"] = _stab().foldx_stability_score_single(
                peptide_pdb,
                foldx_bin=foldx_bin,
                workdir_root=str(foldx_work_root),
                timeout_s=600,
            )
        except Exception as e:
            out["logs"].append(f"[foldx error] {e}")
    else:
        out["logs"].append(f"[skip] foldx missing: {foldx_bin}")

    if Path(proteinsol_wrapper).is_file():
        try:
            seq = _solu().extract_peptide_seq(peptide_pdb)
            with _SOL_LOCK:
                out["solubility"] = _solu().solubility_score_from_seq_single(
                    seq, proteinsol_wrapper=proteinsol_wrapper
                )
        except Exception as e:
            out["logs"].append(f"[solubility error] {e}")
    else:
        out["logs"].append(f"[skip] proteinsol missing: {proteinsol_wrapper}")

    return out


def success_triple(
    aff: Optional[float],
    stab: Optional[float],
    sol: Optional[float],
    th: dict[str, float],
) -> Optional[bool]:
    """True if all three pass; None if any score missing."""
    if aff is None or stab is None or sol is None:
        return None
    return bool(aff < th["hdock_max"] and stab > th["stability_min"] and sol > th["solubility_min"])


def to_higher_better(name: str, val: Optional[float]) -> Optional[float]:
    """Unified direction: higher is better for curve/drop (affinity uses -HDOCK)."""
    if val is None:
        return None
    if name == "affinity_hdock":
        return -float(val)
    return float(val)

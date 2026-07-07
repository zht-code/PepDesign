from __future__ import annotations

import hashlib
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import Chain, Model, PDBIO, Structure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_2_SOTA_DIR = PROJECT_ROOT / "results" / "2_SOTA"
if str(RESULTS_2_SOTA_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_2_SOTA_DIR))

from metrics_affinity import contact_map_consistency  # noqa: E402
from metrics_structure import clash_score_structure, load_structure, ramachandran_compliance_structure  # noqa: E402
from utils_native_complex import merge_receptor_and_peptide_to_pdb  # noqa: E402


LOGGER = logging.getLogger(__name__)
ESMFOLD_BATCH_SCRIPT = PROJECT_ROOT / "scripts" / "run_esmfold_plddt_batch.py"
MOLPROBITY_BATCH_SCRIPT = PROJECT_ROOT / "scripts" / "run_molprobity_metrics_batch.py"


def _polymer_residues(chain) -> List:
    return [res for res in chain if res.id[0] == " "]


def _chain_lengths(structure) -> List[Tuple[object, int]]:
    out: List[Tuple[object, int]] = []
    model = next(structure.get_models())
    for chain in model:
        residues = _polymer_residues(chain)
        if residues:
            out.append((chain, len(residues)))
    return out


def _select_peptide_chain(structure, reference_peptide_path: Optional[str] = None):
    chain_infos = _chain_lengths(structure)
    if not chain_infos:
        raise ValueError("No polymer chains found in structure")
    if len(chain_infos) == 1:
        return chain_infos[0][0]

    ref_len = None
    if reference_peptide_path:
        try:
            ref_structure = load_structure(reference_peptide_path)
            ref_infos = _chain_lengths(ref_structure)
            if ref_infos:
                ref_len = min(length for _, length in ref_infos)
        except Exception:
            ref_len = None

    if ref_len is not None:
        chain_infos.sort(key=lambda item: (abs(item[1] - ref_len), item[1]))
    else:
        chain_infos.sort(key=lambda item: item[1])
    return chain_infos[0][0]


def _single_chain_structure(chain) -> Structure.Structure:
    structure = Structure.Structure("peptide")
    model = Model.Model(0)
    structure.add(model)
    new_chain = Chain.Chain(chain.id if isinstance(chain.id, str) else "A")
    for residue in chain:
        new_chain.add(residue.copy())
    model.add(new_chain)
    return structure


def peptide_structure_from_file(pdb_path: str, reference_peptide_path: Optional[str] = None):
    structure = load_structure(pdb_path)
    peptide_chain = _select_peptide_chain(structure, reference_peptide_path=reference_peptide_path)
    return _single_chain_structure(peptide_chain)


def save_peptide_chain_pdb(
    pdb_path: str,
    output_path: Path,
    reference_peptide_path: Optional[str] = None,
) -> str:
    peptide_structure = peptide_structure_from_file(pdb_path, reference_peptide_path=reference_peptide_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(peptide_structure)
    io.save(str(output_path))
    return str(output_path.resolve())


def extract_peptide_sequence(pdb_path: str, reference_peptide_path: Optional[str] = None) -> str:
    structure = load_structure(pdb_path)
    peptide_chain = _select_peptide_chain(structure, reference_peptide_path=reference_peptide_path)
    three_to_one = {
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
    }
    seq_chars: List[str] = []
    for residue in peptide_chain:
        if residue.id[0] != " ":
            continue
        aa = three_to_one.get(residue.resname.upper())
        if aa:
            seq_chars.append(aa)
    return "".join(seq_chars)


def mean_plddt_for_peptide(pdb_path: str, reference_peptide_path: Optional[str] = None) -> float:
    peptide_structure = peptide_structure_from_file(pdb_path, reference_peptide_path=reference_peptide_path)
    values = [atom.bfactor for atom in peptide_structure.get_atoms() if atom.element != "H"]
    return float(np.mean(values)) if values else float("nan")


def ramachandran_compliance_for_peptide(pdb_path: str, reference_peptide_path: Optional[str] = None) -> float:
    peptide_structure = peptide_structure_from_file(pdb_path, reference_peptide_path=reference_peptide_path)
    return float(ramachandran_compliance_structure(peptide_structure))


def clash_score_for_peptide(pdb_path: str, reference_peptide_path: Optional[str] = None) -> float:
    peptide_structure = peptide_structure_from_file(pdb_path, reference_peptide_path=reference_peptide_path)
    return float(clash_score_structure(peptide_structure))


def cached_native_complex_path(
    dataset: str,
    target_id: str,
    receptor_pdb: Optional[str],
    reference_peptide_pdb: Optional[str],
    cache_dir: Path,
) -> Optional[str]:
    if not receptor_pdb or not reference_peptide_pdb:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"native_{dataset}_{target_id}.pdb"
    if out_path.exists():
        return str(out_path.resolve())

    try:
        merge_receptor_and_peptide_to_pdb(receptor_pdb, reference_peptide_pdb, str(out_path))
        return str(out_path.resolve())
    except Exception as exc:
        LOGGER.warning("Failed to build native complex for %s/%s: %s", dataset, target_id, exc)
        return None


def compute_structure_metrics_for_row(row: dict, native_complex_cache_dir: Path) -> Dict[str, float]:
    metrics = {
        "plddt": float("nan"),
        "ramachandran_compliance": float("nan"),
        "clash_score": float("nan"),
        "contact_consistency": float("nan"),
    }

    pdb_path = row.get("pdb_path")
    if pdb_path:
        try:
            metrics["plddt"] = mean_plddt_for_peptide(pdb_path, reference_peptide_path=row.get("reference_peptide_pdb"))
        except Exception as exc:
            LOGGER.warning("Failed pLDDT for %s: %s", pdb_path, exc)
        try:
            metrics["ramachandran_compliance"] = ramachandran_compliance_for_peptide(
                pdb_path, reference_peptide_path=row.get("reference_peptide_pdb")
            )
        except Exception as exc:
            LOGGER.warning("Failed Ramachandran metric for %s: %s", pdb_path, exc)
        try:
            metrics["clash_score"] = clash_score_for_peptide(pdb_path, reference_peptide_path=row.get("reference_peptide_pdb"))
        except Exception as exc:
            LOGGER.warning("Failed clash score for %s: %s", pdb_path, exc)

    native_complex = cached_native_complex_path(
        dataset=str(row.get("dataset")),
        target_id=str(row.get("target_id")),
        receptor_pdb=row.get("receptor_pdb"),
        reference_peptide_pdb=row.get("reference_peptide_pdb"),
        cache_dir=native_complex_cache_dir,
    )
    pred_complex = row.get("pred_complex_pdb")
    if native_complex and pred_complex and Path(pred_complex).exists():
        try:
            metrics["contact_consistency"] = float(safe_contact_consistency(native_complex, pred_complex))
        except Exception as exc:
            LOGGER.warning("Failed contact consistency for %s vs %s: %s", native_complex, pred_complex, exc)

    return metrics


def compute_contact_consistency_for_row(row: dict, native_complex_cache_dir: Path) -> float:
    native_complex = cached_native_complex_path(
        dataset=str(row.get("dataset")),
        target_id=str(row.get("target_id")),
        receptor_pdb=row.get("receptor_pdb"),
        reference_peptide_pdb=row.get("reference_peptide_pdb"),
        cache_dir=native_complex_cache_dir,
    )
    pred_complex = row.get("pred_complex_pdb")
    if native_complex and pred_complex and Path(pred_complex).exists():
        return float(safe_contact_consistency(native_complex, pred_complex))
    return float("nan")


def _query_stem(query_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in query_id)


def _sequence_key(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]


def _run_external_python(
    python_executable: str,
    script_path: Path,
    args: List[str],
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    env = os.environ.copy()
    exe_dir = str(Path(python_executable).resolve().parent)
    env["PATH"] = f"{exe_dir}:{env.get('PATH', '')}"
    if extra_env:
        env.update({key: value for key, value in extra_env.items() if value})
    command = [python_executable, str(script_path), *args]
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(command)}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )


def batch_compute_structure_metrics(
    per_candidate_df: pd.DataFrame,
    output_dir: Path,
    esmfold_python: str,
    molprobity_python: str,
    esmfold_torch_home: Optional[str] = None,
    esmfold_chunk_size: int = 128,
) -> pd.DataFrame:
    backend_dir = output_dir / "_structure_backend"
    peptide_cache_dir = backend_dir / "peptide_only_pdbs"
    peptide_cache_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame({"query_id": per_candidate_df["query_id"].astype(str)})
    results["plddt"] = float("nan")
    results["ramachandran_compliance"] = float("nan")
    results["clash_score"] = float("nan")
    results["peptide_only_pdb"] = None

    valid_sequences = (
        per_candidate_df.loc[per_candidate_df["sequence"].fillna("").astype(str).str.len() > 0, ["sequence"]]
        .drop_duplicates()
        .copy()
    )
    if not valid_sequences.empty:
        valid_sequences["sequence"] = valid_sequences["sequence"].astype(str)
        valid_sequences["sequence_id"] = valid_sequences["sequence"].apply(_sequence_key)
        esmfold_input = backend_dir / "esmfold_input.csv"
        esmfold_output = backend_dir / "esmfold_output.csv"
        valid_sequences[["sequence_id", "sequence"]].to_csv(esmfold_input, index=False)
        _run_external_python(
            python_executable=esmfold_python,
            script_path=ESMFOLD_BATCH_SCRIPT,
            args=[
                "--input-csv",
                str(esmfold_input),
                "--output-csv",
                str(esmfold_output),
                "--chunk-size",
                str(esmfold_chunk_size),
            ],
            extra_env={"TORCH_HOME": esmfold_torch_home or ""},
        )
        esmfold_df = pd.read_csv(esmfold_output)
        seq_to_plddt = dict(zip(esmfold_df["sequence_id"].astype(str), pd.to_numeric(esmfold_df["plddt"], errors="coerce")))
        results["plddt"] = per_candidate_df["sequence"].astype(str).map(lambda seq: seq_to_plddt.get(_sequence_key(seq), float("nan")))

    peptide_rows: List[dict] = []
    for row in per_candidate_df.to_dict(orient="records"):
        query_id = str(row["query_id"])
        pdb_path = row.get("pdb_path")
        if not pdb_path or not Path(str(pdb_path)).exists():
            continue
        peptide_path = peptide_cache_dir / f"{_query_stem(query_id)}.pdb"
        try:
            save_peptide_chain_pdb(
                pdb_path=str(pdb_path),
                output_path=peptide_path,
                reference_peptide_path=row.get("reference_peptide_pdb"),
            )
            peptide_rows.append({"query_id": query_id, "pdb_path": str(peptide_path.resolve())})
        except Exception as exc:
            LOGGER.warning("Failed to prepare peptide-only PDB for %s: %s", query_id, exc)

    if peptide_rows:
        peptide_df = pd.DataFrame(peptide_rows).drop_duplicates(subset=["query_id"])
        molprobity_input = backend_dir / "molprobity_input.csv"
        molprobity_output = backend_dir / "molprobity_output.csv"
        peptide_df.to_csv(molprobity_input, index=False)
        _run_external_python(
            python_executable=molprobity_python,
            script_path=MOLPROBITY_BATCH_SCRIPT,
            args=["--input-csv", str(molprobity_input), "--output-csv", str(molprobity_output)],
        )
        molprobity_df = pd.read_csv(molprobity_output)
        results = results.merge(
            molprobity_df[["query_id", "ramachandran_compliance", "clash_score", "peptide_only_pdb"]],
            on="query_id",
            how="left",
            suffixes=("", "_new"),
        )
        for column in ("ramachandran_compliance", "clash_score", "peptide_only_pdb"):
            new_column = f"{column}_new"
            if new_column in results.columns:
                results[column] = results[new_column].where(results[new_column].notna(), results[column])
                results = results.drop(columns=[new_column])

    return results


def _safe_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _parse_atoms_manually(pdb_path: str) -> Dict[str, List[Tuple[Tuple[str, int], np.ndarray]]]:
    chains: Dict[str, List[Tuple[Tuple[str, int], np.ndarray]]] = {}
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name.startswith("H"):
                continue
            chain_id = line[21].strip() or "_"
            resseq = int(line[22:26].strip())
            insertion = line[26].strip()
            x = _safe_float(line[30:38].strip())
            y = _safe_float(line[38:46].strip())
            z = _safe_float(line[46:54].strip())
            if any(math.isnan(v) for v in (x, y, z)):
                continue
            chains.setdefault(chain_id, []).append(((f"{chain_id}{insertion}", resseq), np.array([x, y, z], dtype=float)))
    return chains


def _manual_interface_contacts(complex_pdb: str, threshold: float = 5.0) -> set:
    chains = _parse_atoms_manually(complex_pdb)
    chain_items = []
    for chain_id, atoms in chains.items():
        residues = {(res_token, resseq) for (res_token, resseq), _ in atoms}
        if residues:
            chain_items.append((chain_id, len(residues), atoms))
    if len(chain_items) < 2:
        return set()

    chain_items.sort(key=lambda item: item[1], reverse=True)
    receptor_atoms = chain_items[0][2]
    peptide_atoms = chain_items[-1][2]
    if not receptor_atoms or not peptide_atoms:
        return set()

    receptor_coords = np.stack([coord for _, coord in receptor_atoms], axis=0)
    receptor_res = [residue for residue, _ in receptor_atoms]
    contacts = set()
    threshold_sq = threshold * threshold
    for peptide_residue, peptide_coord in peptide_atoms:
        distances_sq = np.sum((receptor_coords - peptide_coord) ** 2, axis=1)
        for idx in np.where(distances_sq <= threshold_sq)[0]:
            contacts.add((receptor_res[idx][1], peptide_residue[1]))
    return contacts


def safe_contact_consistency(native_complex_pdb: str, pred_complex_pdb: str, threshold: float = 5.0) -> float:
    try:
        return float(contact_map_consistency(native_complex_pdb, pred_complex_pdb, threshold=threshold))
    except Exception:
        native_contacts = _manual_interface_contacts(native_complex_pdb, threshold=threshold)
        pred_contacts = _manual_interface_contacts(pred_complex_pdb, threshold=threshold)
        if not native_contacts and not pred_contacts:
            return 1.0
        union = native_contacts | pred_contacts
        inter = native_contacts & pred_contacts
        return len(inter) / len(union) if union else float("nan")

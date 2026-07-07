from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from Bio.PDB import NeighborSearch, PDBParser


def parse_hdock_score(hdock_result: Optional[str]) -> float:
    if hdock_result is None:
        return float("nan")
    p = Path(hdock_result)
    if not p.exists():
        return float("nan")
    text = p.read_text(encoding="utf-8", errors="ignore")

    patterns = [
        r"HDOCK score\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"Score\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"Docking Score\s*[:=]\s*(-?\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return float("nan")


def run_hdock(hdock_bin: str, receptor_pdb: str, peptide_pdb: str, out_txt: str) -> None:
    cmd = [hdock_bin, receptor_pdb, peptide_pdb]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    Path(out_txt).write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")


def get_receptor_peptide_atoms(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)

    model = next(structure.get_models())
    chains = list(model.get_chains())
    if len(chains) < 2:
        raise ValueError(f"Expected at least 2 chains in complex PDB: {pdb_path}")

    chain_sizes = []
    for c in chains:
        residues = [r for r in c if r.id[0] == " "]
        chain_sizes.append((c, len(residues)))
    chain_sizes.sort(key=lambda x: x[1], reverse=True)

    receptor_chain = chain_sizes[0][0]
    peptide_chain = chain_sizes[-1][0]

    receptor_atoms = [a for a in receptor_chain.get_atoms() if a.element != "H"]
    peptide_atoms = [a for a in peptide_chain.get_atoms() if a.element != "H"]
    return receptor_atoms, peptide_atoms


def interface_contacts_from_complex(complex_pdb: str, threshold: float = 5.0) -> set:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", complex_pdb)
    model = next(structure.get_models())
    chains = list(model.get_chains())
    if len(chains) < 2:
        return set()

    chain_sizes = []
    for c in chains:
        residues = [r for r in c if r.id[0] == " "]
        chain_sizes.append((c, len(residues)))
    chain_sizes.sort(key=lambda x: x[1], reverse=True)

    receptor = chain_sizes[0][0]
    peptide = chain_sizes[-1][0]

    rec_atoms = [a for a in receptor.get_atoms() if a.element != "H"]
    pep_atoms = [a for a in peptide.get_atoms() if a.element != "H"]

    ns = NeighborSearch(rec_atoms + pep_atoms)
    contacts = set()
    for atom in pep_atoms:
        neighbors = ns.search(atom.coord, threshold)
        for nb in neighbors:
            if nb.get_parent().get_parent().id == receptor.id:
                rec_res = nb.get_parent().id[1]
                pep_res = atom.get_parent().id[1]
                contacts.add((rec_res, pep_res))
    return contacts


def contact_map_consistency(native_complex_pdb: str, pred_complex_pdb: str, threshold: float = 5.0) -> float:
    native = interface_contacts_from_complex(native_complex_pdb, threshold=threshold)
    pred = interface_contacts_from_complex(pred_complex_pdb, threshold=threshold)

    if not native and not pred:
        return 1.0
    union = native | pred
    inter = native & pred
    return len(inter) / len(union) if union else float("nan")

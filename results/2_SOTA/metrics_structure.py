from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser, PPBuilder


RAMA_FAVORED = {
    "general": [(-180, -100, -180, 180), (-100, -30, -80, 60), (-180, -100, 90, 180)],
    "gly": [(-180, 180, -180, 180)],
    "pro": [(-100, -35, -80, 160)],
}

VDW_RADII = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80
}


def load_structure(path: Union[str, Path]):
    """Load PDB or mmCIF; same Structure object type for downstream metrics."""
    p = Path(path).expanduser().resolve()
    suf = p.suffix.lower()
    if suf in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
        return parser.get_structure("x", str(p))
    parser = PDBParser(QUIET=True)
    return parser.get_structure("x", str(p))


def mean_plddt_from_structure_file(path: Union[str, Path]) -> float:
    structure = load_structure(path)
    vals = [a.bfactor for a in structure.get_atoms() if a.element != "H"]
    return float(np.mean(vals)) if vals else float("nan")


def mean_plddt_from_pdb(pdb_path: str) -> float:
    return mean_plddt_from_structure_file(pdb_path)


def _angle_in_boxes(phi_deg: float, psi_deg: float, boxes) -> bool:
    for phi_lo, phi_hi, psi_lo, psi_hi in boxes:
        if phi_lo <= phi_deg <= phi_hi and psi_lo <= psi_deg <= psi_hi:
            return True
    return False


def ramachandran_compliance_structure(structure) -> float:
    ppb = PPBuilder()

    total = 0
    favored = 0
    for pp in ppb.build_peptides(structure):
        residues = pp
        angles = pp.get_phi_psi_list()
        for residue, (phi, psi) in zip(residues, angles):
            if phi is None or psi is None:
                continue
            total += 1
            phi_deg = math.degrees(phi)
            psi_deg = math.degrees(psi)
            resname = residue.get_resname().upper()
            if resname == "GLY":
                key = "gly"
            elif resname == "PRO":
                key = "pro"
            else:
                key = "general"
            if _angle_in_boxes(phi_deg, psi_deg, RAMA_FAVORED[key]):
                favored += 1
    return favored / total if total else float("nan")


def ramachandran_compliance(pdb_path: str) -> float:
    return ramachandran_compliance_structure(load_structure(pdb_path))


def clash_score_structure(structure, overlap_tolerance: float = 0.4) -> float:
    atoms = [a for a in structure.get_atoms() if a.element != "H"]
    if not atoms:
        return float("nan")

    ns = NeighborSearch(atoms)
    clashes = 0
    checked = set()
    for atom in atoms:
        center = atom.coord
        near = ns.search(center, 4.0)
        for nb in near:
            if atom is nb:
                continue
            a1 = atom.serial_number
            a2 = nb.serial_number
            key = tuple(sorted((a1, a2)))
            if key in checked:
                continue
            checked.add(key)

            # Skip same residue
            if atom.get_parent() == nb.get_parent():
                continue

            e1 = atom.element.strip().upper()
            e2 = nb.element.strip().upper()
            r1 = VDW_RADII.get(e1, 1.7)
            r2 = VDW_RADII.get(e2, 1.7)
            dist = np.linalg.norm(atom.coord - nb.coord)
            if dist < (r1 + r2 - overlap_tolerance):
                clashes += 1

    # Common convention: clashes per 1000 atoms
    return clashes * 1000.0 / len(atoms)


def clash_score(pdb_path: str, overlap_tolerance: float = 0.4) -> float:
    return clash_score_structure(load_structure(pdb_path), overlap_tolerance=overlap_tolerance)

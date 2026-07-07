from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Atom:
    name: str
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float
    altloc: str
    line_index: int


def _parse_atom_line(line: str, idx: int) -> Atom | None:
    if len(line) < 54:
        return None
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None
    atom_name = line[12:16].strip()
    altloc = line[16] if len(line) > 16 else " "
    resname = line[17:20].strip()
    chain = line[21].strip() or " "
    try:
        resseq = int(line[22:26])
    except ValueError:
        resseq = 0
    return Atom(atom_name, resname, chain, resseq, x, y, z, altloc, idx)


def read_atoms(path: Path) -> list[Atom]:
    atoms: list[Atom] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            rec = line[:6]
            if rec in ("ATOM  ", "HETATM"):
                if line[16] not in (" ", "A", "1", ""):
                    continue
                a = _parse_atom_line(line, i)
                if a:
                    atoms.append(a)
    return atoms


def split_ter_segments(atoms: list[Atom]) -> list[list[Atom]]:
    """Split on TER records implied by file order: caller passes atoms between TER as separate reads."""
    # We read sequentially; PDB TER is separate line — approximate by detecting HEADER lig after big jump
    return [atoms]


def read_atom_segments(path: Path) -> list[list[Atom]]:
    """Split PDB into segments separated by TER lines (Hdock: receptor then ligand)."""
    segments: list[list[Atom]] = []
    current: list[Atom] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("TER"):
                if current:
                    segments.append(current)
                    current = []
                continue
            rec = line[:6]
            if rec in ("ATOM  ", "HETATM"):
                if len(line) > 16 and line[16] not in (" ", "A", "1"):
                    continue
                a = _parse_atom_line(line, i)
                if a:
                    current.append(a)
        if current:
            segments.append(current)
    if not segments:
        return []
    return segments


def ca_atoms(segment: list[Atom]) -> list[Atom]:
    return [a for a in segment if a.name.strip() == "CA"]


def guess_receptor_peptide_segments(segments: list[list[Atom]]) -> tuple[list[Atom], list[Atom] | None]:
    """Heuristic: shortest CA count segment = peptide; if single segment, peptide only."""
    if not segments:
        return [], None
    if len(segments) == 1:
        return segments[0], None
    ca_counts = [(len(ca_atoms(s)), i) for i, s in enumerate(segments)]
    ca_counts.sort()
    pep_idx = ca_counts[0][1]
    rec_idx = ca_counts[-1][1]
    if pep_idx == rec_idx:
        pep_idx = len(segments) - 1
        rec_idx = 0
    peptide = segments[pep_idx]
    receptor = segments[rec_idx]
    return receptor, peptide


def vec_sub(a: Atom, b: Atom) -> tuple[float, float, float]:
    return (a.x - b.x, a.y - b.y, a.z - b.z)


def vec_len(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def centroid(atoms: Iterable[Atom]) -> tuple[float, float, float]:
    xs = [a.x for a in atoms]
    ys = [a.y for a in atoms]
    zs = [a.z for a in atoms]
    n = max(len(xs), 1)
    return (sum(xs) / n, sum(ys) / n, sum(zs) / n)


def radius_of_gyration(ca: list[Atom]) -> float | None:
    if len(ca) < 2:
        return None
    cx, cy, cz = centroid(ca)
    acc = 0.0
    for a in ca:
        dx, dy, dz = a.x - cx, a.y - cy, a.z - cz
        acc += dx * dx + dy * dy + dz * dz
    return math.sqrt(acc / len(ca))


def end_to_end_distance(ca: list[Atom]) -> float | None:
    if len(ca) < 2:
        return None
    return vec_len(vec_sub(ca[-1], ca[0]))


def is_docked_complex_path(p: Path) -> bool:
    parts = {x.lower() for x in p.parts}
    return "hdock_work" in parts and p.name.lower().startswith("model_")


def is_likely_peptide_only_path(p: Path) -> bool:
    s = str(p).lower()
    return "clean_inputs" in s and p.suffix.lower() == ".pdb"

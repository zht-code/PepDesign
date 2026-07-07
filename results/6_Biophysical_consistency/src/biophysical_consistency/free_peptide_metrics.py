from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .pdb_io import (
    Atom,
    ca_atoms,
    end_to_end_distance,
    guess_receptor_peptide_segments,
    radius_of_gyration,
    read_atom_segments,
    vec_len,
    vec_sub,
)
from .sequence_biophysics import kd_for_resname


def _angle(a: Atom, b: Atom, c: Atom) -> float:
    v1 = vec_sub(a, b)
    v2 = vec_sub(c, b)
    n1 = vec_len(v1)
    n2 = vec_len(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    dot = -(v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
    cos = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos))


def _torsion(a: Atom, b: Atom, c: Atom, d: Atom) -> float:
    """Proper torsion in degrees."""

    def ucross(p, q):
        return (
            p[1] * q[2] - p[2] * q[1],
            p[2] * q[0] - p[0] * q[2],
            p[0] * q[1] - p[1] * q[0],
        )

    def udot(p, q):
        return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]

    b1 = vec_sub(b, a)
    b2 = vec_sub(c, b)
    b3 = vec_sub(d, c)
    n1 = ucross(b1, b2)
    n2 = ucross(b2, b3)
    m1 = vec_len(n1)
    m2 = vec_len(n2)
    if m1 < 1e-8 or m2 < 1e-8:
        return float("nan")
    n1 = (n1[0] / m1, n1[1] / m1, n1[2] / m1)
    n2 = (n2[0] / m2, n2[1] / m2, n2[2] / m2)
    m3 = vec_len(b2)
    if m3 < 1e-8:
        return float("nan")
    ub2 = (-b2[0] / m3, -b2[1] / m3, -b2[2] / m3)
    x = udot(n1, n2)
    y = udot(ucross(n1, ub2), n2)
    return math.degrees(math.atan2(y, x))


def ca_clash_count(ca: list[Atom], cutoff: float) -> int:
    n = len(ca)
    cnt = 0
    for i in range(n):
        for j in range(i + 2, n):
            d = vec_len(vec_sub(ca[i], ca[j]))
            if d < cutoff:
                cnt += 1
    return cnt


def ca_bond_stats(ca: list[Atom]) -> tuple[float | None, float | None, int]:
    if len(ca) < 2:
        return None, None, 0
    dists = [vec_len(vec_sub(ca[i + 1], ca[i])) for i in range(len(ca) - 1)]
    mean = sum(dists) / len(dists)
    var = sum((d - mean) ** 2 for d in dists) / max(len(dists) - 1, 1)
    std = math.sqrt(var)
    outliers = sum(1 for d in dists if d < 3.0 or d > 4.2)
    return mean, std, outliers


def analyze_peptide_segment(peptide: list[Atom], clash_cutoff: float) -> dict[str, Any]:
    ca = ca_atoms(peptide)
    rg = radius_of_gyration(ca)
    ete = end_to_end_distance(ca)
    clashes = ca_clash_count(ca, clash_cutoff)
    mean_d, std_d, out_bonds = ca_bond_stats(ca)

    pseudo_phi_like: list[float] = []
    for i in range(1, len(ca) - 1):
        pseudo_phi_like.append(_angle(ca[i - 1], ca[i], ca[i + 1]))
    mean_ba = (
        sum(pseudo_phi_like) / len(pseudo_phi_like) if pseudo_phi_like else float("nan")
    )

    dihs: list[float] = []
    for i in range(len(ca) - 3):
        dihs.append(_torsion(ca[i], ca[i + 1], ca[i + 2], ca[i + 3]))
    helix_like = sum(1 for t in dihs if not math.isnan(t) and 30 < t < 90)
    strand_like = sum(1 for t in dihs if not math.isnan(t) and (t < -90 or t > 90))
    denom = max(len(dihs), 1)

    kd_mean = None
    if ca:
        kds = [kd_for_resname(a.resname) for a in ca]
        kd_mean = sum(kds) / len(kds)

    return {
        "n_ca": len(ca),
        "radius_gyration_ca": rg,
        "end_to_end_ca": ete,
        "ca_clash_pairs": clashes,
        "ca_bond_length_mean": mean_d,
        "ca_bond_length_std": std_d,
        "ca_bond_outlier_count": out_bonds,
        "pseudo_bend_angle_mean": mean_ba,
        "fraction_dihedral_helix_like": helix_like / denom,
        "fraction_dihedral_strand_like": strand_like / denom,
        "mean_kd_ca_residues": kd_mean,
    }


def analyze_pdb_path(pdb_path: Path, clash_cutoff: float) -> dict[str, Any]:
    segs = read_atom_segments(pdb_path)
    if not segs:
        return {"status": "empty", "error": "no_atoms"}
    if len(segs) == 1:
        pep = segs[0]
        role = "peptide_only"
        metrics = analyze_peptide_segment(pep, clash_cutoff)
        metrics["pdb_role"] = role
        metrics["n_segments"] = 1
        metrics["receptor_n_ca"] = None
        metrics["status"] = "ok"
        return metrics

    rec, pep = guess_receptor_peptide_segments(segs)
    if pep is None:
        pep = segs[-1]
    role = "complex_shortest_segment_peptide"
    metrics = analyze_peptide_segment(pep, clash_cutoff)
    metrics["pdb_role"] = role
    metrics["n_segments"] = len(segs)
    metrics["receptor_n_ca"] = len(ca_atoms(rec))
    metrics["status"] = "ok"
    return metrics

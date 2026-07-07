from __future__ import annotations

from pathlib import Path
from typing import Any

from .pdb_io import (
    Atom,
    ca_atoms,
    guess_receptor_peptide_segments,
    read_atom_segments,
    vec_len,
    vec_sub,
)
from .sequence_biophysics import is_polar_resname, kd_for_resname


def _heavy_atoms(seg: list[Atom]) -> list[Atom]:
    return [a for a in seg if a.name.strip() not in ("H", "HA", "HN")]


def _min_distance_heavy(a: list[Atom], b: list[Atom]) -> float:
    best = float("inf")
    for x in a:
        for y in b:
            d = vec_len(vec_sub(x, y))
            if d < best:
                best = d
    return best


def interface_residue_indices(
    receptor: list[Atom], peptide: list[Atom], cutoff: float
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    """Return sets of (chain, resseq) for residues with any heavy-atom contact < cutoff."""
    rec_h = _heavy_atoms(receptor)
    pep_h = _heavy_atoms(peptide)
    rec_if: set[tuple[str, int]] = set()
    pep_if: set[tuple[str, int]] = set()
    # neighbor list naive O(N^2) — acceptable for single complex
    for pr in pep_h:
        for rr in rec_h:
            if vec_len(vec_sub(pr, rr)) < cutoff:
                pep_if.add((pr.chain, pr.resseq))
                rec_if.add((rr.chain, rr.resseq))
    return rec_if, pep_if


def simple_hbond_pairs(
    receptor: list[Atom], peptide: list[Atom], cutoff: float
) -> int:
    def is_donor(atom: Atom) -> bool:
        n = atom.name.strip()
        return n == "N" or n.startswith("NH")

    def is_acceptor(atom: Atom) -> bool:
        n = atom.name.strip()
        return n == "O" or n.startswith("OD") or n.startswith("OE") or n.startswith("OG") or n == "OXT"

    cnt = 0
    for p in receptor + peptide:
        if not is_donor(p):
            continue
        for q in receptor + peptide:
            if p is q:
                continue
            if not is_acceptor(q):
                continue
            if vec_len(vec_sub(p, q)) <= cutoff:
                cnt += 1
    return cnt // 2


def hydrophobic_complementarity_score(
    rec_if_res: set[tuple[str, int]],
    pep_if_res: set[tuple[str, int]],
    receptor: list[Atom],
    peptide: list[Atom],
    delta: float,
) -> float | None:
    """Fraction of interface pairs (approx) where |KD_rec - KD_pep| < delta."""
    def res_kd_map(seg: list[Atom]) -> dict[tuple[str, int], float]:
        m: dict[tuple[str, int], float] = {}
        for a in seg:
            key = (a.chain, a.resseq)
            if key not in m:
                m[key] = kd_for_resname(a.resname)
        return m

    rk = res_kd_map(receptor)
    pk = res_kd_map(peptide)
    if not rec_if_res or not pep_if_res:
        return None
    # approximate: compare mean KD on interface patches
    mr = sum(rk[k] for k in rec_if_res if k in rk) / max(len(rec_if_res), 1)
    mp = sum(pk[k] for k in pep_if_res if k in pk) / max(len(pep_if_res), 1)
    return abs(mr - mp)


def analyze_complex(
    pdb_path: Path, cutoff: float, hydro_delta: float
) -> dict[str, Any]:
    segs = read_atom_segments(pdb_path)
    if len(segs) < 2:
        return {
            "status": "skipped",
            "reason": "single_segment_not_a_docked_complex",
            "n_segments": len(segs),
        }
    rec, pep = guess_receptor_peptide_segments(segs)
    if pep is None:
        return {"status": "error", "reason": "no_peptide_segment"}
    rec_if, pep_if = interface_residue_indices(rec, pep, cutoff)
    hb = simple_hbond_pairs(rec, pep, 3.5)
    comp = hydrophobic_complementarity_score(rec_if, pep_if, rec, pep, hydro_delta)

    pep_ca = ca_atoms(pep)
    rec_ca = ca_atoms(rec)
    pep_polar = sum(1 for a in pep_ca if is_polar_resname(a.resname)) / max(
        len(pep_ca), 1
    )
    pep_if_ca = [a for a in pep_ca if (a.chain, a.resseq) in pep_if]
    pep_if_polar = (
        sum(1 for a in pep_if_ca if is_polar_resname(a.resname)) / max(len(pep_if_ca), 1)
        if pep_if_ca
        else None
    )

    return {
        "status": "ok",
        "n_segments": len(segs),
        "receptor_n_ca": len(rec_ca),
        "peptide_n_ca": len(pep_ca),
        "n_interface_peptide_residues": len(pep_if),
        "n_interface_receptor_residues": len(rec_if),
        "interface_area_proxy": len(pep_if) + len(rec_if),
        "simple_hbond_pairs_3p5A": hb,
        "hydrophobic_mismatch_abs_mean_kd_interface": comp,
        "peptide_overall_polar_fraction_ca": pep_polar,
        "peptide_interface_polar_fraction_ca": pep_if_polar,
    }

"""Target perturbations for robustness evaluation (structure / pocket / sequence)."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from .paths import PROJECT_ROOT  # noqa: F401 — 确保 sys.path 含 Peptide_3D 根目录

from model.esm.utils.structure.protein_chain import ProteinChain

PerturbKind = Literal["clean", "structure_missing", "pocket_noise", "sequence_trunc"]


def _chain_copy(chain: ProteinChain) -> ProteinChain:
    return replace(
        chain,
        atom37_positions=np.array(chain.atom37_positions, dtype=np.float64, copy=True),
        atom37_mask=np.array(chain.atom37_mask, dtype=bool, copy=True),
        confidence=np.array(chain.confidence, dtype=np.float64, copy=True),
        residue_index=np.array(chain.residue_index, dtype=np.int64, copy=True),
        insertion_code=np.array(chain.insertion_code, copy=True),
    )


def pocket_residue_indices(
    chain: ProteinChain,
    peptide_pdb: str,
    radius_A: float,
) -> list[int]:
    """
    Receptor residues with any heavy atom within radius_A of any peptide CA.
    Falls back to all residue indices if peptide PDB is missing or empty.
    """
    from pathlib import Path

    pep_path = Path(peptide_pdb)
    if not pep_path.is_file():
        return list(range(len(chain)))

    pep_ca: list[np.ndarray] = []
    with open(pep_path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            pep_ca.append(np.array([x, y, z], dtype=np.float64))
    if not pep_ca:
        return list(range(len(chain)))

    pep_pts = np.stack(pep_ca, axis=0)
    pocket: set[int] = set()
    pos = chain.atom37_positions
    mask = chain.atom37_mask
    for i in range(len(chain)):
        for a in range(pos.shape[1]):
            if not mask[i, a]:
                continue
            d = np.linalg.norm(pos[i, a] - pep_pts, axis=1).min()
            if d <= radius_A:
                pocket.add(i)
                break
    return sorted(pocket) if pocket else list(range(len(chain)))


def apply_structure_missing(
    chain: ProteinChain,
    drop_pct: float,
    rng: np.random.Generator,
) -> ProteinChain:
    """Randomly remove structure for floor(drop_pct/100 * L) residues (mask + nan coords)."""
    c = _chain_copy(chain)
    L = len(c)
    if L == 0 or drop_pct <= 0:
        return c
    k = int(np.floor(L * (drop_pct / 100.0)))
    k = min(max(k, 0), L)
    if k == 0:
        return c
    idx = rng.choice(L, size=k, replace=False)
    c.atom37_mask[idx, :] = False
    c.atom37_positions[idx, :, :] = np.nan
    return c


def apply_pocket_noise(
    chain: ProteinChain,
    pocket_idx: list[int],
    sigma_A: float,
    rng: np.random.Generator,
) -> ProteinChain:
    """Gaussian noise on N, CA, C (atom37 slots 0,1,2) for pocket residues."""
    c = _chain_copy(chain)
    if sigma_A <= 0 or not pocket_idx:
        return c
    noise = rng.normal(scale=sigma_A, size=(len(pocket_idx), 3, 3))
    backbone = (0, 1, 2)
    for j, i in enumerate(pocket_idx):
        if i < 0 or i >= len(c):
            continue
        for b in backbone:
            if c.atom37_mask[i, b]:
                c.atom37_positions[i, b, :] += noise[j, b, :]
    return c


def apply_sequence_truncation(
    chain: ProteinChain,
    trunc_pct: float,
    rng: np.random.Generator,
    min_len: int,
) -> ProteinChain | None:
    """
    Random contiguous crop keeping (1 - trunc_pct/100) * L residues (at least min_len).
    Returns None if resulting length < min_len.
    """
    L = len(chain)
    if L == 0:
        return None
    if trunc_pct <= 0:
        return chain
    Lk = int(round(L * (1.0 - trunc_pct / 100.0)))
    Lk = max(Lk, min_len)
    Lk = min(Lk, L)
    if Lk < min_len:
        return None
    if Lk >= L:
        return chain
    start = int(rng.integers(0, L - Lk + 1))
    idx = np.arange(start, start + Lk, dtype=np.int64)
    return chain[idx]


def apply_perturbation(
    chain: ProteinChain,
    kind: PerturbKind,
    level_value: float,
    rng: np.random.Generator,
    *,
    pocket_idx: list[int] | None,
    min_len_after_trunc: int,
) -> ProteinChain | None:
    """
    level_value:
      structure_missing / sequence_trunc: percentage 0..100
      pocket_noise: sigma in Angstrom
    """
    if kind == "clean":
        return chain
    if kind == "structure_missing":
        return apply_structure_missing(chain, level_value, rng)
    if kind == "pocket_noise":
        return apply_pocket_noise(chain, pocket_idx or [], level_value, rng)
    if kind == "sequence_trunc":
        return apply_sequence_truncation(chain, level_value, rng, min_len=min_len_after_trunc)
    raise ValueError(f"Unknown perturbation {kind}")

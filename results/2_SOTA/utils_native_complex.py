from __future__ import annotations

import string
from pathlib import Path
from typing import Set

from Bio.PDB import Chain, Model, PDBIO, Structure

from metrics_structure import load_structure


def _used_chain_ids(structure) -> Set[str]:
    used = set()
    for model in structure:
        for chain in model:
            cid = chain.id
            if isinstance(cid, str) and len(cid) == 1:
                used.add(cid)
    return used


def _pick_peptide_chain_id(used: Set[str]) -> str:
    for c in string.ascii_uppercase:
        if c not in used:
            return c
    raise ValueError("No free single-letter chain id for peptide merge")


def merge_receptor_and_peptide_to_pdb(
    receptor_path: str,
    peptide_path: str,
    out_pdb: str,
) -> str:
    """
    Write a single-model PDB with receptor chains preserved and peptide placed on a new chain id.
    Used as a native complex for interface contact map consistency.
    """
    rec = load_structure(receptor_path)
    pep = load_structure(peptide_path)
    used = _used_chain_ids(rec)
    new_id = _pick_peptide_chain_id(used)

    combined = Structure.Structure("complex")
    rec_model = next(rec.get_models())
    new_model = Model.Model(0)
    for chain in rec_model:
        new_model.add(chain.copy())
    combined.add(new_model)

    pep_model = next(pep.get_models())
    pep_chain = next(iter(pep_model))
    nc = Chain.Chain(new_id)
    for res in pep_chain:
        nc.add(res.copy())
    new_model.add(nc)

    io = PDBIO()
    io.set_structure(combined)
    out = Path(out_pdb)
    out.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(out))
    return str(out)

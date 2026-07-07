"""Generate ranked peptides for one PPDbench target under a perturbation setting."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from .paths import PROJECT_ROOT  # 先将 Peptide_3D 根目录加入 sys.path，再 import model
from .encode_perturbed import encode_perturbed_target, write_chain_temp_pdb
from model.esm.utils.structure.protein_chain import ProteinChain
from .perturbations import (
    apply_pocket_noise,
    apply_sequence_truncation,
    apply_structure_missing,
    pocket_residue_indices,
)

if TYPE_CHECKING:
    from models_DPO import ProteinPeptideModel

# Reuse OpenMM / peptide building from the official PPDbench generator
_PARETO = PROJECT_ROOT / "results" / "3_Pareto_improved"
if str(_PARETO) not in sys.path:
    sys.path.insert(0, str(_PARETO))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ppdbench_generate_core as pgc  # noqa: E402


def _placement_pdb_path(
    receptor_pdb: str,
    peptide_pdb: str,
    mode: str,
    perturb_kind: str,
    level_value: float,
    rng: np.random.Generator,
    pocket_radius_A: float,
    tmp_root: Path,
    min_residues: int,
) -> str:
    """PDB path passed to rigid placement (CA / pocket center)."""
    if mode == "sequence_only":
        return str(Path(receptor_pdb).resolve())

    chain0 = ProteinChain.from_pdb(receptor_pdb)
    pocket_idx = pocket_residue_indices(chain0, peptide_pdb, pocket_radius_A)
    if perturb_kind == "clean":
        chain = chain0
    elif perturb_kind == "structure_missing":
        chain = apply_structure_missing(chain0, level_value, rng)
    elif perturb_kind == "pocket_noise":
        chain = apply_pocket_noise(chain0, pocket_idx, level_value, rng)
    elif perturb_kind == "sequence_trunc":
        ch = apply_sequence_truncation(chain0, level_value, rng, min_len=min_residues)
        if ch is None:
            raise ValueError("truncation too aggressive for placement chain")
        chain = ch
    else:
        raise ValueError(perturb_kind)
    return write_chain_temp_pdb(chain, tmp_root)


def score_interface(model: ProteinPeptideModel, seq: str, enc: torch.Tensor, mask: torch.Tensor) -> float:
    return float(pgc._score_interface(model, seq, enc, mask))


def generate_for_target(
    model: ProteinPeptideModel,
    prot_dir: str,
    *,
    encoder_mode: str,
    perturb_kind: str,
    level_value: float,
    rng: np.random.Generator,
    pocket_radius_A: float,
    min_residues: int,
    num_keep: int,
    top_k: int,
    max_len: int,
    temperature: float,
    oversample_factor: int,
    tmp_root: Path,
) -> tuple[list[str], str]:
    """
    Returns list of peptide sequences (length num_keep) and writes PDBs to prot_dir/out_subdir
    via caller-provided out_dir.
    """
    prot_dir = Path(prot_dir)
    rec = prot_dir / "receptor.pdb"
    pep_ref = prot_dir / "peptide.pdb"
    if not rec.is_file():
        raise FileNotFoundError(rec)

    enc, mask, meta = encode_perturbed_target(
        model,
        str(rec),
        str(pep_ref) if pep_ref.is_file() else "",
        encoder_mode,  # type: ignore[arg-type]
        perturb_kind,
        level_value,
        rng,
        pocket_radius_A,
        min_residues,
    )
    _ = meta

    pdb_for_placement = _placement_pdb_path(
        str(rec),
        str(pep_ref) if pep_ref.is_file() else "",
        encoder_mode,
        perturb_kind,
        level_value,
        rng,
        pocket_radius_A,
        tmp_root,
        min_residues,
    )

    num_cand = max(num_keep, num_keep * oversample_factor)
    seqs: list[str] = []
    for _ in range(num_cand):
        toks = model.esmc.sample_topk(
            encoder_embeddings=enc,
            cross_attention_mask=mask,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
        )
        s = model.esmc._detokenize(toks)[0]
        seqs.append(s)

    scored = []
    for s in seqs:
        try:
            sc = score_interface(model, s, enc, mask)
        except Exception:
            sc = -1e9
        scored.append((sc, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:num_keep]], pdb_for_placement


def write_peptide_pdbs(
    seqs: list[str],
    *,
    receptor_pdb_for_placement: str,
    out_dir: Path,
) -> None:
    """For each sequence build fullatom peptide + minimize; reuse ppdbench_generate_core helpers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_ca, pocket_center = pgc._pocket_center_from_receptor(receptor_pdb_for_placement, pocket_idx=None)

    for i, seq in enumerate(seqs, start=1):
        Lcap = 64
        Lwant = min(len(seq), Lcap)
        seq_used = pgc.sanitize_sequence(seq[:Lwant])

        if pgc.INIT_GEOM_MODE == "helix":
            struct = pgc._build_fullatom_peptide_helix(seq_used)
            ca_list = []
            for atom in struct.get_atoms():
                if atom.get_name() == "CA":
                    ca_list.append(atom.get_coord())
            ca_xyz = torch.tensor(np.array(ca_list, dtype=np.float32), device=rec_ca.device)
            placed_ca = pgc._rigid_place_near_pocket(ca_xyz, rec_ca, pocket_center, tries=48, margin=2.5)
            placed_ca_numpy = placed_ca.cpu().numpy()
            src = ca_xyz.cpu().numpy()
            dst = placed_ca_numpy
            R, t = pgc._kabsch_align(src, dst)
            for atom in struct.get_atoms():
                x = atom.get_coord().astype(np.float64)
                atom.set_coord((R @ x) + t)
        else:
            raise RuntimeError("Only helix INIT_GEOM_MODE supported in robustness writer")

        final_pdb = out_dir / f"pep_{i:02d}.pdb"
        with tempfile.TemporaryDirectory(dir=pgc.TEMP_ROOT) as tdir:
            tmp_raw = os.path.join(tdir, "pep_raw.pdb")
            pgc._save_structure(struct, tmp_raw)
            if pgc.OPENMM_OK:
                try:
                    pgc._openmm_minimize_with_hard_frame(
                        in_pdb=tmp_raw,
                        out_pdb=str(final_pdb),
                        placed_ca_xyz_A=placed_ca_numpy,
                        ca_k=2.0,
                        max_steps=2000,
                        ph=7.0,
                        add_alpha_restraints=True,
                    )
                except Exception:
                    pgc._save_structure(struct, str(final_pdb))
            else:
                pgc._save_structure(struct, str(final_pdb))

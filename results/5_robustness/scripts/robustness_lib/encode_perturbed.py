"""Perturbed protein encoding for robustness (geometry-aware or sequence-only)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from .paths import PROJECT_ROOT  # noqa: F401 — 确保 sys.path 含 Peptide_3D 根目录

from model.esm.utils.structure.protein_chain import ProteinChain

if TYPE_CHECKING:
    from models_DPO import ProteinPeptideModel

EncoderMode = Literal["geometry", "sequence_only"]


def _backbone_NCAC(chain: ProteinChain) -> np.ndarray:
    """(L, 3, 3) for N, CA, C; nan if any backbone atom missing."""
    pos = chain.atom37_positions
    mask = chain.atom37_mask
    L = len(chain)
    out = np.full((L, 3, 3), np.nan, dtype=np.float32)
    for i in range(L):
        ok = True
        bb = []
        for j in (0, 1, 2):
            if not mask[i, j]:
                ok = False
                break
            bb.append(pos[i, j].astype(np.float32))
        if ok:
            out[i] = np.stack(bb, axis=0)
    return out


def _pad_structure_coords_for_tokens(
    backbone: np.ndarray,
    seq: str,
    tokens: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map per-residue backbone (L_aa, 3, 3) to token length [1, L_tok, 3, 3] with nan padding."""
    L_aa = len(seq)
    L_tok = int(tokens.shape[1])
    full = torch.full((1, L_tok, 3, 3), float("nan"), device=device, dtype=dtype)
    if L_aa == 0:
        return full
    if L_tok == L_aa + 2:
        t = torch.tensor(backbone, device=device, dtype=dtype)
        full[0, 1 : 1 + L_aa, :, :] = t
    elif L_tok == L_aa:
        full[0, :L_aa, :, :] = torch.tensor(backbone, device=device, dtype=dtype)
    else:
        n = min(L_aa, L_tok)
        full[0, :n, :, :] = torch.tensor(backbone[:n], device=device, dtype=dtype)
    return full


def encode_from_sequence_tokens(
    model: ProteinPeptideModel,
    sequence: str,
    structure_coords: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of encode_protein_from_pdb after sequence is fixed; optional structure_coords for ESM3."""
    # 必须与 esm3 权重同设备（esm3 默认 CPU、esmc 可能在 GPU 时，勿仅用 model.device）
    esm_dev = next(model.esm3_model.parameters()).device
    dec_dev = next(model.esmc.parameters()).device
    tokens = model.esmc._tokenize([sequence]).to(esm_dev)
    pad_id = model.esmc.tokenizer.pad_token_id
    cross_attention_mask = (tokens != pad_id).to(torch.int8)

    model.esm3_model.eval()
    dtype_esm = next(model.esm3_model.parameters()).dtype

    if structure_coords is not None:
        sc = structure_coords.to(device=esm_dev, dtype=dtype_esm)
        esm_out = model.esm3_model(sequence_tokens=tokens, structure_coords=sc)
    else:
        esm_out = model.esm3_model(sequence_tokens=tokens)

    esm_structure_features = esm_out.structure_logits
    if esm_structure_features.dim() == 4:
        struct_feats = esm_structure_features.mean(dim=2)
    elif esm_structure_features.dim() == 3:
        struct_feats = esm_structure_features
    else:
        raise RuntimeError(f"Unexpected struct shape: {esm_structure_features.shape}")

    encoder_embeddings = model.linear_proj(struct_feats.to(next(model.linear_proj.parameters()).device))
    dtype_dec = next(model.esmc.parameters()).dtype
    encoder_embeddings = encoder_embeddings.to(device=dec_dev, dtype=dtype_dec)
    cross_attention_mask = cross_attention_mask.to(dec_dev)
    return encoder_embeddings, cross_attention_mask


def mask_sequence_fraction(seq: str, mask_pct: float, rng: np.random.Generator, mask_char: str = "#") -> str:
    """Replace floor(mask_pct/100 * L) positions with mask token (length preserved)."""
    L = len(seq)
    if L == 0 or mask_pct <= 0:
        return seq
    k = int(np.floor(L * (mask_pct / 100.0)))
    k = min(max(k, 0), L)
    if k == 0:
        return seq
    idx = rng.choice(L, size=k, replace=False)
    chars = list(seq)
    for i in idx:
        chars[i] = mask_char
    return "".join(chars)


def random_contiguous_sequence_crop(seq: str, keep_pct: float, rng: np.random.Generator) -> str:
    """Keep random contiguous segment of length round(keep_pct/100 * L)."""
    L = len(seq)
    if L == 0 or keep_pct >= 100:
        return seq
    Lk = int(round(L * (keep_pct / 100.0)))
    Lk = max(1, min(Lk, L))
    if Lk >= L:
        return seq
    start = int(rng.integers(0, L - Lk + 1))
    return seq[start : start + Lk]


def encode_perturbed_target(
    model: ProteinPeptideModel,
    receptor_pdb: str,
    peptide_pdb: str,
    mode: EncoderMode,
    perturb_kind: str,
    level_value: float,
    rng: np.random.Generator,
    pocket_radius_A: float,
    min_residues: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Returns encoder_embeddings, cross_attention_mask, meta dict.
    level_value: % for structure_missing / sequence_trunc, Angstrom for pocket_noise.
    """
    meta: dict = {"encoder_mode": mode, "perturb_kind": perturb_kind, "level": level_value}
    rec_path = str(Path(receptor_pdb).resolve())
    pep_path = str(Path(peptide_pdb).resolve()) if Path(peptide_pdb).is_file() else ""

    chain0 = ProteinChain.from_pdb(rec_path)
    if len(chain0) < min_residues:
        raise ValueError(f"receptor too short: {len(chain0)} < {min_residues}")

    from .perturbations import (
        apply_pocket_noise,
        apply_sequence_truncation,
        apply_structure_missing,
        pocket_residue_indices,
    )

    pocket_idx = pocket_residue_indices(chain0, pep_path, pocket_radius_A)

    if mode == "sequence_only":
        seq = chain0.sequence
        if perturb_kind == "clean":
            pass
        elif perturb_kind == "structure_missing" and level_value > 0:
            seq = mask_sequence_fraction(seq, level_value, rng)
        elif perturb_kind == "pocket_noise" and level_value > 0:
            meta["note"] = (
                "sequence_only: pocket noise does not change ESM3 conditioning; "
                "use encoder_mode=geometry for pocket-coordinate perturbations at the encoder."
            )
        elif perturb_kind == "sequence_trunc" and level_value > 0:
            seq = random_contiguous_sequence_crop(seq, 100.0 - level_value, rng)
        elif perturb_kind not in ("clean", "structure_missing", "pocket_noise", "sequence_trunc"):
            raise ValueError(f"Unknown perturb_kind {perturb_kind}")
        if len(seq) < min_residues:
            raise ValueError(f"sequence too short after perturbation: {len(seq)}")
        enc, mask = encode_from_sequence_tokens(model, seq, None)
        meta["sequence_len"] = len(seq)
        return enc, mask, meta

    # geometry mode
    if perturb_kind == "clean":
        chain = chain0
    elif perturb_kind == "structure_missing":
        chain = apply_structure_missing(chain0, level_value, rng)
    elif perturb_kind == "pocket_noise":
        chain = apply_pocket_noise(chain0, pocket_idx, level_value, rng)
    elif perturb_kind == "sequence_trunc":
        ch = apply_sequence_truncation(chain0, level_value, rng, min_len=min_residues)
        if ch is None:
            raise ValueError("sequence_trunc produced chain shorter than min_residues")
        chain = ch
    else:
        raise ValueError(f"Unknown perturb_kind {perturb_kind}")

    seq = chain.sequence
    if len(seq) < min_residues:
        raise ValueError(f"receptor sequence too short: {len(seq)}")

    bb = _backbone_NCAC(chain)
    esm_dev = next(model.esm3_model.parameters()).device
    tokens = model.esmc._tokenize([seq]).to(esm_dev)
    dtype_esm = next(model.esm3_model.parameters()).dtype
    sc = _pad_structure_coords_for_tokens(bb, seq, tokens, esm_dev, dtype_esm)
    enc, mask = encode_from_sequence_tokens(model, seq, sc)
    meta["sequence_len"] = len(seq)
    meta["pocket_n_residues"] = len(pocket_idx)
    return enc, mask, meta


def write_chain_temp_pdb(chain: ProteinChain, tmp_root: Path) -> str:
    """Write ProteinChain to a temporary PDB path."""
    tmp_root.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix="_rec.pdb", dir=str(tmp_root))
    os.close(fd)
    path = Path(name)
    chain.to_pdb(path)
    return str(path)

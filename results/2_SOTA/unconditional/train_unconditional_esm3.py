#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from Bio.PDB import PDBParser


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
RESULTS_ROOT = THIS_DIR.parent
REFERENCE_ROOT = PROJECT_ROOT / "utils" / "reference"

for path in [PROJECT_ROOT, RESULTS_ROOT, THIS_DIR, REFERENCE_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from model.esm.pretrained import load_local_model
from models import _ensure_pairwise
from utils_io import ensure_dir, set_seed, write_fasta
import test_data_generate_top10 as ref_gen


DISTOGRAM_BINS = 64
DISTOGRAM_LEN = 64
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
    "SEC": "U", "PYL": "O",
}


class PeptideStructureDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


def is_distributed_mode() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not is_distributed_mode():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return 0, 1, 0, device

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return local_rank, world_size, rank, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def set_requires_grad(module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def freeze_unconditional_unused_modules(decoder) -> None:
    model = unwrap_model(decoder)

    if hasattr(model, "iface_q"):
        set_requires_grad(model.iface_q, False)
    if hasattr(model, "iface_k"):
        set_requires_grad(model.iface_k, False)

    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        for block in model.transformer.blocks:
            for name in ("attr_protein_cross_attn", "peptide_attr_cross_attn", "cross_attn"):
                if hasattr(block, name):
                    set_requires_grad(getattr(block, name), False)


def count_trainable_parameters(model) -> int:
    return sum(param.numel() for param in unwrap_model(model).parameters() if param.requires_grad)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Train or run an unconditional peptide baseline that reuses the current "
            "ESMC tokenizer/decoder/sampling stack."
        )
    )
    ap.add_argument("--mode", choices=["train", "generate"], default="generate")
    ap.add_argument("--input-csv", default=None, help="Test split csv, e.g. protein_level_test.csv.")
    ap.add_argument("--outdir", default=None, help="Output root for generated candidates and manifest.")
    ap.add_argument(
        "--ckpt-path",
        default=None,
        help=(
            "Optional initialization checkpoint. Supports ProteinPeptideModel checkpoints "
            "with `esmc.*` keys or checkpoints saved by `--mode train`."
        ),
    )
    ap.add_argument("--train-root", default="/root/autodl-tmp/train_data", help="Training set root.")
    ap.add_argument(
        "--train-peptide-pdb",
        action="append",
        default=[],
        help="Optional peptide pdb path to include explicitly. Can be passed multiple times.",
    )
    ap.add_argument(
        "--save-dir",
        default="/root/autodl-tmp/Peptide_3D/log_unconditional",
        help="Directory for trained unconditional checkpoints.",
    )
    ap.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    ap.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    ap.add_argument("--train-num-workers", type=int, default=0, help="Training dataloader workers per process.")
    ap.add_argument("--lr", type=float, default=2e-5, help="Training learning rate.")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    ap.add_argument(
        "--structure-loss-weight",
        type=float,
        default=0.25,
        help="Relative weight for unconditional structure-head supervision.",
    )
    ap.add_argument(
        "--train-max-residues",
        type=int,
        default=62,
        help="Truncate training peptides to keep token length within the 64-bin structure head.",
    )
    ap.add_argument("--save-every", type=int, default=1, help="Checkpoint frequency in epochs.")
    ap.add_argument(
        "--method",
        default="unconditional_esm3",
        choices=["unconditional_esm3"],
        help="Manifest method name.",
    )
    ap.add_argument("--dataset", default="internal", help="Manifest dataset column.")
    ap.add_argument("--split-name", default=None, help="Manifest split_name; defaults to input csv stem.")
    ap.add_argument("--num-candidates", type=int, default=5, help="Generated candidates per target.")
    ap.add_argument("--top-k", type=int, default=12, help="Top-k sampling for the decoder.")
    ap.add_argument("--max-len", type=int, default=30, help="Maximum decoded peptide length.")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    ap.add_argument(
        "--oversample-factor",
        type=int,
        default=3,
        help="Extra unconditional draws used to reduce duplicates before keeping N candidates.",
    )
    ap.add_argument(
        "--save-pdb",
        choices=["mds", "helix", "none"],
        default="mds",
        help=(
            "`mds`: use decoder structure_head + existing MDS rebuild; "
            "`helix`: reuse current helix builder; `none`: sequence only."
        ),
    )
    ap.add_argument("--num-gpus", type=int, default=1, help="How many GPUs to shard targets across.")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return ap.parse_args()


def load_targets(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["sample_id", "receptor_pdb", "peptide_pdb"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input csv missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"Input csv has no rows: {csv_path}")
    df = df.copy()
    df["_row_order"] = np.arange(len(df), dtype=np.int64)
    return df


def _extract_decoder_state(raw_state: Any) -> dict[str, torch.Tensor]:
    state = raw_state
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise RuntimeError("Unsupported checkpoint format for unconditional decoder loading.")

    if any(str(key).startswith("esmc.") for key in state.keys()):
        return {
            str(key)[len("esmc."):]: value
            for key, value in state.items()
            if str(key).startswith("esmc.")
        }
    return state


def load_unconditional_decoder(device: torch.device, ckpt_path: str | None):
    decoder = load_local_model(model_name="esmc_300m", device=device).to(device)

    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        esmc_state = _extract_decoder_state(state)
        if not esmc_state:
            raise RuntimeError(
                f"No decoder weights found in checkpoint: {ckpt_path}."
            )
        decoder.load_state_dict(esmc_state, strict=False)

    decoder.eval()
    return decoder


def extract_sequence_and_ca_from_pdb(pdb_path: str, max_residues: int | None = None) -> tuple[str, np.ndarray]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pep", pdb_path)

    seq_chars: list[str] = []
    ca_coords: list[np.ndarray] = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                aa = THREE_TO_ONE.get(res.resname.upper())
                if aa is None or "CA" not in res:
                    continue
                seq_chars.append(aa)
                ca_coords.append(np.asarray(res["CA"].coord, dtype=np.float32))
            if seq_chars:
                break
        if seq_chars:
            break

    if max_residues is not None:
        seq_chars = seq_chars[:max_residues]
        ca_coords = ca_coords[:max_residues]

    seq = "".join(seq_chars)
    coords = np.stack(ca_coords, axis=0) if ca_coords else np.zeros((0, 3), dtype=np.float32)
    return seq, coords


def discover_training_pairs(train_root: str, explicit_peptide_pdbs: list[str]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    seen: set[str] = set()

    for pdb_path in explicit_peptide_pdbs:
        pep = Path(pdb_path)
        rec = pep.with_name("receptor.pdb")
        if pep.is_file() and rec.is_file() and str(pep.resolve()) not in seen:
            pairs.append((pep, rec))
            seen.add(str(pep.resolve()))

    root = Path(train_root)
    if root.is_dir():
        for sample_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            pep = sample_dir / "peptide.pdb"
            rec = sample_dir / "receptor.pdb"
            if pep.is_file() and rec.is_file() and str(pep.resolve()) not in seen:
                pairs.append((pep, rec))
                seen.add(str(pep.resolve()))

    if not pairs:
        raise RuntimeError("No train samples with both peptide.pdb and receptor.pdb were found.")
    return pairs


def load_training_items(
    train_root: str,
    explicit_peptide_pdbs: list[str],
    max_residues: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for pep_pdb, rec_pdb in discover_training_pairs(train_root, explicit_peptide_pdbs):
        sequence, ca_coords = extract_sequence_and_ca_from_pdb(str(pep_pdb), max_residues=max_residues)
        sequence = ref_gen.sanitize_sequence(sequence)
        if not sequence or len(sequence) < 2 or len(ca_coords) != len(sequence):
            continue
        items.append(
            {
                "sample_id": pep_pdb.parent.name,
                "peptide_pdb": str(pep_pdb),
                "receptor_pdb": str(rec_pdb),
                "sequence": sequence,
                "ca_coords": ca_coords,
            }
        )

    if not items:
        raise RuntimeError("No valid training peptide sequences could be extracted from the training set.")
    return items


def train_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "sequence": [item["sequence"] for item in batch],
        "ca_coords": [item["ca_coords"] for item in batch],
    }


def build_batch_structure_labels(
    sequences: list[str],
    ca_coords_list: list[np.ndarray],
    padded_token_len: int,
    device: torch.device,
) -> torch.Tensor:
    labels = torch.full(
        (len(sequences), padded_token_len, padded_token_len),
        -100,
        dtype=torch.long,
        device=device,
    )

    for i, (seq, ca_coords) in enumerate(zip(sequences, ca_coords_list)):
        L = min(len(seq), len(ca_coords))
        if L == 0:
            continue
        dists = np.linalg.norm(ca_coords[:L, None, :] - ca_coords[None, :L, :], axis=-1)
        bins = np.digitize(dists, ref_gen.BIN_EDGES[1:-1], right=False)
        bins = np.clip(bins, 0, DISTOGRAM_BINS - 1)
        labels[i, 1 : 1 + L, 1 : 1 + L] = torch.as_tensor(bins, dtype=torch.long, device=device)

    return labels


def save_training_checkpoint(
    decoder,
    optimizer: torch.optim.Optimizer,
    save_path: Path,
    *,
    epoch: int,
    avg_loss: float,
    avg_seq_loss: float,
    avg_struct_loss: float,
    train_args: dict[str, Any],
) -> None:
    ensure_dir(save_path.parent)
    torch.save(
        {
            "model_state_dict": unwrap_model(decoder).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_seq_loss": avg_seq_loss,
            "avg_struct_loss": avg_struct_loss,
            "train_args": train_args,
        },
        save_path,
    )


def run_training(args: argparse.Namespace) -> None:
    local_rank, world_size, rank, device = setup_distributed()
    set_seed(args.seed + rank)
    save_dir = ensure_dir(args.save_dir)

    try:
        items = load_training_items(args.train_root, args.train_peptide_pdb, args.train_max_residues)
        dataset = PeptideStructureDataset(items)
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
            if world_size > 1
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.train_num_workers,
            collate_fn=train_collate,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
        )

        decoder = load_unconditional_decoder(device, args.ckpt_path)
        freeze_unconditional_unused_modules(decoder)
        if world_size > 1:
            decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        decoder.train()
        optimizer = torch.optim.AdamW(
            [param for param in unwrap_model(decoder).parameters() if param.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        pad_id = unwrap_model(decoder).tokenizer.pad_token_id
        assert pad_id is not None
        best_loss = float("inf")
        avg_loss = avg_seq_loss = avg_struct_loss = float("nan")

        if is_main_process(rank):
            print(
                f"Training unconditional model on {len(dataset)} samples "
                f"with world_size={world_size}, batch_size_per_gpu={args.batch_size}, "
                f"trainable_params={count_trainable_parameters(decoder)}"
            )

        for epoch in range(1, args.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)

            total_loss = torch.tensor(0.0, device=device)
            total_seq_loss = torch.tensor(0.0, device=device)
            total_struct_loss = torch.tensor(0.0, device=device)
            steps = torch.tensor(0.0, device=device)

            for batch in loader:
                sequences = batch["sequence"]
                ca_coords_list = batch["ca_coords"]

                tokens = unwrap_model(decoder)._tokenize(sequences).to(device)
                out = decoder.forward(sequence_tokens=tokens)

                seq_logits = out.sequence_logits[:, :-1, :].contiguous()
                seq_targets = tokens[:, 1:].contiguous()
                seq_loss = F.cross_entropy(
                    seq_logits.view(-1, seq_logits.size(-1)),
                    seq_targets.view(-1),
                    ignore_index=pad_id,
                )

                struct_loss = torch.tensor(0.0, device=device)
                if args.structure_loss_weight > 0:
                    struct_logits = _ensure_pairwise(out.structure_logits, DISTOGRAM_BINS, tokens.size(1))
                    struct_labels = build_batch_structure_labels(
                        sequences,
                        ca_coords_list,
                        struct_logits.size(1),
                        device,
                    )
                    valid = struct_labels != -100
                    if valid.any():
                        struct_loss = F.cross_entropy(
                            struct_logits[valid],
                            struct_labels[valid],
                        )

                loss = seq_loss + args.structure_loss_weight * struct_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unwrap_model(decoder).parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.detach()
                total_seq_loss += seq_loss.detach()
                total_struct_loss += struct_loss.detach()
                steps += 1

            if world_size > 1:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_seq_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_struct_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(steps, op=dist.ReduceOp.SUM)

            avg_loss = float((total_loss / torch.clamp(steps, min=1.0)).item())
            avg_seq_loss = float((total_seq_loss / torch.clamp(steps, min=1.0)).item())
            avg_struct_loss = float((total_struct_loss / torch.clamp(steps, min=1.0)).item())

            if is_main_process(rank):
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"loss={avg_loss:.4f} seq={avg_seq_loss:.4f} struct={avg_struct_loss:.4f}"
                )

                if epoch % args.save_every == 0:
                    save_training_checkpoint(
                        decoder,
                        optimizer,
                        save_dir / f"epoch_{epoch:03d}.pt",
                        epoch=epoch,
                        avg_loss=avg_loss,
                        avg_seq_loss=avg_seq_loss,
                        avg_struct_loss=avg_struct_loss,
                        train_args=vars(args),
                    )

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_training_checkpoint(
                        decoder,
                        optimizer,
                        save_dir / "best_unconditional_esm3.pt",
                        epoch=epoch,
                        avg_loss=avg_loss,
                        avg_seq_loss=avg_seq_loss,
                        avg_struct_loss=avg_struct_loss,
                        train_args=vars(args),
                    )

        if is_main_process(rank):
            save_training_checkpoint(
                decoder,
                optimizer,
                save_dir / "last_unconditional_esm3.pt",
                epoch=args.epochs,
                avg_loss=avg_loss,
                avg_seq_loss=avg_seq_loss,
                avg_struct_loss=avg_struct_loss,
                train_args=vars(args),
            )
            print(f"Saved unconditional checkpoints to: {save_dir}")
    finally:
        cleanup_distributed()


def _sampling_special_token_ids(decoder) -> set[int]:
    tok = decoder.tokenizer
    ids = []
    for name in ("pad_token_id", "bos_token_id", "cls_token_id", "mask_token_id"):
        value = getattr(tok, name, None)
        if value is not None:
            ids.append(int(value))
    return set(ids)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    history_tokens: list[int],
    *,
    penalty: float,
    special_ids: set[int],
) -> None:
    if penalty is None or penalty <= 1.0:
        return
    used = {int(tok) for tok in history_tokens if int(tok) not in special_ids}
    if not used:
        return
    token_idx = torch.as_tensor(sorted(used), dtype=torch.long, device=logits.device)
    selected = logits[:, token_idx]
    adjusted = torch.where(selected < 0, selected * penalty, selected / penalty)
    logits[:, token_idx] = adjusted


def _apply_no_repeat_ngram(
    logits: torch.Tensor,
    history_tokens: list[int],
    *,
    ngram_size: int,
    special_ids: set[int],
) -> None:
    if ngram_size is None or ngram_size <= 1:
        return
    clean = [int(tok) for tok in history_tokens if int(tok) not in special_ids]
    if len(clean) < ngram_size - 1:
        return

    prefix = tuple(clean[-(ngram_size - 1):])
    banned: set[int] = set()
    for i in range(len(clean) - ngram_size + 1):
        ngram = tuple(clean[i : i + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(int(ngram[-1]))
    if banned:
        logits[:, list(sorted(banned))] = torch.finfo(logits.dtype).min


def _apply_consecutive_token_block(
    logits: torch.Tensor,
    history_tokens: list[int],
    *,
    max_consecutive: int,
    special_ids: set[int],
) -> None:
    if max_consecutive is None or max_consecutive < 1:
        return
    clean = [int(tok) for tok in history_tokens if int(tok) not in special_ids]
    if len(clean) < max_consecutive:
        return
    tail = clean[-max_consecutive:]
    if len(set(tail)) == 1:
        logits[:, tail[-1]] = torch.finfo(logits.dtype).min


@torch.no_grad()
def sample_unconditional_tokens(
    decoder,
    *,
    max_len: int,
    top_k: int,
    temperature: float,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
    max_consecutive_aa: int = 2,
    min_length: int = 6,
) -> torch.Tensor:
    decoder.eval()
    tok = decoder.tokenizer
    device = next(decoder.parameters()).device

    bos_id = getattr(tok, "bos_token_id", None)
    cls_id = getattr(tok, "cls_token_id", None)
    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None)
    start_id = bos_id if bos_id is not None else (cls_id if cls_id is not None else (pad_id if pad_id is not None else 0))

    seq = torch.tensor([[start_id]], dtype=torch.long, device=device)
    special_ids = _sampling_special_token_ids(decoder)

    banned_always = set(special_ids)
    if eos_id is not None:
        banned_always.discard(int(eos_id))

    for _ in range(max_len):
        out = decoder.forward(sequence_tokens=seq)
        logits = out.sequence_logits[:, -1, :].clone()

        if temperature is not None and temperature > 0:
            logits = logits / temperature

        if banned_always:
            logits[:, list(sorted(banned_always))] = torch.finfo(logits.dtype).min

        generated_len = max(0, seq.size(1) - 1)
        if eos_id is not None and generated_len < min_length:
            logits[:, int(eos_id)] = torch.finfo(logits.dtype).min

        history = seq[0].tolist()
        _apply_repetition_penalty(
            logits,
            history,
            penalty=repetition_penalty,
            special_ids=special_ids,
        )
        _apply_no_repeat_ngram(
            logits,
            history,
            ngram_size=no_repeat_ngram_size,
            special_ids=special_ids,
        )
        _apply_consecutive_token_block(
            logits,
            history,
            max_consecutive=max_consecutive_aa,
            special_ids=special_ids,
        )

        if top_k is not None and top_k > 0 and top_k < logits.size(-1):
            topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
            probs = torch.softmax(topk_vals, dim=-1)
            nxt_rel = torch.multinomial(probs, num_samples=1)
            nxt = topk_idx.gather(-1, nxt_rel)
        else:
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)

        seq = torch.cat([seq, nxt], dim=1)
        if eos_id is not None and int(nxt.item()) == int(eos_id):
            break

    return seq


def sample_unconditional_sequences(
    decoder,
    *,
    num_candidates: int,
    top_k: int,
    max_len: int,
    temperature: float,
    oversample_factor: int,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
    max_consecutive_aa: int = 2,
    min_length: int = 6,
) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    raw_pool: list[str] = []
    max_attempts = max(num_candidates * max(1, oversample_factor), num_candidates)
    hard_cap = max(max_attempts, num_candidates * 10)
    attempts = 0

    while len(kept) < num_candidates and attempts < hard_cap:
        toks = sample_unconditional_tokens(
            decoder,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_consecutive_aa=max_consecutive_aa,
            min_length=min_length,
        )
        try:
            pad_id = decoder.tokenizer.pad_token_id
            assert pad_id is not None
            non_pad = toks[0][toks[0] != pad_id]
            if non_pad.numel() <= 2:
                attempts += 1
                continue
            seq = decoder._detokenize(toks)[0].strip()
        except Exception:
            attempts += 1
            continue
        seq = ref_gen.sanitize_sequence(seq)
        attempts += 1
        if not seq:
            continue
        raw_pool.append(seq)
        if seq not in seen:
            seen.add(seq)
            kept.append(seq)

    if len(kept) < num_candidates:
        for seq in raw_pool:
            kept.append(seq)
            if len(kept) >= num_candidates:
                break

    if len(kept) < num_candidates and kept:
        while len(kept) < num_candidates:
            kept.append(kept[len(kept) % len(kept)])

    if len(kept) < num_candidates:
        raise RuntimeError("Unconditional decoder failed to produce any non-empty sequence.")

    return kept[:num_candidates]


@torch.no_grad()
def predict_unconditional_struct_logits(decoder, sequence: str) -> torch.Tensor:
    device = next(decoder.parameters()).device
    toks = decoder._tokenize([sequence]).to(device)
    pad_id = decoder.tokenizer.pad_token_id
    assert pad_id is not None

    L = toks.size(1)
    if L > DISTOGRAM_LEN:
        toks_eval = toks[:, :DISTOGRAM_LEN]
        L = DISTOGRAM_LEN
    elif L < DISTOGRAM_LEN:
        pad = torch.full(
            (1, DISTOGRAM_LEN - L),
            pad_id,
            dtype=toks.dtype,
            device=device,
        )
        toks_eval = torch.cat([toks, pad], dim=1)
    else:
        toks_eval = toks

    out = decoder.forward(sequence_tokens=toks_eval)
    return _ensure_pairwise(out.structure_logits, DISTOGRAM_BINS, L)


def save_unconditional_pdb(
    decoder,
    sequence: str,
    out_pdb: Path,
    save_mode: str,
) -> str:
    if save_mode == "none":
        return ""

    seq_used = ref_gen.sanitize_sequence(sequence[:DISTOGRAM_LEN])
    if not seq_used:
        return ""

    ensure_dir(out_pdb.parent)
    tmp_dir_root = getattr(ref_gen, "TEMP_ROOT", None)

    with tempfile.TemporaryDirectory(dir=tmp_dir_root) as tmpdir:
        tmp_raw = Path(tmpdir) / "peptide_raw.pdb"

        try:
            if save_mode == "mds":
                struct_logits = predict_unconditional_struct_logits(decoder, seq_used)
                coords = ref_gen._distogram_to_coords(struct_logits, ref_gen.BIN_EDGES)
                if getattr(ref_gen, "PB_OK", False):
                    struct = ref_gen._build_fullatom_peptide(seq_used)
                    struct = ref_gen._set_structure_coords_by_ca(struct, coords.cpu().numpy())
                    ref_gen._save_structure(struct, str(tmp_raw))
                else:
                    ref_gen._write_pdb_from_coords(seq_used, coords.cpu(), str(tmp_raw))
            elif save_mode == "helix":
                struct = ref_gen._build_fullatom_peptide_helix(seq_used)
                ref_gen._save_structure(struct, str(tmp_raw))
            else:
                raise ValueError(f"Unsupported save mode: {save_mode}")
        except Exception:
            if save_mode == "mds" and getattr(ref_gen, "PB_OK", False):
                struct = ref_gen._build_fullatom_peptide_helix(seq_used)
                ref_gen._save_structure(struct, str(tmp_raw))
            else:
                raise

        if getattr(ref_gen, "OPENMM_OK", False):
            try:
                ref_gen._openmm_minimize_pdb_simple(str(tmp_raw), str(out_pdb))
            except Exception:
                shutil.copyfile(tmp_raw, out_pdb)
        else:
            shutil.copyfile(tmp_raw, out_pdb)

    return str(out_pdb.resolve())


def build_manifest_rows(
    row: dict[str, Any],
    sequences: list[str],
    pdb_paths: list[str],
    *,
    dataset: str,
    split_name: str,
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, (seq, pdb_path) in enumerate(zip(sequences, pdb_paths), start=1):
        item = {
            "dataset": dataset,
            "split_name": split_name,
            "target_id": str(row["sample_id"]),
            "method": method,
            "candidate_rank": rank,
            "receptor_pdb": str(row["receptor_pdb"]),
            "reference_peptide_pdb": str(row["peptide_pdb"]),
            "generated_peptide_pdb": pdb_path,
            "generated_sequence": seq,
            "hdock_result": "",
            "native_complex_pdb": "",
            "pred_complex_pdb": "",
            "_row_order": int(row["_row_order"]),
        }
        for extra_col in ["protein_id", "sample_dir", "family_id"]:
            if extra_col in row:
                item[extra_col] = row[extra_col]
        rows.append(item)
    return rows


def process_target(
    decoder,
    row: dict[str, Any],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    sample_id = str(row["sample_id"])
    target_dir = ensure_dir(Path(cfg["generation_root"]) / sample_id)

    sequences = sample_unconditional_sequences(
        decoder,
        num_candidates=cfg["num_candidates"],
        top_k=cfg["top_k"],
        max_len=cfg["max_len"],
        temperature=cfg["temperature"],
        oversample_factor=cfg["oversample_factor"],
    )

    fasta_records = []
    pdb_paths: list[str] = []
    for rank, seq in enumerate(sequences, start=1):
        cand_name = f"candidate_{rank:03d}"
        cand_dir = ensure_dir(target_dir / cand_name)
        fasta_records.append((f"{sample_id}_{cand_name}", seq))
        write_fasta([(f"{sample_id}_{cand_name}", seq)], cand_dir / "sequence.fasta")

        pdb_path = ""
        if cfg["save_pdb"] != "none":
            out_pdb = cand_dir / "peptide.pdb"
            try:
                pdb_path = save_unconditional_pdb(decoder, seq, out_pdb, cfg["save_pdb"])
            except Exception:
                pdb_path = ""
        pdb_paths.append(pdb_path)

    write_fasta(fasta_records, target_dir / "generated_sequences.fasta")
    return build_manifest_rows(
        row,
        sequences,
        pdb_paths,
        dataset=cfg["dataset"],
        split_name=cfg["split_name"],
        method=cfg["method"],
    )


def worker(worker_id: int, shard: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    set_seed(int(cfg["seed"]) + int(worker_id))
    device = torch.device(f"cuda:{worker_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    decoder = load_unconditional_decoder(device, cfg["ckpt_path"])
    rows: list[dict[str, Any]] = []
    for row in shard:
        rows.extend(process_target(decoder, row, cfg))

    shard_path = Path(cfg["shard_root"]) / f"manifest_rank{worker_id}.csv"
    pd.DataFrame(rows).to_csv(shard_path, index=False)


def run_generation(args: argparse.Namespace) -> None:
    if not args.input_csv:
        raise ValueError("`--input-csv` is required in generate mode.")
    if not args.outdir:
        raise ValueError("`--outdir` is required in generate mode.")

    df = load_targets(args.input_csv)
    split_name = args.split_name or Path(args.input_csv).stem
    outdir = ensure_dir(args.outdir)
    generation_root = ensure_dir(outdir / "generated" / args.method / split_name)
    shard_root = ensure_dir(outdir / "_manifest_shards" / args.method / split_name)

    records = df.to_dict("records")
    avail = torch.cuda.device_count()
    if avail == 0:
        world_size = 1
        shards = [records]
    else:
        world_size = min(max(1, args.num_gpus), avail, max(1, len(records)))
        indices = np.array_split(np.arange(len(records)), world_size)
        shards = [[records[i] for i in idx.tolist()] for idx in indices]

    cfg = {
        "ckpt_path": args.ckpt_path,
        "dataset": args.dataset,
        "split_name": split_name,
        "method": args.method,
        "num_candidates": args.num_candidates,
        "top_k": args.top_k,
        "max_len": args.max_len,
        "temperature": args.temperature,
        "oversample_factor": args.oversample_factor,
        "save_pdb": args.save_pdb,
        "seed": args.seed,
        "generation_root": str(generation_root),
        "shard_root": str(shard_root),
    }

    if world_size == 1:
        worker(0, shards[0], cfg)
    else:
        ctx = get_context("spawn")
        procs = []
        for rank in range(world_size):
            p = ctx.Process(target=worker, args=(rank, shards[rank], cfg), daemon=False)
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker {p.pid} failed with exit code {p.exitcode}")

    shard_files = sorted(shard_root.glob("manifest_rank*.csv"))
    if not shard_files:
        raise RuntimeError("No shard manifests were written.")

    manifest = pd.concat([pd.read_csv(path) for path in shard_files], ignore_index=True)
    manifest = manifest.sort_values(["_row_order", "candidate_rank"]).drop(columns=["_row_order"])

    manifest_path = outdir / f"{args.method}_{split_name}_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"Saved manifest: {manifest_path}")
    print(f"Generated candidates root: {generation_root}")


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_training(args)
        return
    run_generation(args)


if __name__ == "__main__":
    main()


'''

torchrun --nproc_per_node=4 /root/autodl-tmp/Peptide_3D/results/2_SOTA/unconditional/train_unconditional_esm3.py \
  --mode train \
  --train-root /root/autodl-tmp/train_data \
  --train-peptide-pdb /root/autodl-tmp/train_data/1A1M/peptide.pdb \
  --save-dir /root/autodl-tmp/Peptide_3D/log_unconditional \
  --epochs 5 \
  --batch-size 8 \
  --lr 2e-5 \
  --train-num-workers 4



'''
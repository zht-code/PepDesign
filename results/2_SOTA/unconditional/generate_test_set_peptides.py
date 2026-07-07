#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from train_unconditional_esm3 import (
    freeze_unconditional_unused_modules,
    load_unconditional_decoder,
    sample_unconditional_sequences,
    save_unconditional_pdb,
    set_seed,
)


DEFAULT_CKPT = "/root/autodl-tmp/Peptide_3D/log_unconditional/best_unconditional_esm3.pt"
DEFAULT_FAMILY_CSV = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/splits/family_level_test.csv"
DEFAULT_PROTEIN_CSV = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/splits/protein_level_test.csv"
DEFAULT_OUTROOT = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/unconditional"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Load the trained unconditional ESM3 baseline and generate 5 peptide PDBs "
            "for each test sample in the family-level and protein-level 2_SOTA test sets."
        )
    )
    ap.add_argument("--ckpt-path", default=DEFAULT_CKPT, help="Path to trained unconditional checkpoint.")
    ap.add_argument("--family-csv", default=DEFAULT_FAMILY_CSV, help="family_level_test split csv.")
    ap.add_argument("--protein-csv", default=DEFAULT_PROTEIN_CSV, help="protein_level_test split csv.")
    ap.add_argument("--out-root", default=DEFAULT_OUTROOT, help="Root directory for generated outputs.")
    ap.add_argument("--num-candidates", type=int, default=5, help="Peptides per test sample.")
    ap.add_argument("--top-k", type=int, default=12, help="Decoder top-k sampling.")
    ap.add_argument("--max-len", type=int, default=30, help="Maximum peptide length.")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    ap.add_argument("--oversample-factor", type=int, default=3, help="Oversampling factor before dedup.")
    ap.add_argument("--repetition-penalty", type=float, default=1.15, help="Penalty for previously used amino-acid tokens.")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=3, help="Disallow repeated ngrams of this size during decoding.")
    ap.add_argument("--max-consecutive-aa", type=int, default=2, help="Maximum allowed consecutive repeats of the same token.")
    ap.add_argument("--min-length", type=int, default=6, help="Minimum generated peptide length before EOS is allowed.")
    ap.add_argument("--save-pdb", choices=["mds", "helix", "none"], default="mds", help="Peptide structure mode.")
    ap.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use for generation.")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return ap.parse_args()


def load_targets(csv_path: str, split_name: str) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    required = ["protein_id", "sample_id", "receptor_pdb", "peptide_pdb"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    df = df.sort_values(["sample_id"]).reset_index(drop=True)
    records = df.to_dict("records")
    for i, row in enumerate(records):
        row["_row_order"] = i
        row["split_name"] = split_name
    return records


def save_sequences_fasta(sequences: list[str], out_path: Path, protein_id: str) -> None:
    lines = []
    for rank, seq in enumerate(sequences, start=1):
        lines.append(f">{protein_id}_candidate_{rank:02d}")
        lines.append(seq)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_for_target(decoder, row: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    protein_id = str(row["protein_id"])
    sample_id = str(row["sample_id"])
    split_name = str(row["split_name"])

    sample_dir = ensure_dir(Path(cfg["out_root"]) / split_name / sample_id)

    sequences = sample_unconditional_sequences(
        decoder,
        num_candidates=cfg["num_candidates"],
        top_k=cfg["top_k"],
        max_len=cfg["max_len"],
        temperature=cfg["temperature"],
        oversample_factor=cfg["oversample_factor"],
        repetition_penalty=cfg["repetition_penalty"],
        no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
        max_consecutive_aa=cfg["max_consecutive_aa"],
        min_length=cfg["min_length"],
    )
    save_sequences_fasta(sequences, sample_dir / "generated_sequences.fasta", sample_id)

    rows: list[dict[str, Any]] = []
    for rank, seq in enumerate(sequences, start=1):
        pdb_path = ""
        if cfg["save_pdb"] != "none":
            out_pdb = sample_dir / f"peptide_{rank:02d}.pdb"
            try:
                pdb_path = save_unconditional_pdb(decoder, seq, out_pdb, cfg["save_pdb"])
            except Exception:
                pdb_path = ""

        rows.append(
            {
                "split_name": split_name,
                "protein_id": protein_id,
                "sample_id": sample_id,
                "receptor_pdb": str(row["receptor_pdb"]),
                "reference_peptide_pdb": str(row["peptide_pdb"]),
                "candidate_rank": rank,
                "generated_sequence": seq,
                "generated_peptide_pdb": pdb_path,
                "_row_order": int(row["_row_order"]),
            }
        )
    return rows


def worker(worker_id: int, shard: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    set_seed(int(cfg["seed"]) + int(worker_id))
    device = torch.device(f"cuda:{worker_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    decoder = load_unconditional_decoder(device, cfg["ckpt_path"])
    freeze_unconditional_unused_modules(decoder)
    decoder.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in shard:
            rows.extend(generate_for_target(decoder, row, cfg))

    shard_path = Path(cfg["shard_root"]) / f"manifest_rank{worker_id}.csv"
    pd.DataFrame(rows).to_csv(shard_path, index=False)


def run_split(records: list[dict[str, Any]], cfg: dict[str, Any]) -> pd.DataFrame:
    shard_root = ensure_dir(cfg["shard_root"])
    avail = torch.cuda.device_count()
    if avail == 0:
        world_size = 1
        shards = [records]
    else:
        world_size = min(max(1, int(cfg["num_gpus"])), avail, max(1, len(records)))
        indices = np.array_split(np.arange(len(records)), world_size)
        shards = [[records[i] for i in idx.tolist()] for idx in indices]

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
        raise RuntimeError(f"No shard manifest was written under {shard_root}")

    df = pd.concat([pd.read_csv(path) for path in shard_files], ignore_index=True)
    df = df.sort_values(["_row_order", "candidate_rank"]).drop(columns=["_row_order"])
    return df


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_root = ensure_dir(args.out_root)

    split_specs = [
        ("family_level_test", args.family_csv),
        ("protein_level_test", args.protein_csv),
    ]

    all_frames: list[pd.DataFrame] = []
    for split_name, csv_path in split_specs:
        records = load_targets(csv_path, split_name)
        split_shard_root = ensure_dir(out_root / "_manifest_shards" / split_name)
        cfg = {
            "ckpt_path": args.ckpt_path,
            "out_root": str(out_root),
            "num_candidates": args.num_candidates,
            "top_k": args.top_k,
            "max_len": args.max_len,
            "temperature": args.temperature,
            "oversample_factor": args.oversample_factor,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_consecutive_aa": args.max_consecutive_aa,
            "min_length": args.min_length,
            "save_pdb": args.save_pdb,
            "num_gpus": args.num_gpus,
            "seed": args.seed,
            "shard_root": str(split_shard_root),
        }
        df = run_split(records, cfg)
        split_manifest = out_root / f"{split_name}_manifest.csv"
        df.to_csv(split_manifest, index=False)
        print(f"Saved {split_name} manifest: {split_manifest}")
        all_frames.append(df)

    merged = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    merged_manifest = out_root / "all_test_sets_manifest.csv"
    merged.to_csv(merged_manifest, index=False)
    print(f"Saved merged manifest: {merged_manifest}")
    print(f"Generated peptides root: {out_root}")


if __name__ == "__main__":
    main()


'''

python /root/autodl-tmp/Peptide_3D/results/2_SOTA/unconditional/generate_test_set_peptides.py \
  --num-gpus 4

'''
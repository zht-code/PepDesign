from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils_io import ensure_dir, set_seed, write_fasta, write_json
from utils_sequence import extract_sequence_from_pdb, mmseqs_cluster


def scan_dataset(dataset_root: str) -> pd.DataFrame:
    rows = []
    root = Path(dataset_root)
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        receptor = sub / "receptor.pdb"
        peptide = sub / "peptide.pdb"
        if not receptor.exists() or not peptide.exists():
            continue
        receptor_seq = extract_sequence_from_pdb(str(receptor))
        peptide_seq = extract_sequence_from_pdb(str(peptide))
        protein_id = sub.name.split("_")[0]
        rows.append({
            "sample_id": sub.name,
            "protein_id": protein_id,
            "sample_dir": str(sub),
            "receptor_pdb": str(receptor),
            "peptide_pdb": str(peptide),
            "receptor_sequence": receptor_seq,
            "peptide_sequence": peptide_seq,
        })
    if not rows:
        raise RuntimeError(f"No valid sample folders found under: {dataset_root}")
    df = pd.DataFrame(rows)
    return df


def build_protein_level_split(df: pd.DataFrame, test_size: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    proteins = df["protein_id"].drop_duplicates().tolist()
    rng.shuffle(proteins)

    test_rows = []
    selected_proteins = []
    count = 0
    for pid in proteins:
        sub = df[df["protein_id"] == pid]
        if count + len(sub) > test_size and count > 0:
            continue
        test_rows.append(sub)
        selected_proteins.append(pid)
        count += len(sub)
        if count >= test_size:
            break

    test_df = pd.concat(test_rows, ignore_index=True).head(test_size).copy()
    heldout_pids = set(test_df["protein_id"].unique())
    train_df = df[~df["protein_id"].isin(heldout_pids)].copy()
    return train_df, test_df


def assign_family_clusters_mmseqs(df: pd.DataFrame, outdir: str, min_seq_id: float = 0.4) -> pd.DataFrame:
    outdir = Path(outdir)
    fasta = outdir / "receptors.fasta"
    write_fasta(list(zip(df["sample_id"], df["receptor_sequence"])), fasta)
    tsv = mmseqs_cluster(str(fasta), str(outdir / "mmseqs_cluster"), min_seq_id=min_seq_id)

    cluster_map = {}
    with open(tsv, "r", encoding="utf-8") as f:
        for line in f:
            rep, member = line.strip().split("\t")[:2]
            cluster_map[member] = rep

    df = df.copy()
    df["family_id"] = df["sample_id"].map(cluster_map)
    missing = df["family_id"].isna()
    df.loc[missing, "family_id"] = df.loc[missing, "sample_id"]
    return df


def assign_family_clusters_fallback(df: pd.DataFrame) -> pd.DataFrame:
    # Conservative fallback: each protein is its own family.
    # This preserves correctness but is weaker than homologous-family stripping.
    df = df.copy()
    df["family_id"] = df["protein_id"]
    return df


def build_family_level_split(df: pd.DataFrame, test_size: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    families = df["family_id"].drop_duplicates().tolist()
    rng.shuffle(families)

    test_rows = []
    selected_families = []
    count = 0
    for fid in families:
        sub = df[df["family_id"] == fid]
        if count + len(sub) > test_size and count > 0:
            continue
        test_rows.append(sub)
        selected_families.append(fid)
        count += len(sub)
        if count >= test_size:
            break

    test_df = pd.concat(test_rows, ignore_index=True).head(test_size).copy()
    heldout_fids = set(test_df["family_id"].unique())
    train_df = df[~df["family_id"].isin(heldout_fids)].copy()
    return train_df, test_df


def save_split(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--protein-test-size", type=int, default=133)
    ap.add_argument("--family-test-size", type=int, default=133)
    ap.add_argument("--family-identity-threshold", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--family-split-seed",
        type=int,
        default=None,
        help="RNG seed for family-level train/test split. Default: seed+1 so family test differs from protein test when grouping differs.",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    metadata = scan_dataset(args.dataset_root)
    metadata.to_csv(outdir / "all_metadata.csv", index=False)

    # Protein-level split
    train_p, test_p = build_protein_level_split(metadata, args.protein_test_size, args.seed)
    train_p.to_csv(outdir / "protein_level_train.csv", index=False)
    test_p.to_csv(outdir / "protein_level_test.csv", index=False)

    # Family-level split
    try:
        metadata_fam = assign_family_clusters_mmseqs(
            metadata, str(outdir / "family_cluster"), min_seq_id=args.family_identity_threshold
        )
        family_mode = "mmseqs"
    except Exception as e:
        print(f"[WARN] MMseqs family clustering failed, fallback to protein-level family proxy. Error: {e}")
        metadata_fam = assign_family_clusters_fallback(metadata)
        family_mode = "fallback_protein_as_family"

    metadata_fam.to_csv(outdir / "all_metadata_with_family.csv", index=False)
    family_split_seed = args.family_split_seed if args.family_split_seed is not None else args.seed + 1
    train_f, test_f = build_family_level_split(metadata_fam, args.family_test_size, family_split_seed)
    train_f.to_csv(outdir / "family_level_train.csv", index=False)
    test_f.to_csv(outdir / "family_level_test.csv", index=False)

    summary = {
        "dataset_root": args.dataset_root,
        "family_assignment_mode": family_mode,
        "protein_level": {
            "train_n": int(len(train_p)),
            "test_n": int(len(test_p)),
            "test_unique_proteins": int(test_p["protein_id"].nunique()),
        },
        "family_level": {
            "train_n": int(len(train_f)),
            "test_n": int(len(test_f)),
            "test_unique_families": int(test_f["family_id"].nunique()) if "family_id" in test_f else None,
        },
        "family_identity_threshold": args.family_identity_threshold,
        "seed": args.seed,
        "family_split_seed": family_split_seed,
    }
    write_json(summary, outdir / "split_summary.json")
    print(summary)


if __name__ == "__main__":
    main()

'''

python /root/autodl-tmp/Peptide_3D/results/2_SOTA/00_build_splits.py \
  --dataset-root /root/autodl-tmp/train_data_augmentation_strong \
  --outdir /root/autodl-tmp/Peptide_3D/results/2_SOTA/splits \
  --protein-test-size 133 \
  --family-test-size 133 \
  --family-identity-threshold 0.4

'''
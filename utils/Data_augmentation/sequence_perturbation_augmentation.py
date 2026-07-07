#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequence Perturbation Augmentation for peptide-receptor pairs

Input:
    /root/autodl-tmp/train_data/
        ├── 1A1M/
        │   ├── receptor.pdb
        │   └── peptide.pdb
        ├── 1A1N/
        │   ├── receptor.pdb
        │   └── peptide.pdb
        ...

Output:
    /root/autodl-tmp/train_data_seq_aug/
        ├── original_1A1M/
        │   ├── receptor.pdb
        │   └── peptide.pdb
        ├── aug_000001_from_1A1M_sub/
        │   ├── receptor.pdb
        │   └── peptide.pdb
        ├── ...
        ├── metadata.csv
        └── summary.json

Strategy:
    - parse peptide sequence and CA coordinates from peptide.pdb
    - perform sequence perturbation:
        substitution / insertion / deletion
    - build approximate CA coordinates for perturbed sequence
    - write a simplified peptide.pdb containing CA atoms only
    - copy receptor.pdb unchanged
"""

import os
import json
import math
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


# =========================================================
# Amino acid mapping
# =========================================================

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K"
}

AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
}

AA_LIST = list("ARNDCQEGHILKMFPSTWYV")


# =========================================================
# Basic I/O
# =========================================================

def scan_dataset(input_root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(input_root.iterdir()):
        if not d.is_dir():
            continue

        receptor = d / "receptor.pdb"
        peptide = d / "peptide.pdb"

        if receptor.exists() and peptide.exists():
            rows.append({
                "sample_id": d.name,
                "sample_dir": str(d),
                "receptor_pdb": str(receptor),
                "peptide_pdb": str(peptide),
            })

    return pd.DataFrame(rows)


# =========================================================
# PDB parsing
# =========================================================

def parse_peptide_ca(pdb_path: Path):
    """
    Parse CA trace from peptide PDB.

    Returns:
        seq: str
        coords: np.ndarray [N, 3]
        chain_id: str
    """
    residues = []
    coords = []
    seen = set()
    chain_id_default = "A"

    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            res_name = line[17:20].strip()
            chain_id = line[21].strip() if line[21].strip() else "A"
            chain_id_default = chain_id

            try:
                res_seq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue

            key = (chain_id, res_seq)
            if key in seen:
                continue
            seen.add(key)

            aa1 = AA3_TO_AA1.get(res_name, "G")
            residues.append(aa1)
            coords.append([x, y, z])

    seq = "".join(residues)
    coords = np.array(coords, dtype=np.float32)

    return seq, coords, chain_id_default


# =========================================================
# Sequence perturbation
# =========================================================

def random_aa(exclude: Optional[str] = None, rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    choices = AA_LIST
    if exclude is not None:
        choices = [aa for aa in choices if aa != exclude]
    return rng.choice(choices)


def mutate_substitution(seq: str, rng: random.Random, max_subs: int = 2) -> Tuple[str, Dict]:
    if len(seq) == 0:
        return seq, {"op": "substitution", "positions": []}

    n_subs = min(len(seq), rng.randint(1, max_subs))
    positions = sorted(rng.sample(range(len(seq)), n_subs))

    seq_list = list(seq)
    detail = []
    for pos in positions:
        old = seq_list[pos]
        new = random_aa(exclude=old, rng=rng)
        seq_list[pos] = new
        detail.append({"pos": pos, "old": old, "new": new})

    return "".join(seq_list), {"op": "substitution", "positions": detail}


def mutate_deletion(seq: str, rng: random.Random) -> Tuple[str, Dict]:
    if len(seq) <= 2:
        return seq, {"op": "deletion", "positions": []}

    pos = rng.randrange(len(seq))
    new_seq = seq[:pos] + seq[pos+1:]
    return new_seq, {"op": "deletion", "positions": [{"pos": pos, "old": seq[pos]}]}


def mutate_insertion(seq: str, rng: random.Random) -> Tuple[str, Dict]:
    pos = rng.randrange(len(seq) + 1)
    aa = random_aa(rng=rng)
    new_seq = seq[:pos] + aa + seq[pos:]
    return new_seq, {"op": "insertion", "positions": [{"pos": pos, "new": aa}]}


def choose_mutation(seq: str, rng: random.Random,
                    p_sub: float, p_del: float, p_ins: float) -> Tuple[str, Dict]:
    ops = ["sub", "del", "ins"]
    probs = np.array([p_sub, p_del, p_ins], dtype=float)
    probs = probs / probs.sum()

    op = rng.choices(ops, weights=probs, k=1)[0]

    if op == "sub":
        return mutate_substitution(seq, rng)
    elif op == "del":
        return mutate_deletion(seq, rng)
    else:
        return mutate_insertion(seq, rng)


# =========================================================
# Coordinate construction
# =========================================================

def unit_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return v / n


def random_small_vector(scale: float) -> np.ndarray:
    v = np.random.normal(0.0, scale, size=(3,)).astype(np.float32)
    return v


def estimate_step(coords: np.ndarray) -> float:
    if len(coords) >= 2:
        d = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        return float(np.mean(d))
    return 3.8  # peptide CA-CA rough spacing


def build_perturbed_coords(orig_seq: str,
                           orig_coords: np.ndarray,
                           new_seq: str,
                           mutation_info: Dict,
                           jitter_sigma: float = 0.35) -> np.ndarray:
    """
    Construct approximate CA coordinates for perturbed sequence.

    Rules:
    - substitution: keep same coordinates, then add small jitter
    - deletion: remove corresponding residue coordinate
    - insertion: insert one coordinate by interpolating neighbor positions
    """
    op = mutation_info["op"]

    if len(orig_coords) == 0:
        # fallback: build a random coil-like CA chain
        coords = []
        current = np.zeros(3, dtype=np.float32)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        for _ in new_seq:
            current = current + direction * 3.8 + random_small_vector(0.5)
            coords.append(current.copy())
            direction = unit_vector(direction + random_small_vector(0.3))
        return np.array(coords, dtype=np.float32)

    if op == "substitution":
        new_coords = orig_coords.copy()

    elif op == "deletion":
        if not mutation_info.get("positions"):
            # no position information: fallback to keeping original chain
            new_coords = orig_coords.copy()
        else:
            pos = mutation_info["positions"][0]["pos"]
            mask = np.ones(len(orig_coords), dtype=bool)
            if 0 <= pos < len(mask):
                mask[pos] = False
            new_coords = orig_coords[mask].copy()

    elif op == "insertion":
        if not mutation_info.get("positions"):
            # no position information: append a random placement at end
            step = estimate_step(orig_coords)
            if len(orig_coords) >= 2:
                direction = unit_vector(orig_coords[-1] - orig_coords[-2])
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            inserted = orig_coords[-1] + direction * step + random_small_vector(0.3)
            new_coords = np.vstack([orig_coords, inserted[None, :]])
        else:
            pos = mutation_info["positions"][0]["pos"]
            step = estimate_step(orig_coords)

        if pos == 0:
            if len(orig_coords) >= 2:
                direction = unit_vector(orig_coords[0] - orig_coords[1])
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            inserted = orig_coords[0] + direction * step + random_small_vector(0.3)
            new_coords = np.vstack([inserted[None, :], orig_coords])

        elif pos == len(orig_coords):
            if len(orig_coords) >= 2:
                direction = unit_vector(orig_coords[-1] - orig_coords[-2])
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            inserted = orig_coords[-1] + direction * step + random_small_vector(0.3)
            new_coords = np.vstack([orig_coords, inserted[None, :]])

        else:
            left = orig_coords[pos - 1]
            right = orig_coords[pos]
            middle = 0.5 * (left + right)
            perp = random_small_vector(0.6)
            inserted = middle + perp
            new_coords = np.vstack([
                orig_coords[:pos],
                inserted[None, :],
                orig_coords[pos:]
            ])
    else:
        raise ValueError(f"Unknown mutation op: {op}")

    # final mild jitter
    new_coords = new_coords + np.random.normal(
        loc=0.0,
        scale=jitter_sigma,
        size=new_coords.shape
    ).astype(np.float32)

    return new_coords


# =========================================================
# PDB writing
# =========================================================

def write_ca_only_pdb(seq: str,
                      coords: np.ndarray,
                      out_path: Path,
                      chain_id: str = "A"):
    """
    Write a simplified peptide PDB containing only CA atoms.
    """
    assert len(seq) == len(coords), f"seq len {len(seq)} != coords len {len(coords)}"

    lines = []
    atom_serial = 1

    for i, (aa1, xyz) in enumerate(zip(seq, coords), start=1):
        aa3 = AA1_TO_AA3.get(aa1, "GLY")
        x, y, z = xyz.tolist()

        line = (
            f"ATOM  {atom_serial:5d}  CA  {aa3:>3s} {chain_id:1s}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00           C"
        )
        lines.append(line)
        atom_serial += 1

    lines.append("TER")
    lines.append("END")

    with open(out_path, "w") as f:
        for line in lines:
            f.write(line + "\n")


# =========================================================
# Save utilities
# =========================================================

def save_pair(receptor_src: Path,
              peptide_seq: str,
              peptide_coords: np.ndarray,
              chain_id: str,
              out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(receptor_src, out_dir / "receptor.pdb")
    write_ca_only_pdb(peptide_seq, peptide_coords, out_dir / "peptide.pdb", chain_id=chain_id)


def copy_original_dataset(df: pd.DataFrame, output_root: Path):
    for _, row in df.iterrows():
        src_dir = Path(row["sample_dir"])
        out_dir = output_root / f"original_{row['sample_id']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_dir / "receptor.pdb", out_dir / "receptor.pdb")
        shutil.copy2(src_dir / "peptide.pdb", out_dir / "peptide.pdb")


# =========================================================
# Main augmentation logic
# =========================================================

def augment_dataset(
    input_root: Path,
    output_root: Path,
    target_total_pairs: int = 56172,
    copy_original: bool = True,
    random_seed: int = 2025,
    p_sub: float = 0.60,
    p_del: float = 0.20,
    p_ins: float = 0.20,
    jitter_sigma: float = 0.35,
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    rng = random.Random(random_seed)

    output_root.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(input_root)
    if len(df) == 0:
        raise ValueError(f"No valid samples found in {input_root}")

    print(f"[Info] Found {len(df)} original pairs.")

    parsed = []
    for _, row in df.iterrows():
        seq, coords, chain_id = parse_peptide_ca(Path(row["peptide_pdb"]))
        if len(seq) == 0 or len(coords) == 0:
            continue

        parsed.append({
            "sample_id": row["sample_id"],
            "sample_dir": row["sample_dir"],
            "receptor_pdb": row["receptor_pdb"],
            "peptide_pdb": row["peptide_pdb"],
            "sequence": seq,
            "coords": coords,
            "chain_id": chain_id,
        })

    parsed_df = pd.DataFrame(parsed)
    if len(parsed_df) == 0:
        raise ValueError("No parsable peptide CA traces found.")

    metadata = []

    if copy_original:
        print("[Info] Copying original pairs...")
        copy_original_dataset(df, output_root)
        for _, row in parsed_df.iterrows():
            metadata.append({
                "out_id": f"original_{row['sample_id']}",
                "type": "original",
                "base_sample": row["sample_id"],
                "mutation_type": "none",
                "orig_seq": row["sequence"],
                "aug_seq": row["sequence"],
                "mutation_info": json.dumps({}),
                "receptor_source": row["receptor_pdb"],
                "peptide_source": row["peptide_pdb"],
            })

    current_total = len(metadata)
    print(f"[Info] Current total after originals: {current_total}")

    if current_total >= target_total_pairs:
        print("[Info] Target total already reached.")
        return

    need_aug = target_total_pairs - current_total
    print(f"[Info] Need to generate {need_aug} augmented pairs.")

    aug_counter = 0
    num_samples = len(parsed_df)
    base_idx = 0

    while current_total < target_total_pairs:
        row = parsed_df.iloc[base_idx % num_samples]
        base_idx += 1

        sample_id = row["sample_id"]
        receptor_pdb = Path(row["receptor_pdb"])
        orig_seq = row["sequence"]
        orig_coords = row["coords"]
        chain_id = row["chain_id"]

        aug_seq, mutation_info = choose_mutation(
            orig_seq, rng,
            p_sub=p_sub,
            p_del=p_del,
            p_ins=p_ins
        )

        # avoid degenerate outputs
        if len(aug_seq) < 2:
            continue

        aug_coords = build_perturbed_coords(
            orig_seq=orig_seq,
            orig_coords=orig_coords,
            new_seq=aug_seq,
            mutation_info=mutation_info,
            jitter_sigma=jitter_sigma
        )

        if len(aug_seq) != len(aug_coords):
            print(f"[Warn] Skip inconsistent augmented sample for {sample_id}")
            continue

        op_name = mutation_info["op"]
        aug_counter += 1
        # 使用唯一的输出目录名，避免同一 sample_id 被多次覆盖
        # 例如: 1A1M_aug_000001, 1A1M_aug_000002, ...
        out_name = f"{sample_id}_aug_{aug_counter:06d}"
        out_dir = output_root / out_name

        try:
            save_pair(
                receptor_src=receptor_pdb,
                peptide_seq=aug_seq,
                peptide_coords=aug_coords,
                chain_id=chain_id,
                out_dir=out_dir
            )
        except Exception as e:
            print(f"[Warn] Failed to save {out_name}: {e}")
            continue

        metadata.append({
            "out_id": out_name,
            "type": "augmented",
            "base_sample": sample_id,
            "mutation_type": op_name,
            "orig_seq": orig_seq,
            "aug_seq": aug_seq,
            "mutation_info": json.dumps(mutation_info, ensure_ascii=False),
            "receptor_source": str(receptor_pdb),
            "peptide_source": row["peptide_pdb"],
        })

        current_total += 1
        if current_total % 1000 == 0:
            print(f"[Info] Generated total pairs: {current_total}/{target_total_pairs}")

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(output_root / "metadata.csv", index=False)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "original_pairs_found": int(len(df)),
        "parsable_pairs": int(len(parsed_df)),
        "target_total_pairs": int(target_total_pairs),
        "final_total_pairs": int(len(meta_df)),
        "num_original_copied": int((meta_df["type"] == "original").sum()),
        "num_augmented_generated": int((meta_df["type"] == "augmented").sum()),
        "random_seed": int(random_seed),
        "mutation_probabilities": {
            "substitution": p_sub,
            "deletion": p_del,
            "insertion": p_ins,
        },
        "jitter_sigma": float(jitter_sigma),
    }

    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[Done] Sequence perturbation augmentation finished.")
    print(json.dumps(summary, indent=2))


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        type=str,
        default="/root/autodl-tmp/train_data",
        help="Root folder of original dataset"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/root/autodl-tmp/train_data_seq_aug",
        help="Output folder of augmented dataset"
    )
    parser.add_argument(
        "--target_total_pairs",
        type=int,
        default=56172,
        help="Total number of pairs in output"
    )
    parser.add_argument(
        "--copy_original",
        action="store_true",
        help="If set, copy original pairs first, then augment until target_total_pairs"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=2025
    )
    parser.add_argument(
        "--p_sub",
        type=float,
        default=0.60,
        help="Probability of substitution augmentation"
    )
    parser.add_argument(
        "--p_del",
        type=float,
        default=0.20,
        help="Probability of deletion augmentation"
    )
    parser.add_argument(
        "--p_ins",
        type=float,
        default=0.20,
        help="Probability of insertion augmentation"
    )
    parser.add_argument(
        "--jitter_sigma",
        type=float,
        default=0.35,
        help="Coordinate jitter sigma in Angstrom"
    )

    args = parser.parse_args()

    augment_dataset(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        target_total_pairs=args.target_total_pairs,
        copy_original=args.copy_original,
        random_seed=args.random_seed,
        p_sub=args.p_sub,
        p_del=args.p_del,
        p_ins=args.p_ins,
        jitter_sigma=args.jitter_sigma,
    )


if __name__ == "__main__":
    main()





'''

情况 1：输出总数 56172，包含原始 9244 对
python sequence_perturbation_augmentation.py \
  --input_root /root/autodl-tmp/train_data \
  --output_root /root/autodl-tmp/train_data_seq_aug \
  --target_total_pairs 56172 \
  --copy_original \
  --random_seed 2025 \
  --p_sub 0.60 \
  --p_del 0.20 \
  --p_ins 0.20 \
  --jitter_sigma 0.35

情况 2：只生成 56172 个增强样本，不包含原始
python /root/autodl-tmp/Peptide_3D/utils/Data_augmentation/sequence_perturbation_augmentation.py \
  --input_root /root/autodl-tmp/train_data \
  --output_root /root/autodl-tmp/train_data_seq_aug_perturbation \
  --target_total_pairs 56172

'''
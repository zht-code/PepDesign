#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Neighbor Augmentation for peptide-receptor pairs

Input structure:
    /root/autodl-tmp/train_data/
        ├── 1A1M/
        │   ├── receptor.pdb
        │   └── peptide.pdb
        ├── 1A1N/
        │   ├── receptor.pdb
        │   └── peptide.pdb
        ...

Output structure:
    /root/autodl-tmp/train_data_augmented/
        ├── original_1A1M/
        │   ├── receptor.pdb
        │   └── peptide.pdb
        ├── aug_000001_from_1A1M_nb_2BCD/
        │   ├── receptor.pdb
        │   └── peptide.pdb
        ├── ...
        ├── metadata.csv
        └── summary.json

Method:
    1) build features from peptide sequence + simple geometry
    2) fit nearest neighbors
    3) for each sample, randomly choose one neighbor from top-k
    4) align neighbor peptide onto current peptide frame
    5) add small coordinate jitter / rotation
    6) save augmented pair
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
from sklearn.neighbors import NearestNeighbors


# =========================
# Config
# =========================

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K"
}
AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}


# =========================
# PDB parsing
# =========================

def parse_pdb_atoms(pdb_path: Path, atom_name_filter: Optional[str] = None):
    """
    Minimal PDB parser.
    Returns list of dict:
        {
            "line": original line,
            "atom_name": str,
            "res_name": str,
            "chain_id": str,
            "res_seq": int,
            "x": float,
            "y": float,
            "z": float
        }
    """
    atoms = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom_name = line[12:16].strip()
            if atom_name_filter is not None and atom_name != atom_name_filter:
                continue

            res_name = line[17:20].strip()
            chain_id = line[21].strip() if line[21].strip() else "A"

            try:
                res_seq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue

            atoms.append({
                "line": line.rstrip("\n"),
                "atom_name": atom_name,
                "res_name": res_name,
                "chain_id": chain_id,
                "res_seq": res_seq,
                "x": x,
                "y": y,
                "z": z,
            })
    return atoms


def extract_sequence_from_pdb(pdb_path: Path) -> str:
    """
    Extract 1-letter peptide sequence from CA trace.
    """
    atoms = parse_pdb_atoms(pdb_path, atom_name_filter="CA")
    residues = []
    seen = set()
    for a in atoms:
        key = (a["chain_id"], a["res_seq"])
        if key in seen:
            continue
        seen.add(key)
        aa1 = AA3_TO_AA1.get(a["res_name"], "X")
        residues.append(aa1)
    return "".join(residues)


def extract_ca_coords(pdb_path: Path) -> np.ndarray:
    """
    Return CA coordinates of shape [N, 3].
    """
    atoms = parse_pdb_atoms(pdb_path, atom_name_filter="CA")
    coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=np.float32)
    return coords


def read_all_atom_lines(pdb_path: Path) -> List[str]:
    with open(pdb_path, "r") as f:
        return [line.rstrip("\n") for line in f]


# =========================
# Feature extraction
# =========================

def aa_composition(seq: str) -> np.ndarray:
    vec = np.zeros(len(AA_ORDER), dtype=np.float32)
    valid = 0
    for aa in seq:
        if aa in AA_INDEX:
            vec[AA_INDEX[aa]] += 1.0
            valid += 1
    if valid > 0:
        vec /= valid
    return vec


def peptide_geometry_features(ca: np.ndarray) -> np.ndarray:
    """
    Simple geometric descriptors from CA coordinates.
    """
    if len(ca) == 0:
        return np.zeros(6, dtype=np.float32)

    centroid = ca.mean(axis=0)
    centered = ca - centroid
    rg = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))

    if len(ca) >= 2:
        end_to_end = np.linalg.norm(ca[0] - ca[-1])
    else:
        end_to_end = 0.0

    if len(ca) >= 2:
        step_lengths = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
        mean_step = float(np.mean(step_lengths))
        std_step = float(np.std(step_lengths))
    else:
        mean_step = 0.0
        std_step = 0.0

    mins = ca.min(axis=0)
    maxs = ca.max(axis=0)
    bbox = maxs - mins
    bbox_mean = float(np.mean(bbox))

    feats = np.array([
        len(ca),
        rg,
        end_to_end,
        mean_step,
        std_step,
        bbox_mean,
    ], dtype=np.float32)
    return feats


def build_pair_feature(peptide_pdb: Path) -> Tuple[np.ndarray, str, np.ndarray]:
    seq = extract_sequence_from_pdb(peptide_pdb)
    ca = extract_ca_coords(peptide_pdb)

    seq_feat = aa_composition(seq)
    geo_feat = peptide_geometry_features(ca)

    if len(seq) > 0:
        length_feat = np.array([len(seq)], dtype=np.float32)
    else:
        length_feat = np.array([0.0], dtype=np.float32)

    feat = np.concatenate([length_feat, seq_feat, geo_feat], axis=0)
    return feat, seq, ca


# =========================
# Geometry ops
# =========================

def safe_centroid(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(3, dtype=np.float32)
    return x.mean(axis=0)


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find R, t so that transformed P aligns to Q:
        P_aligned = P @ R + t

    P, Q: [N, 3], N must be equal
    """
    assert P.shape == Q.shape and P.shape[1] == 3

    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = Q.mean(axis=0) - P.mean(axis=0) @ R
    return R, t


def random_rotation_matrix(max_angle_deg: float, rng: random.Random) -> np.ndarray:
    angle = math.radians(rng.uniform(-max_angle_deg, max_angle_deg))
    axis = np.array([rng.random(), rng.random(), rng.random()], dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c

    R = np.array([
        [x * x * C + c,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c]
    ], dtype=np.float32)
    return R


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return coords @ R + t


def add_jitter(coords: np.ndarray,
               coord_sigma: float,
               max_rotate_deg: float,
               rng: random.Random) -> np.ndarray:
    """
    Small local perturbation.
    """
    if len(coords) == 0:
        return coords.copy()

    center = coords.mean(axis=0, keepdims=True)
    centered = coords - center

    Rj = random_rotation_matrix(max_rotate_deg, rng)
    rotated = centered @ Rj
    noise = np.random.normal(loc=0.0, scale=coord_sigma, size=coords.shape).astype(np.float32)

    return rotated + center + noise


# =========================
# PDB coordinate rewriting
# =========================

def replace_pdb_coords(pdb_lines: List[str], new_coords: np.ndarray) -> List[str]:
    """
    Replace coordinates in all ATOM/HETATM lines using new_coords in order.
    Number of atom lines must match len(new_coords).
    """
    out = []
    idx = 0
    for line in pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            if idx >= len(new_coords):
                raise ValueError("new_coords shorter than number of atom lines")
            x, y, z = new_coords[idx]
            new_line = (
                line[:30]
                + f"{x:8.3f}{y:8.3f}{z:8.3f}"
                + line[54:]
            )
            out.append(new_line)
            idx += 1
        else:
            out.append(line)

    if idx != len(new_coords):
        raise ValueError("new_coords longer than number of atom lines")
    return out


def extract_all_atom_coords(pdb_path: Path) -> np.ndarray:
    coords = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
                except ValueError:
                    continue
    return np.array(coords, dtype=np.float32)


def transform_entire_pdb(pdb_path: Path,
                         R: np.ndarray,
                         t: np.ndarray,
                         coord_sigma: float,
                         max_rotate_deg: float,
                         rng: random.Random) -> List[str]:
    """
    Apply rigid transform + slight jitter to all atoms in peptide pdb.
    """
    lines = read_all_atom_lines(pdb_path)
    coords = extract_all_atom_coords(pdb_path)

    transformed = apply_transform(coords, R, t)
    transformed = add_jitter(transformed, coord_sigma=coord_sigma, max_rotate_deg=max_rotate_deg, rng=rng)

    new_lines = replace_pdb_coords(lines, transformed)
    return new_lines


# =========================
# Dataset scan
# =========================

def scan_dataset(input_root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(input_root.iterdir()):
        if not d.is_dir():
            continue

        receptor = d / "receptor.pdb"
        peptide = d / "peptide.pdb"
        if not receptor.exists() or not peptide.exists():
            continue

        rows.append({
            "sample_id": d.name,
            "sample_dir": str(d),
            "receptor_pdb": str(receptor),
            "peptide_pdb": str(peptide),
        })

    df = pd.DataFrame(rows)
    return df


# =========================
# Augmentation core
# =========================

def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for _, row in df.iterrows():
        peptide_pdb = Path(row["peptide_pdb"])
        feat, seq, ca = build_pair_feature(peptide_pdb)
        feats.append({
            "sample_id": row["sample_id"],
            "feature": feat,
            "sequence": seq,
            "peptide_len": len(seq),
            "ca_coords": ca,
        })
    feat_df = pd.DataFrame(feats)
    return feat_df


def standardize_features(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X - mean) / std


def select_neighbor(indices_row: np.ndarray,
                    self_idx: int,
                    rng: random.Random) -> int:
    """
    Randomly choose one neighbor from nearest-neighbor list excluding self.
    """
    candidates = [i for i in indices_row if i != self_idx]
    if not candidates:
        return self_idx
    return rng.choice(candidates)


def prepare_alignment(source_ca: np.ndarray,
                      target_ca: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align source peptide onto target peptide using CA atoms.
    If lengths differ, use min length prefix for Kabsch.
    """
    if len(source_ca) == 0 or len(target_ca) == 0:
        return np.eye(3, dtype=np.float32), safe_centroid(target_ca)

    n = min(len(source_ca), len(target_ca))
    P = source_ca[:n]
    Q = target_ca[:n]

    R, t = kabsch_align(P, Q)
    return R, t


def save_pair(receptor_src: Path,
              peptide_lines: List[str],
              out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(receptor_src, out_dir / "receptor.pdb")
    with open(out_dir / "peptide.pdb", "w") as f:
        for line in peptide_lines:
            f.write(line + "\n")


def copy_original_dataset(df: pd.DataFrame, output_root: Path):
    """
    Optional: copy original dataset into output_root/original_xxx
    """
    for _, row in df.iterrows():
        src_dir = Path(row["sample_dir"])
        out_dir = output_root / f"original_{row['sample_id']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_dir / "receptor.pdb", out_dir / "receptor.pdb")
        shutil.copy2(src_dir / "peptide.pdb", out_dir / "peptide.pdb")


def augment_dataset(
    input_root: Path,
    output_root: Path,
    target_total_pairs: int = 56172,
    knn_k: int = 16,
    coord_sigma: float = 0.35,
    max_rotate_deg: float = 8.0,
    random_seed: int = 2025,
    copy_original: bool = True,
):
    rng = random.Random(random_seed)
    np.random.seed(random_seed)

    output_root.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(input_root)
    if len(df) == 0:
        raise ValueError(f"No valid samples found in {input_root}")

    print(f"[Info] Found {len(df)} original pairs.")

    feat_df = build_feature_table(df)
    merged = df.merge(feat_df, on="sample_id", how="inner")

    # Build features
    X = np.stack(merged["feature"].values, axis=0)
    X = standardize_features(X)

    # Fit nearest neighbors
    n_neighbors = min(knn_k + 1, len(merged))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Copy originals if needed
    metadata = []

    if copy_original:
        print("[Info] Copying original pairs...")
        copy_original_dataset(df, output_root)
        for _, row in merged.iterrows():
            metadata.append({
                "out_id": f"original_{row['sample_id']}",
                "type": "original",
                "base_sample": row["sample_id"],
                "neighbor_sample": row["sample_id"],
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

    num_samples = len(merged)
    aug_counter = 0
    base_pointer = 0

    while current_total < target_total_pairs:
        i = base_pointer % num_samples
        base_pointer += 1

        base_row = merged.iloc[i]
        nb_idx = select_neighbor(indices[i], self_idx=i, rng=rng)
        nb_row = merged.iloc[nb_idx]

        base_id = base_row["sample_id"]
        nb_id = nb_row["sample_id"]

        base_receptor = Path(base_row["receptor_pdb"])
        base_peptide = Path(base_row["peptide_pdb"])
        nb_peptide = Path(nb_row["peptide_pdb"])

        base_ca = base_row["ca_coords"]
        nb_ca = nb_row["ca_coords"]

        # Align neighbor peptide onto base peptide
        R, t = prepare_alignment(nb_ca, base_ca)

        try:
            new_peptide_lines = transform_entire_pdb(
                nb_peptide,
                R=R,
                t=t,
                coord_sigma=coord_sigma,
                max_rotate_deg=max_rotate_deg,
                rng=rng
            )
        except Exception as e:
            print(f"[Warn] Skip {base_id} <- {nb_id} due to transform error: {e}")
            continue

        aug_counter += 1
        out_name = f"{base_id}_nb_{nb_id}"
        out_dir = output_root / out_name

        try:
            save_pair(base_receptor, new_peptide_lines, out_dir)
        except Exception as e:
            print(f"[Warn] Failed to save {out_name}: {e}")
            continue

        metadata.append({
            "out_id": out_name,
            "type": "augmented",
            "base_sample": base_id,
            "neighbor_sample": nb_id,
            "receptor_source": str(base_receptor),
            "peptide_source": str(nb_peptide),
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
        "target_total_pairs": int(target_total_pairs),
        "final_total_pairs": int(len(meta_df)),
        "num_original_copied": int((meta_df["type"] == "original").sum()),
        "num_augmented_generated": int((meta_df["type"] == "augmented").sum()),
        "knn_k": int(knn_k),
        "coord_sigma": float(coord_sigma),
        "max_rotate_deg": float(max_rotate_deg),
        "random_seed": int(random_seed),
    }
    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[Done] Augmentation finished.")
    print(json.dumps(summary, indent=2))


# =========================
# Main
# =========================

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
        default="/root/autodl-tmp/train_data_augmented",
        help="Output folder of augmented dataset"
    )
    parser.add_argument(
        "--target_total_pairs",
        type=int,
        default=56172,
        help="Total number of pairs wanted in output (including originals if copy_original=True)"
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=16,
        help="Number of nearest neighbors to build candidate set"
    )
    parser.add_argument(
        "--coord_sigma",
        type=float,
        default=0.35,
        help="Gaussian coordinate noise std in Angstrom"
    )
    parser.add_argument(
        "--max_rotate_deg",
        type=float,
        default=8.0,
        help="Maximum random rotation angle in degrees for local jitter"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=2025
    )
    parser.add_argument(
        "--copy_original",
        action="store_true",
        help="If set, copy original pairs to output and then append augmentations until target_total_pairs"
    )

    args = parser.parse_args()

    augment_dataset(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        target_total_pairs=args.target_total_pairs,
        knn_k=args.knn_k,
        coord_sigma=args.coord_sigma,
        max_rotate_deg=args.max_rotate_deg,
        random_seed=args.random_seed,
        copy_original=args.copy_original,
    )


if __name__ == "__main__":
    main()





'''
情况 1：输出总数 56172，包含原始样本
python random_neighbor_augmentation.py \
  --input_root /root/autodl-tmp/train_data \
  --output_root /root/autodl-tmp/train_data_augmented \
  --target_total_pairs 56172 \
  --knn_k 16 \
  --coord_sigma 0.35 \
  --max_rotate_deg 8.0 \
  --random_seed 2025 \
  --copy_original

  

情况 2：只想生成 56172 个增强样本，不含原始
python /root/autodl-tmp/Peptide_3D/utils/Data_augmentation/random_neighbor_augmentation.py \
  --input_root /root/autodl-tmp/train_data \
  --output_root /root/autodl-tmp/train_data_augmented_random_neighbor \
  --target_total_pairs 56172

'''
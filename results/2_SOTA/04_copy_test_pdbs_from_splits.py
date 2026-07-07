#!/usr/bin/env python3
"""Copy receptor.pdb / peptide.pdb for split test rows into flat test-set folders."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def resolve_sample_dir(sample_dir: str, sample_id: str, source_root: Path) -> Path:
    """Prefer CSV sample_dir; fall back to source_root / sample_id."""
    p = Path(sample_dir.strip())
    if p.is_dir():
        return p.resolve()
    sid = sample_id.strip()
    alt = (source_root / sid).resolve()
    if alt.is_dir():
        return alt
    raise FileNotFoundError(f"No sample folder for {sid!r}: tried {p} and {alt}")


def copy_split(csv_path: Path, dest_root: Path, source_root: Path) -> None:
    dest_root = dest_root.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: empty or no header")
        required = {"sample_id", "sample_dir"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        n_ok = 0
        for row in reader:
            sid = str(row["sample_id"]).strip()
            src_dir = resolve_sample_dir(str(row["sample_dir"]), sid, source_root)
            rec = src_dir / "receptor.pdb"
            pep = src_dir / "peptide.pdb"
            if not rec.is_file():
                raise FileNotFoundError(f"Missing receptor: {rec}")
            if not pep.is_file():
                raise FileNotFoundError(f"Missing peptide: {pep}")

            out_dir = dest_root / sid
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rec, out_dir / "receptor.pdb")
            shutil.copy2(pep, out_dir / "peptide.pdb")
            n_ok += 1

    print(f"{csv_path.name}: copied {n_ok} samples -> {dest_root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    ap.add_argument(
        "--splits-dir",
        type=Path,
        default=base / "splits",
        help="Directory containing protein_level_test.csv and family_level_test.csv",
    )
    ap.add_argument(
        "--source-root",
        type=Path,
        default=Path("/root/autodl-tmp/train_data_augmentation_strong"),
        help="If sample_dir in CSV is missing, try source_root / sample_id",
    )
    ap.add_argument(
        "--protein-outdir",
        type=Path,
        default=base / "protein_level_test",
        help="Output root for protein-level test copies",
    )
    ap.add_argument(
        "--family-outdir",
        type=Path,
        default=base / "family_level_test",
        help="Output root for family-level test copies",
    )
    args = ap.parse_args()

    splits = args.splits_dir.resolve()
    protein_csv = splits / "protein_level_test.csv"
    family_csv = splits / "family_level_test.csv"
    if not protein_csv.is_file():
        raise FileNotFoundError(protein_csv)
    if not family_csv.is_file():
        raise FileNotFoundError(family_csv)

    source_root = args.source_root.resolve()
    copy_split(protein_csv, args.protein_outdir, source_root)
    copy_split(family_csv, args.family_outdir, source_root)


if __name__ == "__main__":
    main()


'''

python /root/autodl-tmp/Peptide_3D/results/2_SOTA/04_copy_test_pdbs_from_splits.py


'''
#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


BASE = Path("/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline").resolve()
RAW = BASE / "raw_results"
CACHE = BASE / "cache"


def count_rows(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    df = pd.read_csv(path)
    return len(df), int(df["target_id"].nunique()) if "target_id" in df.columns else 0


def main() -> int:
    print("Baseline progress")
    print("")
    for method in ["proteingenerator", "bindcraft"]:
        mdir = RAW / method
        print(f"[{method}]")
        if not mdir.is_dir():
            print("missing")
            print("")
            continue
        files = sorted(mdir.glob("samples_*.csv"))
        print(f"sample_tables={len(files)}")
        for path in files[:6]:
            rows, targets = count_rows(path)
            print(f"{path.name}: rows={rows} targets={targets}")
        print("")

    rfd_csv = RAW / "rfdiffusion_mpnn_sequences.csv"
    rows, targets = count_rows(rfd_csv)
    recovered = sum(1 for _ in (CACHE / "recovered_structures" / "rfdiffusion").rglob("*.pdb"))
    hf_cache_bytes = sum(p.stat().st_size for p in (CACHE / "hf_cache").rglob("*") if p.is_file()) if (CACHE / "hf_cache").exists() else 0
    print("[rfdiffusion]")
    print(f"mapping_rows={rows} targets={targets}")
    print(f"recovered_pdbs={recovered}")
    print(f"hf_cache_gb={hf_cache_bytes / (1024**3):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

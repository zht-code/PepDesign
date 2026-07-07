from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.split_csv)
    out = pd.DataFrame({
        "dataset": ["internal"] * len(df),
        "split_name": [Path(args.split_csv).stem] * len(df),
        "target_id": df["sample_id"],
        "method": [args.method] * len(df),
        "candidate_rank": [1] * len(df),
        "receptor_pdb": df["receptor_pdb"],
        "reference_peptide_pdb": df["peptide_pdb"],
        "generated_peptide_pdb": ["" for _ in range(len(df))],
        "generated_sequence": ["" for _ in range(len(df))],
        "hdock_result": ["" for _ in range(len(df))],
        "native_complex_pdb": ["" for _ in range(len(df))],
        "pred_complex_pdb": ["" for _ in range(len(df))],
    })
    out.to_csv(args.out_csv, index=False)
    print(f"Template saved to {args.out_csv}")


if __name__ == "__main__":
    main()

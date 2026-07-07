from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
import esm


LOGGER = logging.getLogger("run_esmfold_plddt_batch")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def mean_bfactor_from_pdb_text(pdb_text: str) -> float:
    values = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = line[76:78].strip().upper()
        if element == "H":
            continue
        try:
            values.append(float(line[60:66].strip()))
        except ValueError:
            continue
    return float(sum(values) / len(values)) if values else float("nan")


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(description="Batch ESMFold pLDDT calculation.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows_in = list(csv.DictReader(handle))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Loading ESMFold model on %s ...", device)
    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    if device == "cuda":
        model = model.cuda()
    if args.chunk_size > 0:
        model.set_chunk_size(args.chunk_size)

    rows = []
    for idx, row in enumerate(rows_in, start=1):
        sequence_id = str(row["sequence_id"])
        sequence = str(row["sequence"]).strip()
        plddt = float("nan")
        if sequence:
            try:
                with torch.no_grad():
                    pdb_text = model.infer_pdb(sequence)
                plddt = mean_bfactor_from_pdb_text(pdb_text)
            except Exception as exc:
                LOGGER.warning("ESMFold failed for %s: %s", sequence_id, exc)
            finally:
                if device == "cuda":
                    torch.cuda.empty_cache()
        rows.append({"sequence_id": sequence_id, "plddt": plddt})
        if idx % 50 == 0:
            LOGGER.info("Processed %d / %d sequences", idx, len(rows_in))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id", "plddt"])
        writer.writeheader()
        writer.writerows(rows)
    finite_count = sum(1 for row in rows if str(row["plddt"]).lower() != "nan")
    LOGGER.info("Saved ESMFold pLDDT results to %s (%d/%d finite)", output_csv, finite_count, len(rows))


if __name__ == "__main__":
    main()

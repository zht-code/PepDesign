#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import logging


BASE_DIR = Path("/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline").resolve()
CACHE_DIR = BASE_DIR / "cache"
HF_CACHE_DIR = CACHE_DIR / "hf_cache"
TORCH_CACHE_DIR = CACHE_DIR / "torch_cache"
TMP_DIR = BASE_DIR / "tmp"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TMPDIR", str(TMP_DIR))
os.environ.setdefault("TEMP", str(TMP_DIR))
os.environ.setdefault("TMP", str(TMP_DIR))
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))

LOGGER = logging.getLogger("recover_rfdiffusion_structures")


def read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    seq_chunks: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        records.append((header, "".join(seq_chunks)))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch recover RFdiffusion peptide structures from top5 FASTA sequences.")
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--recovered-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    setup_logging()
    args = parse_args()
    for path in [HF_CACHE_DIR, TORCH_CACHE_DIR, TMP_DIR, Path(args.recovered_root)]:
        path.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(args.index_csv)
    rfd = index_df[index_df["method"] == "rfdiffusion"].copy()
    if rfd.empty:
        pd.DataFrame(columns=["target_id", "candidate_id", "pdb_path", "mpnn_sequence", "status", "notes"]).to_csv(
            args.output_csv, index=False
        )
        return 0

    rfd["source_score_numeric"] = pd.to_numeric(rfd["source_score"], errors="coerce")
    best = (
        rfd.sort_values(["target_id", "source_score_numeric", "candidate_id"], na_position="last")
        .groupby("target_id", as_index=False)
        .first()
    )

    existing_map = {}
    output_csv = Path(args.output_csv)
    if args.skip_existing and output_csv.is_file():
        old = pd.read_csv(output_csv)
        for row in old.to_dict(orient="records"):
            existing_map[str(row["candidate_id"])] = row

    tasks = []
    passthrough_rows = []
    for row in best.to_dict(orient="records"):
        candidate_id = str(row["candidate_id"])
        target_id = str(row["target_id"]).lower()
        fasta_path = Path(str(row["original_pdb_path"]))
        recovered_pdb = Path(args.recovered_root) / target_id / f"{candidate_id}.pdb"
        if candidate_id in existing_map and recovered_pdb.is_file():
            passthrough_rows.append(existing_map[candidate_id])
            continue
        seq = ""
        if fasta_path.is_file():
            records = read_fasta_records(fasta_path)
            if records:
                seq = records[-1][1].strip()
        if not seq:
            passthrough_rows.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "pdb_path": str(recovered_pdb),
                    "mpnn_sequence": "",
                    "status": "missing_sequence",
                    "notes": "No usable designed sequence parsed from FASTA.",
                }
            )
            continue
        tasks.append((target_id, candidate_id, seq, recovered_pdb))

    if not tasks:
        pd.DataFrame(passthrough_rows).to_csv(output_csv, index=False)
        return 0

    import torch
    from transformers import EsmForProteinFolding

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    LOGGER.info("loading EsmForProteinFolding on %s for %d RFdiffusion targets", device, len(tasks))
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.eval()
    if use_cuda:
        model = model.cuda()
    LOGGER.info("model loaded; starting recovery batches batch_size=%d", batch_size)

    rows = list(passthrough_rows)
    batch_size = max(1, int(args.batch_size))

    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        seqs = [seq for _, _, seq, _ in batch]
        try:
            with torch.no_grad():
                pdb_texts = model.infer_pdbs(seqs)
            for (target_id, candidate_id, seq, recovered_pdb), pdb_text in zip(batch, pdb_texts):
                recovered_pdb.parent.mkdir(parents=True, exist_ok=True)
                recovered_pdb.write_text(pdb_text, encoding="utf-8")
                rows.append(
                    {
                        "target_id": target_id,
                        "candidate_id": candidate_id,
                        "pdb_path": str(recovered_pdb),
                        "mpnn_sequence": seq,
                        "status": "esmfold_recovered",
                        "notes": "Recovered with batched transformers EsmForProteinFolding.infer_pdbs.",
                    }
                )
        except Exception as exc:
            for target_id, candidate_id, seq, recovered_pdb in batch:
                rows.append(
                    {
                        "target_id": target_id,
                        "candidate_id": candidate_id,
                        "pdb_path": str(recovered_pdb),
                        "mpnn_sequence": seq,
                        "status": "recovery_failed",
                        "notes": str(exc),
                    }
                )
        if use_cuda:
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        LOGGER.info("recovered %d / %d RFdiffusion targets", min(start + len(batch), len(tasks)), len(tasks))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_2_SOTA_DIR = PROJECT_ROOT / "results" / "2_SOTA"
if str(RESULTS_2_SOTA_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_2_SOTA_DIR))

from utils_sequence import sequence_identity  # noqa: E402
from utils.structure_metrics import extract_peptide_sequence  # noqa: E402


LOGGER = logging.getLogger(__name__)


def write_fasta(records: Iterable[Tuple[str, str]], fasta_path: Path) -> None:
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w", encoding="utf-8") as handle:
        for record_id, sequence in records:
            handle.write(f">{record_id}\n{sequence}\n")


def build_train_fasta_from_split(split_csv: Path, output_fasta: Path) -> Path:
    df = pd.read_csv(split_csv)
    seq_col = "peptide_sequence" if "peptide_sequence" in df.columns else "generated_sequence"
    id_col = "sample_id" if "sample_id" in df.columns else df.columns[0]
    records = [
        (str(row[id_col]), str(row[seq_col]).strip())
        for _, row in df.iterrows()
        if str(row.get(seq_col, "")).strip() and str(row.get(seq_col, "")).strip().lower() != "nan"
    ]
    write_fasta(records, output_fasta)
    return output_fasta


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.exists() and path_str.startswith("/autodl-tmp/"):
        alt = Path("/root") / path_str.lstrip("/")
        if alt.exists():
            path = alt
    return path


def _read_first_fasta_sequence(fasta_path: Path) -> str:
    parts: List[str] = []
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if parts:
                    break
                continue
            parts.append(line)
    return "".join(parts)


def build_train_fasta_from_root(train_root: Path, output_fasta: Path) -> Path:
    records: List[Tuple[str, str]] = []
    for sample_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
        seq = ""
        fasta_path = sample_dir / "peptide.fasta"
        peptide_pdb = sample_dir / "peptide.pdb"
        if fasta_path.is_file():
            try:
                seq = _read_first_fasta_sequence(fasta_path)
            except OSError:
                seq = ""
        if not seq and peptide_pdb.is_file():
            try:
                seq = extract_peptide_sequence(str(peptide_pdb))
            except Exception:
                seq = ""
        if seq:
            records.append((sample_dir.name, seq))

    if not records:
        raise RuntimeError(f"No training peptide sequences found under {train_root}")

    write_fasta(records, output_fasta)
    LOGGER.info("Built train FASTA from %s with %d peptide sequences.", train_root, len(records))
    return output_fasta


def prepare_train_fastas(
    split_files: Dict[str, Path],
    output_dir: Path,
    train_fasta: Optional[str] = None,
    train_root: Optional[str] = None,
) -> Dict[str, Path]:
    train_dir = output_dir / "_train_fastas"
    train_dir.mkdir(parents=True, exist_ok=True)

    if train_fasta:
        train_path = _resolve_path(train_fasta)
        if not train_path.exists():
            raise FileNotFoundError(f"Provided --train-fasta does not exist: {train_fasta}")
        return {dataset: train_path.resolve() for dataset in split_files}

    if train_root:
        root_path = _resolve_path(train_root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"Provided --train-root does not exist: {train_root}")
        shared_fasta = build_train_fasta_from_root(root_path, train_dir / "train_data_train.fasta")
        return {dataset: shared_fasta.resolve() for dataset in split_files}

    return {
        dataset: build_train_fasta_from_split(split_path, train_dir / f"{dataset}_train.fasta")
        for dataset, split_path in split_files.items()
    }


def _run_mmseqs_easy_search(
    mmseqs_path: Path,
    query_fasta: Path,
    target_fasta: Path,
    work_dir: Path,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    result_tsv = work_dir / "nearest.tsv"
    tmp_dir = work_dir / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    cmd = [
        str(mmseqs_path),
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(result_tsv),
        str(tmp_dir),
        "--max-seqs",
        "1",
        "--format-output",
        "query,target,pident",
        "--min-seq-id",
        "0",
        "-e",
        "1000000",
        "-s",
        "1",
        "--comp-bias-corr",
        "0",
        "--mask",
        "0",
        "--min-ungapped-score",
        "0",
        "--exhaustive-search",
        "1",
    ]
    LOGGER.info("Running MMseqs2: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result_tsv


def _fallback_similarity(
    query_records: Sequence[Tuple[str, str]],
    train_sequences: Sequence[str],
) -> Dict[str, float]:
    LOGGER.warning("MMseqs2 failed; falling back to Python sequence identity. This can be slow.")
    unique_train = [seq for seq in dict.fromkeys(train_sequences) if seq]
    out: Dict[str, float] = {}
    for query_id, query_sequence in query_records:
        if not query_sequence:
            out[query_id] = float("nan")
            continue
        best = 0.0
        for train_sequence in unique_train:
            similarity = sequence_identity(query_sequence, train_sequence)
            if similarity == similarity and similarity > best:
                best = similarity
        out[query_id] = best
    return out


def compute_train_similarity(
    dataset_to_queries: Dict[str, Sequence[Tuple[str, str]]],
    dataset_to_train_fasta: Dict[str, Path],
    mmseqs: str,
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    mmseqs_path = Path(mmseqs).expanduser()
    if not mmseqs_path.exists() and mmseqs.startswith("/autodl-fs/"):
        mmseqs_path = Path("/root") / mmseqs.lstrip("/")
    if not mmseqs_path.exists():
        raise FileNotFoundError(f"MMseqs2 binary not found: {mmseqs}")

    results: Dict[str, Dict[str, float]] = {}
    for dataset, query_records in dataset_to_queries.items():
        valid_queries = [(query_id, sequence) for query_id, sequence in query_records if sequence]
        dataset_result: Dict[str, float] = {
            query_id: (0.0 if sequence else float("nan")) for query_id, sequence in query_records
        }
        if not valid_queries:
            results[dataset] = dataset_result
            continue

        query_fasta = output_dir / "_mmseqs" / dataset / "queries.fasta"
        write_fasta(valid_queries, query_fasta)

        train_fasta = dataset_to_train_fasta[dataset]
        train_sequences = []
        try:
            with open(train_fasta, "r", encoding="utf-8") as handle:
                current = []
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if current:
                            train_sequences.append("".join(current))
                            current = []
                    else:
                        current.append(line)
                if current:
                    train_sequences.append("".join(current))
        except OSError:
            train_sequences = []

        try:
            nearest_tsv = _run_mmseqs_easy_search(
                mmseqs_path=mmseqs_path,
                query_fasta=query_fasta,
                target_fasta=train_fasta,
                work_dir=output_dir / "_mmseqs" / dataset,
            )
            with open(nearest_tsv, "r", encoding="utf-8") as handle:
                reader = csv.reader(handle, delimiter="\t")
                for query_id, _target_id, pident, *_rest in reader:
                    dataset_result[query_id] = float(pident) / 100.0
        except Exception as exc:
            LOGGER.warning("MMseqs2 search failed for %s: %s", dataset, exc)
            dataset_result.update(_fallback_similarity(valid_queries, train_sequences))

        unresolved = sum(1 for query_id, _ in valid_queries if dataset_result.get(query_id, float("nan")) == 0.0)
        LOGGER.info("MMseqs2 similarity finished for %s: %d/%d queries had no detectable hit and were left at 0.0.",
                    dataset, unresolved, len(valid_queries))
        results[dataset] = dataset_result

    return results


def novelty_from_similarity(similarity: float, threshold: float) -> float:
    if similarity != similarity:
        return float("nan")
    return float(similarity < threshold)

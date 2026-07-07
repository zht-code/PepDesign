from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from Bio import pairwise2
from utils_io import write_fasta

try:
    from metrics_structure import load_structure
except ImportError:  # pragma: no cover
    load_structure = None  # type: ignore


THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "MSE":"M","SEC":"U","PYL":"O"
}


def extract_sequence_from_structure_file(path: str, prefer_longest_chain: bool = True) -> str:
    if load_structure is None:
        raise RuntimeError("metrics_structure.load_structure unavailable")
    structure = load_structure(path)
    return _sequence_from_structure(structure, prefer_longest_chain=prefer_longest_chain)


def _sequence_from_structure(structure, prefer_longest_chain: bool = True) -> str:

    chain_seqs = []
    for model in structure:
        for chain in model:
            seq_chars = []
            for res in chain:
                if res.id[0] != " ":
                    continue
                resname = res.resname.upper()
                aa = THREE_TO_ONE.get(resname)
                if aa is not None:
                    seq_chars.append(aa)
            if seq_chars:
                chain_seqs.append(("".join(seq_chars), chain.id))
        break

    if not chain_seqs:
        return ""

    if prefer_longest_chain:
        chain_seqs.sort(key=lambda x: len(x[0]), reverse=True)
    return chain_seqs[0][0]


def extract_sequence_from_pdb(pdb_path: str, prefer_longest_chain: bool = True) -> str:
    return extract_sequence_from_structure_file(pdb_path, prefer_longest_chain=prefer_longest_chain)


def sequence_identity(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return float("nan")
    aln = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True, score_only=False)
    if not aln:
        return 0.0
    best = aln[0]
    a1, a2 = best.seqA, best.seqB
    matches = sum(c1 == c2 for c1, c2 in zip(a1, a2) if c1 != "-" and c2 != "-")
    denom = max(len(seq1), len(seq2))
    return matches / denom if denom else 0.0


def novelty_against_train(seq: str, train_sequences: Sequence[str], threshold: float = 0.8) -> Tuple[float, int]:
    if not train_sequences:
        return float("nan"), 1
    sims = [sequence_identity(seq, tr) for tr in train_sequences if tr]
    max_sim = max(sims) if sims else 0.0
    is_novel = int(max_sim < threshold)
    return max_sim, is_novel


def write_receptor_fasta_from_metadata(metadata_csv: str, fasta_out: str) -> None:
    import pandas as pd
    df = pd.read_csv(metadata_csv)
    records = list(zip(df["sample_id"].astype(str), df["receptor_sequence"].astype(str)))
    write_fasta(records, fasta_out)


def mmseqs_cluster(fasta_path: str, outdir: str, min_seq_id: float = 0.4) -> str:
    outdir = str(Path(outdir))
    tmpdir = str(Path(outdir) / "tmp")
    cluster_db = str(Path(outdir) / "clusterDB")
    tsv_out = str(Path(outdir) / "clusters.tsv")

    Path(outdir).mkdir(parents=True, exist_ok=True)

    cmd_cluster = [
        "/root/autodl-fs/mmseqs-linux-gpu/mmseqs/bin/mmseqs", "easy-cluster", fasta_path, cluster_db, tmpdir,
        "--min-seq-id", str(min_seq_id), "-c", "0.8", "--cov-mode", "1"
    ]
    subprocess.run(cmd_cluster, check=True)

    cmd_tsv = [
        "/root/autodl-fs/mmseqs-linux-gpu/mmseqs/bin/mmseqs", "createtsv", fasta_path, fasta_path, cluster_db, tsv_out
    ]
    subprocess.run(cmd_tsv, check=True)
    return tsv_out


def _unlink_mmseqs_db_stem(outdir: Path, stem: str) -> None:
    """Remove MMseqs DB files for a given path stem (e.g. resultDB); search refuses to overwrite."""
    for p in outdir.glob(f"{stem}*"):
        if p.is_file():
            p.unlink(missing_ok=True)


def mmseqs_search(query_fasta: str, target_fasta: str, outdir: str) -> str:
    outdir = str(Path(outdir))
    tmpdir = str(Path(outdir) / "tmp")
    qdb = str(Path(outdir) / "queryDB")
    tdb = str(Path(outdir) / "targetDB")
    rdb = str(Path(outdir) / "resultDB")
    tsv = str(Path(outdir) / "search.tsv")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    subprocess.run(["/root/autodl-fs/mmseqs-linux-gpu/mmseqs/bin/mmseqs", "createdb", query_fasta, qdb], check=True)
    subprocess.run(["/root/autodl-fs/mmseqs-linux-gpu/mmseqs/bin/mmseqs", "createdb", target_fasta, tdb], check=True)
    _unlink_mmseqs_db_stem(Path(outdir), Path(rdb).name)
    subprocess.run(["/root/autodl-fs/mmseqs-linux-gpu/mmseqs/bin/mmseqs", "search", qdb, tdb, rdb, tmpdir], check=True)
    subprocess.run(["/root/autodl-fs/mmseqs-linux-gpu/mmseqs/bin/mmseqs", "convertalis", qdb, tdb, rdb, tsv], check=True)
    return tsv

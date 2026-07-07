from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils_sequence import novelty_against_train, sequence_identity


def repetition_rate(seq: str, k: int = 3) -> float:
    if not seq or len(seq) < k:
        return 0.0
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    if not kmers:
        return 0.0
    c = Counter(kmers)
    repeated = sum(v for v in c.values() if v > 1)
    return repeated / len(kmers)


def pairwise_dedup_rate(seqs: Sequence[str], threshold: float = 0.95) -> float:
    if len(seqs) < 2:
        return 0.0
    dup = 0
    total = 0
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            total += 1
            sim = sequence_identity(seqs[i], seqs[j])
            if sim >= threshold:
                dup += 1
    return dup / total if total else 0.0


class PerplexityScorer:
    """
    Generic adapter. Replace `score_sequence_logprob` with your actual decoder scorer.

    Expected behavior:
        returns token-average negative log-likelihood for a sequence
    """

    def __init__(self, model_name_or_path: Optional[str] = None):
        self.model_name_or_path = model_name_or_path

    def score_sequence_logprob(self, seq: str) -> float:
        # Placeholder. Replace with real ESM3 decoder or HF model scoring.
        # Returning NaN is safer than a fake perplexity.
        return float("nan")

    def perplexity(self, seq: str) -> float:
        nll = self.score_sequence_logprob(seq)
        if nll is None or (isinstance(nll, float) and math.isnan(nll)):
            return float("nan")
        return float(math.exp(nll))


def add_generation_metrics(
    df: pd.DataFrame,
    train_sequences: Sequence[str],
    novelty_threshold: float = 0.8,
    perplexity_scorer: Optional[PerplexityScorer] = None,
) -> pd.DataFrame:
    df = df.copy()

    df["repetition_rate"] = df["generated_sequence"].astype(str).apply(repetition_rate)

    max_sim_list = []
    novel_list = []
    for seq in df["generated_sequence"].astype(str):
        max_sim, is_novel = novelty_against_train(seq, train_sequences, novelty_threshold)
        max_sim_list.append(max_sim)
        novel_list.append(is_novel)
    df["max_train_similarity"] = max_sim_list
    df["is_novel"] = novel_list

    if perplexity_scorer is None:
        df["perplexity"] = float("nan")
    else:
        df["perplexity"] = df["generated_sequence"].astype(str).apply(perplexity_scorer.perplexity)

    return df

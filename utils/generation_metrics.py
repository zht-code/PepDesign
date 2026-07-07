from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Optional

from utils.structure_metrics import extract_peptide_sequence


LOGGER = logging.getLogger(__name__)


def repetition_rate(sequence: str, k: int = 3) -> float:
    """
    Classic repeated n-gram ratio: among all overlapping k-mers,
    count how many belong to a k-mer type that appears more than once.
    """
    if not sequence or len(sequence) < k:
        return 0.0
    kmers = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
    counts = Counter(kmers)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(kmers) if kmers else 0.0


class PerplexityScorer:
    """
    Placeholder interface for project LM scorers.
    The current repository does not expose a ready-to-call peptide perplexity scorer.
    """

    def __init__(self, model_name_or_path: Optional[str] = None):
        self.model_name_or_path = model_name_or_path
        self.available = False

    def score_sequence_logprob(self, sequence: str) -> float:
        return float("nan")

    def perplexity(self, sequence: str) -> float:
        nll = self.score_sequence_logprob(sequence)
        if nll is None or (isinstance(nll, float) and math.isnan(nll)):
            return float("nan")
        return float(math.exp(nll))


def resolve_candidate_sequence(
    pdb_path: Optional[str],
    reference_peptide_pdb: Optional[str],
    existing_sequence: Optional[str] = None,
) -> str:
    if existing_sequence:
        sequence = str(existing_sequence).strip()
        if sequence and sequence.lower() != "nan":
            return sequence

    if not pdb_path:
        return ""
    try:
        return extract_peptide_sequence(pdb_path, reference_peptide_path=reference_peptide_pdb)
    except Exception as exc:
        LOGGER.warning("Failed to extract peptide sequence from %s: %s", pdb_path, exc)
        return ""

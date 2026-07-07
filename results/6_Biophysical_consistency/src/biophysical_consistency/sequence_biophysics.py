from __future__ import annotations

import re
from typing import Sequence

# Kyte & Doolittle (1982)
KD: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
    "X": 0.0,
}


def normalize_sequence(seq: str | None) -> str:
    if not seq:
        return ""
    s = seq.strip().upper()
    s = re.sub(r"[^A-Z]", "", s)
    return s


def gravy(seq: str) -> float | None:
    s = normalize_sequence(seq)
    if not s:
        return None
    vals = [KD.get(aa, 0.0) for aa in s]
    return sum(vals) / len(vals)


def net_charge_ph7(seq: str) -> float | None:
    s = normalize_sequence(seq)
    if not s:
        return None
    pos = s.count("K") + s.count("R") + 0.5 * s.count("H")
    neg = s.count("D") + s.count("E")
    return pos - neg


def aromatic_fraction(seq: str) -> float | None:
    s = normalize_sequence(seq)
    if not s:
        return None
    arom = sum(s.count(x) for x in ("F", "W", "Y"))
    return arom / len(s)


def aliphatic_index(seq: str) -> float | None:
    """Ikai (1980) aliphatic index scaled to percentage-like contribution (0–100 scale proxy)."""
    s = normalize_sequence(seq)
    n = len(s)
    if n == 0:
        return None
    x_ala = s.count("A")
    x_val = s.count("V")
    x2_ile_leu = 2.9 * (s.count("I") + s.count("L"))
    return (x_ala + 2.9 * x_val + x2_ile_leu) / n * 100


def aggregation_hotspot_max_score(seq: str, window: int = 5) -> float | None:
    """Sliding window hydrophobic / beta-branched density (0–1). Proxy for aggregation-prone patches."""
    s = normalize_sequence(seq)
    if len(s) < window:
        return None
    hydro_set = set("AILMFWYV")
    best = 0.0
    for i in range(0, len(s) - window + 1):
        w = s[i : i + window]
        score = sum(1 for c in w if c in hydro_set) / window
        best = max(best, score)
    return best


def proline_fraction(seq: str) -> float | None:
    s = normalize_sequence(seq)
    if not s:
        return None
    return s.count("P") / len(s)


def cysteine_fraction(seq: str) -> float | None:
    s = normalize_sequence(seq)
    if not s:
        return None
    return s.count("C") / len(s)


def summarize_sequence(seq: str | None) -> dict[str, float | None]:
    s = normalize_sequence(seq)
    return {
        "seq_len": float(len(s)) if s else None,
        "gravy": gravy(s),
        "net_charge_ph7": net_charge_ph7(s),
        "aromatic_fraction": aromatic_fraction(s),
        "aliphatic_index": aliphatic_index(s),
        "aggregation_hotspot_max": aggregation_hotspot_max_score(s),
        "proline_fraction": proline_fraction(s),
        "cysteine_fraction": cysteine_fraction(s),
    }


def kd_for_resname(resname: str) -> float:
    r = resname.strip().upper()
    if len(r) == 3:
        three = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
        }
        one = three.get(r, "X")
        return KD.get(one, 0.0)
    if len(r) == 1:
        return KD.get(r, 0.0)
    return 0.0


def is_polar_resname(resname: str) -> bool:
    return kd_for_resname(resname) < 0

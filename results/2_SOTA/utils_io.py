from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj, path: os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_fasta(records: Iterable[Tuple[str, str]], path: os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rid, seq in records:
            f.write(f">{rid}\n{seq}\n")


def safe_mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

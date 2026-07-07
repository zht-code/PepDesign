"""解析可选的 docking 汇总表（TSV/CSV）。"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def load_docking_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        log.warning("未找到 docking 汇总文件，将仅使用 candidate_peptides.csv 中的对接列（若有）。")
        return pd.DataFrame()
    try:
        if path.suffix.lower() in (".tsv", ".txt"):
            return pd.read_csv(path, sep="\t")
        return pd.read_csv(path)
    except Exception as exc:
        log.warning("读取 docking 文件失败 %s: %s", path, exc)
        return pd.DataFrame()

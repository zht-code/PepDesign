"""第一层辅助：多肽理化性质与 binding 综合分。"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
except ImportError:
    ProteinAnalysis = None  # type: ignore


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def sanitize_sequence(seq: str) -> str:
    s = re.sub(r"\s+", "", str(seq).upper())
    return "".join(c for c in s if c in VALID_AA)


def compute_sequence_features(sequence: str) -> dict[str, float]:
    seq = sanitize_sequence(sequence)
    if not seq:
        return {
            "length": 0.0,
            "molecular_weight": 0.0,
            "net_charge": 0.0,
            "hydrophobic_ratio": 0.0,
            "instability_index": 0.0,
            "gravy": 0.0,
            "aromaticity": 0.0,
        }

    L = len(seq)
    if ProteinAnalysis is None:
        log.warning("Biopython ProtParam 不可用，使用近似理化特征。")
        # 近似
        kd = {
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
        }
        gravy = sum(kd.get(a, 0) for a in seq) / L
        hydrophobic = sum(1 for a in seq if kd.get(a, 0) > 0) / L
        charge_p = sum(seq.count(x) for x in "KRH")
        charge_n = sum(seq.count(x) for x in "DE")
        mw_approx = sum(
            {"A": 89.1, "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2, "E": 147.1, "Q": 146.2, "G": 75.1, "H": 155.2, "I": 131.2, "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1, "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1}.get(a, 110.0)
            for a in seq
        ) - (L - 1) * 18.015  # 粗略脱水
        aromaticity = sum(seq.count(x) for x in "FWY") / L
        return {
            "length": float(L),
            "molecular_weight": float(mw_approx),
            "net_charge": float(charge_p - charge_n),
            "hydrophobic_ratio": float(hydrophobic),
            "instability_index": 40.0,  # placeholder
            "gravy": float(gravy),
            "aromaticity": float(aromaticity),
        }

    pa = ProteinAnalysis(seq)
    return {
        "length": float(L),
        "molecular_weight": float(pa.molecular_weight()),
        "net_charge": float(pa.charge_at_pH(7.0)),
        "hydrophobic_ratio": sum(1 for a in seq if a in "AILMVFWYP") / L,
        "instability_index": float(pa.instability_index()),
        "gravy": float(pa.gravy()),
        "aromaticity": float(pa.aromaticity()),
    }


def _minmax_inv(series: pd.Series) -> pd.Series:
    """数值越小越好 -> 得分越高。"""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    norm = (hi - s) / (hi - lo)
    return norm.clip(0, 1)


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def compute_binding_score_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    docking_score / mmgbsa 越低越好；distance_to_PD1_interface 越小越好。
    缺失列时用默认 0.5。
    """
    out = df.copy()
    parts = []
    if "docking_score" in out.columns:
        parts.append(_minmax_inv(out["docking_score"]).rename("docking_component"))
    if "mmgbsa_score" in out.columns:
        parts.append(_minmax_inv(out["mmgbsa_score"]).rename("mmgbsa_component"))
    if "distance_to_PD1_interface" in out.columns:
        parts.append(_minmax_inv(out["distance_to_PD1_interface"]).rename("dist_component"))

    if parts:
        mat = pd.concat(parts, axis=1)
        out["binding_score"] = mat.mean(axis=1, skipna=True).fillna(0.5)
    else:
        log.warning("无 docking 列，binding_score 置为 0.5")
        out["binding_score"] = 0.5

    return out


def run_layer1(
    peptide_csv: Path,
    docking_extra: pd.DataFrame | None,
    out_csv: Path,
) -> pd.DataFrame:
    df = pd.read_csv(peptide_csv)
    records = []
    for _, row in df.iterrows():
        feats = compute_sequence_features(str(row.get("sequence", "")))
        rec = {
            "peptide_id": row.get("peptide_id"),
            "sequence": sanitize_sequence(str(row.get("sequence", ""))),
            "source": row.get("source", ""),
            "predicted_structure_path": row.get("predicted_structure_path", ""),
            **feats,
        }
        records.append(rec)
    res = pd.DataFrame(records)
    for col in ("docking_score", "mmgbsa_score", "distance_to_PD1_interface", "interface_residues"):
        if col in df.columns:
            res[col] = df[col].values

    if docking_extra is not None and not docking_extra.empty:
        key = "peptide_id"
        if key in docking_extra.columns:
            res = res.merge(docking_extra, on=key, how="left", suffixes=("", "_dock"))
        else:
            log.warning("docking 表缺少 peptide_id，忽略合并")

    res = compute_binding_score_table(res)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    log.info("第一层输出: %s", out_csv)
    return res

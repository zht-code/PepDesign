"""第二层：PD-1/PD-L1 通路与免疫激活评分（单细胞）。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ranksums

log = logging.getLogger(__name__)

PATHWAY_GENES = {
    "human": {
        "PD1_PDL1_core": [
            "PDCD1",
            "CD274",
            "PDCD1LG2",
            "PTPN11",
            "PTPN6",
            "LCK",
            "ZAP70",
            "CD3D",
            "CD3E",
            "CD3G",
            "LAT",
            "NFATC1",
            "JUN",
            "FOS",
            "IL2",
            "IFNG",
        ],
        "TCR_activation": ["CD3D", "CD3E", "CD3G", "LCK", "ZAP70", "LAT"],
        "NFAT": ["NFATC1", "NFATC2", "NFATC3"],
        "IFNG_response": ["IFNG", "STAT1", "IRF1", "CXCL9", "CXCL10"],
        "exhaustion": ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT"],
    },
    "mouse": {
        "PD1_PDL1_core": [
            "Pdcd1",
            "Cd274",
            "Pdcd1lg2",
            "Ptpn11",
            "Ptpn6",
            "Lck",
            "Zap70",
            "Cd3d",
            "Cd3e",
            "Cd3g",
            "Lat",
            "Nfatc1",
            "Jun",
            "Fos",
            "Il2",
            "Ifng",
        ],
        "TCR_activation": ["Cd3d", "Cd3e", "Cd3g", "Lck", "Zap70", "Lat"],
        "NFAT": ["Nfatc1", "Nfatc2", "Nfatc3"],
        "IFNG_response": ["Ifng", "Stat1", "Irf1", "Cxcl9", "Cxcl10"],
        "exhaustion": ["Pdcd1", "Ctla4", "Lag3", "Havcr2", "Tigit"],
    },
}


def _present_genes(adata, genes: list[str]) -> list[str]:
    vmap = {str(v).upper(): str(v) for v in adata.var_names}
    return [vmap[g.upper()] for g in genes if g.upper() in vmap]


def score_pathways(adata, config: dict[str, Any]) -> ad.AnnData:
    species = config.get("species", "human")
    sets = PATHWAY_GENES.get(species, PATHWAY_GENES["human"])
    adata = adata.copy()
    use_raw = adata.raw is not None
    for name, genes in sets.items():
        g = _present_genes(adata, genes)
        if len(g) < 2:
            log.warning("通路 %s 可用基因过少 (%s)，跳过 score_genes", name, g)
            adata.obs[f"score_{name}"] = 0.0
            continue
        sc.tl.score_genes(adata, gene_list=g, score_name=f"score_{name}", use_raw=use_raw)
    return adata


def summarize_pathway_blockade(
    adata: ad.AnnData,
    config: dict[str, Any],
    out_csv: Path,
) -> tuple[pd.DataFrame, float]:
    cond = config["condition_column"]
    ctrl = config["control_label"]
    treat = config["treatment_label"]
    ct_col = config["celltype_column"]
    targets = set(config.get("target_celltypes", []))

    rows = []
    for name in [
        "score_PD1_PDL1_core",
        "score_TCR_activation",
        "score_NFAT",
        "score_IFNG_response",
        "score_exhaustion",
    ]:
        if name not in adata.obs:
            continue
        sub = adata.obs[[cond, ct_col, name]].dropna()
        for ct in sorted(sub[ct_col].unique()):
            if targets and ct not in targets:
                continue
            a = sub[(sub[cond] == ctrl) & (sub[ct_col] == ct)][name].values
            b = sub[(sub[cond] == treat) & (sub[ct_col] == ct)][name].values
            if len(a) < 3 or len(b) < 3:
                continue
            delta = float(np.nanmean(b) - np.nanmean(a))
            if "exhaustion" in name:
                effect = -delta
            else:
                effect = delta
            rows.append(
                dict(
                    score=name,
                    cell_type=str(ct),
                    control_mean=float(np.nanmean(a)),
                    treatment_mean=float(np.nanmean(b)),
                    delta=delta,
                    signed_effect=effect,
                    pvalue=ranksums(b, a).pvalue,
                )
            )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if df.empty:
        log.warning("第二层汇总为空，pathway_blockade 使用中性分 0.5")
        return df, 0.5

    pos = df[~df["score"].str.contains("exhaustion")]
    neg = df[df["score"].str.contains("exhaustion")]
    pos_score = np.clip(pos["signed_effect"].mean() / 2 + 0.5, 0, 1) if len(pos) else 0.5
    neg_score = np.clip(neg["signed_effect"].mean() / 2 + 0.5, 0, 1) if len(neg) else 0.5
    pathway_blockade_global = float(np.clip(0.7 * pos_score + 0.3 * neg_score, 0, 1))
    log.info("pathway_blockade_reference (0-1) ~= %.3f", pathway_blockade_global)
    return df, pathway_blockade_global

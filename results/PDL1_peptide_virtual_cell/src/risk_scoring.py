"""第五层：毒性 / 增殖 / 炎症 / EMT / stemness 风险。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ranksums

log = logging.getLogger(__name__)

RISK_GENES = {
    "human": {
        "toxicity_dna": ["GADD45A", "CDKN1A", "ATM", "ATR", "TP53", "BAX"],
        "proliferation": ["MKI67", "TOP2A", "PCNA", "CCNB1", "CDK1", "MYC"],
        "apoptosis": ["CASP3", "CASP8", "BAX", "BCL2", "PARP1"],
        "inflammatory": ["IL6", "TNF", "IL1B", "CCL2", "CXCL10", "IFNB1"],
        "emt": ["VIM", "SNAI1", "SNAI2", "ZEB1", "TWIST1"],
        "stemness": ["SOX2", "NANOG", "POU5F1", "ALDH1A1"],
    },
    "mouse": {
        "toxicity_dna": ["Gadd45a", "Cdkn1a", "Atm", "Atr", "Trp53", "Bax"],
        "proliferation": ["Mki67", "Top2a", "Pcna", "Ccnb1", "Cdk1", "Myc"],
        "apoptosis": ["Casp3", "Casp8", "Bax", "Bcl2", "Parp1"],
        "inflammatory": ["Il6", "Tnf", "Il1b", "Ccl2", "Cxcl10", "Ifnb1"],
        "emt": ["Vim", "Snai1", "Snai2", "Zeb1", "Twist1"],
        "stemness": ["Sox2", "Nanog", "Pou5f1", "Aldh1a1"],
    },
}


def _present(adata, genes: list[str]) -> list[str]:
    vmap = {str(v).upper(): str(v) for v in adata.var_names}
    out = []
    for g in genes:
        u = g.upper()
        if u in vmap:
            out.append(vmap[u])
    return out


def score_risks(adata: ad.AnnData, config: dict[str, Any]) -> ad.AnnData:
    species = config.get("species", "human")
    sets = RISK_GENES.get(species, RISK_GENES["human"])
    out = adata.copy()
    use_raw = out.raw is not None
    for name, genes in sets.items():
        g = _present(out, genes)
        if len(g) < 2:
            out.obs[f"risk_{name}"] = 0.0
            continue
        sc.tl.score_genes(out, gene_list=g, score_name=f"risk_{name}", use_raw=use_raw)
    return out


def summarize_risks(
    adata: ad.AnnData,
    config: dict[str, Any],
    out_csv: Path,
) -> pd.DataFrame:
    cond = config["condition_column"]
    ctrl = config["control_label"]
    treat = config["treatment_label"]
    ct_col = config["celltype_column"]

    cols = [c for c in adata.obs.columns if c.startswith("risk_")]
    rows = []
    for c in cols:
        for ct in sorted(adata.obs[ct_col].unique()):
            sub = adata.obs[[cond, ct_col, c]].dropna()
            a = sub[(sub[cond] == ctrl) & (sub[ct_col] == ct)][c].values
            b = sub[(sub[cond] == treat) & (sub[ct_col] == ct)][c].values
            if len(a) < 2 or len(b) < 2:
                continue
            delta = float(np.nanmean(b) - np.nanmean(a))
            rows.append(
                dict(
                    risk_category=c,
                    cell_type=str(ct),
                    control_mean=float(np.nanmean(a)),
                    treatment_mean=float(np.nanmean(b)),
                    delta=delta,
                    pvalue=ranksums(b, a).pvalue,
                )
            )
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def peptide_toxicity_prediction(
    layer1: pd.DataFrame,
    risk_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    将队列 risk（treatment vs control）与多肽理化性质结合，得到 0-1 toxicity_risk_score。
    """
    if risk_summary.empty:
        cohort = 0.25
    else:
        bad = risk_summary[risk_summary["risk_category"].isin(["risk_inflammatory", "risk_toxicity_dna"])]
        cohort = float(np.clip((bad["delta"].clip(lower=0).mean() if len(bad) else 0) / 2 + 0.2, 0, 1))

    out_rows = []
    for _, row in layer1.iterrows():
        charge = abs(float(row.get("net_charge", 0)))
        inst = float(row.get("instability_index", 40))
        hyd = float(row.get("hydrophobic_ratio", 0.5))
        chem = np.clip((charge / 10.0) * 0.35 + (inst / 100.0) * 0.35 + hyd * 0.3, 0, 1)
        tox = np.clip(0.55 * cohort + 0.45 * chem, 0, 1)
        out_rows.append(
            dict(
                peptide_id=row["peptide_id"],
                cohort_inflammatory_like_risk=cohort,
                peptide_chem_risk=chem,
                toxicity_risk_score=tox,
                safety_score=float(np.clip(1.0 - tox, 0, 1)),
            )
        )
    return pd.DataFrame(out_rows)

"""单细胞读取、预处理与 demo 数据。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

log = logging.getLogger(__name__)

sc.settings.verbosity = 1


def _ensure_genes_for_panel(genes: list[str], required: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for g in genes + required:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def create_demo_anndata(
    out_path: Path,
    config: dict[str, Any],
    seed: int = 42,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    species = config.get("species", "human")
    cond_col = config.get("condition_column", "condition")
    ct_col = config.get("celltype_column", "cell_type")
    ctrl = config.get("control_label", "control")
    treat = config.get("treatment_label", "treatment")

    if species == "mouse":
        required_core = [
            "Pdcd1",
            "Cd274",
            "Ifng",
            "Il2",
            "Cd3e",
            "Gzmb",
            "Prf1",
            "Mki67",
            "Il6",
        ]
        genes = [f"gene_mock_{i}" for i in range(400)]
        genes = _ensure_genes_for_panel(genes, required_core)
    else:
        required_core = [
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
            "NFATC2",
            "NFATC3",
            "JUN",
            "FOS",
            "IL2",
            "IFNG",
            "CTLA4",
            "LAG3",
            "HAVCR2",
            "TIGIT",
            "GZMB",
            "PRF1",
            "MKI67",
            "IL6",
            "TNF",
            "TP53",
            "MYC",
            "VIM",
            "GADD45A",
            "STAT1",
            "IRF1",
            "CXCL9",
            "CXCL10",
            "CDKN1A",
            "ATM",
            "ATR",
            "BAX",
            "TOP2A",
            "PCNA",
            "CCNB1",
            "CDK1",
            "CASP3",
            "CASP8",
            "BCL2",
            "PARP1",
            "IL1B",
            "CCL2",
            "IFNB1",
            "SNAI1",
            "SNAI2",
            "ZEB1",
            "TWIST1",
            "SOX2",
            "NANOG",
            "POU5F1",
            "ALDH1A1",
        ]
        genes = [f"GM{i:05d}" for i in range(450)]
        genes = _ensure_genes_for_panel(genes, required_core)

    n_genes = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    n_cells = 600
    obs = pd.DataFrame(
        {
            cond_col: [ctrl] * (n_cells // 2) + [treat] * (n_cells - n_cells // 2),
            ct_col: rng.choice(["CD8_T", "T_cell", "Tumor", "Myeloid"], size=n_cells, p=[0.35, 0.25, 0.25, 0.15]),
        }
    )
    X = rng.negative_binomial(5, 0.35, size=(n_cells, n_genes)).astype(np.float32)

    def bump(gene: str, mask: np.ndarray, delta: float):
        if gene not in gene_to_idx:
            return
        j = gene_to_idx[gene]
        X[mask, j] = X[mask, j] * delta + rng.poisson(3, size=mask.sum())

    treat_mask = obs[cond_col].values == treat
    cd8 = (obs[ct_col].values == "CD8_T") & treat_mask
    tcell = (obs[ct_col].str.startswith("T")) & treat_mask

    if species != "mouse":
        bump("IFNG", cd8 | tcell, 2.2)
        bump("IL2", cd8, 1.8)
        bump("CD3E", tcell, 1.4)
        bump("NFATC1", cd8, 1.6)
        bump("GZMB", cd8, 2.0)
        bump("PRF1", cd8, 1.9)
        bump("PDCD1", cd8, 0.65)
        bump("LAG3", cd8, 0.7)
        bump("IL6", treat_mask & (obs[ct_col].values == "Myeloid"), 1.5)
        bump("MKI67", treat_mask & (obs[ct_col].values == "Tumor"), 0.85)
    else:
        bump("Ifng", cd8 | tcell, 2.2)
        bump("Il2", cd8, 1.8)

    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    log.warning("已写入 demo AnnData: %s （非真实实验）", out_path)
    return adata


def load_scrna(
    project_root: Path,
    config: dict[str, Any],
) -> ad.AnnData:
    seed = int(config.get("random_seed", 42))
    inp = (config.get("scrna_input_path") or "").strip()
    fmt = (config.get("input_format") or "auto").lower()
    demo_path = project_root / "data" / "processed" / "demo_scrna.h5ad"

    if inp:
        p = project_root / inp
        if p.is_file():
            log.info("加载 h5ad: %s", p)
            return ad.read_h5ad(p)
        log.warning("配置 scrna_input_path 不存在: %s", p)

    tenx = (config.get("tenx_matrix_dir") or "").strip()
    if tenx:
        p = project_root / tenx
        if p.is_dir():
            log.info("读取 10x: %s", p)
            return sc.read_10x_mtx(str(p), var_names="gene_symbols", cache=True)
        log.warning("tenx_matrix_dir 不存在: %s", p)

    csvp = (config.get("scrna_csv_path") or "").strip()
    if csvp:
        p = project_root / csvp
        if p.is_file():
            log.info("读取 CSV: %s", p)
            df = pd.read_csv(p, index_col=0)
            if config.get("csv_genes_are_rows", False):
                df = df.T
            return ad.AnnData(X=df.values, obs=pd.DataFrame(index=df.index), var=pd.DataFrame(index=df.columns))
        log.warning("scrna_csv_path 不存在: %s", p)

    if demo_path.is_file() and fmt != "force_new_demo":
        log.info("加载已有 demo: %s", demo_path)
        return ad.read_h5ad(demo_path)

    log.warning("创建/刷新 demo 单细胞数据用于跑通流程")
    return create_demo_anndata(demo_path, config, seed=seed)


def preprocess(
    adata: ad.AnnData,
    config: dict[str, Any],
) -> ad.AnnData:
    cond_col = config["condition_column"]
    if cond_col not in adata.obs:
        raise KeyError(f"obs 缺少 {cond_col}")

    n_top = int(config.get("n_top_genes", 2000))
    n_pcs = int(config.get("n_pcs", 30))
    nn = int(config.get("n_neighbors", 15))
    res = float(config.get("leiden_resolution", 0.5))

    adata_out = adata.copy()
    sc.pp.filter_cells(adata_out, min_genes=50)
    sc.pp.filter_genes(adata_out, min_cells=3)
    sc.pp.normalize_total(adata_out, target_sum=1e4)
    sc.pp.log1p(adata_out)
    sc.pp.highly_variable_genes(adata_out, n_top_genes=min(n_top, adata_out.n_vars), subset=False)
    # scale 原地改 X。若 raw 仍指向与 adata_out 共享的矩阵，.raw 会一同被缩放（出现负值），
    # 第三层 rank_genes / logFC 与火山图会错误。scale 前必须冻结 log1p 归一化矩阵的独立副本。
    adata_out.raw = adata_out.copy()
    sc.pp.scale(adata_out, max_value=10)
    sc.tl.pca(adata_out, n_comps=min(n_pcs, adata_out.n_vars - 1, adata_out.n_obs - 1))
    sc.pp.neighbors(
        adata_out,
        n_neighbors=min(nn, adata_out.n_obs - 1),
        n_pcs=min(n_pcs, adata_out.obsm["X_pca"].shape[1]),
    )
    sc.tl.umap(adata_out)
    sc.tl.leiden(adata_out, resolution=res)
    return adata_out


def preprocess_for_deg_reference(adata: ad.AnnData, config: dict[str, Any]) -> ad.AnnData:
    """
    仅用于第三层 rank_genes_groups：filter → normalize_total → log1p → .raw，
    不做 scale/PCA/UMAP（与主流程 preprocess 中 raw 存储的语义一致）。
    """
    min_genes = int(config.get("deg_ref_filter_min_genes_per_cell", 50))
    min_cells_gene = int(config.get("deg_ref_filter_min_cells_per_gene", 3))
    adata_out = adata.copy()
    sc.pp.filter_cells(adata_out, min_genes=max(1, min_genes))
    sc.pp.filter_genes(adata_out, min_cells=max(1, min_cells_gene))
    sc.pp.normalize_total(adata_out, target_sum=1e4)
    sc.pp.log1p(adata_out)
    adata_out.raw = adata_out.copy()
    return adata_out

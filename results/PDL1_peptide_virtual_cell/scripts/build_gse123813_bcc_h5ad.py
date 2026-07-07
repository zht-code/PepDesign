#!/usr/bin/env python3
"""
将 GEO GSE123813（Yost et al., Nature 2019）BCC 补充文件转为 h5ad。

数据来源（已下载到 data/raw/gse123813_bcc_icb/）:
  - GSE123813_bcc_scRNA_counts.txt.gz  首行为细胞 barcode，后续每行一基因、制表符分隔计数
  - GSE123813_bcc_all_metadata.txt.gz 含 patient, treatment(pre/post), cluster 等

输出: data/processed/GSE123813_BCC_prepost_antiPD1.h5ad
obs 列: patient, treatment (pre/post), sort, cluster, cell_type(=cluster), condition(=treatment 便于 deg_reference)
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "gse123813_bcc_icb"
COUNTS = RAW / "GSE123813_bcc_scRNA_counts.txt.gz"
META = RAW / "GSE123813_bcc_all_metadata.txt.gz"
OUT = PROJECT_ROOT / "data" / "processed" / "GSE123813_BCC_prepost_antiPD1.h5ad"


def main() -> None:
    if not COUNTS.is_file() or not META.is_file():
        print("缺少原始文件，请先下载到:", RAW, file=sys.stderr)
        sys.exit(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(gzip.open(META, "rt"), sep="\t", index_col=0)
    meta["cell_type"] = meta["cluster"].astype(str)
    meta["condition"] = meta["treatment"].astype(str)

    with gzip.open(COUNTS, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        obs_all = header

    keep = [c for c in obs_all if c in meta.index]
    miss = len(obs_all) - len(keep)
    if miss:
        print(f"剔除 counts 有而 metadata 无的 barcode: {miss}", file=sys.stderr)
    idx_map = {cid: i for i, cid in enumerate(obs_all)}
    pick_idx = [idx_map[c] for c in keep]
    obs_names = keep
    n_cells = len(obs_names)

    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    gene_names: list[str] = []

    with gzip.open(COUNTS, "rt") as f:
        f.readline()  # skip header again
        j = 0
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(obs_all) + 1:
                print(f"跳过畸形行 j={j} nfields={len(parts)}", file=sys.stderr)
                continue
            gene = parts[0]
            vals_full = np.asarray(parts[1 : len(obs_all) + 1], dtype=np.int32)
            vals = vals_full[pick_idx]
            nz = np.flatnonzero(vals)
            if nz.size:
                rows.extend(nz.tolist())
                cols.extend([j] * int(nz.size))
                data.extend(vals[nz].astype(np.int32).tolist())
            gene_names.append(gene)
            j += 1
            if j % 4000 == 0:
                print("genes", j, flush=True)

    n_genes = len(gene_names)
    X = sparse.coo_matrix(
        (np.asarray(data, dtype=np.int32), (np.asarray(rows), np.asarray(cols))),
        shape=(n_cells, n_genes),
        dtype=np.int32,
    ).tocsr()

    obs = meta.reindex(obs_names)

    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=gene_names))
    adata.obs_names = obs_names
    adata.var_names_make_unique()
    adata.uns["GSE123813"] = {
        "title": "Clonal replacement of tumor-specific T cells following PD-1 blockade (BCC)",
        "reference": "Yost et al., Nature 2019; GEO GSE123813",
        "treatment_column": "treatment",
        "pre_post_labels": ["pre", "post"],
        "note": "Site-matched tumor scRNA pre vs on anti-PD-1; suitable for deg_reference within cell_type.",
    }
    adata.write_h5ad(OUT, compression="gzip")
    print("写入", OUT, "obs", adata.n_obs, "var", adata.n_vars)


if __name__ == "__main__":
    main()

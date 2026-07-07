#!/usr/bin/env python3
"""
将 AnnData（h5ad）转为 Geneformer ``TranscriptomeTokenizer`` 所需的磁盘 ``*.dataset``，
供 ``InSilicoPerturber.perturb_data(..., input_data_dir, ...)`` 使用。

前提（与官方 Geneformer 文档一致）：

- **表达矩阵**：近似 **原始计数**（tokenizer 内部按 ``X / n_counts * 1e4 / gene_median`` 归一）。
  若你只有 log1p 归一化矩阵，请使用 ``--assume-log1p``（会 ``expm1``，仅作近似），或提供 ``--counts-layer``。
- **基因**：``adata.var['ensembl_id']`` 为 Ensembl gene id（如 ENSG00000196776）。
  若仅有 symbol，可用 ``--symbol-to-ensembl-tsv``（列 ``symbol``, ``ensembl_id``），或安装 ``mygene`` 后加 ``--map-symbols-mygene``。
- **细胞**：``adata.obs['n_counts']`` 为每细胞总计数（缺省则从所用 X 按行求和）。
- **可选**：``adata.obs['filter_pass']`` 为 0/1，仅 tokenize 为 1 的细胞。

依赖::

    pip install "git+https://huggingface.co/ctheodoris/Geneformer"  # 或镜像可访问的等价安装
    # 另需：datasets, pyarrow, loompy, tqdm, torch, transformers 等（以官方为准）

词典与 median 文件须与模型版本一致（默认 V2），通常来自
``models/perturb_virtual_cell/geneformer/`` 的 HuggingFace 整包快照。

用法示例::

cd /root/autodl-tmp/Peptide_3D/results/PDL1_peptide_virtual_cell
python3 scripts/tokenize_h5ad_for_geneformer.py \
  --h5ad data/processed/GSE115978.h5ad \
  --config config.yaml \
  --model-dir models/perturb_virtual_cell/geneformer \
  --out-dir results/geneformer_tokenized \
  --output-prefix gse115978 \
  --map-symbols-mygene
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

log = logging.getLogger("tokenize_h5ad_for_geneformer")


def _subsample_mask(
    obs: pd.DataFrame,
    cond_col: str,
    ctrl: str,
    treat: str,
    max_cells: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(obs)
    if n <= max_cells:
        return np.ones(n, dtype=bool)
    condv = obs[cond_col].astype(str).values
    ic = np.where(condv == str(ctrl))[0]
    it = np.where(condv == str(treat))[0]
    half = max_cells // 2
    take_c = min(len(ic), half)
    take_t = min(len(it), max_cells - take_c)
    if take_t < len(it) and take_c < len(ic):
        take_c = min(len(ic), max_cells - take_t)
    sel_c = rng.choice(ic, size=take_c, replace=False) if take_c else np.array([], dtype=int)
    sel_t = rng.choice(it, size=take_t, replace=False) if take_t else np.array([], dtype=int)
    m = np.zeros(n, dtype=bool)
    m[np.concatenate([sel_c, sel_t])] = True
    return m


def _row_sum(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X, dtype=np.float64).sum(axis=1).ravel()


def _resolve_tokenizer_pickles(model_dir: Path, args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    med = Path(args.gene_median_file) if args.gene_median_file else None
    tok = Path(args.token_dictionary_file) if args.token_dictionary_file else None
    ens = Path(args.gene_mapping_file) if args.gene_mapping_file else None

    if med is None or not med.is_file():
        med = next(
            (
                model_dir / n
                for n in (
                    "gene_median_dictionary.pkl",
                    "gene_median_dictionary_gc104M.pkl",
                    "genecorpus_median_dictionary.pkl",
                    "median_dictionary.pkl",
                )
                if (model_dir / n).is_file()
            ),
            None,
        )
        if med is None or not med.is_file():
            cands = sorted(model_dir.glob("gene_median*.pkl"))
            if cands:
                med = cands[0]
    if tok is None or not tok.is_file():
        tok = next(
            (
                model_dir / n
                for n in (
                    "token_dictionary.pkl",
                    "token_dictionary_gc104M.pkl",
                    "gene_token_dictionary.pkl",
                )
                if (model_dir / n).is_file()
            ),
            None,
        )
        if tok is None or not tok.is_file():
            cands = sorted(model_dir.glob("token_dictionary*.pkl"))
            if cands:
                tok = cands[0]
    if ens is None or not ens.is_file():
        ens = next(
            (
                model_dir / n
                for n in (
                    "ensembl_mapping_dict.pkl",
                    "ensembl_mapping_dict_gc104M.pkl",
                    "ensembl_id_mapping_dict.pkl",
                    "gene_mapping_dict.pkl",
                )
                if (model_dir / n).is_file()
            ),
            None,
        )
        if ens is None or not ens.is_file():
            cands = sorted(model_dir.glob("ensembl_mapping*.pkl"))
            if cands:
                ens = cands[0]

    if med is None or not med.is_file():
        raise FileNotFoundError(
            f"未在 {model_dir} 找到 gene median pickle；请指定 --gene-median-file 或完整下载 Geneformer 快照。"
        )
    if tok is None or not tok.is_file():
        raise FileNotFoundError(
            f"未在 {model_dir} 找到 token_dictionary pickle；请指定 --token-dictionary-file 或完整下载快照。"
        )
    return med, tok, ens


def _map_symbols_mygene(symbols: list[str], species: str) -> dict[str, str]:
    try:
        import mygene  # type: ignore
    except ImportError as e:
        raise RuntimeError("需要 pip install mygene 才能使用 --map-symbols-mygene") from e

    mg = mygene.MyGeneInfo()
    scopes = "symbol,alias"
    q = mg.querymany(
        list(set(symbols)),
        scopes=scopes,
        fields="ensembl.gene",
        species=species,
        verbose=False,
    )
    out: dict[str, str] = {}
    for item in q:
        sym = str(item.get("query", "")).upper()
        ens = item.get("ensembl")
        gid = None
        if isinstance(ens, dict):
            gid = ens.get("gene")
        elif isinstance(ens, list) and ens:
            gid = ens[0].get("gene") if isinstance(ens[0], dict) else None
        if gid:
            out[sym] = str(gid)
    return out


def _prepare_adata(
    adata: ad.AnnData,
    args: argparse.Namespace,
    cfg: dict | None,
) -> ad.AnnData:
    """筛选细胞 / 基因，构造 counts 矩阵与 ensembl_id、n_counts。"""
    ad = adata.copy()
    if cfg:
        targets = [str(x).strip() for x in (cfg.get("target_celltypes") or []) if str(x).strip()]
        ct_col = str(cfg.get("celltype_column", "cell_type"))
        if targets and ct_col in ad.obs.columns:
            ad = ad[ad.obs[ct_col].astype(str).isin(targets)].copy()

        cond = str(cfg.get("condition_column", "condition"))
        ctrl = str(cfg.get("control_label", "control"))
        treat = str(cfg.get("treatment_label", "treatment"))
        if cond in ad.obs.columns:
            mask = _subsample_mask(
                ad.obs,
                cond,
                ctrl,
                treat,
                max_cells=int(args.max_cells),
                seed=int(args.seed),
            )
            ad = ad[mask].copy()

    # counts matrix
    X_counts = None
    if args.counts_layer:
        layer = args.counts_layer.strip()
        if layer not in ad.layers:
            raise KeyError(f"layers 中不存在 {layer!r}")
        X_counts = ad.layers[layer]
    else:
        X_counts = ad.X

    if sparse.issparse(X_counts):
        max_x = float(X_counts.data.max()) if X_counts.nnz else 0.0
    else:
        max_x = float(np.max(X_counts)) if X_counts.size else 0.0

    if args.assume_log1p or (max_x < 25.0 and not args.counts_layer):
        log.warning(
            "对表达矩阵使用 expm1（假定当前为 log1p 域）。Geneformer 官方建议使用原始计数；请尽量提供 --counts-layer 或未 log 的 h5ad。"
        )
        if sparse.issparse(X_counts):
            X_counts = sparse.csr_matrix(X_counts)
            X_counts.data = np.expm1(X_counts.data).astype(np.float32)
        else:
            X_counts = np.expm1(np.asarray(X_counts, dtype=np.float64)).astype(np.float32)

    ad.X = X_counts
    if sparse.issparse(ad.X):
        ad.X.eliminate_zeros()

    # n_counts
    if args.n_counts_column and args.n_counts_column in ad.obs.columns:
        ad.obs["n_counts"] = pd.to_numeric(ad.obs[args.n_counts_column], errors="coerce").fillna(0).astype(np.float64)
    else:
        ad.obs["n_counts"] = _row_sum(ad.X)

    # ensembl_id in var
    if args.var_ensembl_column and args.var_ensembl_column in ad.var.columns:
        ad.var["ensembl_id"] = ad.var[args.var_ensembl_column].astype(str)
    elif "ensembl_id" in ad.var.columns:
        ad.var["ensembl_id"] = ad.var["ensembl_id"].astype(str)
    else:
        symbols = ad.var_names.astype(str).tolist()
        sym_to_ens: dict[str, str] = {}
        if args.symbol_to_ensembl_tsv:
            mp = pd.read_csv(args.symbol_to_ensembl_tsv, sep=None, engine="python")
            cols = {c.lower(): c for c in mp.columns}
            scol = cols.get("symbol") or cols.get("gene_symbol") or mp.columns[0]
            ecol = cols.get("ensembl_id") or cols.get("ensembl") or mp.columns[1]
            for _, r in mp.iterrows():
                sym_to_ens[str(r[scol]).strip().upper()] = str(r[ecol]).strip()
        if args.map_symbols_mygene:
            species = (args.species or (cfg or {}).get("species") or "human").lower()
            sp = "mouse" if species == "mouse" else "human"
            sym_to_ens.update(_map_symbols_mygene(symbols, sp))
        ens_ids = []
        for s in symbols:
            su = s.upper()
            e = sym_to_ens.get(su, "")
            if not e and (su.startswith("ENSG") or su.startswith("ENSMUSG")):
                e = s
            ens_ids.append(e if e else "NA")
        ad.var["ensembl_id"] = ens_ids
        n_ok = sum(1 for x in ens_ids if str(x).startswith("ENS"))
        if n_ok < 1000:
            log.warning("仅 %s / %s 个基因映射到 Ensembl id；请检查 TSV 或网络映射。", n_ok, len(ens_ids))

    ad.var["ensembl_id"] = ad.var["ensembl_id"].astype(str).str.strip()
    # drop genes without valid Ensembl
    ok_gene = ad.var["ensembl_id"].str.startswith("ENS")
    ad = ad[:, ok_gene].copy()

    if ad.n_obs < 10 or ad.n_vars < 500:
        raise RuntimeError(f"过滤后维度过小: n_obs={ad.n_obs}, n_vars={ad.n_vars}")

    if args.filter_pass_all:
        ad.obs["filter_pass"] = 1

    return ad


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="h5ad -> Geneformer tokenized .dataset on disk")
    ap.add_argument("--h5ad", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None, help="可选：读取 target_celltypes / condition 等与主流程一致")
    ap.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "perturb_virtual_cell" / "geneformer")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "geneformer_tokenized")
    ap.add_argument("--output-prefix", type=str, default="tokenized_cells")
    ap.add_argument("--model-version", type=str, default="V2", choices=("V1", "V2"))
    ap.add_argument("--max-cells", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--counts-layer", type=str, default="", help="使用 adata.layers[name] 作为计数矩阵")
    ap.add_argument("--assume-log1p", action="store_true", help="对 X 或 counts layer 做 expm1")
    ap.add_argument("--var-ensembl-column", type=str, default="", help="adata.var 中 Ensembl id 列名")
    ap.add_argument("--symbol-to-ensembl-tsv", type=Path, default=None)
    ap.add_argument("--map-symbols-mygene", action="store_true")
    ap.add_argument("--species", type=str, default="")
    ap.add_argument("--n-counts-column", type=str, default="", help="已存在的总 UMI 列名；缺省则对 X 求和")
    ap.add_argument("--filter-pass-all", action="store_true", help="写入 filter_pass=1（全部纳入）")
    ap.add_argument("--gene-median-file", type=str, default="")
    ap.add_argument("--token-dictionary-file", type=str, default="")
    ap.add_argument("--gene-mapping-file", type=str, default="")
    ap.add_argument("--nproc", type=int, default=4)
    args = ap.parse_args()

    cfg = None
    if args.config and args.config.is_file():
        import yaml

        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    try:
        from geneformer import TranscriptomeTokenizer
    except ImportError as e:
        # 常见原因：pip 与 python3 不是同一环境（如 pip 装到 miniconda，脚本被系统 python3 执行）
        vend = (PROJECT_ROOT / "vendor" / "Geneformer").resolve()
        log.error(
            "无法 import geneformer：%s\n"
            "当前实际使用的解释器: %s\n"
            "请用**与运行本脚本相同**的 Python 安装，例如:\n"
            "  %s -m pip install -e %s\n"
            "或先 `which python3` / `which pip` 确认二者是否来自同一 prefix；"
            "conda 环境需先 `conda activate` 后再 pip install。",
            e,
            sys.executable,
            sys.executable,
            vend if vend.is_dir() else "<Geneformer 源码目录>",
        )
        raise SystemExit(1)

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model-dir 不存在: {model_dir}")

    med, tok, ens_map = _resolve_tokenizer_pickles(model_dir, args)

    adata = sc.read_h5ad(str(args.h5ad))
    adata = _prepare_adata(adata, args, cfg)

    custom = None
    if cfg:
        ct = str(cfg.get("celltype_column", "cell_type"))
        cond = str(cfg.get("condition_column", "condition"))
        d = {}
        if ct in adata.obs.columns:
            d[ct] = ct
        if cond in adata.obs.columns:
            d[cond] = cond
        custom = d if d else None

    staging = args.out_dir / "_staging_h5ad"
    staging.mkdir(parents=True, exist_ok=True)
    stem = Path(args.output_prefix).stem or "tokenized"
    staged_path = staging / f"{stem}.h5ad"
    adata.write_h5ad(staged_path)

    tk_kw: dict = dict(
        custom_attr_name_dict=custom,
        nproc=int(args.nproc),
        model_version=str(args.model_version),
        gene_median_file=med,
        token_dictionary_file=tok,
    )
    if ens_map is not None:
        tk_kw["gene_mapping_file"] = ens_map

    tokenizer = TranscriptomeTokenizer(**tk_kw)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("开始 tokenize -> %s / %s.dataset", args.out_dir, args.output_prefix)
    tokenizer.tokenize_data(staging, args.out_dir, args.output_prefix, file_format="h5ad")
    log.info("完成。请将 config 中 geneformer_isp_token_data_dir 设为: %s", args.out_dir.resolve())


if __name__ == "__main__":
    main()

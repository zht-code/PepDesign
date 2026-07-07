"""第三层：差异表达与 PD-L1 blockade reference signature。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import preprocess_scrna

log = logging.getLogger(__name__)


def _warn_if_condition_confounds_celltype(
    sub: anndata.AnnData,
    cond_col: str,
    ct_col: str | None,
) -> None:
    """PBMC→仅 T、TIL→仅 Tumor 等设计下，condition 与 cell_type 一一对应，DEG 非纯药效对比。"""
    if not ct_col or cond_col not in sub.obs.columns or ct_col not in sub.obs.columns:
        return
    try:
        nx = sub.obs.groupby(cond_col, observed=True)[ct_col].nunique(dropna=False)
        if nx.shape[0] >= 2 and bool((nx == 1).all()):
            log.warning(
                "DEG：各 condition 内仅出现单一 cell_type（与解剖/分选设计共线），"
                "logFC 混合细胞组成与微环境；CD274（PD-L1）等未必呈药物阻断型显著下调。"
            )
    except Exception:
        return


def _flatten_volcano_label_genes(config: dict[str, Any]) -> list[str]:
    """从火山图配置收集基因符号，用于 filter_genes 后补回列。"""
    out: list[str] = []
    for grp in config.get("deg_volcano_annotation_groups") or []:
        if not isinstance(grp, dict):
            continue
        for item in grp.get("genes") or []:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                sym = str(item.get("symbol", "")).strip()
                if sym:
                    out.append(sym)
    for g in config.get("deg_volcano_label_genes") or []:
        g = str(g).strip()
        if g:
            out.append(g)
    for g in (config.get("deg_volcano_gene_display") or {}).keys():
        g = str(g).strip()
        if g:
            out.append(g)
    return list(dict.fromkeys(out))


def _deg_reference_dict(config: dict[str, Any]) -> dict[str, Any]:
    dr = config.get("deg_reference")
    return dr if isinstance(dr, dict) else {}


def _deg_rank_method(config: dict[str, Any], dr: dict[str, Any]) -> str:
    rm = dr.get("rank_method")
    if isinstance(rm, str) and rm.strip():
        return rm.strip()
    return str(config.get("deg_rank_method", "t-test_overestim_var")).strip()


def _mask_deg_reference(
    ad: anndata.AnnData,
    dr: dict[str, Any],
    config: dict[str, Any],
    *,
    cond: str,
    ctrl: str,
    treat: str,
) -> tuple[np.ndarray, str | None]:
    """deg_reference：可选按 cell_type 子集，并可选仅保留双臂细胞数均足够的类型。"""
    ct_col = (dr.get("celltype_column") or "").strip()
    if not ct_col or ct_col not in ad.obs.columns:
        log.info("deg_reference: 未配置有效 celltype_column，对全体细胞做 DEG")
        return np.ones(ad.n_obs, dtype=bool), (ct_col if ct_col in ad.obs.columns else None)

    targets = [str(x).strip() for x in (dr.get("target_celltypes") or config.get("target_celltypes") or []) if str(x).strip()]
    require = bool(dr.get("require_celltypes_in_both_groups", True))
    min_arm = int(dr.get("deg_min_cells_per_arm", 15))

    if not targets:
        log.info("deg_reference: target_celltypes 为空，对全体细胞做 DEG")
        return np.ones(ad.n_obs, dtype=bool), ct_col

    if require:
        ok_types: list[str] = []
        condv = ad.obs[cond].astype(str)
        ctv = ad.obs[ct_col].astype(str)
        for ct in targets:
            m = ctv == ct
            n_ctrl = int((m & (condv == str(ctrl))).sum())
            n_treat = int((m & (condv == str(treat))).sum())
            if n_ctrl >= min_arm and n_treat >= min_arm:
                ok_types.append(ct)
            else:
                log.warning(
                    "deg_reference: 跳过细胞类型 %s（%s=%d, %s=%d；每臂至少 %d）",
                    ct,
                    ctrl,
                    n_ctrl,
                    treat,
                    n_treat,
                    min_arm,
                )
        if not ok_types:
            log.error("deg_reference: 无一细胞类型满足双臂细胞数，退回全体细胞（统计可能仍不理想）")
            return np.ones(ad.n_obs, dtype=bool), ct_col
        log.info("deg_reference: 双臂均 >=%d 的细胞类型: %s", min_arm, ", ".join(ok_types))
        return ctv.isin(ok_types).values, ct_col

    mask = ad.obs[ct_col].astype(str).isin(targets).values
    if int(mask.sum()) < 20:
        log.warning("deg_reference: target_celltypes 子集 <20 细胞，退回全体细胞")
        return np.ones(ad.n_obs, dtype=bool), ct_col
    return mask, ct_col


def _log2_fold_change_from_means(
    sub: anndata.AnnData,
    cond: str,
    ctrl: str,
    treat: str,
) -> pd.Series:
    """
    基于当前 ``sub.X``（与 rank_genes 所用一致，通常为 log1p 归一化计数）的组均值，
    计算 log2((mean_treat+eps)/(mean_ctrl+eps))。用于填补 scanpy t-test 在零均值处产生的 NaN/inf，
    避免误填为 0 导致火山图全部挤在 x=0。
    """
    condv = sub.obs[cond].astype(str).values
    ic = condv == str(ctrl)
    it = condv == str(treat)
    idx = sub.var_names.astype(str)
    if ic.sum() < 1 or it.sum() < 1:
        return pd.Series(np.nan, index=idx)
    X = sub.X
    if sparse.issparse(X):
        mt = np.asarray(X[it].mean(axis=0)).ravel()
        mc = np.asarray(X[ic].mean(axis=0)).ravel()
    else:
        mt = np.mean(np.asarray(X[it], dtype=np.float64), axis=0).ravel()
        mc = np.mean(np.asarray(X[ic], dtype=np.float64), axis=0).ravel()
    eps = 1e-9
    mt = np.maximum(mt, 0.0)
    mc = np.maximum(mc, 0.0)
    lfc = np.log2(mt + eps) - np.log2(mc + eps)
    return pd.Series(lfc, index=idx)


def build_reference_signature(
    adata: anndata.AnnData,
    config: dict[str, Any],
    deg_csv: Path,
    up_txt: Path,
    down_txt: Path,
    *,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """
    第三层 DEG。默认使用主流程 adata（与 UMAP/pathway 同源）。

    若 config['deg_reference']['scrna_input_path'] 非空，则从该 h5ad 单独构建 DEG
    （需在同一 cell_type 内双臂可比的实验设计），以获得如 CD274 等基因的**可解释 FDR**。
    """
    dr = _deg_reference_dict(config)
    use_ref = bool((dr.get("scrna_input_path") or "").strip())
    if use_ref and project_root is None:
        # 默认从 results/tables/deg_csv 反推项目根（与 run_pipeline 默认 output_dir=results 一致）
        project_root = deg_csv.resolve().parents[2]
        log.info("deg_reference: 未传入 project_root，使用 %s", project_root)

    if use_ref:
        for key in ("condition_column", "control_label", "treatment_label"):
            if not (dr.get(key) or "").strip():
                raise ValueError(f"启用 deg_reference 时必须在 config.deg_reference 中设置非空 {key!r}")
        rel = (dr["scrna_input_path"] or "").strip()
        path = Path(project_root) / rel
        if not path.is_file():
            raise FileNotFoundError(f"deg_reference.scrna_input_path 不存在: {path}")
        log.info("第三层 DEG 使用 deg_reference 专用 h5ad（与主 scrna_input_path 分离）: %s", path)
        deg_src = anndata.read_h5ad(path)
        deg_pp = preprocess_scrna.preprocess_for_deg_reference(deg_src, config)
        ad = deg_pp.raw.to_adata() if deg_pp.raw is not None else deg_pp
        cond = str(dr["condition_column"]).strip()
        ctrl = str(dr["control_label"]).strip()
        treat = str(dr["treatment_label"]).strip()
        mask, ct_eff = _mask_deg_reference(ad, dr, config, cond=cond, ctrl=ctrl, treat=treat)
    else:
        cond = str(config["condition_column"])
        ctrl = str(config["control_label"])
        treat = str(config["treatment_label"])
        ct_col = str(config["celltype_column"])
        ad = adata.copy()
        if ad.raw is not None:
            ad = ad.raw.to_adata()
        mask = ad.obs[ct_col].isin(config.get("target_celltypes", []))
        if int(mask.sum()) < 20:
            log.warning("目标细胞类型子集过小，使用全体细胞做 DEG")
            mask = np.ones(ad.n_obs, dtype=bool)
        ct_eff = ct_col

    sub = ad[mask].copy()
    sub.obs = sub.obs.copy()
    _warn_if_condition_confounds_celltype(sub, cond, ct_eff)

    method = _deg_rank_method(config, dr)
    min_cells = max(1, int(0.1 * sub.n_obs))
    sc.pp.filter_genes(sub, min_cells=min_cells)
    log.info("filter_genes(min_cells=%d) 后: %d 基因；rank_genes method=%s", min_cells, sub.n_vars, method)

    extra_syms = _flatten_volcano_label_genes(config)
    extra_syms += [str(x).strip() for x in (config.get("deg_always_keep_genes") or []) if str(x).strip()]
    extra_syms = list(dict.fromkeys(extra_syms))
    vnames = sub.var_names.astype(str)
    miss = [g for g in extra_syms if g in ad.var_names.astype(str) and g not in set(vnames)]
    if miss:
        add = ad[mask, miss].copy()
        sub = anndata.concat([sub, add], axis=1, merge="unique")
        log.info("为火山/解读补回 %d 个基因列: %s", len(miss), ", ".join(miss[:15]) + ("…" if len(miss) > 15 else ""))

    try:
        sc.tl.rank_genes_groups(
            sub,
            groupby=cond,
            reference=str(ctrl),
            method=method,
            key_added="rank_cond",
        )
    except Exception as exc:
        log.warning("rank_genes_groups 失败: %s — 返回空签名", exc)
        empty = pd.DataFrame(columns=["gene", "logfoldchanges", "pvals", "pvals_adj"])
        empty.to_csv(deg_csv, index=False)
        up_txt.write_text("")
        down_txt.write_text("")
        return empty

    res = sc.get.rank_genes_groups_df(sub, key="rank_cond", group=str(treat))
    res = res.rename(columns={"names": "gene"})
    if "scores" not in res.columns:
        res["scores"] = 0.0

    lfc_scanpy = pd.to_numeric(res["logfoldchanges"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    lfc_mean = _log2_fold_change_from_means(sub, cond, ctrl, treat)
    lfc_from_expr = res["gene"].astype(str).map(lfc_mean)
    n_bad = int((~np.isfinite(lfc_scanpy)).sum())
    if n_bad:
        log.info(
            "rank_genes_groups：%d 个基因 logfoldchanges 非有限（组内均表达为 0 等），"
            "已用组均值 log2FC 回填，供火山图与上下调列表",
            n_bad,
        )
    lfc = lfc_scanpy.where(np.isfinite(lfc_scanpy), lfc_from_expr)
    lfc = lfc.replace([np.inf, -np.inf], np.nan).fillna(lfc_from_expr)
    lfc = lfc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    res["logfoldchanges"] = np.clip(lfc.astype(np.float64), -5.0, 5.0)

    deg_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(deg_csv, index=False)

    padj_thr = float(config.get("deg_padj_max", 0.05))
    lfc_thr = float(config.get("deg_abs_logfc_min", 0.5))
    padj = res["pvals_adj"].fillna(1.0)
    lfcv = res["logfoldchanges"]
    sig = res[(padj < padj_thr) & (np.isfinite(lfcv)) & (lfcv.abs() > lfc_thr)]
    up_genes = sig[sig["logfoldchanges"] > 0]["gene"].astype(str).tolist()
    down_genes = sig[sig["logfoldchanges"] < 0]["gene"].astype(str).tolist()
    if not up_genes and not down_genes:
        log.warning("无基因满足 padj<%.2f 且 |logFC|>%.2f，宽松模式取 top", padj_thr, lfc_thr)
        r = res[np.isfinite(res["logfoldchanges"])].copy()
        r["_p"] = r["pvals_adj"].fillna(1.0)
        up_genes = (
            r[r["logfoldchanges"] > 0].nsmallest(80, "_p")["gene"].astype(str).tolist()
        )
        down_genes = (
            r[r["logfoldchanges"] < 0].nsmallest(80, "_p")["gene"].astype(str).tolist()
        )
    up_txt.write_text("\n".join(up_genes[:500]))
    down_txt.write_text("\n".join(down_genes[:500]))
    log.info(
        "签名: %d up, %d down (padj<%.2f, |logFC|>%.2f；DE=%s, logFC∈[-5,5])",
        len(up_genes),
        len(down_genes),
        padj_thr,
        lfc_thr,
        method,
    )
    return res

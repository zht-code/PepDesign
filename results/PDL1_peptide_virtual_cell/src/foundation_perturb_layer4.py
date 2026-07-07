"""第四层：GeneFormer-pert / scFoundation-pert 显式 CD274（PD-L1）虚拟扰动读出。

设计要点（与「真扰动」文献流程对齐的工程实现）：

1. **参考单细胞**：使用主流程传入的 ``adata``（优先 ``.raw`` 的 log1p 归一化表达，与 DEG 层一致），
   按 ``condition_column`` 与 ``control_label`` / ``treatment_label`` 计算基因级 **伪批量均值**
   ``μ_ctrl``、``μ_treat``，并在 **log1p** 域得到参考对比向量 ``δ = log1p(μ_treat) - log1p(μ_ctrl)``。

2. **多肽扰动**：多肽不作为 transformer token 直接输入（Geneformer 官方 InSilicoPerturber 需全基因 rank
   流水线与可选 pip 包）；此处将 Layer1 的 **结合 / 对接 / 界面** 特征压缩为扰动强度 ``s∈[0,1]``，
   沿 ``δ`` 构造 **虚拟 log 表达** ``λ_v = λ_ctrl + clip(s)·δ``，其中 ``λ_ctrl=log1p(μ_ctrl)``。

3. **PD-L1 靶向项**：对 ``perturb_vc_target_gene``（默认 CD274）施加额外 **转录下调** 项，强度由 ``s``、
   界面分及 backend 校准系数（``geneformer_pert`` vs ``scfoundation_pert``）控制，模拟 PDL1 轴药理占位。

4. **权重目录**：``models/perturb_virtual_cell/{geneformer,scfoundation_cell}`` 由
   ``scripts/download_perturb_virtual_cell_models.py`` 下载，用于实验可复现与后续接入完整前向。

输出列含 ``predicted_cd274_*`` 及 ``predicted_cd274_down``（1 表示相对对照伪批量下调超过阈值）。

完整 Geneformer ``InSilicoPerturber`` 删除/过表达单基因请见 HuggingFace 仓库示例 notebook，可在本层替换
``_virtual_expression_vector`` 为官方调用。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

log = logging.getLogger(__name__)

BackendName = Literal["geneformer_pert", "scfoundation_pert"]


def _dense_mean_rows(X: np.ndarray | sparse.spmatrix, idx: np.ndarray) -> np.ndarray:
    if idx.size == 0:
        raise ValueError("子集细胞数为 0")
    if sparse.issparse(X):
        sub = X[idx]
        return np.asarray(sub.mean(axis=0)).ravel()
    return np.mean(np.asarray(X[idx], dtype=np.float64), axis=0).ravel()


def _get_expression_matrix(adata: ad.AnnData) -> tuple[np.ndarray, list[str]]:
    """返回用于伪批量的矩阵（与 .raw 一致时为 log1p 归一化）及基因名。"""
    if adata.raw is not None:
        X = adata.raw.X
        names = adata.raw.var_names.astype(str).tolist()
    else:
        X = adata.X
        names = adata.var_names.astype(str).tolist()
    return X, names


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


def _foundation_dirs(project_root: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    root = project_root / (config.get("perturb_vc_models_dir") or "models/perturb_virtual_cell")
    return root / "geneformer", root / "scfoundation_cell"


def _virtual_expression_vector(
    lc: np.ndarray,
    delta: np.ndarray,
    cd274_j: int,
    pep_strength: float,
    iface: float,
    backend: BackendName,
    cfg: dict[str, Any],
    gene_sym: str,
    deg_df: pd.DataFrame | None,
) -> np.ndarray:
    """在 log1p 域构造虚拟表达向量，并对 CD274 施加额外下调。"""
    s = float(np.clip(pep_strength, 0.0, 1.0))
    lv = lc + s * delta
    if backend == "geneformer_pert":
        extra = float(cfg.get("geneformer_pert_cd274_extra_scale", 0.35))
    else:
        extra = float(cfg.get("scfoundation_pert_cd274_extra_scale", 0.28))
    deg_lfc = 0.0
    if deg_df is not None and len(deg_df) > 0 and "gene" in deg_df.columns:
        hit = deg_df[deg_df["gene"].astype(str) == gene_sym]
        if not hit.empty:
            deg_lfc = float(pd.to_numeric(hit.iloc[0].get("logfoldchanges", 0.0), errors="coerce") or 0.0)
    # 参考 DEG 中 CD274 若倾向下调，则略增强 knock（仅缩放，不改变符号结构）
    if deg_lfc < 0:
        extra *= 1.0 + float(cfg.get("perturb_vc_cd274_deg_align_boost", 0.15))
    knock = extra * s * iface * (1.0 + max(0.0, float(lc[cd274_j])))
    lv[cd274_j] -= knock
    return lv


def run_layer4_foundation_perturb(
    project_root: Path,
    adata: ad.AnnData,
    deg_df: pd.DataFrame,
    layer1: pd.DataFrame,
    pathway_blockade_ref: float,
    config: dict[str, Any],
    out_csv: Path,
    backend: BackendName,
) -> pd.DataFrame:
    cond = str(config["condition_column"])
    ctrl = str(config["control_label"])
    treat = str(config["treatment_label"])
    if cond not in adata.obs:
        raise KeyError(f"adata.obs 缺少 {cond}")

    gene_sym = str(config.get("perturb_vc_target_gene", "CD274")).strip()

    gf_dir, sf_dir = _foundation_dirs(project_root, config)
    if backend == "geneformer_pert" and not gf_dir.is_dir():
        log.warning("未找到 GeneFormer 权重目录 %s，请先运行 scripts/download_perturb_virtual_cell_models.py", gf_dir)
    if backend == "scfoundation_pert" and not sf_dir.is_dir():
        log.warning("未找到 scFoundation 权重目录 %s，请先运行 scripts/download_perturb_virtual_cell_models.py", sf_dir)

    targets = [str(x).strip() for x in (config.get("target_celltypes") or []) if str(x).strip()]
    ad = adata
    if targets:
        ct_col = str(config.get("celltype_column", "cell_type"))
        if ct_col in ad.obs.columns:
            ad = ad[ad.obs[ct_col].astype(str).isin(targets)].copy()

    mask = _subsample_mask(
        ad.obs,
        cond,
        ctrl,
        treat,
        max_cells=int(config.get("perturb_vc_max_cells", 4000)),
        seed=int(config.get("random_seed", 42)),
    )
    ad = ad[mask].copy()

    X, gnames = _get_expression_matrix(ad)
    if gene_sym not in gnames:
        raise KeyError(f"表达矩阵中缺少靶基因 {gene_sym}，请检查 h5ad 基因符号")
    cd274_j = gnames.index(gene_sym)

    condv = ad.obs[cond].astype(str).values
    ic = condv == str(ctrl)
    it = condv == str(treat)
    if ic.sum() < 5 or it.sum() < 5:
        raise RuntimeError("foundation_perturb: 对照或处理组细胞过少")

    mu_c = _dense_mean_rows(X, np.where(ic)[0])
    mu_t = _dense_mean_rows(X, np.where(it)[0])
    mu_c = np.maximum(mu_c, 0.0)
    mu_t = np.maximum(mu_t, 0.0)
    lc = np.log1p(mu_c)
    lt = np.log1p(mu_t)
    delta = lt - lc

    down_logfc_thr = float(config.get("perturb_vc_cd274_down_logfc_thr", 0.05))
    eps_lin = float(config.get("perturb_vc_cd274_down_linear_eps", 0.0))

    from similarity_analysis import peptide_perturbation_features

    rows: list[dict[str, Any]] = []
    for _, row in layer1.iterrows():
        feats = peptide_perturbation_features(row)
        bind = float(row.get("binding_score", 0.5))
        iface = float(feats["interface_blocking_score"])
        pep_strength = float(
            pathway_blockade_ref * 0.35 + bind * 0.35 + iface * 0.30
        )

        lv = _virtual_expression_vector(
            lc, delta, cd274_j, pep_strength, iface, backend, config, gene_sym, deg_df
        )
        mu_v = np.expm1(lv)
        mu_v = np.maximum(mu_v, 0.0)

        ctrl_cd = float(mu_c[cd274_j])
        virt_cd = float(mu_v[cd274_j])
        logfc_vs_ctrl = float(lv[cd274_j] - lc[cd274_j])
        if eps_lin > 0:
            down_lin = bool(virt_cd < ctrl_cd * (1.0 - eps_lin))
        else:
            down_lin = bool(virt_cd < ctrl_cd - 1e-12)
        down_log = bool(logfc_vs_ctrl < -down_logfc_thr)
        down = bool(down_log and down_lin)

        # 与旧第四层兼容的综合分：沿 δ 的 cosine 相似度 + CD274 下调奖励
        dv = lv - lc
        nd = np.linalg.norm(delta) + 1e-9
        nv = np.linalg.norm(dv) + 1e-9
        cos = float(np.dot(dv, delta) / (nd * nv))
        blockade_sim = float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))
        if down_log:
            blockade_sim = float(np.clip(blockade_sim + 0.08 * bind, 0.0, 1.0))

        t_act = float(np.clip(0.35 + 0.65 * blockade_sim * feats["peptide_perturbation_score"], 0, 1))
        ifng = float(np.clip(0.3 + 0.7 * blockade_sim * bind, 0, 1))
        cyto = float(np.clip(0.3 + 0.65 * blockade_sim * feats["stability_proxy"], 0, 1))
        tum_sup = float(np.clip(0.25 + 0.75 * blockade_sim * (1.0 - float(row.get("hydrophobic_ratio", 0.5))), 0, 1))
        exh_down = float(np.clip(0.35 + 0.65 * bind * iface * blockade_sim, 0, 1))
        immune_activation = float(np.mean([t_act, ifng, cyto, exh_down]))

        rows.append(
            dict(
                peptide_id=row["peptide_id"],
                sequence=row.get("sequence", ""),
                binding_score=bind,
                peptide_perturbation_score=feats["peptide_perturbation_score"],
                interface_blocking_score=iface,
                blockade_similarity_score=blockade_sim,
                T_cell_activation_prediction=t_act,
                IFNG_prediction=ifng,
                cytotoxicity_prediction=cyto,
                tumor_suppression_prediction=tum_sup,
                exhaustion_down_prediction=exh_down,
                immune_activation_score=immune_activation,
                foundation_perturb_backend=backend,
                predicted_cd274_expr_ctrl_mean=ctrl_cd,
                predicted_cd274_expr_virtual=virt_cd,
                predicted_cd274_logfc_vs_ctrl=logfc_vs_ctrl,
                predicted_cd274_down=int(down),
                predicted_cd274_down_rule="logfc_thr_and_linear_drop_vs_ctrl",
            )
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("foundation_perturb (%s) 已写入 %s", backend, out_csv)
    return df

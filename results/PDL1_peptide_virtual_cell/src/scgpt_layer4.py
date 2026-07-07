"""第四层 scGPT：真实细胞嵌入 + 肽级「虚拟表达」细胞嵌入，沿 treat–ctrl 轴评分。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

log = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def peptide_perturbation_features(row: pd.Series) -> dict[str, float]:
    """与 similarity_analysis 中定义一致，避免循环导入。"""
    bind = float(row.get("binding_score", 0.5))
    dock = row.get("docking_score")
    mmg = row.get("mmgbsa_score")
    dist = row.get("distance_to_PD1_interface")
    inst = float(row.get("instability_index", 40))
    gravy = float(row.get("gravy", 0.0))
    instability_norm = np.clip(inst / 100.0, 0, 1)
    sol_proxy = np.clip(1.0 - abs(gravy) / 2.5, 0, 1)
    stab_proxy = np.clip(1.0 - instability_norm, 0, 1)

    dock_strength = 0.5
    if pd.notna(dock):
        dock_strength = _sigmoid(-(float(dock) + 6.0))
    mmg_strength = 0.5
    if pd.notna(mmg):
        mmg_strength = _sigmoid(-(float(mmg) + 40.0) / 15.0)
    dist_score = 0.5
    if pd.notna(dist):
        dist_score = np.clip(1.0 - float(dist) / 12.0, 0, 1)

    interface_block = 0.25 * dock_strength + 0.25 * mmg_strength + 0.25 * dist_score + 0.25 * bind
    pep_pert = np.clip(0.45 * bind + 0.35 * interface_block + 0.1 * stab_proxy + 0.1 * sol_proxy, 0, 1)
    return dict(
        peptide_perturbation_score=float(pep_pert),
        interface_blocking_score=float(interface_block),
        stability_proxy=float(stab_proxy),
        solubility_proxy=float(sol_proxy),
    )


def _dense_rows(X: np.ndarray | sparse.spmatrix) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X, dtype=np.float64)


def _resolve_condition_labels(
    obs: pd.DataFrame, cond_col: str, ctrl: str, treat: str
) -> tuple[str, str]:
    """若配置标签在数据中不存在，则回退为出现次数最多的两个类别。"""
    vals = obs[cond_col].astype(str).values
    u, c = np.unique(vals, return_counts=True)
    if str(ctrl) in u and str(treat) in u:
        return str(ctrl), str(treat)
    if len(u) >= 2:
        order = np.argsort(-c)
        a, b = str(u[order[0]]), str(u[order[1]])
        log.warning(
            "condition 标签 %s / %s 不在数据中，改用 %s / %s",
            ctrl,
            treat,
            a,
            b,
        )
        return a, b
    raise RuntimeError(f"obs[{cond_col}] 中有效分组不足 2 个: {u!r}")


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
    i_ctrl = np.where(obs[cond_col].astype(str).values == str(ctrl))[0]
    i_treat = np.where(obs[cond_col].astype(str).values == str(treat))[0]
    half = max_cells // 2
    take_c = min(len(i_ctrl), half)
    take_t = min(len(i_treat), max_cells - take_c)
    if take_t < len(i_treat) and take_c < len(i_ctrl):
        take_c = min(len(i_ctrl), max_cells - take_t)
    sel_c = rng.choice(i_ctrl, size=take_c, replace=False) if take_c else np.array([], int)
    sel_t = rng.choice(i_treat, size=take_t, replace=False) if take_t else np.array([], int)
    m = np.zeros(n, dtype=bool)
    m[np.concatenate([sel_c, sel_t])] = True
    return m


def _prepare_vocab_columns(adata: ad.AnnData, gene_col: str) -> ad.AnnData:
    adata = adata.copy()
    if gene_col == "index":
        adata.var["index"] = adata.var.index.astype(str)
    elif gene_col not in adata.var:
        raise KeyError(f"adata.var 缺少基因列 {gene_col}")
    return adata


def _load_scgpt_model(
    model_dir: Path,
    device: torch.device,
    use_fast_transformer: bool,
):
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    model_dir = Path(model_dir)
    with open(model_dir / "args.json", encoding="utf-8") as f:
        model_configs = json.load(f)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for s in ("<pad>", "<cls>", "<eoc>"):
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device)
    load_pretrained(model, ckpt, verbose=False)
    model.to(device)
    model.eval()
    return model, vocab, model_configs


def _embed_matrix(
    X: np.ndarray,
    var_names: list[str],
    model,
    vocab,
    model_configs: dict,
    gene_col: str,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    from scgpt.tasks.cell_emb import get_batch_cell_embeddings

    obs = pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=var_names)
    adata = ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=obs, var=var)
    adata = _prepare_vocab_columns(adata, gene_col)
    adata.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in adata.var[gene_col].astype(str)]
    adata = adata[:, np.array(adata.var["id_in_vocab"]) >= 0]
    if adata.n_vars < 50:
        raise RuntimeError(f"与 scGPT 词表匹配的基因过少: {adata.n_vars}")
    genes = adata.var[gene_col].astype(str).tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    return get_batch_cell_embeddings(
        adata,
        cell_embedding_mode="cls",
        model=model,
        vocab=vocab,
        max_length=max_length,
        batch_size=batch_size,
        model_configs=model_configs,
        gene_ids=gene_ids,
        use_batch_labels=False,
    )


def run_layer4_scgpt(
    project_root: Path,
    deg_df: pd.DataFrame,
    layer1: pd.DataFrame,
    pathway_blockade_ref: float,
    config: dict[str, Any],
    out_csv: Path,
    model_dir: Path,
) -> pd.DataFrame:
    from preprocess_scrna import load_scrna

    scgpt_h5ad = (config.get("scgpt_scrna_input_path") or "").strip()
    load_cfg = dict(config)
    if scgpt_h5ad:
        load_cfg["scrna_input_path"] = scgpt_h5ad
        log.info("scGPT 使用独立单细胞 h5ad: %s", scgpt_h5ad)

    cond_col = (config.get("scgpt_condition_column") or "").strip() or str(config["condition_column"])
    ctrl = (config.get("scgpt_control_label") or "").strip() or str(config["control_label"])
    treat = (config.get("scgpt_treatment_label") or "").strip() or str(config["treatment_label"])
    seed = int(config.get("random_seed", 42))
    max_cells = int(config.get("scgpt_max_cells", 4000))
    batch_size = int(config.get("scgpt_batch_size", 32))
    max_length = int(config.get("scgpt_max_length", 1200))
    gene_col = str(config.get("scgpt_gene_col", "index"))
    delta = float(config.get("scgpt_virtual_delta", 0.45))
    use_flash = bool(config.get("scgpt_use_flash_attention", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log.warning("未检测到 GPU，scGPT 推理将使用 CPU（较慢）")

    ad_full = load_scrna(project_root, load_cfg)
    if cond_col not in ad_full.obs:
        raise KeyError(f"全量 h5ad 缺少 obs 列 {cond_col}")

    ctrl, treat = _resolve_condition_labels(ad_full.obs, cond_col, ctrl, treat)
    m = _subsample_mask(ad_full.obs, cond_col, ctrl, treat, max_cells, seed)
    ad_sub = ad_full[m].copy()
    ad_sub = _prepare_vocab_columns(ad_sub, gene_col)

    log.info("scGPT 子集: %d 细胞 × %d 基因", ad_sub.n_obs, ad_sub.n_vars)

    model, vocab, model_configs = _load_scgpt_model(
        model_dir, device, use_fast_transformer=use_flash
    )

    Xr = _dense_rows(ad_sub.X)
    emb_real = _embed_matrix(
        Xr,
        ad_sub.var_names.astype(str).tolist(),
        model,
        vocab,
        model_configs,
        gene_col,
        device,
        max_length,
        batch_size,
    )

    condv = ad_sub.obs[cond_col].astype(str).values
    ic = condv == ctrl
    it = condv == treat
    if ic.sum() < 5 or it.sum() < 5:
        raise RuntimeError("对照或处理组细胞过少，无法估计 scGPT blockade 轴")

    ec = emb_real[ic].mean(axis=0)
    et = emb_real[it].mean(axis=0)
    axis = et - ec
    an = np.linalg.norm(axis)
    if an < 1e-8:
        raise RuntimeError("scGPT 嵌入上 treat≈ctrl，无法定义 blockade 轴")
    axis = axis / an

    proj_c = emb_real[ic] @ axis
    proj_t = emb_real[it] @ axis
    med_c = float(np.median(proj_c))
    med_t = float(np.median(proj_t))
    if med_t < med_c:
        axis = -axis
        proj_c = emb_real[ic] @ axis
        proj_t = emb_real[it] @ axis
        med_c = float(np.median(proj_c))
        med_t = float(np.median(proj_t))
    spread = max(med_t - med_c, 1e-6)

    # 虚拟表达：对照均值 + 肽相关强度 × (处理–对照) 表达位移，并对 DEG 加权微调
    base = Xr[ic].mean(axis=0)
    shift = Xr[it].mean(axis=0) - base
    gnames = ad_sub.var_names.astype(str).tolist()
    g_to_j = {g: j for j, g in enumerate(gnames)}

    lfc_map: dict[str, float] = {}
    if deg_df is not None and len(deg_df) > 0 and "gene" in deg_df.columns:
        for _, r in deg_df.iterrows():
            g = str(r["gene"])
            lfc_map[g] = float(pd.to_numeric(r.get("logfoldchanges", 0.0), errors="coerce") or 0.0)

    top_deg = []
    if len(deg_df) > 0 and "pvals_adj" in deg_df.columns and "gene" in deg_df.columns:
        dd = deg_df.copy()
        dd["_p"] = pd.to_numeric(dd["pvals_adj"], errors="coerce").fillna(1.0)
        dd = dd[np.isfinite(dd["_p"])].nsmallest(400, "_p")
        top_deg = [str(x) for x in dd["gene"].tolist()]

    virt_rows: list[np.ndarray] = []
    for _, row in layer1.iterrows():
        feats = peptide_perturbation_features(row)
        bind = float(row.get("binding_score", 0.5))
        iface = float(feats["interface_blocking_score"])
        pep_strength = float(
            pathway_blockade_ref * 0.35 + bind * 0.35 + iface * 0.30
        )
        v = base + delta * np.clip(pep_strength, 0, 1) * shift
        for g in top_deg:
            j = g_to_j.get(g)
            if j is None:
                continue
            lg = lfc_map.get(g, 0.0)
            bump = 0.15 * np.sign(lg) * np.log1p(abs(lg)) * bind * iface
            v[j] = max(0.0, v[j] + bump)
        virt_rows.append(np.clip(v, 0.0, None))

    Xv = np.stack(virt_rows, axis=0).astype(np.float32)
    emb_virt = _embed_matrix(
        Xv,
        gnames,
        model,
        vocab,
        model_configs,
        gene_col,
        device,
        max_length,
        min(batch_size, max(8, Xv.shape[0])),
    )

    emb_center = emb_real.mean(axis=0)
    treat_pull = float(np.mean((emb_real[it] - emb_center) @ axis))

    rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(layer1.iterrows()):
        feats = peptide_perturbation_features(row)
        bind = float(row.get("binding_score", 0.5))
        pv = float((emb_virt[i] - emb_center) @ axis)
        blockade_sim = float(np.clip((pv - med_c) / spread, 0.0, 1.0))
        align = float(np.dot(emb_virt[i] - ec, et - ec) / (np.linalg.norm(et - ec) + 1e-9) / (np.linalg.norm(emb_virt[i] - ec) + 1e-9))
        align = float(np.clip((align + 1) / 2, 0, 1))

        t_act = float(np.clip(0.35 + 0.65 * blockade_sim * align * feats["peptide_perturbation_score"], 0, 1))
        ifng = float(np.clip(0.3 + 0.7 * blockade_sim * bind * (0.5 + 0.5 * float(treat_pull > 0)), 0, 1))
        cyto = float(np.clip(0.3 + 0.65 * blockade_sim * feats["stability_proxy"], 0, 1))
        tum_sup = float(np.clip(0.25 + 0.75 * blockade_sim * (1.0 - float(row.get("hydrophobic_ratio", 0.5))), 0, 1))
        exh_down = float(np.clip(0.35 + 0.65 * bind * feats["interface_blocking_score"] * blockade_sim, 0, 1))
        immune_activation = float(np.mean([t_act, ifng, cyto, exh_down]))

        rows.append(
            dict(
                peptide_id=row["peptide_id"],
                sequence=row.get("sequence", ""),
                binding_score=bind,
                peptide_perturbation_score=feats["peptide_perturbation_score"],
                interface_blocking_score=feats["interface_blocking_score"],
                blockade_similarity_score=blockade_sim,
                T_cell_activation_prediction=t_act,
                IFNG_prediction=ifng,
                cytotoxicity_prediction=cyto,
                tumor_suppression_prediction=tum_sup,
                exhaustion_down_prediction=exh_down,
                immune_activation_score=immune_activation,
                scgpt_axis_projection=pv,
                scgpt_blockade_axis_spread=spread,
            )
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("scGPT 第四层已写入 %s", out_csv)
    return df

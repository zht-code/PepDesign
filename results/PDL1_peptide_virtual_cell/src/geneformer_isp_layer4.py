"""第四层：Geneformer 官方 ``InSilicoPerturber`` 完整前向 + 方案 A（多肽强度缩放）。

Geneformer 仅支持 **基因级** delete/overexpress 等扰动；多肽 **不进入** Transformer。
本模块对 CD274（默认 Ensembl ``ENSG00000196776``）做 ``delete`` 扰动后，从 ISP 输出的
``*_raw.pickle`` 中读取 **CLS 嵌入余弦相似度**（扰动前后），定义基线位移强度
``embed_shift = 1 - mean(cosine)``，再对每条肽用 Layer1 导出的强度 ``s∈[0,1]`` 做::

    embed_shift_peptide = s * embed_shift

并据此构造与 ``foundation_perturb_layer4`` 兼容的 ``predicted_cd274_*`` 与第四层综合分列。

**前置**：先用 ``scripts/tokenize_h5ad_for_geneformer.py`` 生成磁盘 ``*.dataset``，
config 中 ``geneformer_isp_token_data_dir`` 指向 **包含** 该 ``.dataset`` 目录的文件夹；
``models/perturb_virtual_cell/geneformer`` 为含权重与 ``token_dictionary.pkl`` 等的完整快照。
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _foundation_geneformer_dir(project_root: Path, config: dict[str, Any]) -> Path:
    root = project_root / (config.get("perturb_vc_models_dir") or "models/perturb_virtual_cell")
    return root / "geneformer"


def _load_isp_cell_cosine_pickles(isp_out_dir: Path, perturb_type: str, output_prefix: str) -> list[float]:
    """合并所有 batch 的 ``(pert_token, 'cell_emb')`` 余弦列表（CLS 模式下每细胞一项）。"""
    prefix = f"in_silico_{perturb_type}_{output_prefix}"
    vals: list[float] = []
    for name in os.listdir(isp_out_dir):
        if not name.startswith(prefix) or "cell_embs_dict" not in name or not name.endswith("_raw.pickle"):
            continue
        path = isp_out_dir / name
        with open(path, "rb") as fp:
            d = pickle.load(fp)
        for key, li in d.items():
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            if key[1] != "cell_emb":
                continue
            for v in li:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
    return vals


def _isp_embed_shift_base(cosines: list[float]) -> tuple[float, float]:
    if not cosines:
        return float("nan"), float("nan")
    arr = np.clip(np.asarray(cosines, dtype=np.float64), -1.0, 1.0)
    mean_cos = float(np.mean(arr))
    shift = float(np.clip(1.0 - mean_cos, 0.0, 2.0))
    return mean_cos, shift


def run_layer4_geneformer_isp(
    project_root: Path,
    layer1: pd.DataFrame,
    pathway_blockade_ref: float,
    config: dict[str, Any],
    out_csv: Path,
) -> pd.DataFrame:
    from similarity_analysis import peptide_perturbation_features

    try:
        from geneformer import InSilicoPerturber
    except ImportError as e:
        raise RuntimeError(
            "未安装 geneformer 包，无法运行 geneformer_isp。请安装官方仓库："
            'pip install "git+https://huggingface.co/ctheodoris/Geneformer"'
        ) from e

    model_dir = Path(config.get("geneformer_isp_model_dir") or _foundation_geneformer_dir(project_root, config))
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Geneformer 模型目录不存在: {model_dir}")

    token_parent = Path(config.get("geneformer_isp_token_data_dir") or "").expanduser()
    if not str(token_parent).strip() or not token_parent.is_dir():
        raise FileNotFoundError(
            "请在 config 中设置 geneformer_isp_token_data_dir 为包含 tokenized ``*.dataset`` 文件夹的目录 "
            "（先运行 scripts/tokenize_h5ad_for_geneformer.py）。"
        )

    isp_out = Path(
        config.get("geneformer_isp_output_dir") or (project_root / "results" / "geneformer_isp")
    )
    isp_out.mkdir(parents=True, exist_ok=True)
    isp_prefix = str(config.get("geneformer_isp_output_prefix") or "cd274_delete").strip() or "cd274_delete"

    ensembl = str(config.get("geneformer_isp_ensembl_gene") or "ENSG00000196776").strip()
    perturb_type = str(config.get("geneformer_isp_perturb_type") or "delete").strip().lower()
    if perturb_type not in ("delete", "overexpress", "inhibit", "activate"):
        raise ValueError(f"不支持的 geneformer_isp_perturb_type: {perturb_type}")

    # V2 tokenized 数据首 token 为 CLS，emb_mode 须含 cls
    emb_mode = str(config.get("geneformer_isp_emb_mode") or "cls").strip().lower()
    if emb_mode not in ("cls", "cls_and_gene"):
        log.warning("geneformer_isp_emb_mode=%s 可能与 V2 CLS 输入不兼容，已改为 cls", emb_mode)
        emb_mode = "cls"

    model_version = str(config.get("geneformer_isp_model_version") or "V2").strip().upper()
    if model_version not in ("V1", "V2"):
        model_version = "V2"

    forward_bs = int(config.get("geneformer_isp_forward_batch_size", 24))
    nproc = int(config.get("geneformer_isp_nproc", 4))
    max_ncells = config.get("geneformer_isp_max_ncells")
    max_ncells_i = None
    if max_ncells is not None and str(max_ncells).strip().lower() not in ("", "none", "null"):
        try:
            max_ncells_i = int(max_ncells)
        except (TypeError, ValueError):
            max_ncells_i = None

    token_dict_path = config.get("geneformer_isp_token_dictionary_file")
    token_dict_path = str(token_dict_path).strip() if token_dict_path else None
    td_file = token_dict_path if token_dict_path and Path(token_dict_path).is_file() else None
    if td_file is None:
        cands = sorted(model_dir.glob("token_dictionary*.pkl"))
        if cands:
            td_file = str(cands[0].resolve())
            log.info("geneformer_isp: 未配置 token 词典，使用模型目录内 %s", td_file)

    filter_data = config.get("geneformer_isp_filter_data")
    if filter_data is not None and not isinstance(filter_data, dict):
        filter_data = None

    force = bool(config.get("geneformer_isp_force_rerun", False))
    pickle_vals = _load_isp_cell_cosine_pickles(isp_out, perturb_type, isp_prefix)
    if not pickle_vals or force:
        isp = InSilicoPerturber(
            perturb_type=perturb_type,
            genes_to_perturb=[ensembl],
            combos=0,
            model_type="Pretrained",
            num_classes=0,
            emb_mode=emb_mode,
            forward_batch_size=forward_bs,
            nproc=nproc,
            model_version=model_version,
            token_dictionary_file=td_file,
            max_ncells=max_ncells_i,
            filter_data=filter_data,
        )
        log.info(
            "运行 InSilicoPerturber.perturb_data(model=%s, input=%s, out=%s, prefix=%s)",
            model_dir,
            token_parent,
            isp_out,
            isp_prefix,
        )
        isp.perturb_data(str(model_dir), str(token_parent), str(isp_out), isp_prefix)
        pickle_vals = _load_isp_cell_cosine_pickles(isp_out, perturb_type, isp_prefix)

    mean_cos, base_shift = _isp_embed_shift_base(pickle_vals)
    if not np.isfinite(base_shift):
        raise RuntimeError(
            f"未能从 {isp_out} 解析 ISP 输出 pickle（前缀 {isp_prefix}）。请检查 perturb 是否成功、GPU/显存与路径。"
        )

    alpha = float(config.get("geneformer_isp_logfc_alpha", 2.5))
    down_thr = float(config.get("geneformer_isp_down_scaled_shift_thr", 0.015))
    scale_mode = str(config.get("geneformer_isp_peptide_scale_mode") or "combined").strip().lower()
    # combined: 与 foundation 类似的 s；或仅用 peptide_perturbation_score
    pathway_ref = float(np.clip(pathway_blockade_ref, 0.0, 1.0))

    rows: list[dict[str, Any]] = []
    for _, row in layer1.iterrows():
        feats = peptide_perturbation_features(row)
        bind = float(row.get("binding_score", 0.5))
        iface = float(feats["interface_blocking_score"])
        if scale_mode == "peptide_perturbation_score":
            s = float(feats["peptide_perturbation_score"])
        else:
            s = float(np.clip(pathway_ref * 0.35 + bind * 0.35 + iface * 0.30, 0.0, 1.0))

        scaled_shift = float(s * base_shift)
        # 代理 logFC：位移越大（删除 CD274 后 CLS 越偏离），越负
        logfc_proxy = float(-alpha * scaled_shift)
        down = int(scaled_shift >= down_thr)

        # 与伪表达列兼容：用余弦及缩放位移占位，便于下游 CSV 理解
        virt_cos = float(np.clip(mean_cos - scaled_shift * 0.08, -1.0, 1.0))

        blockade_sim = float(
            np.clip(0.5 + 0.5 * (scaled_shift / (base_shift + 1e-9)) * (0.25 + 0.75 * bind), 0.0, 1.0)
        )
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
                geneformer_isp_ensembl_gene=ensembl,
                geneformer_isp_mean_cls_cosine=mean_cos,
                geneformer_isp_embed_shift_base=base_shift,
                geneformer_isp_peptide_strength_s=s,
                geneformer_isp_embed_shift_scaled=scaled_shift,
                foundation_perturb_backend="geneformer_isp",
                predicted_cd274_expr_ctrl_mean=mean_cos,
                predicted_cd274_expr_virtual=virt_cos,
                predicted_cd274_logfc_vs_ctrl=logfc_proxy,
                predicted_cd274_down=down,
                predicted_cd274_down_rule="isp_cls_shift_x_peptide_strength>=thr",
            )
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info(
        "geneformer_isp: mean_cls_cosine=%.4f base_shift=%.4f -> %s",
        mean_cos,
        base_shift,
        out_csv,
    )
    return df

"""第四层：候选肽 virtual cell / signature 相似性。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def peptide_perturbation_features(row: pd.Series) -> dict[str, float]:
    bind = float(row.get("binding_score", 0.5))
    dock = row.get("docking_score")
    mmg = row.get("mmgbsa_score")
    dist = row.get("distance_to_PD1_interface")
    # 稳定性 / 溶解度代理
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


def run_layer4_simple(
    layer1: pd.DataFrame,
    pathway_blockade_ref: float,
    config: dict[str, Any],
    out_csv: Path,
) -> pd.DataFrame:
    rows = []
    for _, row in layer1.iterrows():
        feats = peptide_perturbation_features(row)
        bind = float(row.get("binding_score", 0.5))
        # 与 blockade reference 对齐：结合强则更接近真实扰动
        blockade_sim = np.clip(0.5 * pathway_blockade_ref + 0.5 * bind * feats["interface_blocking_score"], 0, 1)
        t_act = np.clip(0.4 + 0.6 * blockade_sim * feats["peptide_perturbation_score"], 0, 1)
        ifng = np.clip(0.35 + 0.65 * blockade_sim * bind, 0, 1)
        cyto = np.clip(0.35 + 0.65 * blockade_sim * feats["stability_proxy"], 0, 1)
        tum_sup = np.clip(0.3 + 0.7 * blockade_sim * (1.0 - float(row.get("hydrophobic_ratio", 0.5))), 0, 1)
        exh_down = np.clip(0.4 + 0.6 * bind * feats["interface_blocking_score"], 0, 1)
        immune_activation = float(np.mean([t_act, ifng, cyto, exh_down]))

        rows.append(
            dict(
                peptide_id=row["peptide_id"],
                sequence=row.get("sequence", ""),
                binding_score=bind,
                peptide_perturbation_score=feats["peptide_perturbation_score"],
                interface_blocking_score=feats["interface_blocking_score"],
                blockade_similarity_score=float(blockade_sim),
                T_cell_activation_prediction=float(t_act),
                IFNG_prediction=float(ifng),
                cytotoxicity_prediction=float(cyto),
                tumor_suppression_prediction=float(tum_sup),
                exhaustion_down_prediction=float(exh_down),
                immune_activation_score=float(immune_activation),
            )
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def run_layer4(
    layer1: pd.DataFrame,
    pathway_blockade_ref: float,
    config: dict[str, Any],
    out_csv: Path,
    *,
    project_root: Path | None = None,
    deg_df: pd.DataFrame | None = None,
    adata: Any | None = None,
) -> pd.DataFrame:
    """
    根据 virtual_cell_backend 选择第四层实现。

    - ``geneformer_isp``：官方 ``InSilicoPerturber`` 完整前向；多肽仅方案 A 缩放（见 ``geneformer_isp_layer4``）。
    - ``geneformer_pert`` / ``scfoundation_pert``：显式 CD274 虚拟扰动读出，需 ``adata`` 与 ``deg_df``。
    - ``scgpt``：原 scGPT 嵌入轴。
    """
    backend = (config.get("virtual_cell_backend") or "simple_signature").strip().lower()

    if backend == "geneformer_isp":
        if project_root is None:
            log.warning("geneformer_isp 需要 project_root，回退 simple_signature")
            return run_layer4_simple(layer1, pathway_blockade_ref, config, out_csv)
        try:
            from geneformer_isp_layer4 import run_layer4_geneformer_isp

            return run_layer4_geneformer_isp(
                project_root,
                layer1,
                pathway_blockade_ref,
                config,
                out_csv,
            )
        except Exception:
            log.exception("geneformer_isp 第四层失败，回退 simple_signature")
            return run_layer4_simple(layer1, pathway_blockade_ref, config, out_csv)

    if backend in ("geneformer_pert", "scfoundation_pert"):
        if project_root is None or deg_df is None:
            log.warning("%s 需要 project_root 与 deg_df，回退 simple_signature", backend)
            return run_layer4_simple(layer1, pathway_blockade_ref, config, out_csv)
        if adata is None:
            log.warning("%s 需要 adata（主流程预处理后的 AnnData），回退 simple_signature", backend)
            return run_layer4_simple(layer1, pathway_blockade_ref, config, out_csv)
        try:
            from foundation_perturb_layer4 import run_layer4_foundation_perturb

            return run_layer4_foundation_perturb(
                project_root,
                adata,
                deg_df,
                layer1,
                pathway_blockade_ref,
                config,
                out_csv,
                backend=backend,  # type: ignore[arg-type]
            )
        except Exception:
            log.exception("%s 第四层失败，回退 simple_signature", backend)
            return run_layer4_simple(layer1, pathway_blockade_ref, config, out_csv)

    if backend != "scgpt":
        if backend not in ("simple_signature", "scvi", "geneformer", "geneformer_isp"):
            log.warning("未知 backend %s，使用 simple_signature", backend)
        elif backend != "simple_signature":
            log.warning("backend=%s 尚未集成，使用 simple_signature。", backend)
        return run_layer4_simple(layer1, pathway_blockade_ref, config, out_csv)

    if project_root is None or deg_df is None:
        log.warning("scGPT 需要 project_root 与 deg_df，回退 simple_signature")
        return run_layer4_simple(layer1, pathway_blockade_ref, config, out_csv)

    from scgpt_checkpoint import ensure_scgpt_checkpoint

    model_dir = ensure_scgpt_checkpoint(project_root, config)
    if model_dir is None:
        log.warning("scGPT 权重未就绪，回退 simple_signature")
        return run_layer4_simple(layer1, pathway_blockade_ref, config, out_csv)

    try:
        from scgpt_layer4 import run_layer4_scgpt

        return run_layer4_scgpt(
            project_root,
            deg_df,
            layer1,
            pathway_blockade_ref,
            config,
            out_csv,
            model_dir,
        )
    except Exception:
        log.exception("scGPT 第四层失败，回退 simple_signature")
        return run_layer4_simple(layer1, pathway_blockade_ref, config, out_csv)

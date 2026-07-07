#!/usr/bin/env python3
"""
五层验证总入口：默认无真实单细胞时使用 demo 数据跑通。

用法：
  cd PDL1_peptide_virtual_cell
  conda activate scgpt   # 第四层 virtual_cell_backend: scgpt 时需要
  python src/run_pipeline.py
  python src/run_pipeline.py --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import docking_parser  # noqa: E402
import download_data  # noqa: E402
import pathway_scoring  # noqa: E402
import peptide_features  # noqa: E402
import preprocess_scrna  # noqa: E402
import risk_scoring  # noqa: E402
import similarity_analysis  # noqa: E402
import visualization  # noqa: E402
import virtual_cell_signature  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("run_pipeline")


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = np.nanmin(s), np.nanmax(s)
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def recommendation(final_score: float, tox: float) -> str:
    if final_score >= 0.75 and tox < 0.30:
        return "Strong candidate"
    if final_score >= 0.60 and tox < 0.50:
        return "Moderate candidate"
    return "Not recommended"


def write_final_report(path: Path, final_df: pd.DataFrame, pathway_ref: float) -> None:
    try:
        md_table = final_df.to_markdown(index=False)
    except Exception:
        md_table = final_df.to_string(index=False)
    lines = [
        "# PD-L1 多肽虚拟细胞五层验证 — 总结报告",
        "",
        "## 参考扰动（阳性对照）",
        "",
        "在缺少「PD-L1 多肽直接处理」单细胞数据时，可使用 **anti-PD-L1、anti-PD-1 或 PD-L1/TGFβ 双抗（如 Bintrafusp alfa）**",
        "治疗前后样本作为 **功能等效阳性扰动**：其共同生物学后果包含削弱 PD-1/PD-L1 轴、恢复 T 细胞效应程序。",
        "本 pipeline 将该类数据用于构建 **reference transcriptional signature**，并与候选多肽的理化/对接特征在虚拟细胞层做类比评分。",
        "",
        f"## 本次 global pathway blockade 参考分 (0–1)：{pathway_ref:.3f}",
        "",
        "## 候选排序",
        "",
        "```",
        md_table,
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    project_root = PROJECT_ROOT
    out_dir = project_root / cfg.get("output_dir", "results")
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    rep_dir = out_dir / "reports"
    for d in (fig_dir, tab_dir, rep_dir):
        d.mkdir(parents=True, exist_ok=True)

    download_data.check_raw_directory(project_root)
    download_data.print_geo_instructions(cfg, project_root)

    ad_raw = preprocess_scrna.load_scrna(project_root, cfg)
    adata = preprocess_scrna.preprocess(ad_raw, cfg)
    adata = pathway_scoring.score_pathways(adata, cfg)
    adata = risk_scoring.score_risks(adata, cfg)

    _layer2_df, pathway_ref = pathway_scoring.summarize_pathway_blockade(
        adata, cfg, tab_dir / "layer2_pathway_blockade.csv"
    )

    deg_df = virtual_cell_signature.build_reference_signature(
        adata,
        cfg,
        tab_dir / "layer3_reference_signature_DEG.csv",
        tab_dir / "layer3_signature_genes_up.txt",
        tab_dir / "layer3_signature_genes_down.txt",
    )

    risk_cohort_df = risk_scoring.summarize_risks(adata, cfg, tab_dir / "layer5_cohort_risk_scores.csv")

    pep_csv = project_root / cfg.get("peptide_table", "peptides/candidate_peptides.csv")
    dock_path = project_root / cfg.get("docking_results_path", "") if cfg.get("docking_results_path") else None
    docking_df = docking_parser.load_docking_table(
        Path(dock_path) if dock_path and str(dock_path).strip() else None
    )
    layer1 = peptide_features.run_layer1(pep_csv, docking_df, tab_dir / "layer1_binding_validation.csv")

    layer4 = similarity_analysis.run_layer4(
        layer1,
        pathway_ref,
        cfg,
        tab_dir / "layer4_peptide_virtual_cell_similarity.csv",
        project_root=project_root,
        deg_df=deg_df,
        adata=adata,
    )

    layer5_pep = risk_scoring.peptide_toxicity_prediction(layer1, risk_cohort_df)
    layer5_pep.to_csv(tab_dir / "layer5_toxicity_risk.csv", index=False)

    merged = layer4.merge(layer5_pep, on="peptide_id", how="left")
    merged["pathway_blockade_score"] = float(np.clip(pathway_ref, 0, 1))
    merged["binding_score_n"] = _norm01(merged["binding_score"])
    merged["pathway_blockade_score_n"] = _norm01(merged["pathway_blockade_score"])
    merged["blockade_similarity_score_n"] = _norm01(merged["blockade_similarity_score"])
    merged["immune_activation_score_n"] = _norm01(merged["immune_activation_score"])
    merged["safety_score_n"] = _norm01(merged["safety_score"])

    w_bind, w_path, w_sim, w_imm, w_safe = 0.25, 0.20, 0.25, 0.15, 0.15
    merged["final_score"] = (
        w_bind * merged["binding_score_n"]
        + w_path * merged["pathway_blockade_score_n"]
        + w_sim * merged["blockade_similarity_score_n"]
        + w_imm * merged["immune_activation_score_n"]
        + w_safe * merged["safety_score_n"]
    )
    merged["recommendation"] = [
        recommendation(fs, float(t)) for fs, t in zip(merged["final_score"], merged["toxicity_risk_score"])
    ]

    final_cols = [
        "peptide_id",
        "sequence",
        "binding_score",
        "pathway_blockade_score",
        "blockade_similarity_score",
        "immune_activation_score",
        "predicted_cd274_logfc_vs_ctrl",
        "predicted_cd274_down",
        "toxicity_risk_score",
        "safety_score",
        "final_score",
        "recommendation",
    ]
    final_df = merged[[c for c in final_cols if c in merged.columns]]
    final_df.to_csv(tab_dir / "final_candidate_ranking.csv", index=False)

    n_plot_pep = int(cfg.get("plot_max_peptides", 50))
    top_peptide_ids = (
        final_df.sort_values("final_score", ascending=False)
        .head(max(1, n_plot_pep))["peptide_id"]
        .tolist()
    )

    write_final_report(rep_dir / "final_report.md", final_df, pathway_ref)

    visualization.plot_umap_condition(adata, cfg["condition_column"], fig_dir / "layer3_umap_condition", config=cfg)
    visualization.plot_umap_celltype(adata, cfg["celltype_column"], fig_dir / "layer3_umap_celltype", config=cfg)
    visualization.plot_pathway_boxplot(adata, cfg, fig_dir / "layer2_pathway_score_boxplot")
    _groups = cfg.get("deg_volcano_annotation_groups")
    _disp = cfg.get("deg_volcano_gene_display")
    _lbl = cfg.get("deg_volcano_label_genes")
    _prior_down = cfg.get("deg_volcano_prior_down_genes")
    _prior_cap = cfg.get("deg_volcano_prior_down_caption")

    def _cfg_float_volc(volc_key: str, main_key: str, default: float) -> float:
        """火山图专用阈值；未配置时回落到第三层签名用 deg_*。"""
        v = cfg.get(volc_key)
        if v is None or (isinstance(v, str) and not str(v).strip()):
            v = cfg.get(main_key, default)
        return float(v)

    def _volcano_xlim(cfg: dict) -> tuple[float, float] | None:
        xl = cfg.get("deg_volcano_xlim")
        if xl is None:
            return None
        if isinstance(xl, (list, tuple)) and len(xl) == 2:
            try:
                return (float(xl[0]), float(xl[1]))
            except (TypeError, ValueError):
                return None
        return None

    def _volcano_figsize(cfg: dict) -> tuple[float, float] | None:
        fs = cfg.get("deg_volcano_figsize")
        if fs is None:
            return None
        if isinstance(fs, (list, tuple)) and len(fs) == 2:
            try:
                return (float(fs[0]), float(fs[1]))
            except (TypeError, ValueError):
                return None
        return None

    _volc_deg = deg_df
    _excl = cfg.get("deg_volcano_exclude_genes")
    if isinstance(_excl, (list, tuple)) and _excl and deg_df is not None and "gene" in deg_df.columns:
        _syms = {str(x).strip() for x in _excl if str(x).strip()}
        if _syms:
            _volc_deg = deg_df[~deg_df["gene"].isin(_syms)].copy()

    visualization.plot_volcano(
        _volc_deg,
        fig_dir / "layer3_volcano_blockade_signature",
        padj_thr=_cfg_float_volc("deg_volcano_padj_max", "deg_padj_max", 0.05),
        abs_lfc_thr=_cfg_float_volc("deg_volcano_abs_logfc_min", "deg_abs_logfc_min", 0.5),
        padj_thr_vline=_cfg_float_volc("deg_volcano_vline_padj", "deg_padj_max", 0.05),
        abs_lfc_thr_vline=_cfg_float_volc("deg_volcano_vline_abs_lfc", "deg_abs_logfc_min", 0.5),
        ymax=float(cfg.get("deg_volcano_ymax", 100)),
        xlim=_volcano_xlim(cfg),
        figsize=_volcano_figsize(cfg),
        label_groups=list(_groups) if isinstance(_groups, list) and len(_groups) > 0 else None,
        gene_display=dict(_disp) if isinstance(_disp, dict) else None,
        label_genes=list(_lbl) if isinstance(_lbl, (list, tuple)) else None,
        prior_down_genes=list(_prior_down) if isinstance(_prior_down, (list, tuple)) else None,
        prior_down_caption=str(_prior_cap).strip() if isinstance(_prior_cap, str) else None,
    )
    visualization.plot_peptide_binding_rank(
        layer1, fig_dir / "layer1_peptide_binding_ranking", peptide_ids=top_peptide_ids, config=cfg
    )
    visualization.plot_blockade_similarity_rank(
        layer4, fig_dir / "layer4_peptide_ranking_barplot", peptide_ids=top_peptide_ids, config=cfg
    )
    _hm_cmap = str(cfg.get("plot_similarity_heatmap_cmap", "viridis")).strip() or "viridis"
    visualization.plot_similarity_heatmap(
        layer4,
        fig_dir / "layer4_similarity_heatmap",
        peptide_ids=top_peptide_ids,
        cmap=_hm_cmap,
    )
    visualization.plot_risk_boxplot(adata, cfg, fig_dir / "layer5_risk_score_boxplot")
    visualization.plot_peptide_safety(
        layer5_pep, fig_dir / "layer5_peptide_safety_ranking", peptide_ids=top_peptide_ids, config=cfg
    )
    visualization.plot_final_ranking(
        final_df, fig_dir / "final_candidate_ranking_barplot", peptide_ids=top_peptide_ids, config=cfg
    )

    log.info("完成。表: %s, 图: %s", tab_dir, fig_dir)


if __name__ == "__main__":
    main()

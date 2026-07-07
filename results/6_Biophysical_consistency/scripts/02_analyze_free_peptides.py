#!/usr/bin/env python3
"""
02 — 游离肽（free-state）结构可行性分析

读取 ``Table_S1_master_sequence_table.csv`` 中 ``usable_for_free_structure_analysis=True`` 的行，
对 ``free_structure_path`` 使用 Bio.PDB + 启发式几何指标进行分析。

输出：
  - ``tables/Table_S2_free_peptide_structure_metrics.csv``
  - ``intermediate/free_peptide_metrics_per_model/*.json``（每条肽一个 JSON）
  - ``intermediate/free_peptide_qc/``（跳过/失败清单）
  - ``logs/free_peptide_analysis.log``
  - ``tables/free_peptide_metric_definitions.md``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.free_peptide_structure import (
    FreePeptideMetrics,
    analyze_free_peptide_pdb,
    write_metric_definitions,
)
from utils.paths import ProjectPaths, load_config


def _truthy(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "yes")


def _setup_file_log(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("free_peptide_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--master-csv",
        type=Path,
        default=None,
        help="默认：tables/Table_S1_master_sequence_table.csv",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="默认：config.paths.tables",
    )
    p.add_argument(
        "--intermediate-dir",
        type=Path,
        default=None,
        help="默认：config.paths.intermediate",
    )
    p.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="默认：config.paths.logs/free_peptide_analysis.log",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0 表示处理全部可用行；>0 用于快速调试。",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_dirs()

    tables = args.tables_dir or paths.tables
    inter = args.intermediate_dir or paths.intermediate
    log_path = args.log_path or (paths.logs / "free_peptide_analysis.log")
    master_csv = args.master_csv or (tables / "Table_S1_master_sequence_table.csv")

    per_dir = inter / "free_peptide_metrics_per_model"
    qc_dir = inter / "free_peptide_qc"
    per_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_file_log(log_path)
    log.info("master_csv=%s", master_csv)

    if not master_csv.exists():
        log.error("Missing master table: %s", master_csv)
        return 2

    write_metric_definitions(tables / "free_peptide_metric_definitions.md")

    df = pd.read_csv(master_csv)
    if "usable_for_free_structure_analysis" not in df.columns:
        log.error("Master table missing usable_for_free_structure_analysis column")
        return 2

    df = df.copy()
    df["_s1_row_index"] = df.index.astype(int)
    mask = df["usable_for_free_structure_analysis"].map(_truthy)
    sub = df.loc[mask].copy()
    skipped = df.loc[~mask].copy()

    skipped_rows = []
    for _, r in skipped.iterrows():
        skipped_rows.append(
            {
                "s1_row_index": int(r["_s1_row_index"]),
                "target_id": r.get("target_id", ""),
                "peptide_id": r.get("peptide_id", ""),
                "reason": "usable_for_free_structure_analysis_false",
            }
        )

    path_missing = []
    rows_out: list[dict] = []
    n_limit = args.max_rows if args.max_rows and args.max_rows > 0 else None

    processed = 0
    for _, r in sub.iterrows():
        if n_limit is not None and processed >= n_limit:
            break
        idx = int(r["_s1_row_index"])
        tid = str(r.get("target_id", "") or "")
        pid = str(r.get("peptide_id", "") or "")
        pth = Path(str(r.get("free_structure_path", "") or "").strip())
        seq_table = str(r.get("sequence", "") or "")

        if not str(pth):
            path_missing.append(
                {
                    "s1_row_index": idx,
                    "target_id": tid,
                    "peptide_id": pid,
                    "reason": "empty_free_structure_path",
                }
            )
            continue
        if not pth.exists():
            path_missing.append(
                {
                    "s1_row_index": idx,
                    "target_id": tid,
                    "peptide_id": pid,
                    "reason": "file_not_found",
                    "path": str(pth),
                }
            )
            continue

        try:
            met = analyze_free_peptide_pdb(pth, tid, pid, idx, seq_table)
        except Exception as e:
            log.exception("row %s failed: %s %s", idx, tid, pid)
            met = FreePeptideMetrics(
                target_id=tid,
                peptide_id=pid,
                free_structure_path=str(pth),
                s1_row_index=idx,
                residue_count=0,
                atom_count=0,
                helix_frac=0.0,
                sheet_frac=0.0,
                coil_frac=0.0,
                n_classified_dihedrals=0,
                clash_count=0,
                severe_clash_count=0,
                approximate_backbone_strain_score=0.0,
                torsion_proxy_score=0.0,
                intrapeptide_hbond_count=0,
                backbone_hbond_count=0,
                sidechain_hbond_count=0,
                hydrophobic_residue_count=0,
                hydrophobic_cluster_count=0,
                longest_hydrophobic_run=0,
                buried_hydrophobic_proxy=0,
                buried_hydrophobic_fraction=0.0,
                hydrophobic_cohesion_score=0.0,
                cysteine_count=0,
                disulfide_candidate_count=0,
                disulfide_feasible_flag=False,
                analysis_status="error",
                error_message=str(e)[:500],
                pdb_sequence_inferred="",
            )

        d = met.as_flat_dict()
        d["group"] = r.get("group", "")
        d["sequence_table"] = seq_table
        d["length_table"] = r.get("length", "")
        if seq_table and met.pdb_sequence_inferred and seq_table.replace(" ", "") != met.pdb_sequence_inferred.replace(
            " ", ""
        ):
            d["notes_s1_vs_pdb"] = "sequence_mismatch_table_vs_pdb"
        else:
            d["notes_s1_vs_pdb"] = ""

        fname = f"{idx:06d}_{tid}_{pid}.json"
        fname = "".join(c if c.isalnum() or c in "._-" else "_" for c in fname)[:200]
        jpath = per_dir / fname
        jpath.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")

        rows_out.append(d)
        processed += 1
        if processed % 200 == 0:
            log.info("processed %s structures", processed)

    pd.DataFrame(skipped_rows).to_csv(qc_dir / "skipped_not_usable.csv", index=False)
    pd.DataFrame(path_missing).to_csv(qc_dir / "skipped_missing_path.csv", index=False)

    out_df = pd.DataFrame(rows_out)
    if not out_df.empty:
        # 列顺序：标识 + 主表信息 + 指标
        preferred = [
            "s1_row_index",
            "target_id",
            "peptide_id",
            "group",
            "sequence_table",
            "length_table",
            "free_structure_path",
            "analysis_status",
            "error_message",
            "residue_count",
            "atom_count",
            "helix_frac",
            "sheet_frac",
            "coil_frac",
            "n_classified_dihedrals",
            "clash_count",
            "severe_clash_count",
            "approximate_backbone_strain_score",
            "torsion_proxy_score",
            "intrapeptide_hbond_count",
            "backbone_hbond_count",
            "sidechain_hbond_count",
            "hydrophobic_residue_count",
            "hydrophobic_cluster_count",
            "longest_hydrophobic_run",
            "buried_hydrophobic_proxy",
            "buried_hydrophobic_fraction",
            "hydrophobic_cohesion_score",
            "cysteine_count",
            "disulfide_candidate_count",
            "disulfide_feasible_flag",
            "pdb_sequence_inferred",
            "notes_s1_vs_pdb",
        ]
        cols = [c for c in preferred if c in out_df.columns] + [
            c for c in out_df.columns if c not in preferred
        ]
        out_df = out_df[cols]
    out_csv = tables / "Table_S2_free_peptide_structure_metrics.csv"
    out_df.to_csv(out_csv, index=False)

    summary = {
        "n_master_rows": int(len(df)),
        "n_usable_flag_true": int(mask.sum()),
        "n_analyzed_written": int(len(out_df)),
        "n_skipped_not_usable": len(skipped_rows),
        "n_skipped_missing_path": len(path_missing),
        "per_model_json_dir": str(per_dir.resolve()),
        "table_s2_csv": str(out_csv.resolve()),
    }
    (qc_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("Done. analyzed=%s table_s2=%s", len(out_df), out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

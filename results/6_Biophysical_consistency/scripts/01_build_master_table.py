#!/usr/bin/env python3
"""
01 — 构建样本主表（Table S1）

依据 ``data_inventory/suggested_inputs.json`` 与 ``all_samples.csv`` / ``baseline_input_index.csv`` /
``clean_properties/*.json`` / 对接 ``model_*.pdb`` 索引，生成后续分析唯一入口主表。

输出（默认 ``tables/``）：
  - ``Table_S1_master_sequence_table.csv``
  - ``Table_S1_master_sequence_table.json``
  - ``master_table_report.md``
  - 若无 decoy：``decoy_generation_plan.md``
  - ``intermediate/01_master_table_meta.json``（构建元数据）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logging_utils import setup_run_logger
from utils.master_table import (
    build_master_table,
    load_suggested_inputs,
    write_decoy_generation_plan,
    write_master_table_report,
)
from utils.paths import ProjectPaths, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--inventory-dir",
        type=Path,
        default=None,
        help="默认：config.paths.data_inventory（读取 suggested_inputs.json）。",
    )
    p.add_argument(
        "--output-intermediate",
        type=Path,
        default=None,
        help="默认：config.paths.intermediate。",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="默认：config.paths.tables。",
    )
    p.add_argument("--log-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_dirs()

    inv = args.inventory_dir or paths.data_inventory
    inter = args.output_intermediate or paths.intermediate
    tables = args.tables_dir or paths.tables
    log_dir = args.log_dir or paths.logs

    log = setup_run_logger(log_dir, "01_build_master_table")
    project_root = Path(cfg.get("project_root", "")).expanduser().resolve()

    suggested_path = inv / "suggested_inputs.json"
    suggested = load_suggested_inputs(suggested_path)
    if not suggested:
        log.warning("Missing suggested_inputs.json at %s — using defaults only", suggested_path)

    log.info("project_root=%s", project_root)
    master, meta = build_master_table(project_root, suggested)

    inter.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    csv_out = tables / "Table_S1_master_sequence_table.csv"
    json_out = tables / "Table_S1_master_sequence_table.json"
    report_out = tables / "master_table_report.md"
    decoy_plan = tables / "decoy_generation_plan.md"

    master.to_csv(csv_out, index=False)
    # JSON：records，布尔保持
    master_json = master.replace({pd.NA: None})
    json_out.write_text(
        master_json.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta_path = inter / "01_master_table_meta.json"
    meta["outputs"] = {
        "csv": str(csv_out.resolve()),
        "json": str(json_out.resolve()),
        "report": str(report_out.resolve()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    write_master_table_report(report_out, master, meta)

    n_decoy = int((master["group"] == "decoy").sum()) if not master.empty else 0
    if n_decoy == 0:
        write_decoy_generation_plan(decoy_plan)
        log.info("No decoy rows — wrote %s", decoy_plan)
    else:
        if decoy_plan.exists():
            decoy_plan.unlink()
        log.info("Decoy rows present (%s) — skipped decoy_generation_plan.md", n_decoy)

    log.info("Wrote Table S1: %s (%s rows)", csv_out, len(master))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

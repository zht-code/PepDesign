#!/usr/bin/env python3
"""
用途：合并 all_samples 与 hdock 复合物模型路径，生成统一样本主表。

输入：
  - Peptide_3D/results/5_robustness/baseline/raw_results/*/all_samples.csv
  - Peptide_3D/results/5_robustness/baseline/tables/baseline_input_index.csv（可选，用于肽链提示）
  - cache/hdock_work/**/model_*.pdb

输出：
  - 01_manifest/sample_master_table.csv
  - 01_manifest/sample_master_table_summary.json

运行示例：
  python scripts/01_build_manifest.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biophysical_consistency.config_loader import load_merged_config
from biophysical_consistency.logging_utils import setup_file_logger
from biophysical_consistency.manifest import build_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    cfg = load_merged_config(ROOT / "config")
    log = setup_file_logger(ROOT / "logs", name="01_build_manifest")
    pr = Path(cfg["project_root"])
    df = build_manifest(pr, ROOT / "00_discovery/discovery_full.json")
    if args.max_samples > 0:
        m = args.max_samples
        cm = df["complex_model_path"]
        has_cpx = cm.notna() & cm.astype(str).str.strip().ne("")
        df_c = df[has_cpx].head(m)
        df_f = df[~has_cpx].head(m)
        df = pd.concat([df_f, df_c], ignore_index=True)
        df.drop_duplicates(subset=["sample_uid"], inplace=True)
    out_dir = ROOT / "01_manifest"
    write_manifest(df, out_dir)
    log.info("Manifest rows: %s", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

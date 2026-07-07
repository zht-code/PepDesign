#!/usr/bin/env python3
"""
用途：扫描 Peptide_3D 项目下与生成肽相关的 PDB/CSV/JSON/FASTA，并写入发现清单。

输入：
  - config/default_config.yaml

输出：
  - 00_discovery/discovered_files.csv
  - 00_discovery/discovery_summary.json
  - 00_discovery/discovery_full.json

运行示例：
  python scripts/00_discover_inputs.py
  python scripts/00_discover_inputs.py --resume   # 发现步骤无状态，参数忽略即可
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biophysical_consistency.config_loader import load_merged_config
from biophysical_consistency.discovery import discover_under_root, write_discovery
from biophysical_consistency.logging_utils import setup_file_logger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    cfg = load_merged_config(ROOT / "config")
    log = setup_file_logger(ROOT / "logs", name="00_discover_inputs")
    pr = Path(cfg["project_root"])
    payload = discover_under_root(
        pr,
        cfg.get("discovery_subdirs", []),
        cfg.get("extra_globs", []),
    )
    out_dir = ROOT / "00_discovery"
    write_discovery(out_dir, payload)
    log.info("Wrote discovery to %s", out_dir)
    (out_dir / "run_meta.json").write_text(
        json.dumps({"resume": args.resume, "max_samples": args.max_samples}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
主控脚本：按步骤调度生物物理一致性分析。

输入：
  - config/default_config.yaml（可选 local_config.yaml）
  - 环境变量 PEPTIDE_3D_ROOT 覆盖 project_root

输出：
  - 各子目录下 CSV / JSON / 日志 / 图

运行示例：
  python run_pipeline.py --steps all --max-samples 200 --resume
  python run_pipeline.py --steps discover,manifest --resume
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from biophysical_consistency.config_loader import load_merged_config
from biophysical_consistency.logging_utils import setup_file_logger

SCRIPTS_DIR = ROOT / "scripts"


def run_step(name: str, extra: list[str], logger) -> int:
    script = SCRIPTS_DIR / f"{name}.py"
    if not script.exists():
        logger.error("Missing script %s", script)
        return 2
    cmd = [sys.executable, str(script), *extra]
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        logger.error("Step %s failed with code %s", name, proc.returncode)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Biophysical consistency master runner")
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma list: discover,manifest,free_peptide,interface,solubility,plot,all",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 = no limit")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_merged_config(ROOT / "config")
    log = setup_file_logger(ROOT / "logs", name="master")

    common = []
    if args.max_samples > 0:
        common += ["--max-samples", str(args.max_samples)]
    if args.resume:
        common.append("--resume")

    raw = [s.strip() for s in args.steps.split(",") if s.strip()]
    alias = {
        "discover": "00_discover_inputs",
        "manifest": "01_build_manifest",
        "free_peptide": "02_analyze_free_peptide_structures",
        "interface": "03_analyze_complex_interface",
        "solubility": "04_analyze_solubility_aggregation",
        "plot": "05_summarize_and_plot",
    }
    if "all" in raw:
        steps = [
            "00_discover_inputs",
            "01_build_manifest",
            "02_analyze_free_peptide_structures",
            "03_analyze_complex_interface",
            "04_analyze_solubility_aggregation",
            "05_summarize_and_plot",
        ]
    else:
        steps = [alias.get(s, s) for s in raw]

    code = 0
    for step in steps:
        rc = run_step(step, common, log)
        if rc != 0:
            code = rc
            log.error("Stopping pipeline due to failure in %s", step)
            break
    return code


if __name__ == "__main__":
    raise SystemExit(main())

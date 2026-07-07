#!/usr/bin/env python3
"""
用途：合并各步输出，写总表，并生成 PNG+PDF 直方图。

输入：
  - 02_free_peptide/tables/free_peptide_metrics.csv（可选）
  - 03_interface/tables/interface_metrics.csv（可选）
  - 04_solubility_aggregation/tables/sequence_biophysics.csv（可选）
  - 01_manifest/sample_master_table.csv

输出：
  - 05_summary_figures/tables/merged_biophysical_master.csv
  - 05_summary_figures/tables/merged_biophysical_master_summary.json
  - 05_summary_figures/figures/*.png 与 .pdf

运行示例：
  python scripts/05_summarize_and_plot.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biophysical_consistency.logging_utils import setup_file_logger
from biophysical_consistency.plotting import plot_histograms


def _load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    log = setup_file_logger(ROOT / "logs", name="05_summarize")

    man = _load_csv(ROOT / "01_manifest/sample_master_table.csv")
    if man is None:
        log.error("Missing manifest")
        return 2
    free = _load_csv(ROOT / "02_free_peptide/tables/free_peptide_metrics.csv")
    iface = _load_csv(ROOT / "03_interface/tables/interface_metrics.csv")
    sol = _load_csv(ROOT / "04_solubility_aggregation/tables/sequence_biophysics.csv")

    merged = man
    if free is not None:
        merged = merged.merge(free, on="sample_uid", how="left", suffixes=("", "_free"))
    if iface is not None:
        merged = merged.merge(iface, on="sample_uid", how="left", suffixes=("", "_iface"))
    if sol is not None:
        merged = merged.merge(sol, on="sample_uid", how="left", suffixes=("", "_sol"))

    if args.max_samples > 0:
        merged = merged.head(args.max_samples)

    out_dir = ROOT / "05_summary_figures/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "merged_biophysical_master.csv"
    merged.to_csv(csv_path, index=False)
    summary = {
        "n_rows": int(len(merged)),
        "columns": list(merged.columns),
    }
    (out_dir / "merged_biophysical_master_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig_dir = ROOT / "05_summary_figures/figures"
    cols = [
        "radius_gyration_ca",
        "ca_clash_pairs",
        "n_interface_peptide_residues",
        "simple_hbond_pairs_3p5A",
        "gravy",
        "net_charge_ph7",
        "aggregation_hotspot_max",
    ]
    plot_histograms(merged, cols, fig_dir / "Figure_biophysical_histograms")

    log.info("Merged table: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

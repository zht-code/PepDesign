#!/usr/bin/env python3
"""
用途：对 Hdock 等输出的多段 PDB（受体+肽）做界面接触、氢键代理与疏水性差异描述。

输入：
  - 01_manifest/sample_master_table.csv 中的 complex_model_path

输出：
  - 03_interface/tables/interface_metrics.csv
  - 03_interface/tables/interface_metrics_summary.json
  - 03_interface/per_sample/*.json
  - state/03_interface_checkpoint.json

运行示例：
  python scripts/03_analyze_complex_interface.py --max-samples 200 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biophysical_consistency.checkpoint import load_set, mark_done, save_set
from biophysical_consistency.config_loader import load_merged_config
from biophysical_consistency.interface_metrics import analyze_complex
from biophysical_consistency.logging_utils import setup_file_logger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    cfg = load_merged_config(ROOT / "config")
    log = setup_file_logger(ROOT / "logs", name="03_interface")
    ck_path = ROOT / "state/03_interface_checkpoint.json"
    done = load_set(ck_path) if args.resume else set()

    man = ROOT / "01_manifest/sample_master_table.csv"
    if not man.exists():
        log.error("Missing manifest %s", man)
        return 2
    df = pd.read_csv(man)
    cm = df["complex_model_path"]
    sub = df[cm.notna() & cm.astype(str).str.strip().ne("")].copy()
    if args.max_samples > 0:
        sub = sub.head(args.max_samples)

    per_dir = ROOT / "03_interface/per_sample"
    per_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cutoff = float(cfg.get("interface_distance_cutoff", 5.0))
    hdelta = float(cfg.get("hydrophobic_match_delta", 1.2))

    for _, r in sub.iterrows():
        uid = str(r["sample_uid"])
        if uid in done:
            pj = per_dir / f"{uid}.json"
            if pj.exists():
                rows.append(json.loads(pj.read_text(encoding="utf-8")))
            continue
        p = Path(str(r["complex_model_path"]))
        if not p.exists():
            row = {"sample_uid": uid, "status": "missing_file", "complex_model_path": str(p)}
        else:
            try:
                row = analyze_complex(p, cutoff, hdelta)
                row["sample_uid"] = uid
                row["complex_model_path"] = str(p)
                row["method"] = r.get("method")
                row["target_id"] = r.get("target_id")
            except Exception as e:
                row = {
                    "sample_uid": uid,
                    "status": "error",
                    "error": str(e),
                    "complex_model_path": str(p),
                }
        (per_dir / f"{uid}.json").write_text(
            json.dumps(row, indent=2, default=str), encoding="utf-8"
        )
        rows.append(row)
        mark_done(ck_path, uid, done)
        save_set(ck_path, done)

    out_csv = ROOT / "03_interface/tables/interface_metrics.csv"
    out_json = ROOT / "03_interface/tables/interface_metrics_summary.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = {
        "n_processed": len(rows),
        "n_ok": sum(1 for x in rows if x.get("status") == "ok"),
        "checkpoint": str(ck_path),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote %s rows to %s", len(rows), out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

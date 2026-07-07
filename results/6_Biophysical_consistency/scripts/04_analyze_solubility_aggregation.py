#!/usr/bin/env python3
"""
用途：基于一级序列计算溶解度/聚集风险代理（GRAVY、电荷、芳香比例、疏水斑块等）。

输入：
  - 01_manifest/sample_master_table.csv 的 sequence 列

输出：
  - 04_solubility_aggregation/tables/sequence_biophysics.csv
  - 04_solubility_aggregation/tables/sequence_biophysics_summary.json
  - 04_solubility_aggregation/per_sequence/*.json
  - state/04_solubility_checkpoint.json

运行示例：
  python scripts/04_analyze_solubility_aggregation.py --max-samples 1000 --resume
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
from biophysical_consistency.logging_utils import setup_file_logger
from biophysical_consistency.sequence_biophysics import summarize_sequence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    log = setup_file_logger(ROOT / "logs", name="04_solubility")
    ck_path = ROOT / "state/04_solubility_checkpoint.json"
    done = load_set(ck_path) if args.resume else set()

    man = ROOT / "01_manifest/sample_master_table.csv"
    if not man.exists():
        log.error("Missing manifest %s", man)
        return 2
    df = pd.read_csv(man)
    sub = df.copy()
    if args.max_samples > 0:
        sub = sub.head(args.max_samples)

    per_dir = ROOT / "04_solubility_aggregation/per_sequence"
    per_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in sub.iterrows():
        uid = str(r["sample_uid"])
        if uid in done:
            pj = per_dir / f"{uid}.json"
            if pj.exists():
                rows.append(json.loads(pj.read_text(encoding="utf-8")))
            continue
        seq = r.get("sequence", "")
        met = summarize_sequence(str(seq) if seq is not None else "")
        row = {
            "sample_uid": uid,
            "method": r.get("method"),
            "target_id": r.get("target_id"),
            "candidate_id": r.get("candidate_id"),
            "sequence": str(seq) if seq is not None else "",
            **{k: v for k, v in met.items()},
        }
        (per_dir / f"{uid}.json").write_text(
            json.dumps(row, indent=2, default=str), encoding="utf-8"
        )
        rows.append(row)
        mark_done(ck_path, uid, done)
        save_set(ck_path, done)

    out_csv = ROOT / "04_solubility_aggregation/tables/sequence_biophysics.csv"
    out_json = ROOT / "04_solubility_aggregation/tables/sequence_biophysics_summary.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = {
        "n_processed": len(rows),
        "checkpoint": str(ck_path),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote %s rows to %s", len(rows), out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

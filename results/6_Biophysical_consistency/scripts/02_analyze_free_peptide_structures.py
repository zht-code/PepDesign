#!/usr/bin/env python3
"""
用途：对 manifest 中的 free_peptide / pdb_path 结构做几何与粗粒度可折叠性代理指标。

输入：
  - 01_manifest/sample_master_table.csv
  - PDB 文件（只读，路径来自表）

输出：
  - 02_free_peptide/tables/free_peptide_metrics.csv
  - 02_free_peptide/tables/free_peptide_metrics_summary.json
  - 02_free_peptide/per_sample/*.json（每个样本一行结果，便于断点）
  - state/02_free_peptide_checkpoint.json

运行示例：
  python scripts/02_analyze_free_peptide_structures.py --max-samples 500 --resume
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
from biophysical_consistency.free_peptide_metrics import analyze_pdb_path
from biophysical_consistency.logging_utils import setup_file_logger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    cfg = load_merged_config(ROOT / "config")
    log = setup_file_logger(ROOT / "logs", name="02_free_peptide")
    ck_path = ROOT / "state/02_free_peptide_checkpoint.json"
    done = load_set(ck_path) if args.resume else set()

    man = ROOT / "01_manifest/sample_master_table.csv"
    if not man.exists():
        log.error("Missing manifest %s — run 01_build_manifest first", man)
        return 2
    df = pd.read_csv(man)
    pp = df["pdb_path"]
    sub = df[pp.notna() & pp.astype(str).str.strip().ne("")].copy()
    sub = sub[sub["pdb_exists"] == True]  # noqa: E712
    if args.max_samples > 0:
        sub = sub.head(args.max_samples)

    per_dir = ROOT / "02_free_peptide/per_sample"
    per_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in sub.iterrows():
        uid = str(r["sample_uid"])
        if uid in done:
            pjson = per_dir / f"{uid}.json"
            if pjson.exists():
                rows.append(json.loads(pjson.read_text(encoding="utf-8")))
            continue
        pdb_path = Path(str(r["pdb_path"]))
        try:
            metrics = analyze_pdb_path(pdb_path, float(cfg.get("clash_ca_cutoff", 3.5)))
        except Exception as e:
            metrics = {"status": "error", "error": str(e)}
        row = {
            "sample_uid": uid,
            "method": r.get("method"),
            "target_id": r.get("target_id"),
            "candidate_id": r.get("candidate_id"),
            "pdb_path": str(pdb_path),
            **metrics,
        }
        (per_dir / f"{uid}.json").write_text(
            json.dumps(row, indent=2, default=str), encoding="utf-8"
        )
        rows.append(row)
        mark_done(ck_path, uid, done)
        save_set(ck_path, done)

    out_csv = ROOT / "02_free_peptide/tables/free_peptide_metrics.csv"
    out_json = ROOT / "02_free_peptide/tables/free_peptide_metrics_summary.json"
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

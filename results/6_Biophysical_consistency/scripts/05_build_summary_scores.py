#!/usr/bin/env python3
"""
05 — 汇总打分（Summary scores）

整合 Table_S2、S5、S7、S8（键：Table_S1 的 target_id + peptide_id + group），
在肽级计算标准化综合分，再聚合到靶标级，并生成分组对比说明。

输出：
  - ``tables/Table_S4_foldability_summary.csv``
  - ``tables/Table_S11_biophysical_summary_scores.csv``
  - ``tables/Table_S12_target_level_summary.csv``
  - ``summary_report.md``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logging_utils import setup_run_logger
from utils.paths import ProjectPaths, load_config
from utils.summary_scoring import (
    attach_ali,
    build_table_s11,
    build_table_s12,
    build_table_s4,
    compute_foldability_components,
    compute_interface_score,
    compute_obcs,
    compute_solubility_score,
    group_comparison_table,
    load_and_merge,
    write_summary_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--tables-dir", type=Path, default=None)
    p.add_argument("--log-dir", type=Path, default=None)
    return p.parse_args()


def _append_fixed_log(logger: logging.Logger, log_dir: Path, name: str = "05_summary_scores.log") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = ROOT / "logs" / "05_build_summary_scores.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info("Appended log: %s", path)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_dirs()
    tabs = args.tables_dir or paths.tables

    th = cfg.get("thresholds") or {}
    w_old = th.get("summary_weights") or {}
    w_obcs = th.get("obcs_weights") or {}
    weights = {
        "fcs": float(w_obcs.get("fcs", w_old.get("foldability", 0.28))),
        "scs": float(w_obcs.get("scs", w_old.get("solubility", 0.28))),
        "ics": float(w_obcs.get("ics", w_old.get("interface", 0.28))),
        "anti_agg": float(w_obcs.get("anti_agg", 0.16)),
    }
    s = sum(weights.values())
    if s > 0:
        weights = {k: v / s for k, v in weights.items()}

    log = setup_run_logger(paths.logs, "05_build_summary_scores")
    _append_fixed_log(log, paths.logs)

    log.info("Loading tables from %s", tabs)
    m = load_and_merge(Path(tabs))
    m = compute_foldability_components(m)
    m = compute_solubility_score(m)
    m = compute_interface_score(m)
    m = attach_ali(m)
    m["OBCS"] = compute_obcs(m, weights)

    s4 = build_table_s4(m)
    s11 = build_table_s11(m, weights)
    s12 = build_table_s12(m)
    gcmp = group_comparison_table(m)

    s4.to_csv(paths.tables / "Table_S4_foldability_summary.csv", index=False)
    s11.to_csv(paths.tables / "Table_S11_biophysical_summary_scores.csv", index=False)
    s12.to_csv(paths.tables / "Table_S12_target_level_summary.csv", index=False)

    n_s2 = int(m["s2_clash_count"].notna().sum())
    n_s8 = int(m["s8_residue_contact_count"].notna().sum())
    write_summary_report(
        paths.root / "summary_report.md",
        n_rows=len(m),
        weights=weights,
        group_cmp=gcmp,
        n_s2=n_s2,
        n_s8=n_s8,
    )

    meta = {
        "n_peptides": len(m),
        "obcs_weights": weights,
        "n_with_s2_clash": n_s2,
        "n_with_s8_interface": n_s8,
        "group_comparison": gcmp.to_dict(orient="records"),
    }
    (paths.intermediate / "05_summary_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    log.info("Wrote Table_S4, S11, S12 and summary_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

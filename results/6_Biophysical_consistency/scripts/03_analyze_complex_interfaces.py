#!/usr/bin/env python3
"""
03 — 肽–靶复合物界面分析（Complex interfaces）

读取 ``Table_S1_master_sequence_table.csv`` 中 ``usable_for_interface_analysis=True`` 的行，
对 ``complex_structure_path`` 做界面互补性 proxy 分析。

输出：
  - ``tables/Table_S3_complex_model_summary.csv``
  - ``tables/Table_S8_interface_metrics.csv``
  - ``tables/interface_metric_definitions.md``
  - ``intermediate/interface_contacts/`` 每复合物残基/原子接触 CSV
  - ``intermediate/interface_qc/`` 失败/跳过说明
  - ``intermediate/interface_hit_frequency_by_target/{target_id}.csv``
  - ``logs/interface_analysis.log``（追加）
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.complex_interface import (
    compute_interface_metrics,
    load_peptide_and_target_atoms,
    write_interface_metric_definitions_md,
)
from utils.logging_utils import setup_run_logger
from utils.paths import ProjectPaths, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--master-table",
        type=Path,
        default=ROOT / "tables" / "Table_S1_master_sequence_table.csv",
    )
    p.add_argument("--max-rows", type=int, default=None, help="仅处理前 N 行（调试用）。")
    p.add_argument(
        "--max-atomic-export",
        type=int,
        default=100_000,
        help="每个复合物原子接触 CSV 最大行数。",
    )
    return p.parse_args()


def _truthy(x) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y")


def _slug(s: str, max_len: int = 180) -> str:
    t = re.sub(r"[^\w.\-]+", "_", str(s).strip())
    return t[:max_len] if len(t) > max_len else t


def _append_fixed_log(logger: logging.Logger, log_dir: Path, name: str = "interface_analysis.log") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
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

    th = cfg.get("thresholds") or {}
    interface_cutoff = float(th.get("interface_distance_cutoff_angstrom", 5.0))
    hb_cutoff = float(th.get("hbond_distance_cutoff_angstrom", 3.5))

    exec_cfg = cfg.get("execution") or {}
    max_samples = exec_cfg.get("max_samples")
    if args.max_rows is not None:
        max_samples = args.max_rows

    contacts_dir = paths.intermediate / "interface_contacts"
    qc_dir = paths.intermediate / "interface_qc"
    hit_dir = paths.intermediate / "interface_hit_frequency_by_target"
    for d in (contacts_dir, qc_dir, hit_dir):
        d.mkdir(parents=True, exist_ok=True)

    log = setup_run_logger(paths.logs, "03_analyze_complex_interfaces")
    _append_fixed_log(log, paths.logs, "interface_analysis.log")

    write_interface_metric_definitions_md(paths.tables / "interface_metric_definitions.md")

    master_path = args.master_table
    if not master_path.is_absolute():
        master_path = paths.root / master_path
    if not master_path.exists():
        log.error("Master table not found: %s", master_path)
        return 1

    df = pd.read_csv(master_path)
    if "usable_for_interface_analysis" not in df.columns:
        log.error("Missing column usable_for_interface_analysis")
        return 1

    iface_mask = df["usable_for_interface_analysis"].map(_truthy)
    sub = df[iface_mask].copy()
    if max_samples is not None:
        sub = sub.head(int(max_samples))

    log.info(
        "Rows in master: %s, usable_for_interface: %s (processing %s)",
        len(df),
        int(iface_mask.sum()),
        len(sub),
    )

    # target_id -> target_res_key -> set(peptide_id)
    hit_tracker: dict[str, dict[tuple, set[str]]] = defaultdict(lambda: defaultdict(set))
    # target_id -> count of successfully analyzed complexes (this run)
    target_success_counts: dict[str, int] = defaultdict(int)

    s3_rows: list[dict] = []
    s8_rows: list[dict] = []

    metric_cols_order = [
        "residue_contact_count",
        "atomic_contact_count",
        "interface_residue_count_peptide",
        "interface_residue_count_target",
        "buried_sasa_proxy",
        "interface_packing_density_proxy",
        "interface_gap_proxy",
        "hydrophobic_contact_count",
        "hydrophobic_patch_overlap_score",
        "hydrophobic_mismatch_penalty",
        "opposite_charge_contact_count",
        "same_charge_contact_count",
        "electrostatic_complementarity_score",
        "salt_bridge_count",
        "unsatisfied_buried_charge_proxy",
        "interface_hbond_count",
        "polar_contact_count",
    ]

    for idx, row in sub.iterrows():
        target_id = str(row.get("target_id", "")).strip()
        peptide_id = str(row.get("peptide_id", "")).strip()
        rank = row.get("rank", "")
        group = row.get("group", "")
        source = row.get("source", "")
        cpath = row.get("complex_structure_path", "")
        cpath_s = str(cpath).strip() if pd.notna(cpath) else ""

        base_uid = _slug(f"{peptide_id}__{target_id}__r{rank}")
        residue_csv = contacts_dir / f"{base_uid}_residue_contacts.csv"
        atomic_csv = contacts_dir / f"{base_uid}_atomic_contacts.csv"

        s3_base = {
            "target_id": target_id,
            "peptide_id": peptide_id,
            "group": group,
            "rank": rank,
            "source": source,
            "complex_structure_path": cpath_s,
            "residue_contact_csv": str(residue_csv.relative_to(paths.root)),
            "atomic_contact_csv": str(atomic_csv.relative_to(paths.root)),
        }

        if not cpath_s:
            reason = "missing_complex_structure_path"
            s3_rows.append(
                {
                    **s3_base,
                    "analysis_status": "failed",
                    "failure_reason": reason,
                    "chain_peptide": "",
                    "chain_target": "",
                    "chain_inference_notes": "",
                    "n_peptide_heavy_atoms": 0,
                    "n_target_heavy_atoms": 0,
                    "n_residue_contacts": 0,
                    "n_atomic_contacts": 0,
                    "atomic_export_truncated": False,
                    "processing_notes": "",
                }
            )
            (qc_dir / f"{base_uid}_qc.json").write_text(
                json.dumps({"status": "failed", "reason": reason}, indent=2),
                encoding="utf-8",
            )
            log.warning("[%s] %s", peptide_id, reason)
            continue

        pdb_path = Path(cpath_s).expanduser()
        pep, tgt, lab_p, lab_t, chain_notes = load_peptide_and_target_atoms(pdb_path)

        if not pep or not tgt:
            reason = chain_notes if chain_notes else "empty_chains_after_load"
            s3_rows.append(
                {
                    **s3_base,
                    "analysis_status": "failed",
                    "failure_reason": reason,
                    "chain_peptide": lab_p,
                    "chain_target": lab_t,
                    "chain_inference_notes": chain_notes,
                    "n_peptide_heavy_atoms": len(pep),
                    "n_target_heavy_atoms": len(tgt),
                    "n_residue_contacts": 0,
                    "n_atomic_contacts": 0,
                    "atomic_export_truncated": False,
                    "processing_notes": "",
                }
            )
            (qc_dir / f"{base_uid}_qc.json").write_text(
                json.dumps({"status": "failed", "reason": reason, "path": str(pdb_path)}, indent=2),
                encoding="utf-8",
            )
            log.warning("[%s] %s", peptide_id, reason)
            continue

        try:
            res = compute_interface_metrics(
                pep,
                tgt,
                interface_cutoff=interface_cutoff,
                hbond_cutoff=hb_cutoff,
                max_atomic_pairs_export=int(args.max_atomic_export),
            )
        except Exception as e:
            reason = f"compute_exception:{e}"
            s3_rows.append(
                {
                    **s3_base,
                    "analysis_status": "failed",
                    "failure_reason": reason,
                    "chain_peptide": lab_p,
                    "chain_target": lab_t,
                    "chain_inference_notes": chain_notes,
                    "n_peptide_heavy_atoms": len(pep),
                    "n_target_heavy_atoms": len(tgt),
                    "n_residue_contacts": 0,
                    "n_atomic_contacts": 0,
                    "atomic_export_truncated": False,
                    "processing_notes": "",
                }
            )
            (qc_dir / f"{base_uid}_qc.json").write_text(
                json.dumps({"status": "failed", "reason": reason, "path": str(pdb_path)}, indent=2),
                encoding="utf-8",
            )
            log.exception("[%s] compute failed", peptide_id)
            continue

        m = res.metrics
        truncated = any("atomic_pairs_truncated" in n for n in res.notes)
        proc_notes = "|".join(res.notes) if res.notes else ""

        pd.DataFrame(res.residue_pairs).to_csv(residue_csv, index=False)
        pd.DataFrame(res.atomic_pairs).to_csv(atomic_csv, index=False)

        s3_rows.append(
            {
                **s3_base,
                "analysis_status": "success",
                "failure_reason": "",
                "chain_peptide": lab_p,
                "chain_target": lab_t,
                "chain_inference_notes": chain_notes,
                "n_peptide_heavy_atoms": len(pep),
                "n_target_heavy_atoms": len(tgt),
                "n_residue_contacts": m.get("residue_contact_count", 0),
                "n_atomic_contacts": m.get("atomic_contact_count", 0),
                "atomic_export_truncated": truncated,
                "processing_notes": proc_notes,
            }
        )

        s8_row = {
            "target_id": target_id,
            "peptide_id": peptide_id,
            "group": group,
            "rank": rank,
            "source": source,
            "complex_structure_path": cpath_s,
            "chain_peptide": lab_p,
            "chain_target": lab_t,
            "chain_inference_notes": chain_notes,
            "interface_distance_cutoff_A": interface_cutoff,
            "hbond_distance_cutoff_A": hb_cutoff,
            "atomic_export_truncated": truncated,
            "processing_notes": proc_notes,
        }
        for c in metric_cols_order:
            s8_row[c] = m.get(c, "")
        s8_rows.append(s8_row)

        target_success_counts[target_id] += 1
        for rp in res.residue_pairs:
            tk = (
                str(rp["target_chain"]),
                int(rp["target_resseq"]),
                str(rp["target_icode"]),
                str(rp.get("target_resname", "")),
            )
            hit_tracker[target_id][tk].add(peptide_id)

        log.info("[%s] contacts: res=%s atom=%s", peptide_id, m.get("residue_contact_count"), m.get("atomic_contact_count"))

    s3_df = pd.DataFrame(s3_rows)
    s3_path = paths.tables / "Table_S3_complex_model_summary.csv"
    s3_df.to_csv(s3_path, index=False)
    log.info("Wrote %s", s3_path)

    s8_path = paths.tables / "Table_S8_interface_metrics.csv"
    if s8_rows:
        s8_df = pd.DataFrame(s8_rows)
    else:
        s8_df = pd.DataFrame(columns=["target_id", "peptide_id"] + metric_cols_order)
        log.warning("No successful interface rows; writing empty Table_S8 with headers")
    s8_df.to_csv(s8_path, index=False)
    log.info("Wrote %s", s8_path)

    # Per-target hit frequency
    for tid, res_map in hit_tracker.items():
        n_succ = max(target_success_counts.get(tid, 1), 1)
        rows = []
        for (ch, rs, ic, rname), pset in sorted(res_map.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            rows.append(
                {
                    "target_id": tid,
                    "target_chain": ch,
                    "target_resseq": rs,
                    "target_icode": ic,
                    "target_resname": rname,
                    "hit_peptide_count": len(pset),
                    "n_analyzed_complexes": target_success_counts[tid],
                    "hit_frequency": len(pset) / n_succ,
                }
            )
        hit_path = hit_dir / f"{_slug(tid, 80)}_target_site_hit_frequency.csv"
        pd.DataFrame(rows).to_csv(hit_path, index=False)

    log.info("Wrote per-target hit tables under %s", hit_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
04 — 溶解度与聚集热点（Solubility & aggregation hotspots）

读取 ``Table_S1_master_sequence_table.csv``，对 ``generated`` / ``reference`` / ``decoy``
分组统一计算序列级与残基级溶解度 proxy、CamSol-like 分数及聚集热点。

输出：
  - ``tables/Table_S5_solubility_global_metrics.csv``
  - ``tables/Table_S6_residue_level_solubility_profile.csv``
  - ``tables/Table_S7_aggregation_hotspot_summary.csv``
  - ``tables/solubility_metric_definitions.md``
  - ``intermediate/solubility_profiles/{peptide_id}.csv``（每条肽残基 profile）
  - ``logs/solubility_analysis.log``（追加）
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logging_utils import setup_run_logger
from utils.paths import ProjectPaths, load_config
from utils.solubility_hotspots import (
    HotspotParams,
    aggregation_liability_index,
    exposure_proxy_from_ca,
    global_metrics,
    load_ca_coords_aligned,
    normalize_sequence,
    per_residue_table,
    summarize_hotspots,
    write_solubility_metric_definitions_md,
)

ALLOWED_GROUPS = frozenset({"generated", "reference", "decoy"})

PROFILE_RESIDUE_COLS = [
    "residue_index",
    "residue",
    "hydrophobicity",
    "charge_state",
    "local_hydrophobic_run",
    "local_charge_balance",
    "camsol_like_local_score",
    "hotspot_score",
    "hotspot_class",
]


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
        "--require-usable-flag",
        action="store_true",
        help="若设置，则仅处理 usable_for_solubility_analysis 为真的行。",
    )
    return p.parse_args()


def _slug(s: str, max_len: int = 180) -> str:
    t = re.sub(r"[^\w.\-]+", "_", str(s).strip())
    return t[:max_len] if len(t) > max_len else t


def _truthy(x) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y")


def _append_fixed_log(logger: logging.Logger, log_dir: Path, name: str = "solubility_analysis.log") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info("Appended log: %s", path)


def _odd_window(w_cfg: int) -> int:
    w = int(max(3, min(w_cfg, 21)))
    if w % 2 == 0:
        w -= 1
    return max(3, w)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_dirs()

    th = cfg.get("thresholds") or {}
    win = _odd_window(int(th.get("aggregation_hotspot_window", 5)))
    half_w = win // 2

    hp = HotspotParams(
        window_half=half_w,
        mild_cut=float(th.get("hotspot_mild_threshold", 0.38)),
        strong_cut=float(th.get("hotspot_strong_threshold", 0.62)),
    )

    exec_cfg = cfg.get("execution") or {}
    max_samples = exec_cfg.get("max_samples")
    if args.max_rows is not None:
        max_samples = args.max_rows

    prof_dir = paths.intermediate / "solubility_profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)

    log = setup_run_logger(paths.logs, "04_analyze_solubility_and_hotspots")
    _append_fixed_log(log, paths.logs, "solubility_analysis.log")

    write_solubility_metric_definitions_md(paths.tables / "solubility_metric_definitions.md", hotspot_params=hp, window=win)

    master_path = args.master_table
    if not master_path.is_absolute():
        master_path = paths.root / master_path
    if not master_path.exists():
        log.error("Master table not found: %s", master_path)
        return 1

    df = pd.read_csv(master_path)
    sub = df[df["group"].isin(ALLOWED_GROUPS)].copy()
    if args.require_usable_flag and "usable_for_solubility_analysis" in sub.columns:
        sub = sub[sub["usable_for_solubility_analysis"].map(_truthy)]
    if max_samples is not None:
        sub = sub.head(int(max_samples))

    log.info(
        "Master rows=%s, allowed_groups rows=%s, processing=%s",
        len(df),
        len(df[df["group"].isin(ALLOWED_GROUPS)]),
        len(sub),
    )

    s5_rows: list[dict] = []
    s6_parts: list[pd.DataFrame] = []
    s7_rows: list[dict] = []

    for _, row in sub.iterrows():
        target_id = str(row.get("target_id", "")).strip()
        peptide_id = str(row.get("peptide_id", "")).strip()
        group = str(row.get("group", "")).strip()
        raw_seq = row.get("sequence", "")
        seq = normalize_sequence(str(raw_seq) if pd.notna(raw_seq) else "")
        fpath = row.get("free_structure_path", "")
        fpath_s = str(fpath).strip() if pd.notna(fpath) else ""

        base = {
            "target_id": target_id,
            "peptide_id": peptide_id,
            "group": group,
            "rank": row.get("rank", ""),
            "source": row.get("source", ""),
        }

        if not seq:
            nan = float("nan")
            s5_rows.append(
                {
                    **base,
                    "analysis_status": "failed",
                    "failure_reason": "empty_sequence",
                    "structure_used": False,
                    "free_structure_path": fpath_s,
                    "structure_alignment_note": "",
                    "length": 0,
                    "gravy": nan,
                    "net_charge_ph74": nan,
                    "positive_residue_fraction": nan,
                    "negative_residue_fraction": nan,
                    "aromatic_fraction": nan,
                    "hydrophobic_fraction": nan,
                    "charge_density": nan,
                    "pI_proxy": nan,
                    "camsol_like_score": nan,
                }
            )
            s7_rows.append(
                {
                    **base,
                    "analysis_status": "failed",
                    "length": 0,
                    "gravy": nan,
                    "net_charge_ph74": nan,
                    "hotspot_window": win,
                    "structure_used": False,
                    "structure_alignment_note": "empty_sequence",
                    "hotspot_count": 0,
                    "strong_hotspot_count": 0,
                    "hotspot_burden": 0.0,
                    "longest_hotspot_span": 0,
                    "aggregation_liability_index": 0.0,
                }
            )
            prof_path = prof_dir / f"{_slug(peptide_id)}.csv"
            pd.DataFrame(columns=["target_id", "peptide_id", "group"] + PROFILE_RESIDUE_COLS).to_csv(
                prof_path, index=False
            )
            log.warning("[%s] empty sequence", peptide_id)
            continue

        glob_m = global_metrics(seq)
        exposure = None
        align_note = "no_structure_path"
        if fpath_s:
            coords, note = load_ca_coords_aligned(Path(fpath_s), seq)
            align_note = note
            if coords is not None:
                exposure = exposure_proxy_from_ca(coords)
            else:
                exposure = None
        else:
            align_note = "no_structure_path"

        structure_used = exposure is not None and len(exposure) == len(seq)

        df_res = per_residue_table(seq, half_window=half_w, exposure=exposure if structure_used else None, hp=hp)
        hs = summarize_hotspots(df_res, len(seq))
        ali = aggregation_liability_index(
            gravy=glob_m["gravy"],
            net_charge=glob_m["net_charge_ph74"],
            length=len(seq),
            hotspot_burden=hs["hotspot_burden"],
            strong_hotspot_count=hs["strong_hotspot_count"],
            longest_hotspot_span=hs["longest_hotspot_span"],
        )

        s5_rows.append(
            {
                **base,
                "analysis_status": "success",
                "failure_reason": "",
                "structure_used": structure_used,
                "free_structure_path": fpath_s,
                "structure_alignment_note": align_note,
                **glob_m,
            }
        )

        prof_path = prof_dir / f"{_slug(peptide_id)}.csv"
        df_out = df_res.copy()
        df_out.insert(0, "group", group)
        df_out.insert(0, "peptide_id", peptide_id)
        df_out.insert(0, "target_id", target_id)
        df_out.to_csv(prof_path, index=False)

        df_s6 = df_out.copy()
        s6_parts.append(df_s6)

        s7_rows.append(
            {
                **base,
                "analysis_status": "success",
                "length": len(seq),
                "gravy": glob_m["gravy"],
                "net_charge_ph74": glob_m["net_charge_ph74"],
                "hotspot_window": win,
                "structure_used": structure_used,
                "structure_alignment_note": align_note,
                **hs,
                "aggregation_liability_index": ali,
            }
        )

        log.info("[%s] len=%s gravy=%.3f ALI=%.3f hotspots=%s", peptide_id, len(seq), glob_m["gravy"], ali, hs["hotspot_count"])

    s5 = pd.DataFrame(s5_rows)
    s5_path = paths.tables / "Table_S5_solubility_global_metrics.csv"
    s5.to_csv(s5_path, index=False)
    log.info("Wrote %s", s5_path)

    if s6_parts:
        s6 = pd.concat(s6_parts, ignore_index=True)
        s6_path = paths.tables / "Table_S6_residue_level_solubility_profile.csv"
        s6.to_csv(s6_path, index=False)
        log.info("Wrote %s (%s rows)", s6_path, len(s6))
    else:
        pd.DataFrame().to_csv(paths.tables / "Table_S6_residue_level_solubility_profile.csv", index=False)
        log.warning("Wrote empty Table_S6")

    s7 = pd.DataFrame(s7_rows)
    s7_path = paths.tables / "Table_S7_aggregation_hotspot_summary.csv"
    s7.to_csv(s7_path, index=False)
    log.info("Wrote %s", s7_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

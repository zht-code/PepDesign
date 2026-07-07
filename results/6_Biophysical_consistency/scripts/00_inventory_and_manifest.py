#!/usr/bin/env python3
"""
00 — 数据盘点与清单（Inventory & Manifest）

对 ``project_root`` 下约定子目录（见 ``config.yaml`` → ``inventory``）做**只读**递归扫描，
索引 PDB/CIF、FASTA、CSV/TSV、JSON、NPZ/PT/PKL 及常见对接输出（*.out），
并启发式标注 ``target_id`` / ``peptide_id`` / ``group``。

输出目录（默认 ``data_inventory/``）：
  - ``file_manifest.csv``、``file_manifest.json``（全量索引 + 汇总）
  - ``pdb_manifest.csv``（PDB/ENT/CIF）
  - ``sequence_manifest.csv``（FASTA + 可能含序列列的 CSV）
  - ``table_manifest.csv``（CSV/TSV）
  - ``suggested_inputs.json``（下游三类分析推荐输入）
  - ``inventory_report.md``（人类可读总结）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inventory_scan import (
    ALLOWED_SUFFIXES,
    build_suggested_inputs,
    classify_file,
    iter_scan_roots,
    records_to_dataframe,
    walk_files,
    write_inventory_report,
)
from utils.logging_utils import setup_run_logger
from utils.paths import ProjectPaths, load_config, project_root as pipeline_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="覆盖 project_root（只读扫描根）。默认取 config.project_root。",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="默认：config.paths.data_inventory。",
    )
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument(
        "--include-this-repo-tables",
        action="store_true",
        default=True,
        help="同时浅层索引本工程 tables/、intermediate/（避免把 data_inventory 输出再次扫入）。",
    )
    return p.parse_args()


def _extra_local_scan_roots(repo: Path, include: bool) -> list[Path]:
    if not include:
        return []
    roots = []
    for sub in ("tables", "intermediate"):
        p = (repo / sub).resolve()
        if p.is_dir():
            roots.append(p)
    return roots


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_dirs()

    log_dir = args.log_dir or paths.logs
    out_dir = args.output_dir or paths.data_inventory
    project_root = Path(args.input_dir or cfg.get("project_root", "")).expanduser().resolve()

    inv_cfg = cfg.get("inventory") or {}
    skip = set(str(x) for x in (inv_cfg.get("skip_dirnames") or []))

    log = setup_run_logger(log_dir, "00_inventory_and_manifest")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_resolved = out_dir.resolve()

    scan_roots = iter_scan_roots(project_root, cfg)
    scan_roots.extend(_extra_local_scan_roots(ROOT, args.include_this_repo_tables))
    scan_roots = list(dict.fromkeys(scan_roots))

    log.info("Scan roots (%d): %s", len(scan_roots), scan_roots)

    records = []
    n_skipped_under_output = 0
    for fp in walk_files(scan_roots, skip):
        try:
            rp = fp.resolve()
        except OSError:
            continue
        if str(rp).startswith(str(out_resolved)) and out_resolved in rp.parents:
            n_skipped_under_output += 1
            continue
        suf = rp.suffix.lower()
        if suf not in ALLOWED_SUFFIXES:
            continue
        records.append(classify_file(rp, project_root))

    df = records_to_dataframe(records)
    scanned_at = datetime.now(timezone.utc).isoformat()

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "file_manifest.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "scanned_at_utc": scanned_at,
        "scan_roots": [str(x) for x in scan_roots],
        "project_root": str(project_root),
        "n_files": int(len(df)),
        "n_skipped_under_output": n_skipped_under_output,
        "by_file_type": df["file_type"].value_counts().to_dict() if len(df) else {},
        "by_group": df["group"].value_counts().to_dict() if len(df) else {},
        "by_ext": df["ext"].value_counts().head(40).to_dict() if len(df) else {},
    }
    json_payload = {
        "summary": summary,
        "files": df.to_dict(orient="records"),
    }
    (out_dir / "file_manifest.json").write_text(
        json.dumps(json_payload, indent=2, default=str), encoding="utf-8"
    )

    pdb_df = df[df["file_type"].isin(["pdb", "cif"])].copy()
    pdb_df.to_csv(out_dir / "pdb_manifest.csv", index=False)

    seq_mask = (df["file_type"] == "fasta") | (
        (df["file_type"] == "csv")
        & df["basename"].str.contains(
            r"sequence|mpnn|fasta|all_samples|samples_", case=True, regex=True
        )
    )
    seq_df = df[seq_mask].copy()
    seq_df.to_csv(out_dir / "sequence_manifest.csv", index=False)

    tab_df = df[df["file_type"].isin(["csv", "tsv"])].copy()
    tab_df.to_csv(out_dir / "table_manifest.csv", index=False)

    suggested = build_suggested_inputs(df, project_root, ROOT)
    suggested["inventory_summary"] = summary
    (out_dir / "suggested_inputs.json").write_text(
        json.dumps(suggested, indent=2, default=str), encoding="utf-8"
    )

    # inventory_report.md 要点
    top_lines = _build_top_findings(df, summary, project_root)
    write_inventory_report(out_dir / "inventory_report.md", summary, suggested, top_lines)

    log.info("Wrote manifests to %s (n=%s)", out_dir, len(df))
    return 0


def _build_top_findings(
    df: pd.DataFrame, summary: dict[str, Any], project_root: Path
) -> list[str]:
    lines: list[str] = []
    pr = str(project_root)
    lines.append(
        f"- **体量**：共索引 **{summary.get('n_files', 0)}** 个相关扩展名文件（不含图片等非目标类型）。"
    )

    n_pdb = int((df["file_type"] == "pdb").sum()) if len(df) else 0
    n_cif = int((df["file_type"] == "cif").sum()) if len(df) else 0
    n_hdock = int(df["absolute_path"].str.contains("/hdock_work/", case=False, regex=False).sum()) if len(df) else 0
    n_clean = int(df["absolute_path"].str.contains("/clean_inputs/", case=False, regex=False).sum()) if len(df) else 0
    n_fasta = int((df["file_type"] == "fasta").sum()) if len(df) else 0
    n_csv = int((df["file_type"] == "csv").sum()) if len(df) else 0
    n_tsv = int((df["file_type"] == "tsv").sum()) if len(df) else 0

    lines.append(
        f"- **结构文件**：PDB/ENT **{n_pdb}**，CIF/MCIF **{n_cif}**；其中 **clean_inputs** 路径约 **{n_clean}** 条，**hdock_work** 路径约 **{n_hdock}** 条（对接复合物模型）。"
    )
    lines.append(
        f"- **序列与表**：FASTA **{n_fasta}**，CSV **{n_csv}**，TSV **{n_tsv}**（含 `all_samples` / `samples_*` 等可解析序列列的候选表）。"
    )

    # 代表性路径
    def first_match(cond) -> str:
        sub = df.loc[cond, "absolute_path"]
        return str(sub.iloc[0]) if len(sub) else ""

    p_all = first_match(df["basename"].eq("all_samples.csv"))
    p_idx = first_match(df["basename"].eq("baseline_input_index.csv"))
    if p_all:
        lines.append(f"- **关键主表**：`all_samples.csv` 示例：`{p_all}`")
    if p_idx:
        lines.append(f"- **索引元数据**：`baseline_input_index.csv`：`{p_idx}`")

    m1 = first_match(
        df["absolute_path"].str.contains("/hdock_work/", case=False, regex=False)
        & df["basename"].str.lower().str.startswith("model_")
    )
    if m1:
        lines.append(f"- **对接模型示例**：`{m1}`")

    c1 = first_match(df["absolute_path"].str.contains("/clean_inputs/", case=False, regex=False))
    if c1:
        lines.append(f"- **clean_inputs 结构示例**：`{c1}`")

    gs = first_match(df["basename"].eq("generated_sequences.fasta"))
    if gs:
        lines.append(f"- **无条件生成序列示例**：`{gs}`")

    lines.append(
        f"- **数据根**：当前扫描以 `{pr}` 下 `results/` 为主；详细路径见 `file_manifest.csv` 与 `suggested_inputs.json`。"
    )
    if len(df):
        nu = int((df["group"] == "unknown").sum())
        if nu:
            lines.append(
                f"- **`unknown` 分组**：约 **{nu}** 条路径未命中生成/参考/诱饵启发式；请结合主表与关键词二次筛选，勿直接丢弃。"
            )
        for label, ft in (("NPZ", "npz"), ("PT/PTH", "pt"), ("PKL", "pkl")):
            n = int((df["file_type"] == ft).sum())
            if n:
                lines.append(f"- **{label} 文件**：**{n}**（可作为模型预测或中间特征接入点）。")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())

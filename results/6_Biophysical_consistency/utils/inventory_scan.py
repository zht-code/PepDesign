from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# PDB 条目式 ID（宽松）
_RE_PDBLIKE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


EXT_MAP: dict[str, str] = {
    ".pdb": "pdb",
    ".ent": "pdb",
    ".cif": "cif",
    ".mcif": "cif",
    ".fasta": "fasta",
    ".fa": "fasta",
    ".faa": "fasta",
    ".csv": "csv",
    ".tsv": "tsv",
    ".json": "json",
    ".npz": "npz",
    ".pt": "pt",
    ".pth": "pt",
    ".pkl": "pkl",
    ".pickle": "pkl",
    ".out": "dock_out",
    ".log": "log",
}

ALLOWED_SUFFIXES: frozenset[str] = frozenset(EXT_MAP.keys())


@dataclass
class FileRecord:
    absolute_path: str
    source_dir: str
    file_type: str
    ext: str
    size_bytes: int | None
    mtime_utc: str | None
    target_id: str
    peptide_id: str
    group: str
    basename: str


def _safe_stat(path: Path) -> tuple[int | None, str | None]:
    try:
        st = path.stat()
        mt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        return st.st_size, mt
    except OSError:
        return None, None


def _classify_group(parts_lower: list[str], name_lower: str) -> str:
    joined = "/".join(parts_lower)
    if "decoy" in joined or "negative" in joined:
        return "decoy"
    if any(
        x in joined
        for x in (
            "clean_inputs",
            "hdock_work",
            "rfdiffusion",
            "bindcraft",
            "proteingenerator",
            "generated_sequences",
            "unconditional",
            "/samples_",
            "raw_results",
            "perturbed_targets",
        )
    ):
        return "generated"
    if any(
        x in joined
        for x in (
            "reference",
            "ground_truth",
            "native",
            "crystal",
            "pdbench",
            "ppdbench",
        )
    ):
        return "reference"
    return "unknown"


def _infer_target_peptide(path: Path, project_root: Path) -> tuple[str, str]:
    """从路径启发式推断 target_id / peptide_id（可能为空）。"""
    parts = path.parts
    pl = [p.lower() for p in parts]
    name = path.name
    stem = path.stem
    target_id = ""
    peptide_id = ""

    if "hdock_work" in pl:
        try:
            i = pl.index("hdock_work")
            # .../hdock_work/<method>/<condition>/<target>/model_n.pdb
            if i + 3 < len(parts):
                cand = parts[i + 3]
                if _RE_PDBLIKE.match(cand):
                    target_id = cand.lower()
        except ValueError:
            pass
        peptide_id = stem if name.lower().startswith("model_") else stem
        return target_id, peptide_id

    if "clean_inputs" in pl:
        try:
            i = pl.index("clean_inputs")
            if i + 2 < len(parts):
                tid = parts[i + 2]
                if _RE_PDBLIKE.match(tid):
                    target_id = tid.lower()
        except ValueError:
            pass
        peptide_id = stem
        return target_id, peptide_id

    if name.lower() == "generated_sequences.fasta":
        parent = path.parent.name
        if re.match(r"^[0-9A-Za-z]{4}_[0-9]+$", parent) or _RE_PDBLIKE.match(
            parent.split("_")[0]
        ):
            target_id = parent
        return target_id, ""

    if "clean_properties" in pl and path.suffix.lower() == ".json":
        target_id = stem.lower() if _RE_PDBLIKE.match(stem) else stem.lower()
        return target_id, ""

    if "receptor" in name.lower() and path.suffix.lower() == ".pdb":
        target_id = path.parent.name.lower() if _RE_PDBLIKE.match(path.parent.name) else ""
        peptide_id = ""
        return target_id, peptide_id

    # 通用：父目录名为 PDB-like
    if _RE_PDBLIKE.match(path.parent.name):
        target_id = path.parent.name.lower()
        peptide_id = stem
        return target_id, peptide_id

    return target_id, peptide_id


def iter_scan_roots(project_root: Path, cfg: dict[str, Any]) -> list[Path]:
    inv = cfg.get("inventory") or {}
    rels = inv.get("relative_scan_roots") or ["results"]
    roots: list[Path] = []
    for r in rels:
        p = (project_root / str(r)).resolve()
        if p.is_dir():
            roots.append(p)
    extra = inv.get("extra_scan_roots") or []
    for e in extra:
        p = Path(str(e)).expanduser().resolve()
        if p.is_dir():
            roots.append(p)
    return roots


def walk_files(roots: Iterable[Path], skip_dirnames: set[str]) -> Iterable[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in skip_dirnames]
            for fn in filenames:
                yield Path(dirpath) / fn


def classify_file(path: Path, project_root: Path) -> FileRecord:
    ext = path.suffix.lower()
    ftype = EXT_MAP.get(ext, "other")
    parts_lower = [p.lower() for p in path.parts]
    group = _classify_group(parts_lower, path.name.lower())
    target_id, peptide_id = _infer_target_peptide(path, project_root)
    size, mtime = _safe_stat(path)
    return FileRecord(
        absolute_path=str(path.resolve()),
        source_dir=str(path.parent.resolve()),
        file_type=ftype,
        ext=ext,
        size_bytes=size,
        mtime_utc=mtime,
        target_id=target_id,
        peptide_id=peptide_id,
        group=group,
        basename=path.name,
    )


def records_to_dataframe(records: list[FileRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])


def build_suggested_inputs(
    df: pd.DataFrame, project_root: Path, this_repo: Path
) -> dict[str, Any]:
    """基于路径与文件名规则，总结下游最可能用到的输入。"""
    paths = df["absolute_path"].astype(str)

    def pick_existing(candidates: list[str]) -> list[dict[str, Any]]:
        out = []
        for c in candidates:
            p = Path(c)
            if p.exists():
                out.append({"absolute_path": str(p.resolve()), "exists": True})
        return out

    pr = str(project_root.resolve())
    suggestions: dict[str, Any] = {
        "project_root": pr,
        "free_peptide_structure": {
            "description": "用于游离肽/输入肽几何与可折叠性代理：clean_inputs 下肽–受体 PDB、或 manifest 中 pdb_path 来源。",
            "high_priority_patterns": [
                f"{pr}/results/5_robustness/baseline/raw_results/*/all_samples.csv",
                f"{pr}/results/5_robustness/baseline/tables/baseline_input_index.csv",
            ],
            "example_paths": [],
        },
        "complex_interface": {
            "description": "用于肽–靶界面：Hdock 输出的多段 model_*.pdb，或含 TER 分隔的对接复合物。",
            "high_priority_patterns": [
                f"{pr}/results/5_robustness/baseline/cache/hdock_work/**/model_*.pdb",
            ],
            "example_paths": [],
        },
        "sequence_solubility": {
            "description": "用于序列层溶解度/聚集代理：all_samples 的 sequence_top1、generated_sequences.fasta、clean_properties JSON。",
            "high_priority_patterns": [
                f"{pr}/results/5_robustness/baseline/raw_results/*/all_samples.csv",
                f"{pr}/results/2_SOTA/**/generated_sequences.fasta",
                f"{pr}/results/5_robustness/baseline/raw_results/rfdiffusion/clean_properties/*.json",
            ],
            "example_paths": [],
        },
    }

    # 从本次扫描中各取最多 N 条示例
    pdb_df = df[df["file_type"] == "pdb"]
    clean = pdb_df[pdb_df["absolute_path"].str.contains("/clean_inputs/", case=False, regex=False)]
    hdock = pdb_df[pdb_df["absolute_path"].str.contains("/hdock_work/", case=False, regex=False)]

    suggestions["free_peptide_structure"]["example_paths"] = (
        clean.head(15)["absolute_path"].tolist()
        if len(clean)
        else pdb_df.head(8)["absolute_path"].tolist()
    )
    suggestions["complex_interface"]["example_paths"] = hdock.head(15)[
        "absolute_path"
    ].tolist()

    fasta_df = df[df["file_type"] == "fasta"]
    seq_csv = df[
        (df["file_type"] == "csv")
        & df["basename"].str.contains("sample|sequence|all_samples", case=False, regex=True)
    ]
    json_props = df[
        (df["file_type"] == "json")
        & df["absolute_path"].str.contains("clean_properties", case=False, regex=False)
    ]
    seq_examples = (
        fasta_df.head(10)["absolute_path"].tolist()
        + seq_csv.head(5)["absolute_path"].tolist()
        + json_props.head(5)["absolute_path"].tolist()
    )
    suggestions["sequence_solubility"]["example_paths"] = seq_examples[:20]

    fixed = [
        f"{pr}/results/5_robustness/baseline/raw_results/rfdiffusion/all_samples.csv",
        f"{pr}/results/5_robustness/baseline/raw_results/bindcraft/all_samples.csv",
        f"{pr}/results/5_robustness/baseline/tables/baseline_input_index.csv",
    ]
    suggestions["canonical_csv_checks"] = pick_existing(fixed)

    suggestions["this_pipeline_repo"] = str(this_repo.resolve())
    return suggestions


def write_inventory_report(
    path: Path,
    summary: dict[str, Any],
    suggested: dict[str, Any],
    top_lines: list[str],
) -> None:
    lines = [
        "# Inventory report（自动生成）",
        "",
        f"- 扫描时间（UTC）：`{summary.get('scanned_at_utc', '')}`",
        f"- 扫描根：`{summary.get('scan_roots', [])}`",
        f"- 记录文件数：**{summary.get('n_files', 0)}**",
        "",
        "## 按类型统计",
        "",
    ]
    by_type = summary.get("by_file_type", {})
    for k, v in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"- **{k}**：{v}")
    lines += ["", "## 按 group 统计", ""]
    by_g = summary.get("by_group", {})
    for k, v in sorted(by_g.items(), key=lambda x: -x[1]):
        lines.append(f"- **{k}**：{v}")
    lines += [
        "",
        "## 后续分析可用性（基于路径与命名规则）",
        "",
        "### Free peptide structure analysis",
        "",
        "- **优先**：`results/5_robustness/baseline/cache/clean_inputs/**.pdb`（生成/清洗肽复合物输入）。",
        "- **辅助**：`raw_results/*/all_samples.csv` 中的 `pdb_path` 列可批量定位同一批肽结构。",
        "",
        "### Complex interface analysis",
        "",
        "- **优先**：`results/5_robustness/baseline/cache/hdock_work/**/model_*.pdb`（对接复合物模型）。",
        "",
        "### Sequence-based solubility analysis",
        "",
        "- **优先**：`all_samples.csv` 等表中的 `sequence_top1` / `sequence` 列。",
        "- **补充**：`results/2_SOTA/**/generated_sequences.fasta`；`clean_properties/*.json` 中的序列字段。",
        "",
        "## 本次扫描最重要的发现（摘要）",
        "",
    ]
    lines.extend(top_lines)
    lines += ["", "## suggested_inputs.json 摘要", ""]
    lines.append(
        f"- 已写入 `suggested_inputs.json`，其中 `example_paths` 为从本次 manifest 抽取的代表路径（每类最多若干条）。"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

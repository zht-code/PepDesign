from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DiscoveredFile:
    path: str
    kind: str
    size_bytes: int


def _safe_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return -1


def discover_under_root(project_root: Path, subdirs: list[str], extra_globs: list[str]) -> dict[str, Any]:
    root = project_root.resolve()
    files: list[DiscoveredFile] = []

    for sub in subdirs:
        base = root / sub
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            kind = "other"
            if suf == ".pdb":
                kind = "pdb"
            elif suf == ".csv":
                kind = "csv"
            elif suf == ".json":
                kind = "json"
            elif suf in (".fasta", ".faa", ".fa"):
                kind = "fasta"
            if kind != "other":
                files.append(DiscoveredFile(str(p), kind, _safe_size(p)))

    for pattern in extra_globs:
        for p in root.glob(pattern):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            kind = "other"
            if suf == ".pdb":
                kind = "pdb"
            elif suf == ".csv":
                kind = "csv"
            elif suf == ".json":
                kind = "json"
            elif suf in (".fasta", ".faa", ".fa"):
                kind = "fasta"
            if kind != "other":
                files.append(DiscoveredFile(str(p), kind, _safe_size(p)))

    # de-duplicate by path
    seen: set[str] = set()
    uniq: list[DiscoveredFile] = []
    for f in files:
        if f.path in seen:
            continue
        seen.add(f.path)
        uniq.append(f)

    summary = {
        "project_root": str(root),
        "n_files": len(uniq),
        "by_kind": {},
    }
    for f in uniq:
        summary["by_kind"][f.kind] = summary["by_kind"].get(f.kind, 0) + 1

    priority_csv_names = (
        "all_samples.csv",
        "baseline_input_index.csv",
        "baseline_best_candidates.csv",
    )
    priority = [f for f in uniq if Path(f.path).name in priority_csv_names]
    samples_like = [
        f
        for f in uniq
        if f.kind == "csv" and Path(f.path).name.startswith("samples_")
    ]
    dock_models = [
        f
        for f in uniq
        if f.kind == "pdb"
        and "hdock_work" in f.path.lower()
        and Path(f.path).name.lower().startswith("model_")
    ]
    clean_peptide_pdbs = [
        f
        for f in uniq
        if f.kind == "pdb" and "clean_inputs" in f.path.lower()
    ]
    clean_props = [
        f
        for f in uniq
        if f.kind == "json" and "clean_properties" in f.path.lower()
    ]

    return {
        "summary": summary,
        "priority_csv": [asdict(x) for x in priority],
        "samples_csv": [asdict(x) for x in samples_like[:5000]],
        "samples_csv_truncated": len(samples_like) > 5000,
        "samples_csv_total": len(samples_like),
        "hdock_model_pdb": [asdict(x) for x in dock_models[:8000]],
        "hdock_model_pdb_truncated": len(dock_models) > 8000,
        "hdock_model_pdb_total": len(dock_models),
        "clean_inputs_pdb": [asdict(x) for x in clean_peptide_pdbs[:8000]],
        "clean_inputs_pdb_truncated": len(clean_peptide_pdbs) > 8000,
        "clean_inputs_pdb_total": len(clean_peptide_pdbs),
        "clean_properties_json": [asdict(x) for x in clean_props],
        "all_files_csv": [asdict(x) for x in uniq],
    }


def write_discovery(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "discovery_summary.json").write_text(
        json.dumps(payload.get("summary", {}), indent=2), encoding="utf-8"
    )
    (out_dir / "discovery_full.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    rows = payload.get("all_files_csv", [])
    pd.DataFrame(rows).to_csv(out_dir / "discovered_files.csv", index=False)

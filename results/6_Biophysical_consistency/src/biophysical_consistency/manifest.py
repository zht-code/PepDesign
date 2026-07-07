from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .pdb_io import is_docked_complex_path, is_likely_peptide_only_path


def stable_uid(parts: list[str]) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return h


def load_priority_tables(project_root: Path) -> dict[str, pd.DataFrame]:
    root = project_root
    out: dict[str, pd.DataFrame] = {}
    candidates = [
        root
        / "results/5_robustness/baseline/raw_results/rfdiffusion/all_samples.csv",
        root
        / "results/5_robustness/baseline/raw_results/bindcraft/all_samples.csv",
        root
        / "results/5_robustness/baseline/raw_results/proteingenerator/all_samples.csv",
        root / "results/5_robustness/baseline/tables/baseline_input_index.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                out[str(p.relative_to(root))] = pd.read_csv(p)
            except Exception:
                continue
    # merge all all_samples style
    frames = []
    for k, df in list(out.items()):
        if k.endswith("all_samples.csv"):
            frames.append(df)
    if frames:
        out["__merged_all_samples__"] = pd.concat(frames, ignore_index=True)
    return out


def build_manifest(project_root: Path, discovery_json: Path | None) -> pd.DataFrame:
    pr = project_root.resolve()
    tables = load_priority_tables(pr)
    rows: list[dict[str, Any]] = []

    merged = tables.get("__merged_all_samples__")
    index_df = None
    for k, df in tables.items():
        if k.endswith("baseline_input_index.csv"):
            index_df = df
            break

    if merged is not None:
        for _, r in merged.iterrows():
            pdb_path = str(r.get("pdb_path", "") or "")
            seq = str(r.get("sequence_top1", r.get("sequence", "")) or "")
            method = str(r.get("method", ""))
            target = str(r.get("target_id", ""))
            cand = str(r.get("candidate_id", ""))
            cond = str(r.get("condition_tag", ""))
            p = Path(pdb_path)
            uid = stable_uid([method, target, cand, cond, pdb_path])
            peptide_chain = None
            if index_df is not None and "pdb_path" in index_df.columns:
                m = index_df[index_df["pdb_path"].astype(str) == pdb_path]
                if not m.empty and "peptide_chain" in m.columns:
                    peptide_chain = str(m.iloc[0]["peptide_chain"])
            pdb_role = "unknown"
            if is_docked_complex_path(p):
                pdb_role = "docked_complex"
            elif is_likely_peptide_only_path(p):
                pdb_role = "free_peptide"
            elif p.suffix.lower() == ".pdb":
                pdb_role = "pdb_other"
            rows.append(
                {
                    "sample_uid": uid,
                    "source_table": "all_samples_merged",
                    "method": method,
                    "target_id": target,
                    "candidate_id": cand,
                    "condition_tag": cond,
                    "sequence": seq,
                    "pdb_path": pdb_path,
                    "pdb_exists": p.exists(),
                    "peptide_chain_hint": peptide_chain,
                    "pdb_role": pdb_role,
                    "complex_model_path": "",
                }
            )

    docked_models: list[Path] = []
    dock_root = pr / "results/5_robustness/baseline/cache/hdock_work"
    if dock_root.exists():
        docked_models = [p for p in dock_root.rglob("model_*.pdb")]
    # cap to avoid explosion in manifest optional second rows
    max_extra = 50000
    for p in docked_models[:max_extra]:
        rel = str(p)
        parts = p.parts
        method = ""
        if "rfdiffusion" in parts:
            method = "rfdiffusion"
        elif "bindcraft" in parts:
            method = "bindcraft"
        elif "proteingenerator" in parts:
            method = "proteingenerator"
        target_id = p.parent.name
        uid = stable_uid([method, target_id, p.name, rel])
        rows.append(
            {
                "sample_uid": uid,
                "source_table": "hdock_model_glob",
                "method": method,
                "target_id": target_id,
                "candidate_id": p.stem,
                "condition_tag": "",
                "sequence": "",
                "pdb_path": "",
                "pdb_exists": False,
                "peptide_chain_hint": None,
                "pdb_role": "docked_complex",
                "complex_model_path": rel,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.drop_duplicates(subset=["sample_uid"], inplace=True)
    return df.reset_index(drop=True)


def write_manifest(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sample_master_table.csv"
    json_path = out_dir / "sample_master_table_summary.json"
    df.to_csv(csv_path, index=False)
    summary = {
        "n_rows": int(len(df)),
        "n_pdb_paths": int(df["pdb_path"].astype(str).str.len().gt(0).sum()),
        "n_complex_paths": int(df["complex_model_path"].astype(str).str.len().gt(0).sum()),
        "by_method": df["method"].value_counts(dropna=False).to_dict(),
        "by_pdb_role": df["pdb_role"].value_counts(dropna=False).to_dict(),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

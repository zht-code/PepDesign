#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
EVAL_ROOT = PROJECT_ROOT / "utils" / "evaluate"
for path in [PROJECT_ROOT, EVAL_ROOT, THIS_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from compute_affinity_auto_cpu import run_hdock_pair

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DEFAULT_SPLITS_DIR = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/splits"
DEFAULT_BASELINE_DIR = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/baseline_data"
DEFAULT_WORK_ROOT = "/root/autodl-tmp/hdock_native_2sota_work"
DEFAULT_HDOCK_BIN = "/root/autodl-fs/HDOCKlite/hdock"
DEFAULT_CREATEPL_BIN = "/root/autodl-fs/HDOCKlite/createpl"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute native receptor-peptide HDOCK affinity for 2_SOTA family/protein test splits."
    )
    ap.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    ap.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    ap.add_argument("--work-root", default=DEFAULT_WORK_ROOT)
    ap.add_argument("--hdock-bin", default=DEFAULT_HDOCK_BIN)
    ap.add_argument("--createpl-bin", default=DEFAULT_CREATEPL_BIN)
    ap.add_argument("--timeout", type=int, default=900, help="Timeout per docking task in seconds.")
    ap.add_argument("--workers", type=int, default=72, help="CPU workers. Default 72 to match your machine.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip rows already saved with a valid score.")
    return ap.parse_args()


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def collect_tasks(
    split_name: str,
    split_csv: Path,
    existing_rows: dict[str, dict[str, Any]],
    skip_existing: bool,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    rows = read_csv_rows(split_csv)
    tasks: list[dict[str, str]] = []
    cached: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        prev = existing_rows.get(sample_id)
        if skip_existing and isinstance(prev, dict):
            prev_score = prev.get("native_hdock_score")
            if prev_score is not None:
                try:
                    float(prev_score)
                    cached[sample_id] = prev
                    continue
                except Exception:
                    pass
        receptor_pdb = Path(str(row["receptor_pdb"]))
        peptide_pdb = Path(str(row["peptide_pdb"]))
        if not receptor_pdb.is_file() or not peptide_pdb.is_file():
            continue
        task = dict(row)
        task["split_name"] = split_name
        tasks.append(task)
    return tasks, cached


def log_path(baseline_dir: Path, split_name: str, sample_id: str) -> Path:
    return baseline_dir / "_native_hdock_logs" / split_name / f"{sample_id}.log"


def task_workdir(work_root: Path, split_name: str, sample_id: str) -> Path:
    return work_root / split_name / sample_id


def run_one(
    row: dict[str, str],
    *,
    baseline_dir: Path,
    work_root: Path,
    hdock_bin: str,
    createpl_bin: str,
    timeout: int,
) -> tuple[str, str, dict[str, Any]]:
    split_name = str(row["split_name"])
    sample_id = str(row["sample_id"])
    protein_id = str(row.get("protein_id", ""))
    receptor_pdb = str(row["receptor_pdb"])
    peptide_pdb = str(row["peptide_pdb"])

    try:
        score, log = run_hdock_pair(
            str(task_workdir(work_root, split_name, sample_id)),
            receptor_pdb,
            peptide_pdb,
            hdock_bin,
            createpl_bin,
            timeout_s=timeout,
        )
    except Exception as exc:
        score = None
        log = f"[ERROR] docking failed: {exc}"

    out_log = log_path(baseline_dir, split_name, sample_id)
    ensure_dir(out_log.parent)
    out_log.write_text(log, encoding="utf-8")

    payload = {
        "split_name": split_name,
        "sample_id": sample_id,
        "protein_id": protein_id,
        "sample_dir": str(row.get("sample_dir", "")),
        "receptor_pdb": receptor_pdb,
        "peptide_pdb": peptide_pdb,
        "native_hdock_score": None if score is None else float(score),
        "hdock_log_path": str(out_log),
    }
    return split_name, sample_id, payload


def progress_iter(futures, total: int, desc: str):
    if tqdm is None:
        return as_completed(futures)
    return tqdm(as_completed(futures), total=total, desc=desc)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    baseline_dir = ensure_dir(args.baseline_dir)
    work_root = ensure_dir(args.work_root)

    split_specs = [
        ("family_level_test", splits_dir / "family_level_test.csv"),
        ("protein_level_test", splits_dir / "protein_level_test.csv"),
    ]

    cpu_count = os.cpu_count() or 1
    workers = max(1, min(int(args.workers), cpu_count))
    print(f"CPU count={cpu_count}, using workers={workers}")
    print(f"HDOCK work root: {work_root}")

    per_split_results: dict[str, dict[str, dict[str, Any]]] = {}
    all_tasks: list[dict[str, str]] = []
    for split_name, split_csv in split_specs:
        out_json = baseline_dir / f"{split_name}_native_hdock.json"
        existing = load_existing_rows(out_json)
        tasks, cached = collect_tasks(split_name, split_csv, existing, args.skip_existing)
        per_split_results[split_name] = dict(cached)
        all_tasks.extend(tasks)
        print(
            f"{split_name}: csv_rows={len(read_csv_rows(split_csv))}, "
            f"pending={len(tasks)}, cached={len(cached)}"
        )

    total_pending = len(all_tasks)
    print(f"Total pending native docking tasks: {total_pending}")

    progress_json = baseline_dir / "native_hdock_progress.json"
    if all_tasks:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_one,
                    row,
                    baseline_dir=baseline_dir,
                    work_root=work_root,
                    hdock_bin=args.hdock_bin,
                    createpl_bin=args.createpl_bin,
                    timeout=args.timeout,
                )
                for row in all_tasks
            ]

            for idx, future in enumerate(progress_iter(futures, len(futures), f"HDOCK native x{workers}"), start=1):
                split_name, sample_id, payload = future.result()
                per_split_results.setdefault(split_name, {})[sample_id] = payload

                # Persist progress after every completed task.
                for save_split_name, split_rows in per_split_results.items():
                    out_json = baseline_dir / f"{save_split_name}_native_hdock.json"
                    ordered = dict(sorted(split_rows.items(), key=lambda kv: kv[0]))
                    save_json(out_json, ordered)
                save_json(
                    progress_json,
                    {
                        "completed": idx,
                        "total_pending": total_pending,
                        "family_level_test_count": len(per_split_results.get("family_level_test", {})),
                        "protein_level_test_count": len(per_split_results.get("protein_level_test", {})),
                    },
                )

                if tqdm is None:
                    print(f"[{idx}/{total_pending}] {split_name} {sample_id} score={payload['native_hdock_score']}")

    combined = {
        split_name: dict(sorted(rows.items(), key=lambda kv: kv[0]))
        for split_name, rows in per_split_results.items()
    }
    save_json(baseline_dir / "all_native_hdock.json", combined)
    save_json(
        baseline_dir / "native_hdock_summary.json",
        {
            "workers": workers,
            "cpu_count": cpu_count,
            "work_root": str(work_root),
            "family_level_test_json": str(baseline_dir / "family_level_test_native_hdock.json"),
            "protein_level_test_json": str(baseline_dir / "protein_level_test_native_hdock.json"),
            "combined_json": str(baseline_dir / "all_native_hdock.json"),
            "progress_json": str(progress_json),
            "family_level_test_count": len(per_split_results.get("family_level_test", {})),
            "protein_level_test_count": len(per_split_results.get("protein_level_test", {})),
        },
    )
    print(f"Saved family native affinity JSON: {baseline_dir / 'family_level_test_native_hdock.json'}")
    print(f"Saved protein native affinity JSON: {baseline_dir / 'protein_level_test_native_hdock.json'}")
    print(f"Saved combined native affinity JSON: {baseline_dir / 'all_native_hdock.json'}")


if __name__ == "__main__":
    main()



'''

python /root/autodl-tmp/Peptide_3D/results/2_SOTA/compute_native_affinity_2sota.py \
  --workers 72 \
  --work-root /root/autodl-tmp/hdock_native_2sota_work \
  --baseline-dir /root/autodl-tmp/Peptide_3D/results/2_SOTA/baseline_data \
  --splits-dir /root/autodl-tmp/Peptide_3D/results/2_SOTA/splits \
  --hdock-bin /root/autodl-fs/HDOCKlite/hdock \
  --createpl-bin /root/autodl-fs/HDOCKlite/createpl \
  --timeout 900 \
  --skip-existing


'''
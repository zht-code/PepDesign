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
PROJECT_ROOT = THIS_DIR.parents[2]
EVAL_ROOT = PROJECT_ROOT / "utils" / "evaluate"

for path in [THIS_DIR, EVAL_ROOT, PROJECT_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from compute_affinity_auto_cpu import run_hdock_pair

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DEFAULT_OUT_ROOT = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/unconditional"
DEFAULT_WORK_ROOT = "/root/autodl-tmp/hdock_unconditional_work"
DEFAULT_PROTEIN_MANIFEST = f"{DEFAULT_OUT_ROOT}/protein_level_test_manifest.csv"
DEFAULT_FAMILY_MANIFEST = f"{DEFAULT_OUT_ROOT}/family_level_test_manifest.csv"
DEFAULT_HDOCK_BIN = "/root/autodl-fs/HDOCKlite/hdock"
DEFAULT_CREATEPL_BIN = "/root/autodl-fs/HDOCKlite/createpl"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compute HDOCK affinity for unconditional peptide generations on "
            "protein_level_test and family_level_test using CPU parallelism."
        )
    )
    ap.add_argument("--protein-manifest", default=DEFAULT_PROTEIN_MANIFEST)
    ap.add_argument("--family-manifest", default=DEFAULT_FAMILY_MANIFEST)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument(
        "--work-root",
        default=DEFAULT_WORK_ROOT,
        help="HDOCK temporary work directory. Put this under /root/autodl-tmp to avoid system disk usage.",
    )
    ap.add_argument("--hdock-bin", default=DEFAULT_HDOCK_BIN)
    ap.add_argument("--createpl-bin", default=DEFAULT_CREATEPL_BIN)
    ap.add_argument("--timeout", type=int, default=900, help="Timeout per docking task in seconds.")
    ap.add_argument(
        "--workers",
        type=int,
        default=72,
        help="CPU worker count. Default 72 to fully utilize a 72-core machine.",
    )
    ap.add_argument("--skip-existing", action="store_true", help="Skip tasks already present in existing output CSV.")
    return ap.parse_args()


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_scores(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.is_file():
        return {}
    rows = read_csv_rows(path)
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["split_name"], row["sample_id"], row["candidate_rank"])
        out[key] = row
    return out


def collect_tasks(
    manifest_path: str | Path,
    split_name: str,
    existing: dict[tuple[str, str, str], dict[str, str]],
    skip_existing: bool,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    manifest_rows = read_csv_rows(manifest_path)
    tasks: list[dict[str, str]] = []
    cached_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        key = (row["split_name"], row["sample_id"], row["candidate_rank"])
        if skip_existing and key in existing:
            cached_rows.append(existing[key])
            continue
        if not row.get("generated_peptide_pdb"):
            continue
        if not Path(row["receptor_pdb"]).is_file():
            continue
        if not Path(row["generated_peptide_pdb"]).is_file():
            continue
        tasks.append(row)
    return tasks, cached_rows


def task_workdir(work_root: Path, row: dict[str, str]) -> Path:
    return work_root / row["split_name"] / row["sample_id"] / f"cand_{int(row['candidate_rank']):02d}"


def task_log_path(out_root: Path, row: dict[str, str]) -> Path:
    return out_root / "_hdock_logs" / row["split_name"] / row["sample_id"] / f"cand_{int(row['candidate_rank']):02d}.log"


def run_one(
    row: dict[str, str],
    out_root: Path,
    work_root: Path,
    hdock_bin: str,
    createpl_bin: str,
    timeout: int,
) -> dict[str, Any]:
    workdir = task_workdir(work_root, row)
    try:
        score, log = run_hdock_pair(
            str(workdir),
            row["receptor_pdb"],
            row["generated_peptide_pdb"],
            hdock_bin,
            createpl_bin,
            timeout_s=timeout,
        )
    except Exception as exc:
        score = None
        log = f"[ERROR] docking failed: {exc}"
    log_path = task_log_path(out_root, row)
    ensure_dir(log_path.parent)
    log_path.write_text(log, encoding="utf-8")

    result = dict(row)
    result["hdock_score"] = "" if score is None else f"{float(score):.6f}"
    result["hdock_log_path"] = str(log_path)
    return result


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def progress_iter(futures, total: int, desc: str):
    if tqdm is None:
        return as_completed(futures)
    return tqdm(as_completed(futures), total=total, desc=desc)


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(args.out_root)
    work_root = ensure_dir(args.work_root)

    cpu_count = os.cpu_count() or 1
    workers = max(1, min(args.workers, cpu_count))
    print(f"CPU count={cpu_count}, using workers={workers}")
    print(f"HDOCK work root: {work_root}")

    all_out_csv = out_root / "all_test_sets_affinity.csv"
    existing = load_existing_scores(all_out_csv)

    split_specs = [
        ("protein_level_test", Path(args.protein_manifest)),
        ("family_level_test", Path(args.family_manifest)),
    ]

    all_tasks: list[dict[str, str]] = []
    completed_rows: list[dict[str, Any]] = []
    per_split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in split_specs}

    for split_name, manifest_path in split_specs:
        tasks, cached_rows = collect_tasks(manifest_path, split_name, existing, args.skip_existing)
        all_tasks.extend(tasks)
        completed_rows.extend(cached_rows)
        per_split_rows[split_name].extend(cached_rows)
        print(
            f"{split_name}: manifest_rows={len(read_csv_rows(manifest_path))}, "
            f"pending={len(tasks)}, cached={len(cached_rows)}"
        )

    print(f"Total pending docking tasks: {len(all_tasks)}")
    checkpoint_json = out_root / "all_test_sets_affinity_progress.json"

    if all_tasks:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_one,
                    row,
                    out_root,
                    work_root,
                    args.hdock_bin,
                    args.createpl_bin,
                    args.timeout,
                )
                for row in all_tasks
            ]

            for idx, future in enumerate(progress_iter(futures, len(futures), f"HDOCK CPU x{workers}"), start=1):
                result = future.result()
                completed_rows.append(result)
                per_split_rows[result["split_name"]].append(result)

                payload = {
                    "completed": idx,
                    "total_pending": len(all_tasks),
                    "rows": completed_rows,
                }
                save_json(checkpoint_json, payload)

                if tqdm is None:
                    print(
                        f"[{idx}/{len(all_tasks)}] "
                        f"{result['split_name']} {result['sample_id']} cand{result['candidate_rank']} "
                        f"score={result['hdock_score']}"
                    )

    def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda r: (r["split_name"], r["sample_id"], int(r["candidate_rank"])),
        )

    completed_rows = sort_rows(completed_rows)
    write_csv_rows(all_out_csv, completed_rows)

    for split_name, _ in split_specs:
        split_rows = sort_rows(per_split_rows[split_name])
        split_csv = out_root / f"{split_name}_affinity.csv"
        write_csv_rows(split_csv, split_rows)

    summary = {
        "workers": workers,
        "cpu_count": cpu_count,
        "total_rows": len(completed_rows),
        "protein_level_test_rows": len(per_split_rows["protein_level_test"]),
        "family_level_test_rows": len(per_split_rows["family_level_test"]),
        "all_output_csv": str(all_out_csv),
        "protein_output_csv": str(out_root / "protein_level_test_affinity.csv"),
        "family_output_csv": str(out_root / "family_level_test_affinity.csv"),
        "progress_json": str(checkpoint_json),
    }
    save_json(out_root / "affinity_summary.json", summary)

    print(f"Saved combined affinity CSV: {all_out_csv}")
    print(f"Saved split affinity CSVs under: {out_root}")
    print(f"Saved progress JSON: {checkpoint_json}")


if __name__ == "__main__":
    main()



'''

python /root/autodl-tmp/Peptide_3D/results/2_SOTA/unconditional/compute_affinity_unconditional.py \
  --workers 72

'''
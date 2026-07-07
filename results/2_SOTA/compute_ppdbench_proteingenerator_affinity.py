#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
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


DEFAULT_PPD_ROOT = "/root/autodl-tmp/PPDbench"
DEFAULT_ZIP = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/baseline_data/proteingenerator.zip"
DEFAULT_OUT_JSON = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/baseline_data/proteingenerator_ppdbench_hdock_scores.json"
DEFAULT_WORK_ROOT = "/root/autodl-tmp/proteingenerator_ppdbench_hdock_work"
DEFAULT_HDOCK_BIN = "/root/autodl-fs/HDOCKlite/hdock"
DEFAULT_CREATEPL_BIN = "/root/autodl-fs/HDOCKlite/createpl"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute PPDbench ProteinGenerator top5/topN HDOCK affinity from proteingenerator.zip.")
    ap.add_argument("--ppdbench-root", default=DEFAULT_PPD_ROOT)
    ap.add_argument("--zip-path", default=DEFAULT_ZIP)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--work-root", default=DEFAULT_WORK_ROOT)
    ap.add_argument("--hdock-bin", default=DEFAULT_HDOCK_BIN)
    ap.add_argument("--createpl-bin", default=DEFAULT_CREATEPL_BIN)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--workers", type=int, default=72)
    ap.add_argument("--topk", type=int, default=5, help="How many candidates per target to process from the zip.")
    ap.add_argument("--target-id", default=None, help="Optional single target id for smoke testing.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on total docking tasks.")
    ap.add_argument("--skip-existing", action="store_true")
    return ap.parse_args()


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def collect_zip_members(zip_path: Path, topk: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in sorted(zf.namelist()):
            parts = member.strip("/").split("/")
            if len(parts) != 3 or parts[0] != "proteingenerator" or not parts[2].endswith(".pdb"):
                continue
            target_id = parts[1]
            out.append((target_id, member))
    grouped: dict[str, list[str]] = {}
    for target_id, member in out:
        grouped.setdefault(target_id, []).append(member)
    selected: list[tuple[str, str]] = []
    for target_id in sorted(grouped):
        for member in sorted(grouped[target_id])[:topk]:
            selected.append((target_id, member))
    return selected


def extract_member_once(zf: zipfile.ZipFile, member: str, extract_root: Path) -> Path:
    out_path = extract_root / member
    if out_path.is_file():
        return out_path
    ensure_dir(out_path.parent)
    with zf.open(member) as src, open(out_path, "wb") as dst:
        dst.write(src.read())
    return out_path


def result_key(member: str) -> str:
    return f"/{member}"


def run_one(
    *,
    target_id: str,
    member: str,
    ppd_root: Path,
    zip_path: Path,
    extract_root: Path,
    hdock_root: Path,
    hdock_bin: str,
    createpl_bin: str,
    timeout: int,
) -> tuple[str, str, dict[str, Any]]:
    receptor_pdb = ppd_root / target_id / "receptor.pdb"
    if not receptor_pdb.is_file():
        raise FileNotFoundError(f"Missing receptor PDB: {receptor_pdb}")

    with zipfile.ZipFile(zip_path) as zf:
        peptide_pdb = extract_member_once(zf, member, extract_root)

    candidate_name = Path(member).stem
    workdir = hdock_root / target_id / candidate_name
    score, log = run_hdock_pair(
        str(workdir),
        str(receptor_pdb),
        str(peptide_pdb),
        hdock_bin,
        createpl_bin,
        timeout_s=timeout,
    )
    log_path = workdir / "dock.log"
    ensure_dir(log_path.parent)
    log_path.write_text(log, encoding="utf-8")
    payload = {
        "protein_id": target_id,
        "receptor_pdb": str(receptor_pdb),
        "peptide_pdb": str(peptide_pdb),
        "score": None if score is None else float(score),
        "log_path": str(log_path),
    }
    return target_id, result_key(member), payload


def progress_iter(futures, total: int, desc: str):
    if tqdm is None:
        return as_completed(futures)
    return tqdm(as_completed(futures), total=total, desc=desc)


def main() -> None:
    args = parse_args()
    ppd_root = Path(args.ppdbench_root)
    zip_path = Path(args.zip_path)
    out_json = Path(args.out_json)
    work_root = ensure_dir(args.work_root)
    extract_root = ensure_dir(work_root / "extracted")
    hdock_root = ensure_dir(work_root / "hdock")

    existing = load_existing(out_json)
    members = collect_zip_members(zip_path, args.topk)
    if args.target_id:
        members = [(target_id, member) for target_id, member in members if target_id == args.target_id]
    if args.limit is not None:
        members = members[: max(0, int(args.limit))]

    tasks: list[tuple[str, str]] = []
    for target_id, member in members:
        key = result_key(member)
        if args.skip_existing:
            prev = existing.get(target_id, {}).get(key, {})
            if isinstance(prev, dict) and prev.get("score") is not None:
                continue
        tasks.append((target_id, member))

    cpu_count = os.cpu_count() or 1
    workers = max(1, min(int(args.workers), cpu_count))
    print(f"CPU count={cpu_count}, using workers={workers}")
    print(f"Pending ProteinGenerator PPDbench docking tasks: {len(tasks)}")
    print(f"Work root: {work_root}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                run_one,
                target_id=target_id,
                member=member,
                ppd_root=ppd_root,
                zip_path=zip_path,
                extract_root=extract_root,
                hdock_root=hdock_root,
                hdock_bin=args.hdock_bin,
                createpl_bin=args.createpl_bin,
                timeout=args.timeout,
            )
            for target_id, member in tasks
        ]

        for idx, future in enumerate(progress_iter(futures, len(futures), f"HDOCK ProteinGenerator x{workers}"), start=1):
            try:
                target_id, key, payload = future.result()
            except Exception as exc:
                target_id, key, payload = "UNKNOWN", f"error_{idx}", {"score": None, "error": str(exc)}
            existing.setdefault(target_id, {})[key] = payload
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            if tqdm is None:
                print(f"[{idx}/{len(futures)}] {target_id} {key} score={payload.get('score')}")

    print(f"Saved ProteinGenerator PPDbench docking JSON: {out_json}")


if __name__ == "__main__":
    main()

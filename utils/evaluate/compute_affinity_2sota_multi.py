#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch HDOCK affinity for 2_SOTA protein/family test sets with progress."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from compute_affinity_auto_cpu import run_hdock_pair


def parse_gpu_list(gpus: str) -> List[Optional[int]]:
    vals = []
    for x in gpus.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals if vals else [None]


def collect_tasks(dataset_root: Path) -> List[Tuple[str, Path, Path, str]]:
    """
    Return tasks: (sample_id, receptor_pdb, peptide_pdb, pep_name)
    Only pep_*.pdb under multi_cands1 are included.
    """
    tasks: List[Tuple[str, Path, Path, str]] = []
    for sample_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        rec = sample_dir / "receptor.pdb"
        cands = sample_dir / "multi_cands1"
        if not rec.is_file() or not cands.is_dir():
            continue
        peps = sorted(cands.glob("pep_*.pdb"))
        for pep in peps:
            tasks.append((sample_dir.name, rec, pep, pep.stem))
    return tasks


def run_one_task(
    dataset_name: str,
    sample_id: str,
    receptor_pdb: Path,
    peptide_pdb: Path,
    pep_name: str,
    work_root: Path,
    hdock_bin: str,
    createpl_bin: str,
    timeout_s: int,
    gpu_id: Optional[int],
) -> Tuple[str, str, str, Optional[float], str]:
    # 独立工作目录，避免同名样本在两个数据集冲突
    workdir = work_root / dataset_name / sample_id / pep_name
    env_vars = {"CUDA_VISIBLE_DEVICES": str(gpu_id)} if gpu_id is not None else None
    score, log = run_hdock_pair(
        str(workdir),
        str(receptor_pdb),
        str(peptide_pdb),
        hdock_bin,
        createpl_bin,
        timeout_s=timeout_s,
    )
    # run_hdock_pair 没有 env 入参；这里保留 gpu_id 到日志标记便于追踪
    log = f"[GPU_HINT]={gpu_id}\n{log}"
    return dataset_name, sample_id, str(peptide_pdb.resolve()), score, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--protein-root",
        default="/root/autodl-tmp/Peptide_3D/results/2_SOTA/protein_level_test",
    )
    ap.add_argument(
        "--family-root",
        default="/root/autodl-tmp/Peptide_3D/results/2_SOTA/family_level_test",
    )
    ap.add_argument(
        "--out-json",
        default="/root/autodl-tmp/Peptide_3D/data/2sota_multi_cands1_hdock_scores.json",
        help="Combined JSON for both datasets.",
    )
    ap.add_argument(
        "--work-root",
        default="/root/autodl-tmp/tmp_hdock_2sota_multi",
    )
    ap.add_argument("--hdock-bin", default="/root/autodl-fs/HDOCKlite/hdock")
    ap.add_argument("--createpl-bin", default="/root/autodl-fs/HDOCKlite/createpl")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--num-workers", type=int, default=36, help="CPU threads; set 36 to max your machine.")
    ap.add_argument("--gpus", type=str, default="0,1", help="GPU ids, round-robin assignment hint.")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    protein_root = Path(args.protein_root)
    family_root = Path(args.family_root)
    work_root = Path(args.work_root)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    if not protein_root.is_dir():
        raise FileNotFoundError(f"Missing protein root: {protein_root}")
    if not family_root.is_dir():
        raise FileNotFoundError(f"Missing family root: {family_root}")

    results: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
        "protein_level_test": {},
        "family_level_test": {},
    }
    logs_map: Dict[str, str] = {}
    if out_json.exists():
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    results.update(loaded)
        except Exception:
            pass

    all_tasks = []
    for dataset_name, root in [
        ("protein_level_test", protein_root),
        ("family_level_test", family_root),
    ]:
        for sample_id, rec, pep, pep_name in collect_tasks(root):
            pep_abs = str(pep.resolve())
            if args.skip_existing:
                prev = results.get(dataset_name, {}).get(sample_id, {}).get(pep_abs, None)
                if prev is not None:
                    continue
            all_tasks.append((dataset_name, sample_id, rec, pep, pep_name))

    print(f"Total peptide affinity tasks: {len(all_tasks)}")
    if not all_tasks:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"No pending tasks. Existing results kept at: {out_json}")
        return

    gpu_list = parse_gpu_list(args.gpus)
    workers = max(1, int(args.num_workers))
    print(f"Using {workers} CPU workers; GPU hint list: {gpu_list}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for i, (dataset_name, sample_id, rec, pep, pep_name) in enumerate(all_tasks):
            gpu_id = gpu_list[i % len(gpu_list)]
            futures.append(
                ex.submit(
                    run_one_task,
                    dataset_name,
                    sample_id,
                    rec,
                    pep,
                    pep_name,
                    work_root,
                    args.hdock_bin,
                    args.createpl_bin,
                    args.timeout,
                    gpu_id,
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"HDOCK 2SOTA (threads x{workers})"):
            dataset_name, sample_id, pep_abs, score, log = fut.result()
            ds = results.setdefault(dataset_name, {})
            sm = ds.setdefault(sample_id, {})
            sm[pep_abs] = (float(score) if score is not None else None)
            logs_map[f"{dataset_name}/{sample_id}/{Path(pep_abs).stem}"] = log

            # 实时保存总体 JSON
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # 每个样本目录写一份 cands_hdock_scores.json，便于下游脚本直接读取
    for dataset_name, root in [
        ("protein_level_test", protein_root),
        ("family_level_test", family_root),
    ]:
        ds = results.get(dataset_name, {})
        for sample_id, score_map in ds.items():
            out_path = root / sample_id / "multi_cands1" / "cands_hdock_scores.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(score_map, f, indent=2, ensure_ascii=False)

    # 保存运行日志（可选排错）
    logs_json = out_json.with_name(out_json.stem + "_logs.json")
    with open(logs_json, "w", encoding="utf-8") as f:
        json.dump(logs_map, f, indent=2, ensure_ascii=False)

    print(f"Done. Combined scores: {out_json}")
    print(f"Done. Per-task logs:   {logs_json}")


if __name__ == "__main__":
    main()


'''

python /root/autodl-tmp/Peptide_3D/utils/evaluate/compute_affinity_2sota_multi.py \
  --num-workers 36 \
  --gpus 0,1 \
  --hdock-bin /root/autodl-fs/HDOCKlite/hdock \
  --createpl-bin /root/autodl-fs/HDOCKlite/createpl \
  --timeout 900

'''
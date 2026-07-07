#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 PPDbench 上 ablation 三种方法生成的多肽（每靶点 pep_01/02/03）批量跑 HDOCK 亲和力。

参考：
- results/3_Pareto_improved/compute_ppdbench_generated_affinity.py 的计算方式与 JSON 保存方式
- utils/evaluate/compute_affinity_auto_cpu.py 的 HDOCK 调用/解析思路

输出：
在本目录（results/4_ablation）写 3 个 JSON：
  - ppdbench_hdock_ablation_base.json
  - ppdbench_hdock_ablation_base_ot.json
  - ppdbench_hdock_ablation_base_dpo.json

并行：
- 默认 workers=os.cpu_count()（你是 36 核就会默认 36）
- 每个 worker 是一个线程，内部启动一个 hdock 子进程（CPU 密集，能把 CPU 打满）
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


# ---------------------------------------------------------------------------
# 与 results/3_Pareto_improved/compute_ppdbench_generated_affinity.py 一致的解析与 HDOCK 调用
# ---------------------------------------------------------------------------
SCORE_RE_LIST = [
    re.compile(r"(?i)\bscore\b\s*:?\s*([+-]?[0-9]+(?:\.[0-9]+)?)"),
    re.compile(r"(?i)\btotal\s*score\b\s*:?\s*([+-]?[0-9]+(?:\.[0-9]+)?)"),
]


def _parse_best_score_in_textfile(path: str):
    if not path or not os.path.exists(path):
        return None
    best = None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            for rgx in SCORE_RE_LIST:
                m = rgx.search(line)
                if m:
                    try:
                        v = float(m.group(1))
                    except Exception:
                        continue
                    best = v if best is None else (v if v < best else best)
    return best


def _find_any_out_file(workdir: str):
    candidates = [
        os.path.join(workdir, "hdock.out"),
        os.path.join(workdir, "Hdock.out"),
        os.path.join(workdir, "HDOCK.out"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    outs = glob.glob(os.path.join(workdir, "*.out"))
    if outs:
        return max(outs, key=lambda x: os.path.getsize(x))
    return None


def _parse_score_from_pdb(pdb_path: str):
    if not pdb_path or not os.path.exists(pdb_path):
        return None
    best = None
    numeric_best = None
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("REMARK"):
                continue
            for rgx in SCORE_RE_LIST:
                m = rgx.search(line)
                if m:
                    try:
                        v = float(m.group(1))
                    except Exception:
                        continue
                    best = v if best is None else (v if v < best else best)

            parts = line.strip().split()
            if len(parts) < 4:
                continue
            floats = []
            for p in parts:
                try:
                    floats.append(float(p))
                except Exception:
                    continue
            if not floats:
                continue
            cand = min(floats)
            if cand < -1.0:
                numeric_best = cand if numeric_best is None else (cand if cand < numeric_best else numeric_best)

    return best if best is not None else numeric_best


def run_hdock_pair(
    workdir: str,
    receptor_pdb: str,
    peptide_pdb: str,
    hdock_bin: str,
    createpl_bin: str,
    timeout_s: int = 900,
) -> Tuple[Optional[float], str]:
    os.makedirs(workdir, exist_ok=True)
    shutil.copy2(receptor_pdb, os.path.join(workdir, "receptor.pdb"))
    shutil.copy2(peptide_pdb, os.path.join(workdir, "peptide.pdb"))

    logs = []
    cmd = [hdock_bin, "receptor.pdb", "peptide.pdb"]
    logs.append(f"[HDOCK] cmd: {' '.join(cmd)} (cwd={workdir})")
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            text=True,
        )
        logs.append(proc.stdout or "")
        if proc.returncode != 0:
            logs.append(f"[HDOCK] exit code {proc.returncode}, stderr={proc.stderr}")
    except subprocess.TimeoutExpired:
        logs.append("[HDOCK] timeout")
    except Exception as e:
        logs.append(f"[HDOCK] failed: {e}")

    best_score = None
    hdock_out = _find_any_out_file(workdir)
    if hdock_out:
        val = _parse_best_score_in_textfile(hdock_out)
        if val is not None:
            best_score = val

    if best_score is None and createpl_bin:
        if hdock_out and os.path.exists(hdock_out):
            cmd2 = [
                createpl_bin,
                os.path.basename(hdock_out),
                "top3.pdb",
                "-nmax",
                "3",
                "-complex",
                "-models",
            ]
            logs.append(f"[CREATEPL] cmd: {' '.join(cmd2)} (cwd={workdir})")
            try:
                proc2 = subprocess.run(
                    cmd2,
                    cwd=workdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_s,
                    text=True,
                )
                logs.append(proc2.stdout or "")
                if proc2.returncode != 0:
                    logs.append(f"[CREATEPL] exit code {proc2.returncode}, stderr={proc2.stderr}")
            except subprocess.TimeoutExpired:
                logs.append("[CREATEPL] timeout")
            except Exception as e:
                logs.append(f"[CREATEPL] failed: {e}")

        pdb_candidates = []
        for p in ("model_1.pdb", "top1.pdb", "complex_1.pdb"):
            absp = os.path.join(workdir, p)
            if os.path.exists(absp):
                pdb_candidates.append(absp)
        if not pdb_candidates:
            pdb_candidates = [
                p
                for p in glob.glob(os.path.join(workdir, "*.pdb"))
                if os.path.basename(p).lower() not in ("receptor.pdb", "peptide.pdb")
            ]

        best_model_score = None
        for p in pdb_candidates:
            val = _parse_score_from_pdb(p)
            if val is not None and (best_model_score is None or val < best_model_score):
                best_model_score = val
        if best_model_score is not None:
            best_score = best_model_score

    return best_score, "\n".join(logs)


# ---------------------------------------------------------------------------
# PPDbench 任务收集与跑批（ablation 三种生成目录）
# ---------------------------------------------------------------------------
METHOD_SPECS: List[Tuple[str, str]] = [
    ("generated_ablation_base", "ppdbench_hdock_ablation_base.json"),
    ("generated_ablation_base_ot", "ppdbench_hdock_ablation_base_ot.json"),
    ("generated_ablation_base_dpo", "ppdbench_hdock_ablation_base_dpo.json"),
]


def collect_tasks(bench_root: Path, gen_subdir: str) -> List[Tuple[str, str, str, str]]:
    """
    返回列表元素: (target_id, receptor_pdb, peptide_pdb, pep_basename)
    """
    tasks = []
    for d in sorted(p for p in bench_root.iterdir() if p.is_dir()):
        rec = d / "receptor.pdb"
        gdir = d / gen_subdir
        if not rec.is_file() or not gdir.is_dir():
            continue
        for pep in sorted(gdir.glob("pep_*.pdb")):
            tasks.append((d.name, str(rec), str(pep), pep.name))
    return tasks


def _task_key(target_id: str, pep_name: str) -> str:
    return f"{target_id}/{pep_name}"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)


def run_method(
    gen_subdir: str,
    out_json_name: str,
    bench_root: Path,
    results_dir: Path,
    work_root: Path,
    hdock_bin: str,
    createpl_bin: str,
    timeout: int,
    max_workers: int,
    skip_existing: bool,
) -> None:
    out_path = results_dir / out_json_name
    results = _load_json(out_path)
    tasks = collect_tasks(bench_root, gen_subdir)
    if not tasks:
        print(f"[WARN] 无任务: bench={bench_root} subdir={gen_subdir}", file=sys.stderr)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        return

    pending = []
    for target_id, rec, pep, pep_name in tasks:
        key = _task_key(target_id, pep_name)
        if skip_existing and key in results:
            prev = results[key]
            if isinstance(prev, dict) and prev.get("score") is not None:
                try:
                    float(prev["score"])
                    continue
                except Exception:
                    pass
        pending.append((target_id, rec, pep, pep_name, key))

    print(f"[{gen_subdir}] 总任务 {len(tasks)}, 待跑 {len(pending)}, workers={max_workers}")
    file_lock = threading.Lock()
    pbar = tqdm(total=len(pending), desc=out_json_name[:40], unit="pair")

    def _one(item):
        target_id, rec, pep, pep_name, key = item
        wdir = work_root / gen_subdir / target_id / Path(pep_name).stem
        score, logs = run_hdock_pair(str(wdir), rec, pep, hdock_bin, createpl_bin, timeout_s=timeout)
        entry = {
            "target_id": target_id,
            "receptor_pdb": rec,
            "peptide_pdb": pep,
            "peptide_basename": pep_name,
            "score": score,
            "log": logs,
        }
        return key, entry

    if not pending:
        pbar.close()
        print(f"[{gen_subdir}] 全部已存在，跳过。")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, it): it for it in pending}
        for fut in as_completed(futs):
            try:
                key, entry = fut.result()
            except Exception as e:
                it = futs[fut]
                key = _task_key(it[0], it[3])
                entry = {
                    "target_id": it[0],
                    "receptor_pdb": it[1],
                    "peptide_pdb": it[2],
                    "peptide_basename": it[3],
                    "score": None,
                    "log": f"[ERROR] {e}",
                }
            with file_lock:
                results[key] = entry
                _atomic_write_json(out_path, results)
            pbar.update(1)

    pbar.close()
    print(f"[{gen_subdir}] 已写入 {out_path}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="PPDbench ablation 三种生成目录的多肽 HDOCK 亲和力（写 3 个 JSON）")
    ap.add_argument("--bench-root", type=str, default="/root/autodl-tmp/PPDbench")
    ap.add_argument(
        "--results-dir",
        type=str,
        default=str(here),
        help="JSON 输出目录（默认为本脚本所在目录 results/4_ablation）",
    )
    ap.add_argument(
        "--work-root",
        type=str,
        default="/root/autodl-tmp/hdock_ppdbench_ablation",
        help="HDOCK 工作目录根（按方法/靶点/肽分子目录隔离）",
    )
    ap.add_argument("--hdock-bin", type=str, default="/root/autodl-fs/HDOCKlite/hdock")
    ap.add_argument("--createpl-bin", type=str, default="/root/autodl-fs/HDOCKlite/createpl")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--workers",
        type=int,
        default=(os.cpu_count() or 36),
        help="并行线程数（每线程起一个 hdock 子进程）；默认=cpu_count()",
    )
    ap.add_argument("--skip-existing", action="store_true", help="已有有效 score 的键则跳过")
    ap.add_argument(
        "--only",
        type=str,
        default="all",
        help="只跑一种: base|base_ot|base_dpo|all",
    )
    args = ap.parse_args()

    bench_root = Path(args.bench_root)
    results_dir = Path(args.results_dir)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    cpu_n = os.cpu_count() or 1
    workers = max(1, int(args.workers))
    if workers > cpu_n:
        print(
            f"[INFO] workers={workers} > cpu_count={cpu_n}；HDOCK 为 CPU 密集，若机器卡顿可调低 --workers",
            file=sys.stderr,
        )

    only = args.only.lower().strip()
    alias = {
        "all": None,
        "base": METHOD_SPECS[0],
        "base_ot": METHOD_SPECS[1],
        "base_dpo": METHOD_SPECS[2],
    }

    to_run = METHOD_SPECS
    if only != "all":
        if only not in alias or alias[only] is None:
            print("ERROR: --only 应为 all | base | base_ot | base_dpo", file=sys.stderr)
            sys.exit(1)
        to_run = [alias[only]]

    for gen_subdir, out_name in to_run:
        run_method(
            gen_subdir,
            out_name,
            bench_root,
            results_dir,
            work_root,
            args.hdock_bin,
            args.createpl_bin,
            args.timeout,
            workers,
            args.skip_existing,
        )

    print("Done.")


if __name__ == "__main__":
    main()

'''

python /root/autodl-tmp/Peptide_3D/results/4_ablation/compute_ppdbench_generated_affinity_ablation.py \
  --bench-root /root/autodl-tmp/PPDbench \
  --workers 36 \
  --skip-existing

'''
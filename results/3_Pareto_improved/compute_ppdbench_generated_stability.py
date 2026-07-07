#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 PPDbench 上四种生成方法的多肽稳定性（FoldX Stability 分数）。
输出到 results/3_Pareto_improved 下的四个 JSON（分别对应四种方法）。

不保存 log（json 中仅保存 score 与 pdb 路径）。
进度条使用 tqdm，并尽量并行（FoldX 调用在独立临时目录下运行，适合并发）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

import Bio  # noqa: F401

from Bio import PDB  # noqa: F401


def foldx_stability_score_single(
    pdb_path: Path,
    *,
    foldx_bin: str,
    workdir_root: Optional[str],
    timeout_s: int = 600,
) -> Optional[float]:
    """
    使用 FoldX --command=Stability 计算稳定性。
    返回：越大越稳定（脚本内返回 -dg）。
    """
    try:
        if workdir_root is not None:
            os.makedirs(workdir_root, exist_ok=True)
            workdir_str = tempfile.mkdtemp(prefix="foldx_", dir=workdir_root)
        else:
            workdir_str = tempfile.mkdtemp(prefix="foldx_")
        workdir = Path(workdir_str)

        local_pdb = workdir / "peptide.pdb"
        local_pdb.write_text(pdb_path.read_text(encoding="utf-8"), encoding="utf-8")

        cmd = [foldx_bin, "--command=Stability", "--pdb=peptide.pdb"]
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        _ = proc.stdout  # 丢弃输出（不保存 log）

        fxout = workdir / "peptide_0_ST.fxout"
        if not fxout.exists():
            return None

        lines = [l.strip() for l in fxout.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
        if not lines:
            return None

        # 参考原脚本：第一行 tab 分割后取第二列为 dg
        parts = lines[0].split("\t")
        if len(parts) < 2:
            return None
        dg = float(parts[1])
        return -dg
    except Exception:
        return None


def collect_tasks(bench_root: Path, gen_subdir: str) -> List[Tuple[str, Path, Path, str]]:
    """
    返回: (target_id, receptor_pdb, peptide_pdb, peptide_basename)
    稳定性只用 peptide_pdb，但保持同样结构方便扩展/对齐。
    """
    tasks: List[Tuple[str, Path, Path, str]] = []
    for d in sorted(p for p in bench_root.iterdir() if p.is_dir()):
        target_id = d.name
        rec = d / "receptor.pdb"
        gdir = d / gen_subdir
        if not rec.is_file() or not gdir.is_dir():
            continue
        for pep in sorted(gdir.glob("pep_*.pdb")):
            tasks.append((target_id, rec, pep, pep.name))
    return tasks


METHOD_SPECS: List[Tuple[str, str]] = [
    ("generated_dpo_affinity_only", "ppdbench_stability_dpo_affinity_only.json"),
    ("generated_dpo_stability_only", "ppdbench_stability_dpo_stability_only.json"),
    ("generated_dpo_weighted_sum", "ppdbench_stability_dpo_weighted_sum.json"),
    ("generated_sft_multi_objective", "ppdbench_stability_sft_multi_objective.json"),
]


def _task_key(target_id: str, pep_basename: str) -> str:
    return f"{target_id}/{pep_basename}"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_method(
    *,
    bench_root: Path,
    gen_subdir: str,
    out_json_path: Path,
    workers: int,
    foldx_bin: str,
    workdir_root: Optional[str],
    skip_existing: bool,
    sync_every: int,
) -> None:
    tasks = collect_tasks(bench_root, gen_subdir)
    if not tasks:
        print(f"[WARN] no tasks for {gen_subdir}")
        _save_json(out_json_path, {})
        return

    results = _load_json(out_json_path)

    pending = []
    for target_id, _rec, pep, pep_name in tasks:
        key = _task_key(target_id, pep_name)
        if skip_existing and key in results:
            prev = results.get(key)
            if isinstance(prev, dict) and prev.get("score") is not None:
                continue
            if isinstance(prev, (float, int)):
                continue
        pending.append((target_id, pep, pep_name, key))

    print(f"[{gen_subdir}] tasks={len(tasks)} pending={len(pending)} workers={workers}")
    if not pending:
        return

    def _one(item: Tuple[str, Path, str, str]) -> Tuple[str, Optional[float]]:
        target_id, pep_path, pep_name, key = item
        score = foldx_stability_score_single(
            pep_path,
            foldx_bin=foldx_bin,
            workdir_root=workdir_root,
        )
        return key, score

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, it): it for it in pending}
        pbar = tqdm(total=len(pending), desc=gen_subdir[:26], unit="pep")
        counter = 0
        for fut in as_completed(futs):
            key, score = fut.result()
            target_id = key.split("/")[0]
            pep_name = futs[fut][2]
            pep_path = futs[fut][1]

            results[key] = {
                "target_id": target_id,
                "peptide_pdb": str(pep_path),
                "peptide_basename": pep_name,
                "score": score,
            }
            counter += 1
            pbar.update(1)
            if counter % sync_every == 0:
                _save_json(out_json_path, results)

        pbar.close()
        _save_json(out_json_path, results)

    print(f"[{gen_subdir}] saved: {out_json_path}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Compute PPDbench peptide stability with FoldX")
    ap.add_argument("--bench-root", type=str, default="/root/autodl-tmp/PPDbench")
    ap.add_argument(
        "--results-dir",
        type=str,
        default=str(here),
        help="JSON 输出目录（默认：本脚本所在目录）",
    )
    ap.add_argument("--workers", type=int, default=72, help="FoldX 并行线程数（每线程起一个 FoldX 进程）")
    ap.add_argument("--foldx-bin", type=str, default="/root/autodl-tmp/foldx_20270131", help="FoldX 可执行文件路径")
    ap.add_argument("--workdir-root", type=str, default="/root/autodl-tmp/tmp_foldx_eval", help="FoldX 临时目录根")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--sync-every", type=int, default=20, help="每跑多少条同步写一次 JSON")
    args = ap.parse_args()

    bench_root = Path(args.bench_root)
    results_dir = Path(args.results_dir)

    cpu_n = os.cpu_count() or 1
    workers = max(1, min(args.workers, cpu_n))
    print(f"[INFO] cpu_count={cpu_n}, workers={workers}")

    for gen_subdir, out_name in METHOD_SPECS:
        out_json_path = results_dir / out_name
        run_method(
            bench_root=bench_root,
            gen_subdir=gen_subdir,
            out_json_path=out_json_path,
            workers=workers,
            foldx_bin=args.foldx_bin,
            workdir_root=args.workdir_root,
            skip_existing=args.skip_existing,
            sync_every=args.sync_every,
        )


if __name__ == "__main__":
    main()

'''

python /root/autodl-tmp/Peptide_3D/results/3_Pareto_improved/compute_ppdbench_generated_stability.py --workers 72 --skip-existing > /root/autodl-tmp/Peptide_3D/results/3_Pareto_improved/compute_ppdbench_generated_stability.log 2>&1 &

'''
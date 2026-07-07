#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 PPDbench 各靶点 <target>/multi_cands/ 下「你的方法」生成的多肽计算溶解性（Protein-Sol），
与 compute_ppdbench_generated_solubility.py 使用相同计算逻辑。

每个靶点计算的条数由 --top-k 控制（默认 3）：
  - top-k <= 0：该靶点 multi_cands 下全部 pep_*.pdb（先按 cands_hdock_scores.json 升序，再补全其余文件序）
  - top-k > 0：先按 HDOCK 升序取，不足则用 pep_*.pdb 补足，至多 top-k 条

结果写入：results/3_Pareto_improved/ppdbench_solubility_multi_cands.json
不保存 log，仅写入 score 与路径字段。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from compute_ppdbench_generated_solubility import (  # noqa: E402
    extract_peptide_seq,
    solubility_score_from_seq_single,
)

OUT_JSON = "ppdbench_solubility_multi_cands.json"
SCORES_NAME = "cands_hdock_scores.json"


def _resolve_pep_path(key: str, mdir: Path) -> Optional[Path]:
    p = Path(key)
    if p.is_file():
        cand = p.resolve()
    else:
        cand = (mdir / Path(key).name).resolve()
        if not cand.is_file():
            return None
    try:
        cand.relative_to(mdir.resolve())
    except ValueError:
        return None
    return cand


def collect_multi_cands_topk(
    bench_root: Path,
    *,
    multi_subdir: str = "multi_cands",
    top_k: int = 3,
) -> List[Tuple[str, Path, Path, str]]:
    """
    返回 (target_id, receptor_pdb, peptide_pdb, peptide_basename)
    top_k <= 0 表示不限制条数（该靶点全部 pep_*.pdb）。
    """
    unlimited = top_k <= 0
    cap: Optional[int] = None if unlimited else top_k

    tasks: List[Tuple[str, Path, Path, str]] = []
    for d in sorted(p for p in bench_root.iterdir() if p.is_dir()):
        target_id = d.name
        rec = d / "receptor.pdb"
        mdir = d / multi_subdir
        if not rec.is_file() or not mdir.is_dir():
            continue

        chosen: List[Path] = []
        score_path = mdir / SCORES_NAME
        if score_path.is_file():
            try:
                raw = json.loads(score_path.read_text(encoding="utf-8"))
            except Exception:
                raw = {}
            ranked: List[Tuple[float, Path]] = []
            if isinstance(raw, dict):
                for k, v in raw.items():
                    pep = _resolve_pep_path(str(k), mdir)
                    if pep is None:
                        continue
                    try:
                        s = float(v)
                    except Exception:
                        continue
                    ranked.append((s, pep))
            ranked.sort(key=lambda x: x[0])
            seen = set()
            for _, pep in ranked:
                if pep.name in seen:
                    continue
                seen.add(pep.name)
                chosen.append(pep)
                if cap is not None and len(chosen) >= cap:
                    break

        if cap is None or len(chosen) < cap:
            all_pep = sorted(mdir.glob("pep_*.pdb"))
            existing = {p.name for p in chosen}
            for pep in all_pep:
                if pep.name in existing:
                    continue
                chosen.append(pep)
                existing.add(pep.name)
                if cap is not None and len(chosen) >= cap:
                    break

        slice_chosen = chosen if cap is None else chosen[:cap]
        for pep in slice_chosen:
            tasks.append((target_id, rec, pep, pep.name))

    return tasks


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


def main():
    ap = argparse.ArgumentParser(description="PPDbench multi_cands top-K solubility (Protein-Sol)")
    ap.add_argument("--bench-root", type=str, default="/root/autodl-tmp/PPDbench")
    ap.add_argument(
        "--results-dir",
        type=str,
        default="/root/autodl-tmp/Peptide_3D/results/3_Pareto_improved",
    )
    ap.add_argument("--multi-subdir", type=str, default="multi_cands")
    ap.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每靶点计算条数；<=0 表示该靶点全部 pep_*.pdb（用于按 stab+sol 重选 top3 绘图）",
    )
    ap.add_argument("--workers", type=int, default=72)
    ap.add_argument(
        "--proteinsol-wrapper",
        type=str,
        default="/root/autodl-tmp/protein-sol/multiple_prediction_wrapper_export.sh",
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--sync-every", type=int, default=20)
    args = ap.parse_args()

    bench_root = Path(args.bench_root)
    results_dir = Path(args.results_dir)
    out_json_path = results_dir / OUT_JSON

    tasks = collect_multi_cands_topk(
        bench_root, multi_subdir=args.multi_subdir, top_k=args.top_k
    )
    if not tasks:
        print("[WARN] no tasks")
        _save_json(out_json_path, {})
        return

    results = _load_json(out_json_path)
    pending = []
    for target_id, _rec, pep, pep_name in tasks:
        key = _task_key(target_id, pep_name)
        if args.skip_existing and key in results:
            prev = results.get(key)
            if isinstance(prev, dict) and prev.get("score") is not None:
                continue
        pending.append((target_id, pep, pep_name, key))

    cpu_n = os.cpu_count() or 1
    workers = max(1, min(args.workers, cpu_n))
    print(f"[INFO] tasks={len(tasks)} pending={len(pending)} workers={workers} -> {out_json_path}")

    proteinsol_lock = threading.Lock()

    def _one(item: Tuple[str, Path, str, str]) -> Tuple[str, Optional[float]]:
        _tid, pep_path, _name, key = item
        seq = extract_peptide_seq(pep_path)
        with proteinsol_lock:
            score = solubility_score_from_seq_single(
                seq, proteinsol_wrapper=args.proteinsol_wrapper
            )
        return key, score

    if not pending:
        return

    counter = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, it): it for it in pending}
        pbar = tqdm(total=len(pending), desc="multi_cands sol", unit="pep")
        for fut in as_completed(futs):
            key, score = fut.result()
            it = futs[fut]
            results[key] = {
                "target_id": it[0],
                "peptide_pdb": str(it[1]),
                "peptide_basename": it[2],
                "score": score,
            }
            counter += 1
            pbar.update(1)
            if counter % args.sync_every == 0:
                _save_json(out_json_path, results)
        pbar.close()
    _save_json(out_json_path, results)
    print(f"[FIN] saved {out_json_path}")


if __name__ == "__main__":
    main()


'''


python /root/autodl-tmp/Peptide_3D/results/3_Pareto_improved/compute_ppdbench_multi_cands_solubility.py --workers 72 --skip-existing


'''
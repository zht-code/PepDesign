#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 PPDbench 上四种生成方法的多肽溶解性（Protein-Sol scaled-sol）。
输出到 results/3_Pareto_improved 下的四个 JSON（分别对应四种方法）。

不保存 log（json 中仅保存 score 与 pdb 路径）。
进度条使用 tqdm，并尽量并行；但 Protein-Sol wrapper 不是并行安全的（会写固定文件名），
因此溶解性预测部分用全局锁串行化，避免输出互相覆盖。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

import numpy as np
from Bio import PDB
from Bio.SeqUtils import seq1 as _seq1


_AA3_CUSTOM = {
    "MSE": "M", "SEC": "U", "PYL": "O",
    "HID": "H", "HIE": "H", "HIP": "H",
    "CYX": "C", "ASX": "B", "GLX": "Z", "UNK": "X",
}


def resname_to_one(resname: str) -> str:
    try:
        return _seq1(resname.strip(), custom_map=_AA3_CUSTOM, undef_code="X")
    except Exception:
        return "X"


def extract_peptide_seq(pdb_path: Path) -> str:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pep", str(pdb_path))
    residues = [
        res for res in structure.get_residues()
        if PDB.is_aa(res, standard=False)
    ]
    seq = "".join(resname_to_one(res.get_resname()) for res in residues)
    return seq


def solubility_score_from_seq_single(
    seq: str,
    *,
    proteinsol_wrapper: str,
) -> Optional[float]:
    """
    单序列调用 Protein-Sol wrapper，返回 scaled-sol（需要 wrapper 正常产生 seq_prediction.txt）。
    注意：Protein-Sol wrapper 会在其目录写固定文件名，因此外部需做互斥锁串行化。
    """
    seq = (seq or "").strip().upper()
    if not seq:
        return None

    ps_bin_path = Path(proteinsol_wrapper).resolve()
    if not ps_bin_path.exists():
        raise FileNotFoundError(f"Protein-Sol wrapper not found: {ps_bin_path}")
    ps_dir = ps_bin_path.parent

    with tempfile.TemporaryDirectory(prefix="proteinsol_eval_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        fasta_path = tmpdir / "input.fasta"
        fasta_path.write_text(f">pep\n{seq}\n", encoding="utf-8")

        # stdout/stderr 不写入 json（不保存 log）
        cmd = [str(ps_bin_path), str(fasta_path)]
        proc = subprocess.run(
            cmd,
            cwd=str(ps_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
        _ = proc.stdout  # 丢弃输出（不保存 log）

        pred_path = ps_dir / "seq_prediction.txt"
        if not pred_path.exists():
            return None

        # 解析：兼容 build_dpo_pairs_stab_solu_cands_json.py 的逻辑（优先 scaled-sol）
        # 读取所有行，找到 HEADERS PREDICTIONS LINE 和 SEQUENCE PREDICTIONS 行
        lines = pred_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        header_cols: Optional[List[str]] = None
        for ln in lines:
            if ln.startswith("HEADERS PREDICTIONS LINE"):
                parts = [p.strip() for p in ln.split(",")]
                try:
                    id_idx = parts.index("ID")
                    header_cols = parts[id_idx:]
                except Exception:
                    header_cols = parts[1:] if len(parts) > 1 else None
                break

        if header_cols is None or len(header_cols) < 3:
            # 清理一下，避免堆积
            try:
                pred_path.unlink()
            except Exception:
                pass
            return None

        # SEQUENCE PREDICTIONS 行一般含：SEQUENCE PREDICTIONS,>pep,percent-sol,scaled-sol,...
        for ln in lines:
            if ln.startswith("SEQUENCE PREDICTIONS"):
                parts = [p.strip() for p in ln.split(",")]
                # 找到第一个 > 开头的位置作为起点
                start_idx = None
                for i, p in enumerate(parts):
                    if p.startswith(">"):
                        start_idx = i
                        break
                if start_idx is None:
                    continue
                values = parts[start_idx : start_idx + len(header_cols)]
                if len(values) < len(header_cols):
                    continue
                colmap = {h: v for h, v in zip(header_cols, values)}
                if "scaled-sol" in colmap:
                    try:
                        return float(colmap["scaled-sol"])
                    except Exception:
                        pass
                if "percent-sol" in colmap:
                    try:
                        return float(colmap["percent-sol"])
                    except Exception:
                        pass

        # 清理
        try:
            pred_path.unlink()
        except Exception:
            pass

    return None


def collect_tasks(bench_root: Path, gen_subdir: str) -> List[Tuple[str, Path, Path, str]]:
    """
    返回: (target_id, receptor_pdb, peptide_pdb, peptide_basename)
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
    ("generated_dpo_affinity_only", "ppdbench_solubility_dpo_affinity_only.json"),
    ("generated_dpo_stability_only", "ppdbench_solubility_dpo_stability_only.json"),
    ("generated_dpo_weighted_sum", "ppdbench_solubility_dpo_weighted_sum.json"),
    ("generated_sft_multi_objective", "ppdbench_solubility_sft_multi_objective.json"),
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
    proteinsol_wrapper: str,
    skip_existing: bool,
    sync_every: int,
    proteinsol_lock: threading.Lock,
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

    out_dir = out_json_path.parent
    _ = out_dir

    # 真正的 Protein-Sol 需要互斥锁：避免写固定文件名 seq_prediction.txt 被并发覆盖
    def _one(item: Tuple[str, Path, str, str]) -> Tuple[str, Optional[float]]:
        target_id, pep_path, pep_name, key = item
        seq = extract_peptide_seq(pep_path)
        with proteinsol_lock:
            score = solubility_score_from_seq_single(seq, proteinsol_wrapper=proteinsol_wrapper)
        return key, score

    # 将 pep 序列提取也并行化，但 proteinsol_wrapper 调用串行化
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, it): it for it in pending}
        pbar = tqdm(total=len(pending), desc=gen_subdir[:26], unit="pep")

        counter = 0
        for fut in as_completed(futs):
            key, score = fut.result()
            results[key] = {
                "target_id": key.split("/")[0],
                "peptide_pdb": str(futs[fut][1]),
                "peptide_basename": futs[fut][2],
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
    ap = argparse.ArgumentParser(description="Compute PPDbench peptide solubility with Protein-Sol")
    ap.add_argument("--bench-root", type=str, default="/root/autodl-tmp/PPDbench")
    ap.add_argument(
        "--results-dir",
        type=str,
        default=str(here),
        help="JSON 输出目录（默认：本脚本所在目录）",
    )
    ap.add_argument("--workers", type=int, default=72, help="线程数（序列提取并行；Protein-Sol 串行）")
    ap.add_argument(
        "--proteinsol-wrapper",
        type=str,
        default="/root/autodl-tmp/protein-sol/multiple_prediction_wrapper_export.sh",
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--sync-every", type=int, default=20, help="每跑多少条同步写一次 JSON")
    args = ap.parse_args()

    bench_root = Path(args.bench_root)
    results_dir = Path(args.results_dir)
    proteinsol_lock = threading.Lock()

    cpu_n = os.cpu_count() or 1
    workers = max(1, min(args.workers, cpu_n))
    print(f"[INFO] cpu_count={cpu_n}, workers={workers}")

    for gen_subdir, out_name in METHOD_SPECS:
        out_json = results_dir / out_name
        run_method(
            bench_root=bench_root,
            gen_subdir=gen_subdir,
            out_json_path=out_json,
            workers=workers,
            proteinsol_wrapper=args.proteinsol_wrapper,
            skip_existing=args.skip_existing,
            sync_every=args.sync_every,
            proteinsol_lock=proteinsol_lock,
        )


if __name__ == "__main__":
    main()

'''

python /root/autodl-tmp/Peptide_3D/results/3_Pareto_improved/compute_ppdbench_generated_solubility.py --workers 72 --skip-existing > /root/autodl-tmp/Peptide_3D/results/3_Pareto_improved/compute_ppdbench_generated_solubility.log 2>&1 &

'''
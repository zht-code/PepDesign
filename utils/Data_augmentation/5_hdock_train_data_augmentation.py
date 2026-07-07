#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, shutil, argparse, subprocess, glob
from typing import Optional, Tuple, Dict, List

import numpy as np
from tqdm import tqdm
from Bio import PDB
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======= 默认路径配置（根据你现在的需求改过） =======
# 增强后的训练集：每个子目录里有 receptor.pdb 和 peptide.pdb
DEFAULT_DATA_ROOT = "/root/autodl-tmp/train_data/train_data_augmentation"

# 汇总所有 (receptor, peptide) HDOCK 结果的 JSON
DEFAULT_OUT_JSON  = "/root/autodl-tmp/train_data/train_data_augmentation_hdock_scores.json"

# HDOCK 工作目录（缓存、中间文件等）
DEFAULT_WORK_ROOT = "/root/autodl-tmp/hdock_tmp"

# HDOCK 可执行文件路径
DEFAULT_HDOCK_BIN    = "/root/autodl-tmp/HDOCKlite/hdock"
DEFAULT_CREATEPL_BIN = "/root/autodl-tmp/HDOCKlite/createpl"

# 解析 score 时用的正则
SCORE_RE_LIST = [
    re.compile(r'(?i)\bscore\b\s*:?\s*([+-]?\d+(?:\.\d+)?)'),
    re.compile(r'(?i)\btotal\s*score\b\s*:?\s*([+-]?\d+(?:\.\d+)?)')
]


# ================== 一些解析 score 的工具函数 ==================

def _parse_best_score_in_textfile(path: str):
    """从 hdock.out 这类文本文件里解析最小的 score（越小越好）"""
    if not path or not os.path.exists(path):
        return None
    best = None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            for rgx in SCORE_RE_LIST:
                m = rgx.search(line)
                if m:
                    v = float(m.group(1))
                    best = v if best is None else (v if v < best else best)
    return best


def _find_any_out_file(workdir: str):
    """在 workdir 里找一个 hdock 输出文件"""
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
        # 选最大的那个
        return max(outs, key=lambda x: os.path.getsize(x))
    return None


def _parse_score_from_pdb(pdb_path: str):
    """从 createpl 生成的模型 pdb 的 REMARK 行中解析 score"""
    if not pdb_path or not os.path.exists(pdb_path):
        return None
    best = None
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("REMARK"):
                for rgx in SCORE_RE_LIST:
                    m = rgx.search(line)
                    if m:
                        v = float(m.group(1))
                        best = v if best is None else (v if v < best else best)
    return best


# ================== 计算几何中心相关 ==================

def is_heavy_atom(atom) -> bool:
    elem = getattr(atom, "element", "").strip().upper()
    name = atom.get_name().strip().upper()
    if elem:
        return elem != "H"
    return not name.startswith("H")


def geom_center(coords: np.ndarray) -> Tuple[float, float, float]:
    c = coords.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])


def coords_of_chain(chain) -> np.ndarray:
    pts = []
    for res in chain:
        for atom in res:
            if is_heavy_atom(atom):
                pts.append(atom.coord)
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def center_of_smallest_chain(pdb_path: str) -> Optional[Tuple[float, float, float]]:
    """在复合物中选“残基数最少”的链作为多肽，取其重原子的几何中心"""
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_path)
    except Exception:
        return None
    model = next(structure.get_models())
    chains = list(model.get_chains())
    if not chains:
        return None

    def key_fn(ch):
        residues = [r for r in ch.get_residues()]
        heavy = sum(1 for r in ch for a in r if is_heavy_atom(a))
        return (len(residues), heavy)

    chains_sorted = sorted(chains, key=key_fn)
    for ch in chains_sorted:
        pts = coords_of_chain(ch)
        if pts.shape[0] > 0:
            return geom_center(pts)
    return None


def center_of_peptide_input(peptide_pdb: str) -> Optional[Tuple[float, float, float]]:
    """直接用输入的 peptide.pdb 计算几何中心（作为回退）"""
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("peptide", peptide_pdb)
    except Exception:
        return None
    pts = []
    for atom in structure.get_atoms():
        if is_heavy_atom(atom):
            pts.append(atom.coord)
    if not pts:
        return None
    return geom_center(np.asarray(pts, dtype=np.float64))


# ================== HDOCK 运行 & 结果解析 ==================

def run_hdock(workdir: str,
              receptor_pdb: str,
              peptide_pdb: str,
              hdock_bin: str,
              createpl_bin: str,
              timeout_s: int = 900) -> Tuple[Optional[float], Optional[str], str]:
    """
    真正调用 hdock + createpl 的函数。
    返回: (best_score, best_model_pdb, logs_str)
    """
    os.makedirs(workdir, exist_ok=True)

    r_fn = os.path.join(workdir, "receptor.pdb")
    l_fn = os.path.join(workdir, "peptide.pdb")
    shutil.copy2(receptor_pdb, r_fn)
    shutil.copy2(peptide_pdb, l_fn)

    logs: List[str] = []

    # 统一把 OpenMP / BLAS 线程数限制为 1，方便多进程把 136 核吃满
    run_env = os.environ.copy()
    run_env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })

    # --- 1) 运行 hdock ---
    cmd = [hdock_bin, "receptor.pdb", "peptide.pdb"]
    logs.append(f"[INFO] run: {' '.join(cmd)} (cwd={workdir})")
    try:
        proc = subprocess.run(
            cmd, cwd=workdir,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout_s, text=True, env=run_env
        )
        logs.append(proc.stdout or "")
        if proc.returncode != 0:
            logs.append(f"[WARN] hdock exit code {proc.returncode}: {proc.stderr}")
    except subprocess.TimeoutExpired:
        logs.append("[WARN] hdock timeout")
    except Exception as e:
        logs.append(f"[WARN] hdock failed: {e}")

    # 先从 *.out 解析一次 best_score
    best_score = None
    hdock_out = _find_any_out_file(workdir)
    if hdock_out:
        val = _parse_best_score_in_textfile(hdock_out)
        if val is not None:
            best_score = val
    else:
        logs.append("[WARN] no *.out file found after hdock")

    # --- 2) 运行 createpl，只要 top3 模型 ---
    docked_model = None
    best_model_pdb = None
    best_model_score = None

    if hdock_out and os.path.exists(hdock_out):
        # 只要 top3
        cmd2 = [
            createpl_bin,
            os.path.basename(hdock_out),
            "top3.pdb",
            "-nmax", "3",
            "-complex",
            "-models",
        ]
        logs.append(f"[INFO] run: {' '.join(cmd2)} (cwd={workdir})")
        try:
            proc2 = subprocess.run(
                cmd2, cwd=workdir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout_s, text=True, env=run_env
            )
            logs.append(proc2.stdout or "")
            if proc2.returncode != 0:
                logs.append(f"[WARN] createpl exit code {proc2.returncode}: {proc2.stderr}")
        except subprocess.TimeoutExpired:
            logs.append("[WARN] createpl timeout")
        except Exception as e:
            logs.append(f"[WARN] createpl failed: {e}")

        # 找到 top3 里最好的那一个模型
        pdb_candidates: List[str] = []
        for pat in ("model_1.pdb", "model_2.pdb", "model_3.pdb", "top3.pdb"):
            p = os.path.join(workdir, pat)
            if os.path.exists(p):
                pdb_candidates.append(p)
        if not pdb_candidates:
            for p in glob.glob(os.path.join(workdir, "*.pdb")):
                base = os.path.basename(p).lower()
                if base not in ("receptor.pdb", "peptide.pdb"):
                    pdb_candidates.append(p)

        for p in pdb_candidates:
            val = _parse_score_from_pdb(p)
            if val is not None:
                if best_model_score is None or val < best_model_score:
                    best_model_score = val
                    best_model_pdb = p

        if best_model_score is not None:
            if best_score is None or best_model_score < best_score:
                best_score = best_model_score
            docked_model = best_model_pdb

        if docked_model is None and pdb_candidates:
            # 若没解析出 score，就选文件最大那个
            docked_model = max(pdb_candidates, key=lambda x: os.path.getsize(x))

    return best_score, docked_model, "\n".join(logs)


# ================== 恢复：根据 workdir 缓存跳过已算过的对 ==================

def try_recover_from_workdir(rid: str,
                             receptor_pdb: str,
                             peptide_pdb: str,
                             work_root: str) -> Optional[Dict]:
    """
    根据 work_root/rid 下已有的 *.out / 模型 等缓存，
    尝试恢复一个 entry（如果成功，就可以跳过重新跑 hdock）。
    """
    workdir = os.path.join(work_root, rid)
    if not os.path.isdir(workdir):
        return None

    hdock_out = _find_any_out_file(workdir)
    if not hdock_out:
        return None

    best_score = _parse_best_score_in_textfile(hdock_out)
    # 找一下之前可能生成的模型
    pdb_candidates: List[str] = []
    for pat in ("model_1.pdb", "model_2.pdb", "model_3.pdb", "top3.pdb"):
        p = os.path.join(workdir, pat)
        if os.path.exists(p):
            pdb_candidates.append(p)
    if not pdb_candidates:
        for p in glob.glob(os.path.join(workdir, "*.pdb")):
            base = os.path.basename(p).lower()
            if base not in ("receptor.pdb", "peptide.pdb"):
                pdb_candidates.append(p)

    docked_model = None
    if pdb_candidates:
        # 优先解析 score
        best_model_score = None
        best_model_pdb = None
        for p in pdb_candidates:
            val = _parse_score_from_pdb(p)
            if val is not None:
                if best_model_score is None or val < best_model_score:
                    best_model_score = val
                    best_model_pdb = p
        docked_model = best_model_pdb or max(pdb_candidates, key=lambda x: os.path.getsize(x))

    # 计算中心
    fallback_center = center_of_peptide_input(peptide_pdb) or (0.0, 0.0, 0.0)
    if docked_model and os.path.exists(docked_model):
        center = center_of_smallest_chain(docked_model) or fallback_center
    else:
        center = fallback_center

    entry = {
        "center": {
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "center_z": float(center[2]),
        },
        "best_model_file": os.path.splitext(os.path.basename(peptide_pdb))[0],
    }
    if best_score is not None:
        entry["score"] = float(best_score)

    return entry


# ================== 线程任务封装 ==================

def task_main_peptide(rid: str,
                      receptor_pdb: str,
                      peptide_pdb: str,
                      work_root: str,
                      hdock_bin: str,
                      createpl_bin: str,
                      timeout_s: int):
    """
    单个 (receptor, peptide) 的任务，在线程池里调用。
    """
    workdir = os.path.join(work_root, rid)
    fallback_center = center_of_peptide_input(peptide_pdb) or (0.0, 0.0, 0.0)

    score, docked_model, logs = run_hdock(
        workdir, receptor_pdb, peptide_pdb,
        hdock_bin, createpl_bin, timeout_s=timeout_s
    )

    if docked_model and os.path.exists(docked_model):
        center = center_of_smallest_chain(docked_model) or fallback_center
    else:
        center = fallback_center

    entry = {
        "center": {
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "center_z": float(center[2]),
        },
        "best_model_file": os.path.splitext(os.path.basename(peptide_pdb))[0],
    }
    if score is not None:
        entry["score"] = float(score)

    # 写日志
    try:
        os.makedirs(workdir, exist_ok=True)
        with open(os.path.join(workdir, "run.log"), "w") as lf:
            lf.write(logs)
    except Exception:
        pass

    return rid, entry


# ================== 主流程 ==================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                    help="train_data_augmentation 根目录")
    ap.add_argument("--out_json", default=DEFAULT_OUT_JSON,
                    help="总体汇总 JSON 输出路径")
    ap.add_argument("--work_root", default=DEFAULT_WORK_ROOT,
                    help="HDOCK 工作目录根")
    ap.add_argument("--hdock_bin", default=DEFAULT_HDOCK_BIN,
                    help="hdock 可执行文件路径")
    ap.add_argument("--createpl_bin", default=DEFAULT_CREATEPL_BIN,
                    help="createpl 可执行文件路径")
    ap.add_argument("--timeout", type=int, default=900,
                    help="每个样本超时（秒），默认 900=15 分钟")
    ap.add_argument("--skip_existing", action="store_true",
                    help="若已有结果或工作目录缓存，则跳过该样本，实现断点续跑")
    ap.add_argument("--num_workers", type=int, default=119,
                    help="并发线程数（建议接近 CPU 核数，如 120–136）")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.work_root, exist_ok=True)

    # 读入已有的总 JSON（方便断点续跑）
    overall: Dict[str, Dict] = {}
    if os.path.exists(args.out_json):
        try:
            with open(args.out_json, "r") as fh:
                overall = json.load(fh)
        except Exception:
            overall = {}

    # 扫描所有子目录（每个目录代表一对 receptor-peptide）
    ids = [d for d in os.listdir(args.data_root)
           if os.path.isdir(os.path.join(args.data_root, d))]
    ids.sort()

    tasks: List[Tuple[str, str, str]] = []

    for rid in ids:
        r_dir = os.path.join(args.data_root, rid)
        receptor_pdb = os.path.join(r_dir, "receptor.pdb")
        peptide_pdb = os.path.join(r_dir, "peptide.pdb")

        if not os.path.exists(receptor_pdb) or not os.path.exists(peptide_pdb):
            continue

        # ====== 断点续跑逻辑 ======
        if args.skip_existing:
            # 1) 如果 JSON 里已经有结果，直接跳过
            if rid in overall and "score" in overall[rid]:
                continue

            # 2) 尝试从 work_root 的缓存恢复
            entry = try_recover_from_workdir(rid, receptor_pdb, peptide_pdb, args.work_root)
            if entry is not None and "score" in entry:
                overall[rid] = entry
                continue

        # 需要真正跑 hdock 的样本
        tasks.append((rid, receptor_pdb, peptide_pdb))

    print(f"Total pairs: {len(ids)}; to run HDOCK: {len(tasks)}")

    num_workers = max(1, args.num_workers)
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for rid, r_pdb, p_pdb in tasks:
            futures.append(ex.submit(
                task_main_peptide,
                rid, r_pdb, p_pdb,
                args.work_root, args.hdock_bin,
                args.createpl_bin, args.timeout
            ))

        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc=f"HDOCK (threads x{num_workers})"):
            rid, entry = fut.result()
            if entry:
                overall[rid] = entry

            # 周期性写盘，防止中途挂掉损失太多
            if len(overall) % 50 == 0:
                try:
                    with open(args.out_json, "w") as fh:
                        json.dump(overall, fh, indent=2)
                except Exception:
                    pass

    # 最终落盘
    with open(args.out_json, "w") as fh:
        json.dump(overall, fh, indent=2)

    print(f"\n[ALL DONE] HDOCK 结果已保存到: {args.out_json}")


if __name__ == "__main__":
    main()

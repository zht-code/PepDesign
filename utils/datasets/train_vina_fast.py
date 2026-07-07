#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, shutil, subprocess, glob, re, math
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from Bio import PDB

# ================== 默认路径（可用命令行覆盖） ==================
DEFAULT_DATA_ROOT = "/root/autodl-tmp/train_data"  # 你的数据根
DEFAULT_OUT_JSON  = "/root/autodl-tmp/Peptide_3D/data/hdock_scores.json"
DEFAULT_WORK_ROOT = "/root/autodl-tmp/hdock"

DEFAULT_HDOCK_BIN    = "/root/autodl-fs/HDOCKlite/hdock"
DEFAULT_CREATEPL_BIN = "/root/autodl-fs/HDOCKlite/createpl"

# ================== 解析工具 ==================
SCORE_RE_LIST = [
    re.compile(r'(?i)\bscore\b\s*:\s*([+-]?\d+(?:\.\d+)?)'),
    re.compile(r'(?i)\btotal\s*score\b\s*:\s*([+-]?\d+(?:\.\d+)?)')
]

def _parse_best_score_in_textfile(path: str) -> Optional[float]:
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

def _find_any_out_file(workdir: str) -> Optional[str]:
    for name in ("hdock.out", "Hdock.out", "HDOCK.out"):
        p = os.path.join(workdir, name)
        if os.path.exists(p):
            return p
    outs = glob.glob(os.path.join(workdir, "*.out"))
    if outs:
        return max(outs, key=lambda x: os.path.getsize(x))
    return None

def _parse_score_from_pdb(pdb_path: str) -> Optional[float]:
    if not pdb_path or not os.path.exists(pdb_path):
        return None
    best = None
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("REMARK"):
                continue
            for rgx in SCORE_RE_LIST:
                m = rgx.search(line)
                if m:
                    v = float(m.group(1))
                    best = v if best is None else (v if v < best else best)
    return best

# ================== 计算几何中心（保持你的逻辑，不做零依赖改写） ==================
def is_heavy_atom(atom) -> bool:
    elem = getattr(atom, "element", "").strip().upper()
    name = atom.get_name().strip().upper()
    if elem:
        return elem != "H"
    return not name.startswith("H")

def geom_center(coords: np.ndarray) -> Tuple[float,float,float]:
    c = coords.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])

def coords_of_chain(chain) -> np.ndarray:
    pts = []
    for res in chain:
        for atom in res:
            if is_heavy_atom(atom):
                pts.append(atom.coord)
    if not pts:
        return np.zeros((0,3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)

def center_of_smallest_chain(pdb_path: str) -> Optional[Tuple[float,float,float]]:
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

def center_of_peptide_input(peptide_pdb: str) -> Optional[Tuple[float,float,float]]:
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

# ================== 文件链接/复制 ==================
def safe_link_or_copy(src: str, dst: str):
    try:
        # 同一文件系统可硬链接，极快且省空间
        os.link(src, dst)
    except Exception:
        try:
            # 否则软链接
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
        except Exception:
            # 实在不行退回复制
            shutil.copy2(src, dst)

# ================== 单个样本运行 ==================
def run_hdock_for_id(rid: str, r_dir: str, work_root: str,
                     hdock_bin: str, createpl_bin: str,
                     timeout_s: int, env_limited: Dict[str,str]) -> Tuple[str, Dict, str]:
    """
    返回: (rid, entry_json, short_log)
    """
    receptor_pdb = os.path.join(r_dir, "receptor.pdb")
    peptide_pdb  = os.path.join(r_dir, "peptide.pdb")
    workdir = os.path.join(work_root, rid)
    os.makedirs(workdir, exist_ok=True)

    # 预备 fallback center
    fallback_center = center_of_peptide_input(peptide_pdb) or (0.0, 0.0, 0.0)

    # 链接/复制输入
    r_fn = os.path.join(workdir, "receptor.pdb")
    l_fn = os.path.join(workdir, "peptide.pdb")
    if not os.path.exists(r_fn): safe_link_or_copy(receptor_pdb, r_fn)
    if not os.path.exists(l_fn): safe_link_or_copy(peptide_pdb,  l_fn)

    logs = []
    # 1) hdock receptor ligand
    cmd = [hdock_bin, "receptor.pdb", "peptide.pdb"]
    logs.append(f"[RUN] {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout_s, text=True, env=env_limited)
        if proc.stdout: logs.append(proc.stdout.strip())
        if proc.returncode != 0:
            logs.append(f"[WARN] hdock exit {proc.returncode}: {proc.stderr.strip()}")
    except subprocess.TimeoutExpired:
        logs.append("[WARN] hdock timeout")
    except Exception as e:
        logs.append(f"[WARN] hdock failed: {e}")

    # 解析 out
    best_score = None
    hdock_out = _find_any_out_file(workdir)
    if hdock_out:
        val = _parse_best_score_in_textfile(hdock_out)
        if val is not None:
            best_score = val
    else:
        logs.append("[WARN] no *.out file found")

    # 2) 只生成 1 个模型，加速
    docked_model = None
    if hdock_out and os.path.exists(hdock_out):
        cmd2 = [createpl_bin, os.path.basename(hdock_out), "top1.pdb", "-nmax", "1", "-complex", "-models"]
        logs.append(f"[RUN] {' '.join(cmd2)}")
        try:
            proc2 = subprocess.run(cmd2, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=timeout_s, text=True, env=env_limited)
            if proc2.stdout: logs.append(proc2.stdout.strip())
            if proc2.returncode != 0:
                logs.append(f"[WARN] createpl exit {proc2.returncode}: {proc2.stderr.strip()}")
        except subprocess.TimeoutExpired:
            logs.append("[WARN] createpl timeout")
        except Exception as e:
            logs.append(f"[WARN] createpl failed: {e}")

        # 选模型
        candidates = []
        for pat in ("model_1.pdb", "top1.pdb", "complex_1.pdb"):
            p = os.path.join(workdir, pat)
            if os.path.exists(p):
                candidates.append(p)
        if not candidates:
            for p in glob.glob(os.path.join(workdir, "*.pdb")):
                base = os.path.basename(p).lower()
                if base not in ("receptor.pdb", "peptide.pdb"):
                    candidates.append(p)
        if candidates:
            prefer = os.path.join(workdir, "model_1.pdb")
            docked_model = prefer if os.path.exists(prefer) else max(candidates, key=lambda x: os.path.getsize(x))

    # 回退：从 PDB REMARK 取分
    if best_score is None:
        try:
            pdb_candidates = []
            pref = os.path.join(workdir, "model_1.pdb")
            if os.path.exists(pref): pdb_candidates.append(pref)
            for p in glob.glob(os.path.join(workdir, "*.pdb")):
                base = os.path.basename(p).lower()
                if base not in ("receptor.pdb", "peptide.pdb") and p not in pdb_candidates:
                    pdb_candidates.append(p)
            for p in pdb_candidates:
                val = _parse_score_from_pdb(p)
                if val is not None:
                    best_score = val if best_score is None else (val if val < best_score else best_score)
            if best_score is None:
                logs.append("[WARN] no score found in PDB REMARKs")
        except Exception as e:
            logs.append(f"[WARN] parse REMARK score failed: {e}")

    # 计算中心
    if docked_model and os.path.exists(docked_model):
        center = center_of_smallest_chain(docked_model) or fallback_center
    else:
        center = fallback_center

    entry = {
        "center": {"center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2])}
    }
    if best_score is not None:
        entry["score"] = float(best_score)

    # 写简要日志
    try:
        with open(os.path.join(workdir, "run.log"), "w") as lf:
            lf.write("\n".join(logs))
    except Exception:
        pass

    return rid, entry, "\n".join(logs)

# ================== 主流程（并行） ==================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out_json",  default=DEFAULT_OUT_JSON)
    ap.add_argument("--work_root", default=DEFAULT_WORK_ROOT)
    ap.add_argument("--hdock_bin", default=DEFAULT_HDOCK_BIN)
    ap.add_argument("--createpl_bin", default=DEFAULT_CREATEPL_BIN)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--workers", type=int, default=max(8, (os.cpu_count() or 16)//2), help="并发任务数")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--flush_every", type=int, default=50, help="每处理多少条写盘一次")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.work_root, exist_ok=True)

    # 限制子进程所用线程，防止过度超订阅
    env_limited = os.environ.copy()
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        env_limited[k] = "1"

    # 载入已有结果
    results: Dict[str, Dict] = {}
    if os.path.exists(args.out_json):
        try:
            with open(args.out_json, "r") as fh:
                results = json.load(fh)
        except Exception:
            results = {}

    # 收集样本
    ids = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    ids.sort()
    id_dirs = {rid: os.path.join(args.data_root, rid) for rid in ids}

    # 可选跳过已存在
    todo: List[str] = []
    for rid in ids:
        if args.skip_existing and rid in results:
            continue
        r_dir = id_dirs[rid]
        if os.path.exists(os.path.join(r_dir, "receptor.pdb")) and os.path.exists(os.path.join(r_dir, "peptide.pdb")):
            todo.append(rid)
    if not todo:
        print("No pending items. Done.")
        return

    # 并发执行
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rid in todo:
            futures.append(ex.submit(
                run_hdock_for_id, rid, id_dirs[rid], args.work_root,
                args.hdock_bin, args.createpl_bin, args.timeout, env_limited
            ))

        pbar = tqdm(total=len(futures), desc=f"HDOCK x{args.workers}")
        done_cnt = 0
        for fut in as_completed(futures):
            rid, entry, _ = fut.result()
            results[rid] = entry
            done_cnt += 1
            pbar.update(1)
            # 分批写盘
            if done_cnt % args.flush_every == 0:
                with open(args.out_json, "w") as fh:
                    json.dump(results, fh, indent=2)
        pbar.close()

    # 最终写盘
    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved JSON to {args.out_json}")

if __name__ == "__main__":
    main()

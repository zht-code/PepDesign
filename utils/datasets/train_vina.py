#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, shutil, argparse, subprocess, glob
from typing import Optional, Tuple, Dict
import numpy as np
from tqdm import tqdm
from Bio import PDB
import re, glob  # 如果文件顶部没引入 re/glob，补上

# ======= 配置默认值（可用命令行覆盖） =======
DEFAULT_DATA_ROOT = "/root/autodl-tmp/train_data"
DEFAULT_OUT_JSON  = "/root/autodl-tmp/Peptide_3D/data/hdock_scores.json"
DEFAULT_WORK_ROOT = "/root/autodl-tmp/hdock"

DEFAULT_HDOCK_BIN   = "/root/autodl-fs/HDOCKlite/hdock"
DEFAULT_CREATEPL_BIN= "/root/autodl-fs/HDOCKlite/createpl"

SCORE_RE_LIST = [
    re.compile(r'(?i)\bscore\b\s*:?\s*([+-]?\d+(?:\.\d+)?)'),
    re.compile(r'(?i)\btotal\s*score\b\s*:?\s*([+-]?\d+(?:\.\d+)?)')
]

def _parse_best_score_in_textfile(path: str):
    """从任意文本文件中提取最优（最负）score。"""
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
    """在工作目录里尽量找到 HDOCK 输出文件。"""
    candidates = [
        os.path.join(workdir, "hdock.out"),
        os.path.join(workdir, "Hdock.out"),
        os.path.join(workdir, "HDOCK.out"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # 兜底：找任意 .out
    outs = glob.glob(os.path.join(workdir, "*.out"))
    if outs:
        # 体积最大的往往更全
        return max(outs, key=lambda x: os.path.getsize(x))
    return None

def _parse_score_from_pdb(pdb_path: str):
    """从 model_*.pdb 的 REMARK 中抓 Score（如：REMARK Score:  -379.91）。"""
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

# ======= 基础工具 =======
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
    """取残基数最少的链作为配体（多肽），计算其重原子几何中心。"""
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_path)
    except Exception:
        return None
    model = next(structure.get_models())
    chains = list(model.get_chains())
    if not chains:
        return None
    # 以“残基数”作为小分子/多肽的判据；如有并列，选重原子更少者
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
    """直接从输入的 peptide.pdb 计算几何中心（不区分链）。"""
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

# ======= 运行 HDOCK + createpl =======
def run_hdock(workdir: str, receptor_pdb: str, peptide_pdb: str,
              hdock_bin: str, createpl_bin: str, timeout_s: int = 900) -> Tuple[Optional[float], Optional[str], str]:
    """
    返回: (best_score, docked_model_path, log_text)
    - best_score: 解析 hdock.out 得到的最优分数（通常越负越好）
    - docked_model_path: 模型文件（优先 model_1.pdb）
    - log_text: 便于调试的合并日志
    """
    os.makedirs(workdir, exist_ok=True)
    r_fn = os.path.join(workdir, "receptor.pdb")
    l_fn = os.path.join(workdir, "peptide.pdb")
    shutil.copy2(receptor_pdb, r_fn)
    shutil.copy2(peptide_pdb, l_fn)

    logs = []
    # 1) hdock receptor ligand （注意顺序！）
    cmd = [hdock_bin, "receptor.pdb", "peptide.pdb"]
    logs.append(f"[INFO] run: {' '.join(cmd)} (cwd={workdir})")
    try:
        proc = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout_s, text=True)
        logs.append(proc.stdout or "")
        if proc.returncode != 0:
            logs.append(f"[WARN] hdock exit code {proc.returncode}: {proc.stderr}")
    except subprocess.TimeoutExpired:
        logs.append("[WARN] hdock timeout")
    except Exception as e:
        logs.append(f"[WARN] hdock failed: {e}")

    # 先尽力找到 out 文件，再解析
    best_score = None
    hdock_out = _find_any_out_file(workdir)
    if hdock_out:
        val = _parse_best_score_in_textfile(hdock_out)
        if val is not None:
            best_score = val
    else:
        logs.append("[WARN] no *.out file found after hdock")


    # 2) createpl 生成模型（至少 1 个）
    docked_model = None
    if os.path.exists(hdock_out):
        cmd2 = [createpl_bin, os.path.basename(hdock_out), "top10.pdb", "-nmax", "10", "-complex", "-models"]
        logs.append(f"[INFO] run: {' '.join(cmd2)} (cwd={workdir})")
        try:
            proc2 = subprocess.run(cmd2, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=timeout_s, text=True)
            logs.append(proc2.stdout or "")
            if proc2.returncode != 0:
                logs.append(f"[WARN] createpl exit code {proc2.returncode}: {proc2.stderr}")
        except subprocess.TimeoutExpired:
            logs.append("[WARN] createpl timeout")
        except Exception as e:
            logs.append(f"[WARN] createpl failed: {e}")

        # 优先 model_1.pdb，否则找最大的 model/top/complex 文件
        candidates = []
        for pat in ("model_1.pdb", "top1.pdb", "complex_1.pdb"):
            p = os.path.join(workdir, pat)
            if os.path.exists(p):
                candidates.append(p)
        if not candidates:
            for p in glob.glob(os.path.join(workdir, "*.pdb")):
                if os.path.basename(p).lower() not in ("receptor.pdb", "peptide.pdb"):
                    candidates.append(p)
        if candidates:
            prefer = os.path.join(workdir, "model_1.pdb")
            docked_model = prefer if os.path.exists(prefer) else max(candidates, key=lambda x: os.path.getsize(x))

    # 如果依然没有从 out 里得到分数，则尝试从对接模型 PDB 的 REMARK 里取
    if best_score is None:
        try:
            # 优先 model_1.pdb；若没有，就遍历工作目录下的 PDB（排除 receptor/peptide）
            pdb_candidates = []
            pref = os.path.join(workdir, "model_1.pdb")
            if os.path.exists(pref):
                pdb_candidates.append(pref)
            for p in glob.glob(os.path.join(workdir, "*.pdb")):
                base = os.path.basename(p).lower()
                if base not in ("receptor.pdb", "peptide.pdb") and p not in pdb_candidates:
                    pdb_candidates.append(p)
            for p in pdb_candidates:
                val = _parse_score_from_pdb(p)
                if val is not None:
                    best_score = val if best_score is None else (val if val < best_score else best_score)
            if best_score is None:
                logs.append("[WARN] no score found in PDB REMARKs either")
        except Exception as e:
            logs.append(f"[WARN] parse score from PDB failed: {e}")

    return (best_score, docked_model, "\n".join(logs))

# ======= 主流程 =======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="train_data 根目录")
    ap.add_argument("--out_json",  default=DEFAULT_OUT_JSON,  help="输出 JSON 文件路径")
    ap.add_argument("--work_root", default=DEFAULT_WORK_ROOT, help="HDOCK 工作目录根")
    ap.add_argument("--hdock_bin", default=DEFAULT_HDOCK_BIN, help="hdock 可执行文件路径")
    ap.add_argument("--createpl_bin", default=DEFAULT_CREATEPL_BIN, help="createpl 可执行文件路径")
    ap.add_argument("--timeout", type=int, default=900, help="每个样本超时（秒）")
    ap.add_argument("--skip_existing", action="store_true", help="若 out_json 已含该ID则跳过")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.work_root, exist_ok=True)

    # 载入已有结果（便于断点续跑）
    results: Dict[str, Dict] = {}
    if os.path.exists(args.out_json):
        try:
            with open(args.out_json, "r") as fh:
                results = json.load(fh)
        except Exception:
            results = {}

    ids = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    ids.sort()

    for rid in tqdm(ids, desc="HDOCK scoring"):
        if args.skip_existing and rid in results:
            continue

        r_dir = os.path.join(args.data_root, rid)
        receptor_pdb = os.path.join(r_dir, "receptor.pdb")
        peptide_pdb  = os.path.join(r_dir, "peptide.pdb")
        if not (os.path.exists(receptor_pdb) and os.path.exists(peptide_pdb)):
            continue

        workdir = os.path.join(args.work_root, rid)
        # 先算输入肽的中心，作为回退
        fallback_center = center_of_peptide_input(peptide_pdb) or (0.0, 0.0, 0.0)

        score, docked_model, logs = run_hdock(workdir, receptor_pdb, peptide_pdb,
                                              args.hdock_bin, args.createpl_bin,
                                              timeout_s=args.timeout)
        # 计算中心：优先对接模型中的“最小链”
        if docked_model and os.path.exists(docked_model):
            center = center_of_smallest_chain(docked_model) or fallback_center
        else:
            center = fallback_center

        entry = {
            "center": {
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
            }
        }
        if score is not None:
            entry["score"] = float(score)
        results[rid] = entry

        # 也把简短日志写在各自工作目录，便于排查
        try:
            with open(os.path.join(workdir, "run.log"), "w") as lf:
                lf.write(logs)
        except Exception:
            pass

        # 每处理若干个就落盘一次
        if len(results) % 20 == 0:
            with open(args.out_json, "w") as fh:
                json.dump(results, fh, indent=2)
    
    # 最终落盘
    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved JSON to {args.out_json}")

if __name__ == "__main__":
    main()

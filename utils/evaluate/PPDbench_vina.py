#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, shutil, argparse, subprocess, glob
from typing import Optional, Tuple, Dict, List
import numpy as np
from tqdm import tqdm
from Bio import PDB

# ======= 配置默认值（可用命令行覆盖） =======
DEFAULT_DATA_ROOT = "/root/autodl-tmp/PPDbench"
DEFAULT_OUT_JSON  = "/root/autodl-tmp/Peptide_3D/data/PPDbench_hdock_scores.json"
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
    - best_score: 解析 hdock.out / 模型 REMARK 得到的最优分数（通常越负越好）
    - docked_model_path: 最优分数对应的模型文件（若能确定）
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

    # 2) createpl 生成模型并在模型 REMARK 中寻找“最优分数对应的文件”
    docked_model = None
    best_model_pdb = None
    best_model_score = None
    if hdock_out and os.path.exists(hdock_out):
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

        # 收集候选模型
        pdb_candidates: List[str] = []
        for pat in ("model_1.pdb", "top1.pdb", "complex_1.pdb"):
            p = os.path.join(workdir, pat)
            if os.path.exists(p):
                pdb_candidates.append(p)
        if not pdb_candidates:
            for p in glob.glob(os.path.join(workdir, "*.pdb")):
                base = os.path.basename(p).lower()
                if base not in ("receptor.pdb", "peptide.pdb"):
                    pdb_candidates.append(p)

        # 从 REMARK 中解析每个模型的 score，记录“最低分对应的文件名”
        for p in pdb_candidates:
            val = _parse_score_from_pdb(p)
            if val is not None:
                if (best_model_score is None) or (val < best_model_score):
                    best_model_score = val
                    best_model_pdb = p

        # 如果模型里找到更优的分数，则用它覆盖 best_score，并把 docked_model 设为该文件
        if best_model_score is not None:
            if (best_score is None) or (best_model_score < best_score):
                best_score = best_model_score
            docked_model = best_model_pdb

        # 若还是没有确定 docked_model，则用体积最大的作为回退
        if docked_model is None and pdb_candidates:
            prefer = os.path.join(workdir, "model_1.pdb")
            docked_model = prefer if os.path.exists(prefer) else max(pdb_candidates, key=lambda x: os.path.getsize(x))

    return (best_score, docked_model, "\n".join(logs))

# ======= cands/ 工具 =======
def list_cand_peptides(r_dir: str) -> List[str]:
    """返回该蛋白目录下 cands/*.pdb 列表（绝对路径，按文件名排序）"""
    cands_dir = os.path.join(r_dir, "multi_cands")
    if not os.path.isdir(cands_dir):
        return []
    peps = sorted(glob.glob(os.path.join(cands_dir, "*.pdb")))
    return [os.path.abspath(p) for p in peps]

def score_one_candidate(rid: str, receptor_pdb: str, pep_pdb: str,
                        work_root: str, hdock_bin: str, createpl_bin: str,
                        timeout_s: int) -> Tuple[Optional[float], Optional[str], str]:
    """
    对单个候选肽 pep_pdb 进行对接评分。
    workdir 布局：<work_root>/<rid>/cands/<pep_basename_without_ext>/
    """
    base = os.path.splitext(os.path.basename(pep_pdb))[0]
    workdir = os.path.join(work_root, rid, "multi_cands", base)
    score, docked_model, logs = run_hdock(workdir, receptor_pdb, pep_pdb,
                                          hdock_bin, createpl_bin, timeout_s=timeout_s)
    # 写各自日志
    try:
        os.makedirs(workdir, exist_ok=True)
        with open(os.path.join(workdir, "run.log"), "w") as lf:
            lf.write(logs)
    except Exception:
        pass
    return score, docked_model, logs

# ======= 主流程 =======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="train_data 根目录")
    ap.add_argument("--out_json",  default=DEFAULT_OUT_JSON,  help="输出 JSON 文件路径（总体汇总）")
    ap.add_argument("--work_root", default=DEFAULT_WORK_ROOT, help="HDOCK 工作目录根")
    ap.add_argument("--hdock_bin", default=DEFAULT_HDOCK_BIN, help="hdock 可执行文件路径")
    ap.add_argument("--createpl_bin", default=DEFAULT_CREATEPL_BIN, help="createpl 可执行文件路径")
    ap.add_argument("--timeout", type=int, default=900, help="每个样本超时（秒）")
    ap.add_argument("--skip_existing", action="store_true", help="若 out_json/各 cands json 已含该ID/候选肽则跳过")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.work_root, exist_ok=True)

    # 载入总体已有结果（便于断点续跑）
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
        r_dir = os.path.join(args.data_root, rid)
        receptor_pdb = os.path.join(r_dir, "receptor.pdb")
        peptide_pdb  = os.path.join(r_dir, "peptide.pdb")
        if not os.path.exists(receptor_pdb):
            continue

        # ---------- 1) 总体条目（仍按原逻辑，用 r_dir/peptide.pdb 若存在） ----------
        if (not args.skip_existing) or (rid not in results):
            if os.path.exists(peptide_pdb):
                workdir_main = os.path.join(args.work_root, rid)
                # 先算输入肽的中心，作为回退
                fallback_center = center_of_peptide_input(peptide_pdb) or (0.0, 0.0, 0.0)

                score, docked_model, logs = run_hdock(workdir_main, receptor_pdb, peptide_pdb,
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
                if docked_model and os.path.exists(docked_model):
                    entry["best_model_file"] = os.path.basename(docked_model)

                results[rid] = entry

                # 写总体日志
                try:
                    with open(os.path.join(workdir_main, "run.log"), "w") as lf:
                        lf.write(logs)
                except Exception:
                    pass

                # 周期性落盘
                if len(results) % 20 == 0:
                    with open(args.out_json, "w") as fh:
                        json.dump(results, fh, indent=2)

        # ---------- 2) cands/ 批量评分并保存该目录下的 JSON ----------
        cand_peps = list_cand_peptides(r_dir)
        if cand_peps:
            cands_dir = os.path.join(r_dir, "multi_cands")
            cands_json_path = os.path.join(cands_dir, "cands_hdock_scores.json")

            # 读入已有 cands JSON（便于断点续跑与增量）
            cand_scores: Dict[str, Optional[float]] = {}
            if os.path.exists(cands_json_path):
                try:
                    with open(cands_json_path, "r") as cf:
                        cand_scores = json.load(cf)
                except Exception:
                    cand_scores = {}

            # 逐个候选肽评分（按需跳过已存在项）
            for pep in cand_peps:
                key = os.path.abspath(pep)
                if args.skip_existing and key in cand_scores:
                    continue
                score, docked_model, _ = score_one_candidate(
                    rid, receptor_pdb, key, args.work_root,
                    args.hdock_bin, args.createpl_bin, args.timeout
                )
                cand_scores[key] = float(score) if score is not None else None

                # 实时落盘以便断点续跑
                try:
                    with open(cands_json_path, "w") as cf:
                        json.dump(cand_scores, cf, indent=2)
                except Exception:
                    pass

            # 最终再落一次盘（保证格式）
            with open(cands_json_path, "w") as cf:
                json.dump(cand_scores, cf, indent=2)

    # 最终落盘（总体）
    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved overall JSON to {args.out_json}")
    print("Per-protein cands JSON saved as cands/cands_hdock_scores.json")

if __name__ == "__main__":
    main()

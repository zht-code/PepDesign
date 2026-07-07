#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, shutil, argparse, subprocess, glob
from typing import Optional, Tuple, Dict, List
import numpy as np
from tqdm import tqdm
from Bio import PDB
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======= 配置默认值 =======
DEFAULT_DATA_ROOT = "/root/autodl-tmp/train_data"
DEFAULT_OUT_JSON  = "/root/autodl-tmp/Peptide_3D/data/PPDbench_hdock_scores.json"
DEFAULT_WORK_ROOT = "/root/autodl-tmp/hdock"

DEFAULT_HDOCK_BIN   = "/root/autodl-fs/HDOCKlite/hdock"
DEFAULT_CREATEPL_BIN= "/root/autodl-fs/HDOCKlite/createpl"

SCORE_RE_LIST = [
    re.compile(r'(?i)\bscore\b\s*:?\s*([+-]?\d+(?:\.\d+)?)'),
    re.compile(r'(?i)\btotal\s*score\b\s*:?\s*([+-]?\d+(?:\.\d+)?)')
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
                    v = float(m.group(1))
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

# ======= 运行 HDOCK + createpl =======
def run_hdock(workdir: str, receptor_pdb: str, peptide_pdb: str,
              hdock_bin: str, createpl_bin: str, timeout_s: int = 900,
              env_vars: Optional[dict] = None) -> Tuple[Optional[float], Optional[str], str]:
    os.makedirs(workdir, exist_ok=True)
    r_fn = os.path.join(workdir, "receptor.pdb")
    l_fn = os.path.join(workdir, "peptide.pdb")
    shutil.copy2(receptor_pdb, r_fn)
    shutil.copy2(peptide_pdb, l_fn)

    logs = []
    run_env = os.environ.copy()
    if env_vars:
        run_env.update(env_vars)

    cmd = [hdock_bin, "receptor.pdb", "peptide.pdb"]
    logs.append(f"[INFO] run: {' '.join(cmd)} (cwd={workdir})")
    try:
        proc = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout_s, text=True, env=run_env)
        logs.append(proc.stdout or "")
        if proc.returncode != 0:
            logs.append(f"[WARN] hdock exit code {proc.returncode}: {proc.stderr}")
    except subprocess.TimeoutExpired:
        logs.append("[WARN] hdock timeout")
    except Exception as e:
        logs.append(f"[WARN] hdock failed: {e}")

    best_score = None
    hdock_out = _find_any_out_file(workdir)
    if hdock_out:
        val = _parse_best_score_in_textfile(hdock_out)
        if val is not None:
            best_score = val
    else:
        logs.append("[WARN] no *.out file found after hdock")

    docked_model = None
    best_model_pdb = None
    best_model_score = None
    if hdock_out and os.path.exists(hdock_out):
        cmd2 = [createpl_bin, os.path.basename(hdock_out), "top10.pdb", "-nmax", "10", "-complex", "-models"]
        logs.append(f"[INFO] run: {' '.join(cmd2)} (cwd={workdir})")
        try:
            proc2 = subprocess.run(cmd2, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=timeout_s, text=True, env=run_env)
            logs.append(proc2.stdout or "")
            if proc2.returncode != 0:
                logs.append(f"[WARN] createpl exit code {proc2.returncode}: {proc2.stderr}")
        except subprocess.TimeoutExpired:
            logs.append("[WARN] createpl timeout")
        except Exception as e:
            logs.append(f"[WARN] createpl failed: {e}")

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

        for p in pdb_candidates:
            val = _parse_score_from_pdb(p)
            if val is not None:
                if (best_model_score is None) or (val < best_model_score):
                    best_model_score = val
                    best_model_pdb = p

        if best_model_score is not None:
            if (best_score is None) or (best_model_score < best_score):
                best_score = best_model_score
            docked_model = best_model_pdb

        if docked_model is None and pdb_candidates:
            prefer = os.path.join(workdir, "model_1.pdb")
            docked_model = prefer if os.path.exists(prefer) else max(pdb_candidates, key=lambda x: os.path.getsize(x))

    return (best_score, docked_model, "\n".join(logs))

# ======= 任务封装 =======
def task_main_peptide(rid: str, receptor_pdb: str, peptide_pdb: str,
                      work_root: str, hdock_bin: str, createpl_bin: str,
                      timeout_s: int, env_vars: Optional[dict]):
    workdir = os.path.join(work_root, rid)
    fallback_center = center_of_peptide_input(peptide_pdb) or (0.0, 0.0, 0.0)
    score, docked_model, logs = run_hdock(workdir, receptor_pdb, peptide_pdb,
                                          hdock_bin, createpl_bin, timeout_s=timeout_s, env_vars=env_vars)
    if docked_model and os.path.exists(docked_model):
        center = center_of_smallest_chain(docked_model) or fallback_center
    else:
        center = fallback_center
    entry = {"center": {"center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2])}}
    if score is not None:
        entry["score"] = float(score)
    entry["best_model_file"] = os.path.splitext(os.path.basename(peptide_pdb))[0]
    # 写日志
    try:
        os.makedirs(workdir, exist_ok=True)
        with open(os.path.join(workdir, "run.log"), "w") as lf:
            lf.write(logs)
    except Exception:
        pass
    return ("main", rid, entry)

def task_cand_peptide(rid: str, receptor_pdb: str, pep_abs: str,
                      work_root: str, hdock_bin: str, createpl_bin: str,
                      timeout_s: int, env_vars: Optional[dict]):
    base = os.path.splitext(os.path.basename(pep_abs))[0]
    workdir = os.path.join(work_root, rid, "cands", base)
    score, docked_model, logs = run_hdock(workdir, receptor_pdb, pep_abs,
                                          hdock_bin, createpl_bin, timeout_s=timeout_s, env_vars=env_vars)
    # 写日志
    try:
        os.makedirs(workdir, exist_ok=True)
        with open(os.path.join(workdir, "run.log"), "w") as lf:
            lf.write(logs)
    except Exception:
        pass
    return ("cand", rid, pep_abs, (float(score) if score is not None else None))

# ======= 主流程 =======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="train_data 根目录")
    ap.add_argument("--out_json",  default=DEFAULT_OUT_JSON,  help="总体汇总 JSON 输出路径")
    ap.add_argument("--work_root", default=DEFAULT_WORK_ROOT, help="HDOCK 工作目录根")
    ap.add_argument("--hdock_bin", default=DEFAULT_HDOCK_BIN, help="hdock 可执行文件路径")
    ap.add_argument("--createpl_bin", default=DEFAULT_CREATEPL_BIN, help="createpl 可执行文件路径")
    ap.add_argument("--timeout", type=int, default=900, help="每个样本超时（秒）")
    ap.add_argument("--skip_existing", action="store_true", help="若已有对应条目则跳过")
    ap.add_argument("--num_workers", type=int, default=75, help="并发线程数")
    # ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5", help="逗号分隔 GPU 列表（仅用于子进程 env 标识）")
    ap.add_argument("--gpus", type=str, default="0,1,2,3", help="逗号分隔 GPU 列表（仅用于子进程 env 标识）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.work_root, exist_ok=True)

    # 读取总体结果
    overall: Dict[str, Dict] = {}
    if os.path.exists(args.out_json):
        try:
            with open(args.out_json, "r") as fh:
                overall = json.load(fh)
        except Exception:
            overall = {}

    # GPU 列表
    gpu_list: List[Optional[int]] = []
    if args.gpus.strip():
        try:
            gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
        except Exception:
            gpu_list = []
    if not gpu_list:
        gpu_list = [None] * max(1, args.num_workers)

    # 准备任务与每蛋白 cands map
    ids = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    ids.sort()

    cands_maps: Dict[str, Dict[str, Optional[float]]] = {}
    cands_json_paths: Dict[str, str] = {}
    rid_to_rdir: Dict[str, str] = {}   # <--- 新增这行

    tasks = []
    assign_idx = 0

    for rid in ids:
        r_dir = os.path.join(args.data_root, rid)
        rid_to_rdir[rid] = r_dir        # <--- 新增这行
        receptor_pdb = os.path.join(r_dir, "receptor.pdb")
        peptide_pdb  = os.path.join(r_dir, "peptide.pdb")
        if not os.path.exists(receptor_pdb):
            continue

        # main peptide
        if os.path.exists(peptide_pdb) and (not args.skip_existing or rid not in overall):
            gpu_id = gpu_list[assign_idx % len(gpu_list)]; assign_idx += 1
            env_vars = {"CUDA_VISIBLE_DEVICES": str(gpu_id)} if gpu_id is not None else None
            tasks.append( ("main", rid, receptor_pdb, peptide_pdb, args.work_root,
                           args.hdock_bin, args.createpl_bin, args.timeout, env_vars) )

        # cands
        cand_peps = sorted(glob.glob(os.path.join(r_dir, "cands", "*.pdb")))
        cand_peps = [os.path.abspath(p) for p in cand_peps]
        cands_dir = os.path.join(r_dir, "cands")
        cands_json = os.path.join(cands_dir, "cands_hdock_scores.json")
        cands_json_paths[rid] = cands_json

        exist_map: Dict[str, Optional[float]] = {}
        if os.path.exists(cands_json):
            try:
                with open(cands_json, "r") as cf:
                    exist_map = json.load(cf)
            except Exception:
                exist_map = {}
        cands_maps[rid] = exist_map

        for pep_abs in cand_peps:
            if args.skip_existing and pep_abs in exist_map:
                continue
            gpu_id = gpu_list[assign_idx % len(gpu_list)]; assign_idx += 1
            env_vars = {"CUDA_VISIBLE_DEVICES": str(gpu_id)} if gpu_id is not None else None
            tasks.append( ("cand", rid, receptor_pdb, pep_abs, args.work_root,
                           args.hdock_bin, args.createpl_bin, args.timeout, env_vars) )

    # 并发执行
    num_workers = max(1, args.num_workers)
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for t in tasks:
            kind = t[0]
            if kind == "main":
                _, rid, r_pdb, p_pdb, work_root, hdock_bin, createpl_bin, timeout_s, env_vars = t
                futures.append(ex.submit(
                    task_main_peptide, rid, r_pdb, p_pdb, work_root, hdock_bin, createpl_bin, timeout_s, env_vars
                ))
            else:
                _, rid, r_pdb, pep_abs, work_root, hdock_bin, createpl_bin, timeout_s, env_vars = t
                futures.append(ex.submit(
                    task_cand_peptide, rid, r_pdb, pep_abs, work_root, hdock_bin, createpl_bin, timeout_s, env_vars
                ))

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"HDOCK (threads x{num_workers})"):
            res = fut.result()
            if res[0] == "main":
                _, rid, entry = res
                if entry:
                    overall[rid] = entry

                    # === 新增：把天然肽写入该蛋白的 cands JSON 映射 ===
                    r_dir = rid_to_rdir.get(rid, None)
                    if r_dir is not None:
                        # 以 <protein_dir>/cands/peptide.pdb 的绝对路径作为键名（与你的示例一致）
                        nat_key = os.path.abspath(os.path.join(r_dir, "cands", "peptide.pdb"))
                        m = cands_maps.get(rid, {})
                        m[nat_key] = entry.get("score", None)  # 分数可能为 None
                        cands_maps[rid] = m
                        # 立刻写该蛋白的 cands JSON
                        try:
                            cj = cands_json_paths[rid]
                            os.makedirs(os.path.dirname(cj), exist_ok=True)
                            with open(cj, "w") as cf:
                                json.dump(m, cf, indent=2)
                        except Exception:
                            pass

                # 周期性写总体 JSON
                if len(overall) % 20 == 0:
                    try:
                        with open(args.out_json, "w") as fh:
                            json.dump(overall, fh, indent=2)
                    except Exception:
                        pass
            else:
                _, rid, pep_abs, score = res
                m = cands_maps.get(rid, {})
                m[pep_abs] = score
                cands_maps[rid] = m
                # 实时写该蛋白的 cands JSON
                try:
                    cj = cands_json_paths[rid]
                    os.makedirs(os.path.dirname(cj), exist_ok=True)
                    with open(cj, "w") as cf:
                        json.dump(m, cf, indent=2)
                except Exception:
                    pass

    # 最终落盘（总体）
    with open(args.out_json, "w") as fh:
        json.dump(overall, fh, indent=2)

    # 兜底：确保每个蛋白的 cands JSON 里包含天然肽
    for rid in ids:
        r_dir = rid_to_rdir.get(rid)
        if not r_dir:
            continue
        nat_key = os.path.abspath(os.path.join(r_dir, "cands", "peptide.pdb"))
        # 如果总体里有分数，就把它同步到 cands
        if rid in overall and ("score" in overall[rid]):
            m = cands_maps.get(rid, {})
            if nat_key not in m or m[nat_key] is None:
                m[nat_key] = overall[rid]["score"]
            cands_maps[rid] = m

    # 再确保每个 cands.json 落一次
    for rid, m in cands_maps.items():
        cj = cands_json_paths.get(rid)
        if not cj:
            continue
        try:
            os.makedirs(os.path.dirname(cj), exist_ok=True)
            with open(cj, "w") as cf:
                json.dump(m, cf, indent=2)
        except Exception:
            pass

    print(f"\nSaved overall JSON to {args.out_json}")
    print(f"Per-protein cands JSON saved as <protein_dir>/cands/cands_hdock_scores.json")

if __name__ == "__main__":
    main()

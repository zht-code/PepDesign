#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
构造用于 DPO 的“亲和力偏好对”数据集（JSONL）。
- 对每个 prompt（receptor_pdb + peptide_seq + candidates_dir），
  读取 candidates_dir 下所有候选多肽 PDB，使用 HDOCK 打分（或复用已有 scores.json），
  在组内做归一化并排序，生成 (chosen, rejected) 成对偏好样本。
- 输出 JSONL：每行包含 prompt / chosen / rejected 及打分元数据。

HDOCK 假设本地可直接命令行调用：
    hdock <receptor.pdb> <ligand.pdb> > run.log
脚本里提供 parse_hdock_log() 的简易解析器，你需要按你机器的 HDOCK 输出格式稍作适配。
如果你已经有现成的 hdock 分数（例如一个 json: {candidate_pdb: score}），
脚本会优先读取，不会重复跑 HDOCK。

使用示例：
    python build_dpo_pairs_hdock.py \
        --index prompts.tsv \
        --out dpo_pairs.jsonl \
        --pairs-per-prompt 3 \
        --min-margin 0.25 \
        --hdock-bin hdock \
        --parallel 8

prompts.tsv 每行三列（制表符分隔）：
    <receptor_pdb>    <peptide_seq>    <candidates_dir>
"""

# import os
# import re
# import json
# import math
# import time
# import glob
# import argparse
# import tempfile
# import subprocess
# import shutil
# from pathlib import Path
# from statistics import mean, pstdev
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # ==== 进度条（tqdm）支持：未安装时自动降级 ====
# try:
#     from tqdm import tqdm
#     _TQDM_AVAILABLE = True
# except Exception:
#     _TQDM_AVAILABLE = False
#     def tqdm(iterable=None, total=None, desc=None, unit=None, leave=True, **kwargs):
#         class _Dummy:
#             def __enter__(self): return self
#             def __exit__(self, *a): return False
#             def update(self, n=1): pass
#             def close(self): pass
#         return iterable if iterable is not None else _Dummy()

# def tprint(msg: str):
#     if _TQDM_AVAILABLE:
#         from tqdm import tqdm as _t
#         _t.write(str(msg))
#     else:
#         print(msg)

# def read_index(tsv_path: str):
#     items = []
#     with open(tsv_path, "r", encoding="utf-8") as f:
#         for ln in f:
#             ln = ln.strip()
#             if not ln or ln.startswith("#"):
#                 continue
#             parts = ln.split("\t")
#             if len(parts) < 3:
#                 raise ValueError(f"Index line needs 3 columns: <receptor_pdb> <peptide_seq> <candidates_dir>, got: {ln}")
#             rec_pdb, pep_seq, cand_dir = parts[0], parts[1], parts[2]
#             items.append({"receptor_pdb": rec_pdb, "peptide_seq": pep_seq, "cand_dir": cand_dir})
#     return items

# def discover_candidates(cand_dir: str):
#     exts = ("*.pdb", "*.ent", "*.pdbqt")
#     files = []
#     for ext in exts:
#         files += glob.glob(os.path.join(cand_dir, ext))
#     files = sorted(set(files))
#     return files

# # ---------- HDOCK 调用与解析（参考你的 hdock+createpl 实现） ----------

# # 允许科学计数法；用于 .out / REMARK 抽取
# _NUM = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"

# SCORE_RE_LIST = [
#     re.compile(rf'(?i)\b(?:final\s+)?(?:total\s+)?(?:docking\s+)?(?:hdock\s+)?score\b\s*:?\s*{_NUM}'),
#     re.compile(rf'(?i)\b(?:energy|E_total|Etot|dG|binding)\b\s*:?\s*{_NUM}'),
# ]

# def _parse_best_score_in_textfile(path: str):
#     """从任意文本文件中提取最优（最负）score。"""
#     if not path or not os.path.exists(path):
#         return None
#     best = None
#     with open(path, "r", errors="ignore") as fh:
#         for line in fh:
#             for rgx in SCORE_RE_LIST:
#                 m = rgx.search(line)
#                 if m:
#                     try:
#                         v = float(m.group(1))
#                         best = v if best is None else (v if v < best else best)
#                     except Exception:
#                         pass
#     return best

# def _find_any_out_file(workdir: str):
#     """在工作目录里尽量找到 HDOCK 输出文件。"""
#     candidates = [
#         os.path.join(workdir, "hdock.out"),
#         os.path.join(workdir, "Hdock.out"),
#         os.path.join(workdir, "HDOCK.out"),
#     ]
#     for p in candidates:
#         if os.path.exists(p):
#             return p
#     outs = glob.glob(os.path.join(workdir, "*.out"))
#     if outs:
#         return max(outs, key=lambda x: os.path.getsize(x))
#     return None

# def _parse_score_from_pdb(pdb_path: str):
#     """从 model_*.pdb 的 REMARK 中抓 Score（如：REMARK Score:  -379.91）。"""
#     if not pdb_path or not os.path.exists(pdb_path):
#         return None
#     best = None
#     with open(pdb_path, "r", errors="ignore") as fh:
#         for line in fh:
#             if not line.startswith("REMARK"):
#                 continue
#             for rgx in SCORE_RE_LIST:
#                 m = rgx.search(line)
#                 if m:
#                     try:
#                         v = float(m.group(1))
#                         best = v if best is None else (v if v < best else best)
#                     except Exception:
#                         pass
#     return best

# def run_hdock(hdock_bin: str, createpl_bin: str, receptor_pdb: str, ligand_pdb: str, work_dir: str, timeout_s: int = 900):
#     """
#     在临时工作目录运行 HDOCK + 可选 createpl，返回 (best_score, run_log_path)。
#     - best_score: 越负越好
#     - run_log_path: 该任务的合并日志文件路径
#     """
#     os.makedirs(work_dir, exist_ok=True)
#     log_path = os.path.join(work_dir, "run.log")
#     logs = []

#     # 拷贝输入，避免不同路径/文件名影响
#     r_fn = os.path.join(work_dir, "receptor.pdb")
#     l_fn = os.path.join(work_dir, "peptide.pdb")
#     try:
#         shutil.copy2(receptor_pdb, r_fn)
#         shutil.copy2(ligand_pdb, l_fn)
#     except Exception as e:
#         with open(log_path, "w") as lf:
#             lf.write(f"[FATAL] copy inputs failed: {e}\n")
#         raise

#     # 1) 运行 hdock
#     cmd = [hdock_bin, "receptor.pdb", "peptide.pdb"]
#     logs.append(f"[INFO] run: {' '.join(cmd)} (cwd={work_dir})")
#     try:
#         proc = subprocess.run(cmd, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
#                               timeout=timeout_s, text=True)
#         logs.append(proc.stdout or "")
#         if proc.returncode != 0:
#             logs.append(f"[WARN] hdock exit code {proc.returncode}")
#     except subprocess.TimeoutExpired:
#         logs.append("[WARN] hdock timeout")
#     except Exception as e:
#         logs.append(f"[WARN] hdock failed: {e}")

#     best_score = None
#     hdock_out = _find_any_out_file(work_dir)
#     if hdock_out:
#         val = _parse_best_score_in_textfile(hdock_out)
#         if val is not None:
#             best_score = val
#     else:
#         logs.append("[WARN] no *.out file found after hdock")

#     # 2) 调用 createpl 产出 model_*.pdb（便于兜底解析）
#     if hdock_out and createpl_bin:
#         cmd2 = [createpl_bin, os.path.basename(hdock_out), "top10.pdb", "-nmax", "10", "-complex", "-models"]
#         logs.append(f"[INFO] run: {' '.join(cmd2)} (cwd={work_dir})")
#         try:
#             proc2 = subprocess.run(cmd2, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
#                                    timeout=timeout_s, text=True)
#             logs.append(proc2.stdout or "")
#             if proc2.returncode != 0:
#                 logs.append(f"[WARN] createpl exit code {proc2.returncode}")
#         except subprocess.TimeoutExpired:
#             logs.append("[WARN] createpl timeout")
#         except Exception as e:
#             logs.append(f"[WARN] createpl failed: {e}")

#     # 3) 若 .out 未解析到分数，尝试从 PDB REMARK 兜底
#     if best_score is None:
#         pdb_candidates = []
#         pref = os.path.join(work_dir, "model_1.pdb")
#         if os.path.exists(pref):
#             pdb_candidates.append(pref)
#         for p in glob.glob(os.path.join(work_dir, "*.pdb")):
#             base = os.path.basename(p).lower()
#             if base not in ("receptor.pdb", "peptide.pdb") and p not in pdb_candidates:
#                 pdb_candidates.append(p)
#         for p in pdb_candidates:
#             val = _parse_score_from_pdb(p)
#             if val is not None:
#                 best_score = val if best_score is None else (val if val < best_score else best_score)
#         if best_score is None:
#             logs.append("[WARN] no score found in PDB REMARKs either")

#     # 写出合并日志
#     try:
#         with open(log_path, "w") as lf:
#             lf.write("\n".join(logs))
#     except Exception:
#         pass

#     if best_score is None:
#         raise RuntimeError(f"Cannot parse HDOCK score; see {log_path}")

#     return best_score, log_path

# # ---------- 分数缓存（避免重复跑） ----------

# def load_cached_scores(cand_dir: str):
#     cache_path = os.path.join(cand_dir, "hdock_scores.json")
#     if os.path.exists(cache_path):
#         try:
#             with open(cache_path, "r") as f:
#                 return json.load(f)
#         except Exception:
#             return {}
#     return {}

# def save_cached_scores(cand_dir: str, d: dict):
#     cache_path = os.path.join(cand_dir, "hdock_scores.json")
#     tmp = cache_path + ".tmp"
#     with open(tmp, "w") as f:
#         json.dump(d, f, indent=2, ensure_ascii=False)
#     os.replace(tmp, cache_path)

# # ---------- 组内标准化与构造偏好对 ----------

# def zscore_in_group(vals):
#     if len(vals) == 0:
#         return []
#     mu = mean(vals)
#     sigma = pstdev(vals) if len(vals) > 1 else 0.0
#     if sigma < 1e-8:
#         return [0.0 for _ in vals]
#     return [(v - mu) / max(sigma, 1e-8) for v in vals]

# def build_pairs_for_prompt(rec_pdb, pep_seq, cand_files, scores_dict, pairs_per_prompt=3, min_margin=0.25):
#     """
#     HDOCK 分数越小（越负）越好；亲和力记分 affinity = -score，再做组内 z-score。
#     """
#     rows = []
#     for f in cand_files:
#         if f not in scores_dict:
#             continue
#         s = scores_dict[f]
#         if s is None:
#             continue
#         rows.append({"pdb": f, "hdock_score": float(s)})

#     if len(rows) < 2:
#         return []

#     aff = [-r["hdock_score"] for r in rows]
#     aff_z = zscore_in_group(aff)
#     for r, rz in zip(rows, aff_z):
#         r["R"] = rz

#     rows.sort(key=lambda x: x["R"], reverse=True)

#     pairs = []
#     M = min(pairs_per_prompt, len(rows) // 2)
#     for i in range(M):
#         chosen = rows[i]
#         rejected = rows[-(i+1)]
#         if chosen["R"] - rejected["R"] < min_margin:
#             continue
#         pairs.append({
#             "prompt": {
#                 "receptor_pdb": rec_pdb,
#                 "peptide_seq": pep_seq
#             },
#             "chosen": {
#                 "type": "pdb",
#                 "pdb_path": chosen["pdb"],
#                 "score": {
#                     "hdock": chosen["hdock_score"],
#                     "R": chosen["R"]
#                 }
#             },
#             "rejected": {
#                 "type": "pdb",
#                 "pdb_path": rejected["pdb"],
#                 "score": {
#                     "hdock": rejected["hdock_score"],
#                     "R": rejected["R"]
#                 }
#             },
#             "pair_weight": float(max(chosen["R"] - rejected["R"], 0.0))
#         })
#     return pairs

# # ---------- 主流程 ----------

# def main(args):
#     prompts = read_index(args.index)
#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     total_pairs = 0

#     with open(out_path, "w", encoding="utf-8") as fout, \
#          tqdm(total=len(prompts), desc="Prompts", unit="prompt") as pbar_prompts:

#         for it in prompts:
#             rec_pdb  = it["receptor_pdb"]
#             pep_seq  = it["peptide_seq"]
#             cand_dir = it["cand_dir"]

#             cand_files = discover_candidates(cand_dir)
#             if len(cand_files) == 0:
#                 tprint(f"[WARN] no candidates in {cand_dir}, skip.")
#                 pbar_prompts.update(1)
#                 continue

#             scores = load_cached_scores(cand_dir)
#             todo = [f for f in cand_files if f not in scores or scores[f] is None]

#             if len(todo) > 0 and args.hdock_bin is not None:
#                 tprint(f"[INFO] scoring {len(todo)} candidates by HDOCK under {cand_dir} ...")
#                 with tqdm(total=len(todo),
#                           desc=f"HDOCK@{Path(cand_dir).name}",
#                           unit="cand", leave=False) as pbar_score:

#                     def _work(fpath):
#                         with tempfile.TemporaryDirectory(prefix="hdock_") as td:
#                             try:
#                                 sc, logp = run_hdock(args.hdock_bin, args.createpl_bin, rec_pdb, fpath, td, timeout_s=args.timeout)
#                                 return (fpath, sc, None)
#                             except Exception as e:
#                                 return (fpath, None, str(e))

#                     with ThreadPoolExecutor(max_workers=args.parallel) as ex:
#                         futures = [ex.submit(_work, f) for f in todo]
#                         for fu in as_completed(futures):
#                             fpath, sc, err = fu.result()
#                             if err is None:
#                                 scores[fpath] = float(sc)
#                             else:
#                                 tprint(f"[WARN] HDOCK failed for {fpath}: {err}")
#                                 scores[fpath] = None
#                             pbar_score.update(1)

#                 save_cached_scores(cand_dir, scores)
#             else:
#                 tprint(f"[INFO] reuse cached scores in {cand_dir} ({len(scores)} items).")

#             pairs = build_pairs_for_prompt(
#                 rec_pdb, pep_seq, cand_files, scores,
#                 pairs_per_prompt=args.pairs_per_prompt,
#                 min_margin=args.min_margin
#             )
#             for p in pairs:
#                 fout.write(json.dumps(p, ensure_ascii=False) + "\n")
#             total_pairs += len(pairs)
#             tprint(f"[OK] prompt@{cand_dir}: {len(pairs)} pairs.")

#             pbar_prompts.update(1)

#     tprint(f"\n[FIN] wrote {total_pairs} pairs to {out_path}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--index", default='/root/autodl-tmp/Peptide_3D/utils/dpo/prompts.tsv', help="TSV: <receptor_pdb> <peptide_seq> <candidates_dir>")
#     ap.add_argument("--out", default='/root/autodl-tmp/Peptide_3D/utils/dpo/dpo_pairs.jsonl', help="output JSONL path")
#     ap.add_argument("--pairs-per-prompt", type=int, default=3, help="最多为每个 prompt 取多少对 (top-vs-bottom)")
#     ap.add_argument("--min-margin", type=float, default=0.25, help="R 分差阈值，小于该阈值的 pair 会被丢弃")
#     ap.add_argument("--hdock-bin", default="/root/autodl-fs/HDOCKlite/hdock", help="HDOCK 可执行文件路径（若仅复用缓存，可置空）")
#     ap.add_argument("--createpl-bin", default="/root/autodl-fs/HDOCKlite/createpl", help="createpl 可执行文件路径（用于从 PDB REMARK 兜底解析分数）")
#     ap.add_argument("--parallel", type=int, default=8, help="并行评分线程数")
#     ap.add_argument("--timeout", type=int, default=900, help="单个候选的超时时间（秒）")
#     args = ap.parse_args()
#     main(args)
'''
多服务器运行脚本，生成的 dpo_pairs.jsonl 可直接合并（行追加）；

'''
# import os
# import re
# import json
# import math
# import time
# import glob
# import argparse
# import tempfile
# import subprocess
# import shutil
# from pathlib import Path
# from statistics import mean, pstdev
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # ==== 进度条（tqdm）支持：未安装时自动降级 ====
# try:
#     from tqdm import tqdm
#     _TQDM_AVAILABLE = True
# except Exception:
#     _TQDM_AVAILABLE = False
#     def tqdm(iterable=None, total=None, desc=None, unit=None, leave=True, **kwargs):
#         class _Dummy:
#             def __enter__(self): return self
#             def __exit__(self, *a): return False
#             def update(self, n=1): pass
#             def close(self): pass
#         return iterable if iterable is not None else _Dummy()

# def tprint(msg: str):
#     if _TQDM_AVAILABLE:
#         from tqdm import tqdm as _t
#         _t.write(str(msg))
#     else:
#         print(msg)

# def read_index(tsv_path: str):
#     items = []
#     with open(tsv_path, "r", encoding="utf-8") as f:
#         for ln in f:
#             ln = ln.strip()
#             if not ln or ln.startswith("#"):
#                 continue
#             parts = ln.split("\t")
#             if len(parts) < 3:
#                 raise ValueError(f"Index line needs 3 columns: <receptor_pdb> <peptide_seq> <candidates_dir>, got: {ln}")
#             rec_pdb, pep_seq, cand_dir = parts[0], parts[1], parts[2]
#             items.append({"receptor_pdb": rec_pdb, "peptide_seq": pep_seq, "cand_dir": cand_dir})
#     return items

# def discover_candidates(cand_dir: str):
#     exts = ("*.pdb", "*.ent", "*.pdbqt")
#     files = []
#     for ext in exts:
#         files += glob.glob(os.path.join(cand_dir, ext))
#     files = sorted(set(files))
#     return files

# # ---------- HDOCK + CREATEPL + 从 PDB 解析分数 ----------

# _NUM = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
# SCORE_RE_LIST = [
#     re.compile(rf'(?i)\b(?:final\s+)?(?:total\s+)?(?:docking\s+)?(?:hdock\s+)?score\b\s*:?\s*{_NUM}'),
#     re.compile(rf'(?i)\b(?:energy|E_total|Etot|dG|binding)\b\s*:?\s*{_NUM}'),
# ]

# def _find_any_out_file(workdir: str):
#     candidates = [os.path.join(workdir, n) for n in ("hdock.out", "Hdock.out", "HDOCK.out")]
#     for p in candidates:
#         if os.path.exists(p):
#             return p
#     outs = glob.glob(os.path.join(workdir, "*.out"))
#     return max(outs, key=lambda x: os.path.getsize(x)) if outs else None

# def _parse_score_from_pdb(pdb_path: str):
#     if not pdb_path or not os.path.exists(pdb_path):
#         return None
#     best = None
#     with open(pdb_path, "r", errors="ignore") as fh:
#         for line in fh:
#             if not line.startswith("REMARK"):
#                 continue
#             for rgx in SCORE_RE_LIST:
#                 m = rgx.search(line)
#                 if m:
#                     try:
#                         v = float(m.group(1))
#                         best = v if best is None else (v if v < best else best)
#                     except Exception:
#                         pass
#     return best

# def run_hdock(hdock_bin: str, createpl_bin: str, receptor_pdb: str, ligand_pdb: str,
#               work_dir: str, timeout_s: int = 900, extra_env: dict | None = None):
#     """
#     在临时工作目录运行 HDOCK + 可选 createpl，返回 (best_score, run_log_path)。
#     - best_score: 越负越好
#     - run_log_path: 合并日志文件路径
#     - extra_env: 附加到子进程的环境变量（如 CUDA_VISIBLE_DEVICES / OMP_NUM_THREADS 等）
#     """
#     os.makedirs(work_dir, exist_ok=True)
#     log_path = os.path.join(work_dir, "run.log")
#     logs = []

#     # 环境
#     env = os.environ.copy()
#     if extra_env:
#         env.update({k: str(v) for k, v in extra_env.items()})

#     # 拷贝输入
#     r_fn = os.path.join(work_dir, "receptor.pdb")
#     l_fn = os.path.join(work_dir, "peptide.pdb")
#     shutil.copy2(receptor_pdb, r_fn)
#     shutil.copy2(ligand_pdb, l_fn)

#     # 1) hdock
#     cmd = [hdock_bin, "receptor.pdb", "peptide.pdb"]
#     logs.append(f"[INFO] run: {' '.join(cmd)} (cwd={work_dir}) env={{CUDA_VISIBLE_DEVICES:{env.get('CUDA_VISIBLE_DEVICES','-')}, OMP_NUM_THREADS:{env.get('OMP_NUM_THREADS','-')}}}")
#     try:
#         proc = subprocess.run(cmd, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
#                               timeout=timeout_s, text=True, env=env)
#         logs.append(proc.stdout or "")
#         if proc.returncode != 0:
#             logs.append(f"[WARN] hdock exit code {proc.returncode}")
#     except subprocess.TimeoutExpired:
#         logs.append("[WARN] hdock timeout")
#     except Exception as e:
#         logs.append(f"[WARN] hdock failed: {e}")

#     best_score = None
#     hdock_out = _find_any_out_file(work_dir)
#     if hdock_out:
#         # 不再从 .out 里取分；直接进入 createpl 并从 PDB REMARK 兜底
#         pass
#     else:
#         logs.append("[WARN] no *.out file found after hdock")

#     # 2) createpl 生成 model_*.pdb
#     if hdock_out and createpl_bin:
#         cmd2 = [createpl_bin, os.path.basename(hdock_out), "top10.pdb", "-nmax", "10", "-complex", "-models"]
#         logs.append(f"[INFO] run: {' '.join(cmd2)} (cwd={work_dir})")
#         try:
#             proc2 = subprocess.run(cmd2, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
#                                    timeout=timeout_s, text=True, env=env)
#             logs.append(proc2.stdout or "")
#             if proc2.returncode != 0:
#                 logs.append(f"[WARN] createpl exit code {proc2.returncode}")
#         except subprocess.TimeoutExpired:
#             logs.append("[WARN] createpl timeout")
#         except Exception as e:
#             logs.append(f"[WARN] createpl failed: {e}")

#     # 3) 从 PDB REMARK 解析分数（优先 model_1.pdb）
#     pref = os.path.join(work_dir, "model_1.pdb")
#     pdb_candidates = []
#     if os.path.exists(pref):
#         pdb_candidates.append(pref)
#     for p in glob.glob(os.path.join(work_dir, "*.pdb")):
#         base = os.path.basename(p).lower()
#         if base not in ("receptor.pdb", "peptide.pdb") and p not in pdb_candidates:
#             pdb_candidates.append(p)
#     for p in pdb_candidates:
#         v = _parse_score_from_pdb(p)
#         if v is not None:
#             best_score = v if best_score is None else (v if v < best_score else best_score)

#     # 写日志
#     try:
#         with open(log_path, "w") as lf:
#             lf.write("\n".join(logs))
#     except Exception:
#         pass

#     if best_score is None:
#         raise RuntimeError(f"Cannot parse HDOCK score; see {log_path}")

#     return best_score, log_path

# # ---------- 分数缓存（避免重复跑） ----------

# def load_cached_scores(cand_dir: str):
#     cache_path = os.path.join(cand_dir, "hdock_scores.json")
#     if os.path.exists(cache_path):
#         try:
#             with open(cache_path, "r") as f:
#                 return json.load(f)
#         except Exception:
#             return {}
#     return {}

# def save_cached_scores(cand_dir: str, d: dict):
#     cache_path = os.path.join(cand_dir, "hdock_scores.json")
#     tmp = cache_path + ".tmp"
#     with open(tmp, "w") as f:
#         json.dump(d, f, indent=2, ensure_ascii=False)
#     os.replace(tmp, cache_path)

# # ---------- 组内标准化与构造偏好对 ----------

# def zscore_in_group(vals):
#     if len(vals) == 0:
#         return []
#     mu = mean(vals)
#     sigma = pstdev(vals) if len(vals) > 1 else 0.0
#     if sigma < 1e-8:
#         return [0.0 for _ in vals]
#     return [(v - mu) / max(sigma, 1e-8) for v in vals]

# def build_pairs_for_prompt(rec_pdb, pep_seq, cand_files, scores_dict, pairs_per_prompt=3, min_margin=0.25):
#     rows = []
#     for f in cand_files:
#         if f not in scores_dict:
#             continue
#         s = scores_dict[f]
#         if s is None:
#             continue
#         rows.append({"pdb": f, "hdock_score": float(s)})

#     if len(rows) < 2:
#         return []

#     # 越负越好 -> 亲和力取负号，再 z-score
#     aff = [-r["hdock_score"] for r in rows]
#     aff_z = zscore_in_group(aff)
#     for r, rz in zip(rows, aff_z):
#         r["R"] = rz

#     rows.sort(key=lambda x: x["R"], reverse=True)

#     pairs = []
#     M = min(pairs_per_prompt, len(rows) // 2)
#     for i in range(M):
#         chosen = rows[i]
#         rejected = rows[-(i+1)]
#         if chosen["R"] - rejected["R"] < min_margin:
#             continue
#         pairs.append({
#             "prompt": {"receptor_pdb": rec_pdb, "peptide_seq": pep_seq},
#             "chosen":  {"type": "pdb", "pdb_path": chosen["pdb"],  "score": {"hdock": chosen["hdock_score"],  "R": chosen["R"]}},
#             "rejected":{"type": "pdb", "pdb_path": rejected["pdb"], "score": {"hdock": rejected["hdock_score"], "R": rejected["R"]}},
#             "pair_weight": float(max(chosen["R"] - rejected["R"], 0.0))
#         })
#     return pairs

# # ---------- 主流程 ----------

# def main(args):
#     prompts = read_index(args.index)
#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     # --- 并发度与线程数推导 ---
#     cpu_cnt = os.cpu_count() or 1
#     gpu_ids = [s.strip() for s in args.gpu_ids.split(",") if s.strip() != ""]
#     if args.num_cards is not None:
#         gpu_ids = gpu_ids[:args.num_cards]
#     total_slots = max(1, (args.num_cards if args.num_cards else len(gpu_ids)) * max(1, args.slots_per_card))
#     # 总并发：若未显式指定，则自动=min(total_slots, cpu_cnt)
#     parallel = args.parallel if args.parallel and args.parallel > 0 else min(total_slots, cpu_cnt)
#     # 每进程线程数：若未显式指定，则尽量均摊 CPU
#     threads_per_proc = args.threads_per_proc if args.threads_per_proc and args.threads_per_proc > 0 else max(1, cpu_cnt // parallel)

#     tprint(f"[INFO] CPU cores={cpu_cnt}, GPUs={gpu_ids}, total_slots={total_slots}, parallel={parallel}, threads_per_proc={threads_per_proc}")

#     total_pairs = 0

#     with open(out_path, "w", encoding="utf-8") as fout, \
#          tqdm(total=len(prompts), desc="Prompts", unit="prompt") as pbar_prompts:

#         for it in prompts:
#             rec_pdb  = it["receptor_pdb"]
#             pep_seq  = it["peptide_seq"]
#             cand_dir = it["cand_dir"]

#             cand_files = discover_candidates(cand_dir)
#             if len(cand_files) == 0:
#                 tprint(f"[WARN] no candidates in {cand_dir}, skip.")
#                 pbar_prompts.update(1)
#                 continue

#             scores = load_cached_scores(cand_dir)
#             todo = [f for f in cand_files if f not in scores or scores[f] is None]

#             if len(todo) > 0 and args.hdock_bin is not None:
#                 tprint(f"[INFO] scoring {len(todo)} candidates by HDOCK under {cand_dir} ...")
#                 with tqdm(total=len(todo),
#                           desc=f"HDOCK@{Path(cand_dir).name}",
#                           unit="cand", leave=False) as pbar_score:

#                     def _work(fpath, gpu_id):
#                         # 为每个任务设置环境变量：GPU 归属 & 线程数限制
#                         extra_env = {
#                             "CUDA_VISIBLE_DEVICES": gpu_id,
#                             "OMP_NUM_THREADS": threads_per_proc,
#                             "MKL_NUM_THREADS": threads_per_proc,
#                             "OPENBLAS_NUM_THREADS": threads_per_proc,
#                             "NUMEXPR_NUM_THREADS": threads_per_proc,
#                         }
#                         with tempfile.TemporaryDirectory(prefix="hdock_") as td:
#                             try:
#                                 sc, logp = run_hdock(
#                                     args.hdock_bin, args.createpl_bin, rec_pdb, fpath, td,
#                                     timeout_s=args.timeout, extra_env=extra_env
#                                 )
#                                 return (fpath, sc, None)
#                             except Exception as e:
#                                 return (fpath, None, str(e))

#                     # 线程池规模 = parallel（全局）
#                     with ThreadPoolExecutor(max_workers=parallel) as ex:
#                         futures = []
#                         for i, f in enumerate(todo):
#                             gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids else ""
#                             futures.append(ex.submit(_work, f, gpu_id))
#                         for fu in as_completed(futures):
#                             fpath, sc, err = fu.result()
#                             if err is None:
#                                 scores[fpath] = float(sc)
#                             else:
#                                 tprint(f"[WARN] HDOCK failed for {fpath}: {err}")
#                                 scores[fpath] = None
#                             pbar_score.update(1)

#                 save_cached_scores(cand_dir, scores)
#             else:
#                 tprint(f"[INFO] reuse cached scores in {cand_dir} ({len(scores)} items).")

#             pairs = build_pairs_for_prompt(
#                 rec_pdb, pep_seq, cand_files, scores,
#                 pairs_per_prompt=args.pairs_per_prompt,
#                 min_margin=args.min_margin
#             )
#             for p in pairs:
#                 fout.write(json.dumps(p, ensure_ascii=False) + "\n")
#             total_pairs += len(pairs)
#             tprint(f"[OK] prompt@{cand_dir}: {len(pairs)} pairs.")

#             pbar_prompts.update(1)

#     tprint(f"\n[FIN] wrote {total_pairs} pairs to {out_path}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--index", default='/root/autodl-tmp/Peptide_3D/utils/dpo/prompts.tsv', help="TSV: <receptor_pdb> <peptide_seq> <candidates_dir>")
#     ap.add_argument("--out", default='/root/autodl-tmp/Peptide_3D/utils/dpo/dpo_pairs.jsonl', help="output JSONL path")
#     ap.add_argument("--pairs-per-prompt", type=int, default=3, help="最多为每个 prompt 取多少对 (top-vs-bottom)")
#     ap.add_argument("--min-margin", type=float, default=0.25, help="R 分差阈值，小于该阈值的 pair 会被丢弃")
#     ap.add_argument("--hdock-bin", default="/root/autodl-fs/HDOCKlite/hdock", help="HDOCK 可执行文件路径")
#     ap.add_argument("--createpl-bin", default="/root/autodl-fs/HDOCKlite/createpl", help="createpl 可执行文件路径")
#     ap.add_argument("--parallel", type=int, default=0, help="总并发；<=0 则自动 = min(num_cards*slots_per_card, CPU核数)")
#     ap.add_argument("--num-cards", type=int, default=5, help="使用的卡数量（用于并发规划与轮询分配）")
#     ap.add_argument("--gpu-ids", type=str, default="0,1,2,3,4", help="要使用的卡编号列表（逗号分隔）")
#     ap.add_argument("--slots-per-card", type=int, default=20, help="每张卡同时运行的任务数")
#     ap.add_argument("--threads-per-proc", type=int, default=0, help="每个 HDOCK 子进程的线程数；<=0 则自动均摊")
#     ap.add_argument("--timeout", type=int, default=900, help="单个候选的超时时间（秒）")
#     args = ap.parse_args()
#     main(args)


"""
并行计算dpo_pairs.jsonl的脚本，适用于单服务器多卡场景；
"""



# import os
# import re
# import json
# import glob
# import argparse
# import tempfile
# import subprocess
# import shutil
# import threading
# from pathlib import Path
# from statistics import mean, pstdev
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # ==== 进度条（tqdm）支持：未安装时自动降级 ====
# try:
#     from tqdm import tqdm
#     _TQDM_AVAILABLE = True
# except Exception:
#     _TQDM_AVAILABLE = False
#     def tqdm(iterable=None, total=None, desc=None, unit=None, leave=True, **kwargs):
#         class _Dummy:
#             def __enter__(self): return self
#             def __exit__(self, *a): return False
#             def update(self, n=1): pass
#             def close(self): pass
#         return iterable if iterable is not None else _Dummy()

# def tprint(msg: str):
#     if _TQDM_AVAILABLE:
#         from tqdm import tqdm as _t
#         _t.write(str(msg))
#     else:
#         print(msg)

# def read_index(tsv_path: str):
#     items = []
#     with open(tsv_path, "r", encoding="utf-8") as f:
#         for ln in f:
#             ln = ln.strip()
#             if not ln or ln.startswith("#"):
#                 continue
#             parts = ln.split("\t")
#             if len(parts) < 3:
#                 raise ValueError(f"Index line needs 3 columns: <receptor_pdb> <peptide_seq> <candidates_dir>, got: {ln}")
#             rec_pdb, pep_seq, cand_dir = parts[0], parts[1], parts[2]
#             items.append({"receptor_pdb": rec_pdb, "peptide_seq": pep_seq, "cand_dir": cand_dir})
#     return items

# def discover_candidates(cand_dir: str):
#     exts = ("*.pdb", "*.ent", "*.pdbqt")
#     files = []
#     for ext in exts:
#         files += glob.glob(os.path.join(cand_dir, ext))
#     files = sorted(set(files))
#     return files

# # ---------- HDOCK + CREATEPL + 从 PDB 解析分数 ----------

# _NUM = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
# SCORE_RE_LIST = [
#     re.compile(rf'(?i)\b(?:final\s+)?(?:total\s+)?(?:docking\s+)?(?:hdock\s+)?score\b\s*:?\s*{_NUM}'),
#     re.compile(rf'(?i)\b(?:energy|E_total|Etot|dG|binding)\b\s*:?\s*{_NUM}'),
# ]

# def _find_any_out_file(workdir: str):
#     candidates = [os.path.join(workdir, n) for n in ("hdock.out", "Hdock.out", "HDOCK.out")]
#     for p in candidates:
#         if os.path.exists(p):
#             return p
#     outs = glob.glob(os.path.join(workdir, "*.out"))
#     return max(outs, key=lambda x: os.path.getsize(x)) if outs else None

# def _parse_score_from_pdb(pdb_path: str):
#     if not pdb_path or not os.path.exists(pdb_path):
#         return None
#     best = None
#     with open(pdb_path, "r", errors="ignore") as fh:
#         for line in fh:
#             if not line.startswith("REMARK"):
#                 continue
#             for rgx in SCORE_RE_LIST:
#                 m = rgx.search(line)
#                 if m:
#                     try:
#                         v = float(m.group(1))
#                         best = v if best is None else (v if v < best else best)
#                     except Exception:
#                         pass
#     return best

# def run_hdock(hdock_bin: str, createpl_bin: str, receptor_pdb: str, ligand_pdb: str,
#               work_dir: str, timeout_s: int = 900, extra_env: dict | None = None):
#     """
#     在临时工作目录运行 HDOCK + 可选 createpl，返回 (best_score, run_log_path)。
#     - best_score: 越负越好
#     - run_log_path: 合并日志文件路径
#     """
#     os.makedirs(work_dir, exist_ok=True)
#     log_path = os.path.join(work_dir, "run.log")
#     logs = []

#     env = os.environ.copy()
#     if extra_env:
#         env.update({k: str(v) for k, v in extra_env.items()})

#     r_fn = os.path.join(work_dir, "receptor.pdb")
#     l_fn = os.path.join(work_dir, "peptide.pdb")
#     shutil.copy2(receptor_pdb, r_fn)
#     shutil.copy2(ligand_pdb, l_fn)

#     # 1) hdock
#     cmd = [hdock_bin, "receptor.pdb", "peptide.pdb"]
#     logs.append(f"[INFO] run: {' '.join(cmd)} (cwd={work_dir}) env={{CUDA_VISIBLE_DEVICES:{env.get('CUDA_VISIBLE_DEVICES','-')}, OMP_NUM_THREADS:{env.get('OMP_NUM_THREADS','-')}}}")
#     try:
#         proc = subprocess.run(cmd, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
#                               timeout=timeout_s, text=True, env=env)
#         logs.append(proc.stdout or "")
#         if proc.returncode != 0:
#             logs.append(f"[WARN] hdock exit code {proc.returncode}")
#     except subprocess.TimeoutExpired:
#         logs.append("[WARN] hdock timeout")
#     except Exception as e:
#         logs.append(f"[WARN] hdock failed: {e}")

#     hdock_out = _find_any_out_file(work_dir)
#     if not hdock_out:
#         logs.append("[WARN] no *.out file found after hdock")

#     # 2) createpl -> 生成 model_*.pdb
#     if hdock_out and createpl_bin:
#         cmd2 = [createpl_bin, os.path.basename(hdock_out), "top10.pdb", "-nmax", "10", "-complex", "-models"]
#         logs.append(f"[INFO] run: {' '.join(cmd2)} (cwd={work_dir})")
#         try:
#             proc2 = subprocess.run(cmd2, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
#                                    timeout=timeout_s, text=True, env=env)
#             logs.append(proc2.stdout or "")
#             if proc2.returncode != 0:
#                 logs.append(f"[WARN] createpl exit code {proc2.returncode}")
#         except subprocess.TimeoutExpired:
#             logs.append("[WARN] createpl timeout")
#         except Exception as e:
#             logs.append(f"[WARN] createpl failed: {e}")

#     # 3) 从 PDB REMARK 解析分数（优先 model_1.pdb）
#     best_score = None
#     pref = os.path.join(work_dir, "model_1.pdb")
#     pdb_candidates = []
#     if os.path.exists(pref):
#         pdb_candidates.append(pref)
#     for p in glob.glob(os.path.join(work_dir, "*.pdb")):
#         base = os.path.basename(p).lower()
#         if base not in ("receptor.pdb", "peptide.pdb") and p not in pdb_candidates:
#             pdb_candidates.append(p)
#     for p in pdb_candidates:
#         v = _parse_score_from_pdb(p)
#         if v is not None:
#             best_score = v if best_score is None else (v if v < best_score else best_score)

#     # 日志
#     try:
#         with open(log_path, "w") as lf:
#             lf.write("\n".join(logs))
#     except Exception:
#         pass

#     if best_score is None:
#         raise RuntimeError(f"Cannot parse HDOCK score; see {log_path}")
#     return best_score, log_path

# # ---------- 分数缓存（避免重复跑） ----------

# def load_cached_scores(cand_dir: str):
#     cache_path = os.path.join(cand_dir, "hdock_scores.json")
#     if os.path.exists(cache_path):
#         try:
#             with open(cache_path, "r") as f:
#                 return json.load(f)
#         except Exception:
#             return {}
#     return {}

# def save_cached_scores(cand_dir: str, d: dict):
#     cache_path = os.path.join(cand_dir, "hdock_scores.json")
#     tmp = cache_path + ".tmp"
#     with open(tmp, "w") as f:
#         json.dump(d, f, indent=2, ensure_ascii=False)
#     os.replace(tmp, cache_path)

# # ---------- 组内标准化与构造偏好对 ----------

# def zscore_in_group(vals):
#     if len(vals) == 0:
#         return []
#     mu = mean(vals)
#     sigma = pstdev(vals) if len(vals) > 1 else 0.0
#     if sigma < 1e-8:
#         return [0.0 for _ in vals]
#     return [(v - mu) / max(sigma, 1e-8) for v in vals]

# def build_pairs_for_prompt(rec_pdb, pep_seq, cand_files, scores_dict, pairs_per_prompt=3, min_margin=0.25):
#     rows = []
#     for f in cand_files:
#         if f not in scores_dict:
#             continue
#         s = scores_dict[f]
#         if s is None:
#             continue
#         rows.append({"pdb": f, "hdock_score": float(s)})

#     if len(rows) < 2:
#         return []

#     aff = [-r["hdock_score"] for r in rows]  # 越负越好 -> 取负为“亲和力”
#     aff_z = zscore_in_group(aff)
#     for r, rz in zip(rows, aff_z):
#         r["R"] = rz

#     rows.sort(key=lambda x: x["R"], reverse=True)

#     pairs = []
#     M = min(pairs_per_prompt, len(rows) // 2)
#     for i in range(M):
#         chosen = rows[i]
#         rejected = rows[-(i+1)]
#         if chosen["R"] - rejected["R"] < min_margin:
#             continue
#         pairs.append({
#             "prompt": {"receptor_pdb": rec_pdb, "peptide_seq": pep_seq},
#             "chosen":  {"type": "pdb", "pdb_path": chosen["pdb"],  "score": {"hdock": chosen["hdock_score"],  "R": chosen["R"]}},
#             "rejected":{"type": "pdb", "pdb_path": rejected["pdb"], "score": {"hdock": rejected["hdock_score"], "R": rejected["R"]}},
#             "pair_weight": float(max(chosen["R"] - rejected["R"], 0.0))
#         })
#     return pairs

# # ---------- 主流程（全局任务队列并发） ----------

# def main(args):
#     prompts = read_index(args.index)

#     # 统计所有候选 & 已有分数；构建“全局待打分任务队列”
#     canddir_to_scores = {}
#     canddir_locks: dict[str, threading.Lock] = {}
#     all_jobs = []  # 每个元素: (rec_pdb, cand_pdb, cand_dir)

#     for it in prompts:
#         rec_pdb  = it["receptor_pdb"]
#         cand_dir = it["cand_dir"]
#         cand_files = discover_candidates(cand_dir)
#         if len(cand_files) == 0:
#             tprint(f"[WARN] no candidates in {cand_dir}, skip.")
#             continue
#         scores = load_cached_scores(cand_dir)
#         canddir_to_scores[cand_dir] = scores
#         canddir_locks[cand_dir] = threading.Lock()

#         for f in cand_files:
#             if f not in scores or scores[f] is None:
#                 all_jobs.append((rec_pdb, f, cand_dir))

#     total_jobs = len(all_jobs)
#     tprint(f"[INFO] total prompts={len(prompts)}, total scoring jobs={total_jobs}")

#     # --- 并发度与线程数推导 ---
#     cpu_cnt = os.cpu_count() or 1
#     gpu_ids = [s.strip() for s in args.gpu_ids.split(",") if s.strip() != ""]
#     if args.num_cards is not None:
#         gpu_ids = gpu_ids[:args.num_cards]
#     total_slots = max(1, (args.num_cards if args.num_cards else len(gpu_ids)) * max(1, args.slots_per_card))
#     parallel = args.parallel if args.parallel and args.parallel > 0 else min(total_slots, cpu_cnt)
#     threads_per_proc = args.threads_per_proc if args.threads_per_proc and args.threads_per_proc > 0 else max(1, cpu_cnt // max(1, parallel))
#     tprint(f"[INFO] CPU cores={cpu_cnt}, GPUs={gpu_ids}, total_slots={total_slots}, parallel={parallel}, threads_per_proc={threads_per_proc}")

#     # === 全局并发打分 ===
#     if total_jobs > 0 and args.hdock_bin is not None:
#         with tqdm(total=total_jobs, desc="Scoring (global)", unit="job") as pbar:
#             def _work(idx_job, rec_pdb, cand_pdb, cand_dir):
#                 gpu_id = gpu_ids[idx_job % len(gpu_ids)] if gpu_ids else ""
#                 extra_env = {
#                     "CUDA_VISIBLE_DEVICES": gpu_id,
#                     "OMP_NUM_THREADS": threads_per_proc,
#                     "MKL_NUM_THREADS": threads_per_proc,
#                     "OPENBLAS_NUM_THREADS": threads_per_proc,
#                     "NUMEXPR_NUM_THREADS": threads_per_proc,
#                 }
#                 with tempfile.TemporaryDirectory(prefix="hdock_") as td:
#                     try:
#                         sc, logp = run_hdock(
#                             args.hdock_bin, args.createpl_bin, rec_pdb, cand_pdb, td,
#                             timeout_s=args.timeout, extra_env=extra_env
#                         )
#                         # 回写分数（加锁）
#                         with canddir_locks[cand_dir]:
#                             canddir_to_scores[cand_dir][cand_pdb] = float(sc)
#                         ok = True
#                     except Exception as e:
#                         with canddir_locks[cand_dir]:
#                             canddir_to_scores[cand_dir][cand_pdb] = None
#                         tprint(f"[WARN] HDOCK failed for {cand_pdb}: {e}")
#                         ok = False
#                 return ok

#             # 线程池规模 = parallel（全局）
#             with ThreadPoolExecutor(max_workers=parallel) as ex:
#                 futures = [ex.submit(_work, i, rec_pdb, cand_pdb, cand_dir)
#                            for i, (rec_pdb, cand_pdb, cand_dir) in enumerate(all_jobs)]
#                 for fu in as_completed(futures):
#                     fu.result()  # 异常已在 _work 内处理
#                     pbar.update(1)

#         # 分目录一次性写缓存（减少频繁 I/O）
#         for cand_dir, scores in canddir_to_scores.items():
#             save_cached_scores(cand_dir, scores)
#     else:
#         tprint("[INFO] no scoring needed; reuse cached scores.")

#     # === 构建偏好对并写 JSONL ===
#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     total_pairs = 0
#     with open(out_path, "w", encoding="utf-8") as fout, \
#          tqdm(total=len(prompts), desc="Build pairs", unit="prompt") as pbar_prompts:
#         for it in prompts:
#             rec_pdb  = it["receptor_pdb"]
#             pep_seq  = it["peptide_seq"]
#             cand_dir = it["cand_dir"]
#             cand_files = discover_candidates(cand_dir)
#             scores = canddir_to_scores.get(cand_dir, load_cached_scores(cand_dir))

#             pairs = build_pairs_for_prompt(
#                 rec_pdb, pep_seq, cand_files, scores,
#                 pairs_per_prompt=args.pairs_per_prompt,
#                 min_margin=args.min_margin
#             )
#             for p in pairs:
#                 fout.write(json.dumps(p, ensure_ascii=False) + "\n")
#             total_pairs += len(pairs)
#             pbar_prompts.update(1)

#     tprint(f"\n[FIN] wrote {total_pairs} pairs to {out_path}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--index", default='/root/autodl-tmp/Peptide_3D/utils/dpo/prompts.tsv', help="TSV: <receptor_pdb> <peptide_seq> <candidates_dir>")
#     ap.add_argument("--out", default='/root/autodl-tmp/Peptide_3D/utils/dpo/dpo_pairs.jsonl', help="output JSONL path")
#     ap.add_argument("--pairs-per-prompt", type=int, default=3, help="最多为每个 prompt 取多少对 (top-vs-bottom)")
#     ap.add_argument("--min-margin", type=float, default=0.25, help="R 分差阈值，小于该阈值的 pair 会被丢弃")
#     ap.add_argument("--hdock-bin", default="/root/autodl-fs/HDOCKlite/hdock", help="HDOCK 可执行文件路径")
#     ap.add_argument("--createpl-bin", default='/root/autodl-fs/HDOCKlite/createpl', help="createpl 可执行文件路径")
#     # —— 并发 / 资源参数 —— #
#     ap.add_argument("--parallel", type=int, default=0, help="全局总并发；<=0 自动 = min(num_cards*slots_per_card, CPU核数)")
#     ap.add_argument("--num-cards", type=int, default=5, help="逻辑分组用的卡数量（仅用于分配标记）")
#     ap.add_argument("--gpu-ids", type=str, default="0,1,2,3,4", help="卡编号（逗号分隔），用于 CUDA_VISIBLE_DEVICES 标记")
#     ap.add_argument("--slots-per-card", type=int, default=20, help="每张卡并发的任务数；决定自动并发上限")
#     ap.add_argument("--threads-per-proc", type=int, default=0, help="每个 HDOCK 子进程占用的 CPU 线程数；<=0 自动均摊")
#     ap.add_argument("--timeout", type=int, default=900, help="单个候选的超时时间（秒）")
#     args = ap.parse_args()
#     main(args)

"""
并行计算dpo_pairs.jsonl的脚本，适用于单服务器多卡场景；
"""
import os
import re
import json
import glob
import argparse
import tempfile
import subprocess
import shutil
import threading
from pathlib import Path
from statistics import mean, pstdev
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==== 进度条（tqdm）支持：未安装时自动降级 ====
try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except Exception:
    _TQDM_AVAILABLE = False
    def tqdm(iterable=None, total=None, desc=None, unit=None, leave=True, **kwargs):
        class _Dummy:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def update(self, n=1): pass
            def close(self): pass
        return iterable if iterable is not None else _Dummy()

def tprint(msg: str):
    if _TQDM_AVAILABLE:
        from tqdm import tqdm as _t
        _t.write(str(msg))
    else:
        print(msg)

def read_index(tsv_path: str):
    items = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                raise ValueError(f"Index line needs 3 columns: <receptor_pdb> <peptide_seq> <candidates_dir>, got: {ln}")
            rec_pdb, pep_seq, cand_dir = parts[0], parts[1], parts[2]
            items.append({"receptor_pdb": rec_pdb, "peptide_seq": pep_seq, "cand_dir": cand_dir})
    return items

def discover_candidates(cand_dir: str):
    exts = ("*.pdb", "*.ent", "*.pdbqt")
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(cand_dir, ext))
    files = sorted(set(files))
    return files

# ---------- HDOCK + CREATEPL + 从 PDB 解析分数（用于可选补打分） ----------

_NUM = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
SCORE_RE_LIST = [
    re.compile(rf'(?i)\b(?:final\s+)?(?:total\s+)?(?:docking\s+)?(?:hdock\s+)?score\b\s*:?\s*{_NUM}'),
    re.compile(rf'(?i)\b(?:energy|E_total|Etot|dG|binding)\b\s*:?\s*{_NUM}'),
]

def _find_any_out_file(workdir: str):
    candidates = [os.path.join(workdir, n) for n in ("hdock.out", "Hdock.out", "HDOCK.out")]
    for p in candidates:
        if os.path.exists(p):
            return p
    outs = glob.glob(os.path.join(workdir, "*.out"))
    return max(outs, key=lambda x: os.path.getsize(x)) if outs else None

def _parse_score_from_pdb(pdb_path: str):
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
                    try:
                        v = float(m.group(1))
                        best = v if best is None else (v if v < best else best)
                    except Exception:
                        pass
    return best

def run_hdock(hdock_bin: str, createpl_bin: str, receptor_pdb: str, ligand_pdb: str,
              work_dir: str, timeout_s: int = 900, extra_env: dict | None = None):
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, "run.log")
    logs = []

    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    r_fn = os.path.join(work_dir, "receptor.pdb")
    l_fn = os.path.join(work_dir, "peptide.pdb")
    shutil.copy2(receptor_pdb, r_fn)
    shutil.copy2(ligand_pdb, l_fn)

    cmd = [hdock_bin, "receptor.pdb", "peptide.pdb"]
    logs.append(f"[INFO] run: {' '.join(cmd)} (cwd={work_dir}) env={{CUDA_VISIBLE_DEVICES:{env.get('CUDA_VISIBLE_DEVICES','-')}, OMP_NUM_THREADS:{env.get('OMP_NUM_THREADS','-')}}}")
    try:
        proc = subprocess.run(cmd, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=timeout_s, text=True, env=env)
        logs.append(proc.stdout or "")
        if proc.returncode != 0:
            logs.append(f"[WARN] hdock exit code {proc.returncode}")
    except subprocess.TimeoutExpired:
        logs.append("[WARN] hdock timeout")
    except Exception as e:
        logs.append(f"[WARN] hdock failed: {e}")

    hdock_out = _find_any_out_file(work_dir)
    if not hdock_out:
        logs.append("[WARN] no *.out file found after hdock")

    if hdock_out and createpl_bin:
        cmd2 = [createpl_bin, os.path.basename(hdock_out), "top10.pdb", "-nmax", "10", "-complex", "-models"]
        logs.append(f"[INFO] run: {' '.join(cmd2)} (cwd={work_dir})")
        try:
            proc2 = subprocess.run(cmd2, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=timeout_s, text=True, env=env)
            logs.append(proc2.stdout or "")
            if proc2.returncode != 0:
                logs.append(f"[WARN] createpl exit code {proc2.returncode}")
        except subprocess.TimeoutExpired:
            logs.append("[WARN] createpl timeout")
        except Exception as e:
            logs.append(f"[WARN] createpl failed: {e}")

    best_score = None
    pref = os.path.join(work_dir, "model_1.pdb")
    pdb_candidates = []
    if os.path.exists(pref):
        pdb_candidates.append(pref)
    for p in glob.glob(os.path.join(work_dir, "*.pdb")):
        base = os.path.basename(p).lower()
        if base not in ("receptor.pdb", "peptide.pdb") and p not in pdb_candidates:
            pdb_candidates.append(p)
    for p in pdb_candidates:
        v = _parse_score_from_pdb(p)
        if v is not None:
            best_score = v if best_score is None else (v if v < best_score else best_score)

    try:
        with open(log_path, "w") as lf:
            lf.write("\n".join(logs))
    except Exception:
        pass

    if best_score is None:
        raise RuntimeError(f"Cannot parse HDOCK score; see {log_path}")
    return best_score, log_path

# ---------- 分数加载（优先 cands_hdock_scores.json） ----------

def load_scores_from_cands_json(cand_dir: str) -> dict:
    """
    读取 cand_dir/cands_hdock_scores.json，并把里面的 key（可能是绝对路径或文件名）
    映射为 cand_dir 下实际候选 PDB 的绝对路径。
    """
    scores_path = os.path.join(cand_dir, "cands_hdock_scores.json")
    mapping = {}
    if not os.path.exists(scores_path):
        return mapping

    try:
        raw = json.load(open(scores_path, "r"))
    except Exception as e:
        tprint(f"[WARN] bad JSON: {scores_path}: {e}")
        return mapping

    # 建 basename -> score 的表，也保留原始绝对路径 -> score
    by_abs = {}
    by_base = {}
    for k, v in raw.items():
        try:
            v = float(v)
        except Exception:
            continue
        by_abs[os.path.normpath(k)] = v
        by_base[os.path.basename(os.path.normpath(k))] = v

    # 遍历实际候选文件，按绝对路径优先匹配，其次按文件名匹配
    for f in discover_candidates(cand_dir):
        fn_abs = os.path.normpath(f)
        fn_base = os.path.basename(fn_abs)
        if fn_abs in by_abs:
            mapping[f] = by_abs[fn_abs]
        elif fn_base in by_base:
            mapping[f] = by_base[fn_base]
        else:
            mapping[f] = None  # 没找到就留空，后面可选择是否补打分
    return mapping

def load_cached_scores(cand_dir: str):
    """兼容旧缓存（cand_dir/hdock_scores.json）"""
    cache_path = os.path.join(cand_dir, "hdock_scores.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cached_scores(cand_dir: str, d: dict):
    cache_path = os.path.join(cand_dir, "hdock_scores.json")
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    os.replace(tmp, cache_path)

# ---------- 组内标准化与构造偏好对 ----------

def zscore_in_group(vals):
    if len(vals) == 0:
        return []
    mu = mean(vals)
    sigma = pstdev(vals) if len(vals) > 1 else 0.0
    if sigma < 1e-8:
        return [0.0 for _ in vals]
    return [(v - mu) / max(sigma, 1e-8) for v in vals]

def build_pairs_for_prompt(rec_pdb, pep_seq, cand_files, scores_dict, pairs_per_prompt=3, min_margin=0.25):
    rows = []
    for f in cand_files:
        if f not in scores_dict:
            continue
        s = scores_dict[f]
        if s is None:
            continue
        rows.append({"pdb": f, "hdock_score": float(s)})

    if len(rows) < 2:
        return []

    # HDOCK 越负越好 -> 取负作为“亲和力”
    aff = [-r["hdock_score"] for r in rows]
    aff_z = zscore_in_group(aff)
    for r, rz in zip(rows, aff_z):
        r["R"] = rz

    rows.sort(key=lambda x: x["R"], reverse=True)

    pairs = []
    M = min(pairs_per_prompt, len(rows) // 2)
    for i in range(M):
        chosen = rows[i]
        rejected = rows[-(i+1)]
        if chosen["R"] - rejected["R"] < min_margin:
            continue
        pairs.append({
            "prompt": {"receptor_pdb": rec_pdb, "peptide_seq": pep_seq},
            "chosen":  {"type": "pdb", "pdb_path": chosen["pdb"],
                        "score": {"hdock": chosen["hdock_score"],  "R": chosen["R"]}},
            "rejected":{"type": "pdb", "pdb_path": rejected["pdb"],
                        "score": {"hdock": rejected["hdock_score"], "R": rejected["R"]}},
            "pair_weight": float(max(chosen["R"] - rejected["R"], 0.0))
        })
    return pairs

# ---------- 主流程（优先用 cands_hdock_scores.json） ----------

def main(args):
    prompts = read_index(args.index)

    # 准备每个 cand_dir 的分数表：优先 cands_hdock_scores.json，其次 hdock_scores.json
    canddir_to_scores = {}
    all_missing_jobs = []  # 可选：缺失分数的补打分队列

    for it in tqdm(prompts, desc="Load scores", unit="prompt"):
        cand_dir = it["cand_dir"]

        # 1) 先从 cands_hdock_scores.json 读取
        scores = load_scores_from_cands_json(cand_dir)

        # 2) 若还缺，再并上旧缓存（hdock_scores.json）
        legacy = load_cached_scores(cand_dir)
        for k, v in legacy.items():
            scores.setdefault(k, v)

        # 3) 找出缺失项（必要时补打分）
        if args.fill_missing_with_hdock and args.hdock_bin:
            for f in discover_candidates(cand_dir):
                if f not in scores or scores[f] is None:
                    all_missing_jobs.append((it["receptor_pdb"], f, cand_dir))

        canddir_to_scores[cand_dir] = scores

    # === （可选）补打分 ===
    if args.fill_missing_with_hdock and len(all_missing_jobs) > 0:
        tprint(f"[INFO] missing scores: {len(all_missing_jobs)} -> run HDOCK to fill (this may take time)")
        # 并发资源设置
        cpu_cnt = os.cpu_count() or 1
        gpu_ids = [s.strip() for s in args.gpu_ids.split(",") if s.strip() != ""]
        if args.num_cards is not None:
            gpu_ids = gpu_ids[:args.num_cards]
        total_slots = max(1, (args.num_cards if args.num_cards else len(gpu_ids)) * max(1, args.slots_per_card))
        parallel = args.parallel if args.parallel and args.parallel > 0 else min(total_slots, cpu_cnt)
        threads_per_proc = args.threads_per_proc if args.threads_per_proc and args.threads_per_proc > 0 else max(1, cpu_cnt // max(1, parallel))
        tprint(f"[INFO] CPU cores={cpu_cnt}, GPUs={gpu_ids}, total_slots={total_slots}, parallel={parallel}, threads_per_proc={threads_per_proc}")

        canddir_locks: dict[str, threading.Lock] = {it["cand_dir"]: threading.Lock() for it in prompts}

        with tqdm(total=len(all_missing_jobs), desc="HDOCK fill", unit="job") as pbar:
            def _work(idx_job, rec_pdb, cand_pdb, cand_dir):
                gpu_id = gpu_ids[idx_job % len(gpu_ids)] if gpu_ids else ""
                extra_env = {
                    "CUDA_VISIBLE_DEVICES": gpu_id,
                    "OMP_NUM_THREADS": threads_per_proc,
                    "MKL_NUM_THREADS": threads_per_proc,
                    "OPENBLAS_NUM_THREADS": threads_per_proc,
                    "NUMEXPR_NUM_THREADS": threads_per_proc,
                }
                with tempfile.TemporaryDirectory(prefix="hdock_") as td:
                    try:
                        sc, _ = run_hdock(args.hdock_bin, args.createpl_bin, rec_pdb, cand_pdb, td,
                                          timeout_s=args.timeout, extra_env=extra_env)
                        with canddir_locks[cand_dir]:
                            canddir_to_scores[cand_dir][cand_pdb] = float(sc)
                    except Exception as e:
                        with canddir_locks[cand_dir]:
                            canddir_to_scores[cand_dir][cand_pdb] = None
                        tprint(f"[WARN] HDOCK failed for {cand_pdb}: {e}")
                return True

            with ThreadPoolExecutor(max_workers=parallel) as ex:
                futures = [ex.submit(_work, i, rec_pdb, cand_pdb, cand_dir)
                           for i, (rec_pdb, cand_pdb, cand_dir) in enumerate(all_missing_jobs)]
                for fu in as_completed(futures):
                    fu.result()
                    pbar.update(1)

        # 把补全后的结果也写回 hdock_scores.json（不覆盖 cands_hdock_scores.json）
        for cand_dir, scores in canddir_to_scores.items():
            save_cached_scores(cand_dir, scores)

    # === 构建偏好对并写 JSONL ===
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    with open(out_path, "w", encoding="utf-8") as fout, \
         tqdm(total=len(prompts), desc="Build pairs", unit="prompt") as pbar_prompts:
        for it in prompts:
            rec_pdb  = it["receptor_pdb"]
            pep_seq  = it["peptide_seq"]
            cand_dir = it["cand_dir"]
            cand_files = discover_candidates(cand_dir)
            scores = canddir_to_scores.get(cand_dir, {})

            pairs = build_pairs_for_prompt(
                rec_pdb, pep_seq, cand_files, scores,
                pairs_per_prompt=args.pairs_per_prompt,
                min_margin=args.min_margin
            )
            for p in pairs:
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")
            total_pairs += len(pairs)
            pbar_prompts.update(1)

    tprint(f"\n[FIN] wrote {total_pairs} pairs to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/root/autodl-tmp/Peptide_3D/utils/dpo/prompts.tsv",
                    help="TSV: <receptor_pdb> <peptide_seq> <candidates_dir>")
    ap.add_argument("--out", default="/root/autodl-tmp/Peptide_3D/utils/dpo/dpo_pairs.jsonl",
                    help="output JSONL path")
    ap.add_argument("--pairs-per-prompt", type=int, default=3,
                    help="最多为每个 prompt 取多少对 (top-vs-bottom)")
    ap.add_argument("--min-margin", type=float, default=0.25,
                    help="R 分差阈值，小于该阈值的 pair 会被丢弃")

    # 读取方式：优先 cands/cands_hdock_scores.json
    ap.add_argument("--fill-missing-with-hdock", action="store_true",
                    help="若 cands_hdock_scores.json 缺少某些候选的分数，则调用 HDOCK 进行补充")

    # HDOCK 相关（仅在 --fill-missing-with-hdock 时用到）
    ap.add_argument("--hdock-bin", default="/root/autodl-fs/HDOCKlite/hdock", help="HDOCK 可执行文件路径")
    ap.add_argument("--createpl-bin", default="/root/autodl-fs/HDOCKlite/createpl", help="createpl 可执行文件路径")
    ap.add_argument("--timeout", type=int, default=900, help="单个候选的超时时间（秒）")

    # 并发 / 资源参数（仅在补打分时生效）
    ap.add_argument("--parallel", type=int, default=0, help="全局总并发；<=0 自动 = min(num_cards*slots_per_card, CPU核数)")
    ap.add_argument("--num-cards", type=int, default=5, help="逻辑分组用的卡数量（仅用于分配标记）")
    ap.add_argument("--gpu-ids", type=str, default="0,1,2,3,4", help="卡编号（逗号分隔），用于 CUDA_VISIBLE_DEVICES 标记")
    ap.add_argument("--slots-per-card", type=int, default=20, help="每张卡并发的任务数；决定自动并发上限")
    ap.add_argument("--threads-per-proc", type=int, default=0, help="每个 HDOCK 子进程占用的 CPU 线程数；<=0 自动均摊")

    args = ap.parse_args()
    main(args)

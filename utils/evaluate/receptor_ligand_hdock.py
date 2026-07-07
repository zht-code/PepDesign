#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hdock_score_pair.py
对“指定 receptor.pdb 与 peptide.pdb”运行 HDOCK，仅输出这对的最优亲和力分数（越负越好）。
"""

import os
import re
import sys
import shutil
import glob
import argparse
import subprocess

# 兼容多种写法的分数字段
SCORE_RE_LIST = [
    re.compile(r'(?i)\bscore\b\s*:?\s*([+-]?\d+(?:\.\d+)?)'),
    re.compile(r'(?i)\btotal\s*score\b\s*:?\s*([+-]?\d+(?:\.\d+)?)'),
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
    outs = glob.glob(os.path.join(workdir, "*.out"))
    if outs:
        return max(outs, key=lambda x: os.path.getsize(x))
    return None

def _parse_score_from_pdb(pdb_path: str):
    """从模型 PDB 的 REMARK 中抓分数（如：REMARK Score:  -379.91）。"""
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

def run_hdock_once(workdir: str, receptor_pdb: str, peptide_pdb: str,
                   hdock_bin: str, createpl_bin: str | None, timeout_s: int = 3600) -> float | None:
    """
    运行 hdock，优先从 hdock.out 解析最优分数；若失败且提供 createpl，则解析生成的 pdb 模型 REMARK 作为兜底。
    仅返回 float 分数（越负越好）或 None。
    """
    os.makedirs(workdir, exist_ok=True)
    r_fn = os.path.join(workdir, "receptor.pdb")
    l_fn = os.path.join(workdir, "peptide.pdb")
    shutil.copy2(receptor_pdb, r_fn)
    shutil.copy2(peptide_pdb, l_fn)

    # 1) 运行 HDOCK
    cmd = [hdock_bin, os.path.basename(r_fn), os.path.basename(l_fn)]
    try:
        proc = subprocess.run(cmd, cwd=workdir,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout_s, text=True)
        if proc.returncode != 0:
            # 不直接失败，仍然尝试解析可能产生的 out 文件
            pass
    except subprocess.TimeoutExpired:
        # 超时也继续尝试解析 out（如果有）
        pass
    except Exception:
        # 出错也继续兜底解析
        pass

    # 2) 解析 hdock.out
    hdock_out = _find_any_out_file(workdir)
    best_score = _parse_best_score_in_textfile(hdock_out) if hdock_out else None
    if best_score is not None:
        return best_score

    # 3) 若需要，再用 createpl 生成模型并从 PDB REMARK 兜底解析
    if createpl_bin:
        # 尝试基于 out 生成 top 模型
        if hdock_out and os.path.exists(hdock_out):
            cmd2 = [createpl_bin, os.path.basename(hdock_out), "top10.pdb", "-nmax", "10", "-complex", "-models"]
            try:
                proc2 = subprocess.run(cmd2, cwd=workdir,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       timeout=timeout_s, text=True)
                # 不论返回码，尝试搜集产生的 pdb
            except Exception:
                pass

            pdb_candidates = []
            for p in ("model_1.pdb", "top1.pdb", "complex_1.pdb"):
                absp = os.path.join(workdir, p)
                if os.path.exists(absp):
                    pdb_candidates.append(absp)
            if not pdb_candidates:
                pdb_candidates = [p for p in glob.glob(os.path.join(workdir, "*.pdb"))
                                  if os.path.basename(p).lower() not in ("receptor.pdb", "peptide.pdb")]

            best_model_score = None
            for p in pdb_candidates:
                val = _parse_score_from_pdb(p)
                if val is not None and (best_model_score is None or val < best_model_score):
                    best_model_score = val
            if best_model_score is not None:
                return best_model_score

    return None

def main():
    parser = argparse.ArgumentParser(description="Run HDOCK for a single receptor-peptide pair and print best score.")
    parser.add_argument("--receptor", default="/root/autodl-tmp/case/nefl_mut_fixed.pdb", help="受体 PDB 路径")
    parser.add_argument("--peptide",  default="/root/autodl-tmp/case/nefl_mut_fixed.pdb", help="多肽 PDB 路径")
    parser.add_argument("--workdir",  default="/root/autodl-tmp/hdock111", help="工作目录（会写入临时文件）")
    parser.add_argument("--hdock",    default="/root/autodl-fs/HDOCKlite/hdock", help="hdock 可执行路径")
    parser.add_argument("--createpl", default="/root/autodl-fs/HDOCKlite/createpl",
                        help="createpl 可执行路径；若不想用兜底解析，可传空字符串")
    parser.add_argument("--timeout",  type=int, default=86400, help="超时（秒）")
    args = parser.parse_args()

    receptor = os.path.abspath(args.receptor)
    peptide  = os.path.abspath(args.peptide)
    workdir  = os.path.abspath(args.workdir)
    hdock_bin = os.path.abspath(args.hdock)
    createpl_bin = os.path.abspath(args.createpl) if args.createpl else None

    if not os.path.exists(receptor):
        print(f"[ERR] receptor not found: {receptor}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(peptide):
        print(f"[ERR] peptide not found:  {peptide}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(hdock_bin):
        print(f"[ERR] hdock not found: {hdock_bin}", file=sys.stderr)
        sys.exit(2)
    if createpl_bin and not os.path.isfile(createpl_bin):
        print(f"[WARN] createpl not found, will skip model-remark fallback: {createpl_bin}", file=sys.stderr)
        createpl_bin = None

    score = run_hdock_once(workdir, receptor, peptide, hdock_bin, createpl_bin, timeout_s=args.timeout)
    if score is None:
        print("[ERR] failed to obtain score", file=sys.stderr)
        sys.exit(1)

    # 仅输出分数（stdout）
    # 例：-379.91
    print(f"{score:.6f}")

if __name__ == "__main__":
    main()

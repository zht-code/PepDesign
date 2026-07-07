#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于 HDOCK 的批量蛋白-多肽亲和力计算（自动识别 CPU 并满载）。"""

import os
import re
import sys
import json
import glob
import shutil
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

SCORE_RE_LIST = [
    re.compile(r'(?i)\bscore\b\s*:?\s*([+-]?[0-9]+(?:\.[0-9]+)?)'),
    re.compile(r'(?i)\btotal\s*score\b\s*:?\s*([+-]?[0-9]+(?:\.[0-9]+)?)'),
]


def _parse_best_score_in_textfile(path: str):
    if not path or not os.path.exists(path):
        return None
    best = None
    numeric_best = None
    with open(path, 'r', errors='ignore') as fh:
        for line in fh:
            for rgx in SCORE_RE_LIST:
                m = rgx.search(line)
                if m:
                    try:
                        v = float(m.group(1))
                    except Exception:
                        continue
                    best = v if best is None else (v if v < best else best)
    return best


def _find_any_out_file(workdir: str):
    candidates = [
        os.path.join(workdir, 'hdock.out'),
        os.path.join(workdir, 'Hdock.out'),
        os.path.join(workdir, 'HDOCK.out'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    outs = glob.glob(os.path.join(workdir, '*.out'))
    if outs:
        return max(outs, key=lambda x: os.path.getsize(x))
    return None


def _parse_score_from_pdb(pdb_path: str):
    if not pdb_path or not os.path.exists(pdb_path):
        return None
    best = None
    with open(pdb_path, 'r', errors='ignore') as fh:
        for line in fh:
            if not line.startswith('REMARK'):
                continue
            for rgx in SCORE_RE_LIST:
                m = rgx.search(line)
                if m:
                    try:
                        v = float(m.group(1))
                    except Exception:
                        continue
                    best = v if best is None else (v if v < best else best)

            # 如果没有在当前行匹配到 "score" 相关字段，尝试用数值列启发式解析
            # 这主要是为 HDOCKlite 这类只有数值表格、没有显式 "score" 文本的输出设计。
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            floats = []
            for p in parts:
                try:
                    floats.append(float(p))
                except Exception:
                    # 非数值字段直接跳过
                    continue
            if not floats:
                continue
            # 经验上，HDOCK 输出中最负的那个能量列就是我们关心的评分
            cand = min(floats)
            # 只接受明显为能量的负值，避免把一些计数类小数当成score
            if cand < -1.0:
                numeric_best = cand if numeric_best is None else (cand if cand < numeric_best else numeric_best)

    return best if best is not None else numeric_best


def run_hdock_pair(workdir: str,
                   receptor_pdb: str,
                   peptide_pdb: str,
                   hdock_bin: str,
                   createpl_bin: str,
                   timeout_s: int = 900) -> Tuple[Optional[float], str]:
    os.makedirs(workdir, exist_ok=True)
    r_fn = os.path.join(workdir, 'receptor.pdb')
    l_fn = os.path.join(workdir, 'peptide.pdb')
    shutil.copy2(receptor_pdb, r_fn)
    shutil.copy2(peptide_pdb, l_fn)

    logs = []
    cmd = [hdock_bin, 'receptor.pdb', 'peptide.pdb']
    logs.append(f"[HDOCK] cmd: {' '.join(cmd)} (cwd={workdir})")
    try:
        proc = subprocess.run(cmd, cwd=workdir,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout_s, text=True)
        logs.append(proc.stdout or '')
        if proc.returncode != 0:
            logs.append(f"[HDOCK] exit code {proc.returncode}, stderr={proc.stderr}")
    except subprocess.TimeoutExpired:
        logs.append('[HDOCK] timeout')
    except Exception as e:
        logs.append(f'[HDOCK] failed: {e}')

    best_score = None
    hdock_out = _find_any_out_file(workdir)
    if hdock_out:
        val = _parse_best_score_in_textfile(hdock_out)
        if val is not None:
            best_score = val

    # 如果没有score, 借助createpl从pdb remark解析
    if best_score is None and createpl_bin:
        if hdock_out and os.path.exists(hdock_out):
            cmd2 = [createpl_bin, os.path.basename(hdock_out), 'top3.pdb', '-nmax', '3', '-complex', '-models']
            logs.append(f"[CREATEPL] cmd: {' '.join(cmd2)} (cwd={workdir})")
            try:
                proc2 = subprocess.run(cmd2, cwd=workdir,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       timeout=timeout_s, text=True)
                logs.append(proc2.stdout or '')
                if proc2.returncode != 0:
                    logs.append(f"[CREATEPL] exit code {proc2.returncode}, stderr={proc2.stderr}")
            except subprocess.TimeoutExpired:
                logs.append('[CREATEPL] timeout')
            except Exception as e:
                logs.append(f'[CREATEPL] failed: {e}')

        pdb_candidates = []
        for p in ('model_1.pdb', 'top1.pdb', 'complex_1.pdb'):
            absp = os.path.join(workdir, p)
            if os.path.exists(absp):
                pdb_candidates.append(absp)
        if not pdb_candidates:
            pdb_candidates = [p for p in glob.glob(os.path.join(workdir, '*.pdb'))
                              if os.path.basename(p).lower() not in ('receptor.pdb', 'peptide.pdb')]

        best_model_score = None
        for p in pdb_candidates:
            val = _parse_score_from_pdb(p)
            if val is not None and (best_model_score is None or val < best_model_score):
                best_model_score = val
        if best_model_score is not None:
            best_score = best_model_score

    return best_score, '\n'.join(logs)


def main():
    parser = argparse.ArgumentParser(description='Batch evaluate peptide-receptor affinity (HDOCK) with auto-CPU parallelism.')
    parser.add_argument('--data_root', default='/root/autodl-tmp/train_data_augmented_random_neighbor', help='根目录（按样本目录）')
    parser.add_argument('--out_json', default='/root/autodl-tmp/Peptide_3D/data/augmented_random_neighbor_hdock_scores.json', help='输出 JSON 路径')
    parser.add_argument('--work_root', default='/root/autodl-tmp/hdock_batch', help='工作目录根')
    parser.add_argument('--hdock_bin', default='/root/autodl-fs/HDOCKlite/hdock', help='hdock 可执行文件')
    parser.add_argument('--createpl_bin', default='/root/autodl-fs/HDOCKlite/createpl', help='createpl 可执行文件')
    parser.add_argument('--timeout', type=int, default=900, help='每个样本超时（秒）')
    parser.add_argument('--skip_existing', action='store_true', help='已有结果则跳过')
    parser.add_argument('--cpu_scale', type=float, default=1.0, help='CPU占用倍数, 默认为1')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f'ERROR: data root not found: {data_root}', file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.work_root, exist_ok=True)

    try:
        with open(args.out_json, 'r') as fh:
            results = json.load(fh)
    except Exception:
        results = {}

    sample_dirs = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    tasks = []
    for d in sample_dirs:
        rec = d / 'receptor.pdb'
        pep = d / 'peptide.pdb'
        if rec.exists() and pep.exists():
            key = os.path.abspath(str(d))
            # 当开启 --skip_existing 时，只跳过已经有“正常 score”的样本；
            # 如果之前跑过但 score 是 None 或解析失败，则会重新计算。
            if args.skip_existing and key in results:
                prev = results.get(key)
                prev_score = None
                if isinstance(prev, dict):
                    prev_score = prev.get('score')
                # 仅当 score 非 None 且能成功转成 float 时才认为是“已完成”
                if prev_score is not None:
                    try:
                        float(prev_score)
                        continue
                    except Exception:
                        pass
            tasks.append((key, str(rec), str(pep)))

    cpu_count = os.cpu_count() or 1
    workers = max(1, int(cpu_count * args.cpu_scale))
    print(f'CPU count {cpu_count}, using {workers} parallel workers')

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_key = {}
        for key, rec, pep in tasks:
            sample_workdir = os.path.join(args.work_root, os.path.basename(key))
            future = executor.submit(run_hdock_pair, sample_workdir, rec, pep, args.hdock_bin, args.createpl_bin, args.timeout)
            future_to_key[future] = key

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                score, logs = future.result()
            except Exception as e:
                score = None
                logs = f'[ERROR] exception {e}'
            results[key] = {'score': score, 'log': logs}
            with open(args.out_json, 'w', encoding='utf-8') as fh:
                json.dump(results, fh, indent=2, ensure_ascii=False)
            print(f'[{len(results)}/{len(sample_dirs)}] {key}, score={score}')

    print(f'Done. results saved to {args.out_json}')


if __name__ == '__main__':
    main()


'''

python /root/autodl-tmp/Peptide_3D/utils/evaluate/compute_affinity_auto_cpu.py \
  --data_root /root/autodl-tmp/train_data_augmented_random_neighbor \
  --out_json /root/autodl-tmp/Peptide_3D/data/train_data_augmented_random_neighbor_hdock.json \
  --work_root /root/autodl-tmp/tmp_hdock_auto_cpu \
  --hdock_bin /root/autodl-fs/HDOCKlite/hdock \
  --createpl_bin /root/autodl-fs/HDOCKlite/createpl \
  --timeout 900 \
  --cpu_scale 1.0
  --skip_existing

nohup python /root/autodl-tmp/Peptide_3D/utils/evaluate/compute_affinity_auto_cpu.py \
  --data_root /root/autodl-tmp/train_data_seq_aug_perturbation \
  --out_json /root/autodl-tmp/Peptide_3D/data/train_data_seq_aug_perturbation_hdock.json \
  --work_root /root/autodl-tmp/tmp_hdock_auto_cpu \
  --hdock_bin /root/autodl-fs/HDOCKlite/hdock \
  --createpl_bin /root/autodl-fs/HDOCKlite/createpl \
  --timeout 900 \
  --cpu_scale 1.0 \
  --skip_existing
  > /root/autodl-tmp/Peptide_3D/utils/evaluate/compute_affinity_auto_cpu_1.log 2>&1 &
echo $! > /root/autodl-tmp/Peptide_3D/utils/evaluate/compute_affinity_auto_cpu.pid
'''
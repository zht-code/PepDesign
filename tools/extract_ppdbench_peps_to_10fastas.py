#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from collections import defaultdict

AA3_TO_AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    # 常见变体/修饰（遇到时尽量不崩）
    "MSE":"M",  # selenomethionine
    "SEC":"U",  # selenocysteine
    "PYL":"O",  # pyrrolysine
    "ASX":"B",  # Asp/Asn
    "GLX":"Z",  # Glu/Gln
    "UNK":"X",
}

def pdb_to_chain_sequences(pdb_path: str):
    """
    从 PDB 文件提取每条链的序列。
    返回: dict(chain_id -> sequence_str)
    """
    # chain -> {(resseq, icode): resname}
    chain_res = defaultdict(dict)

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            # PDB 固定列格式（1-based）
            # resname: 18-20, chain: 22, resseq: 23-26, icode: 27
            resname = line[17:20].strip().upper()
            chain_id = line[21].strip() or "_"
            resseq_str = line[22:26].strip()
            icode = line[26].strip() or ""

            if not resseq_str:
                continue
            try:
                resseq = int(resseq_str)
            except ValueError:
                continue

            key = (resseq, icode)
            # 同一残基会出现多行 ATOM（不同原子），我们只存一次
            if key not in chain_res[chain_id]:
                chain_res[chain_id][key] = resname

    chain_seq = {}
    for chain_id, resmap in chain_res.items():
        # 按残基编号排序（icode 也纳入排序，避免乱序）
        keys_sorted = sorted(resmap.keys(), key=lambda x: (x[0], x[1]))
        seq = []
        for k in keys_sorted:
            aa3 = resmap[k]
            aa1 = AA3_TO_AA1.get(aa3, "X")
            seq.append(aa1)
        if seq:
            chain_seq[chain_id] = "".join(seq)

    return chain_seq

def find_protein_folders(root_dir: str):
    """
    root_dir 下的一级子目录默认为 protein_id 文件夹。
    过滤掉非目录。
    """
    subs = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p):
            subs.append(name)
    subs.sort()
    return subs

def main():
    ap = argparse.ArgumentParser(
        description="Extract peptide sequences from PPDbench test set and save into 10 FASTA files."
    )
    ap.add_argument("--root", type=str, required=True,
                    help="PPDbench root dir, e.g. /root/autodl-tmp/PPDbench")
    ap.add_argument("--out", type=str, required=True,
                    help="Output directory for FASTA files")
    ap.add_argument("--cands_dir", type=str, default="multi_cands",
                    help="Candidates folder name under each protein folder (default: multi_cands)")
    ap.add_argument("--pattern", type=str, default=r"pep_(\d{2})\.pdb",
                    help=r"Regex filename pattern (default: pep_(\d{2})\.pdb )")
    args = ap.parse_args()

    root_dir = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    pep_re = re.compile(args.pattern)

    # 准备 10 个输出句柄：01..10
    handles = {}
    for i in range(1, 11):
        tag = f"{i:02d}"
        fasta_path = os.path.join(out_dir, f"pep_{tag}.fasta")
        handles[tag] = open(fasta_path, "w", encoding="utf-8")

    protein_ids = find_protein_folders(root_dir)

    total_written = {f"{i:02d}": 0 for i in range(1, 11)}
    missing = 0
    bad = 0

    for pid in protein_ids:
        cands_path = os.path.join(root_dir, pid, args.cands_dir)
        if not os.path.isdir(cands_path):
            continue

        for fname in sorted(os.listdir(cands_path)):
            m = pep_re.fullmatch(fname)
            if not m:
                continue
            idx = m.group(1)  # "01".."10" (如果你目录里超出也会写到对应 idx；本脚本默认只开了01-10)
            pdb_path = os.path.join(cands_path, fname)

            if idx not in handles:
                # 只写 01..10；其他忽略
                continue

            if not os.path.isfile(pdb_path):
                missing += 1
                continue

            try:
                chain_seqs = pdb_to_chain_sequences(pdb_path)
                if not chain_seqs:
                    bad += 1
                    continue

                # 每条链单独写一条（避免多个链拼在一起导致歧义）
                for chain_id, seq in chain_seqs.items():
                    header = f">{pid}|{fname}|chain={chain_id}"
                    handles[idx].write(header + "\n")
                    # 80列换行
                    for j in range(0, len(seq), 80):
                        handles[idx].write(seq[j:j+80] + "\n")
                    total_written[idx] += 1

            except Exception as e:
                bad += 1
                # 不中断全局
                continue

    for h in handles.values():
        h.close()

    print("Done.")
    for i in range(1, 11):
        tag = f"{i:02d}"
        print(f"pep_{tag}.fasta  entries: {total_written[tag]}")
    print(f"missing pdb files: {missing}")
    print(f"failed/empty parses: {bad}")
    print(f"output dir: {out_dir}")

if __name__ == "__main__":
    main()



'''

python /root/autodl-tmp/Peptide_3D/tools/extract_ppdbench_peps_to_10fastas.py \
  --root /root/autodl-tmp/PPDbench \
  --out  /root/autodl-tmp/Peptide_3D/data/PPDbench_pep_fastas


'''
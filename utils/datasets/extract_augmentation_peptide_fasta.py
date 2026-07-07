#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import Optional, List

from Bio.PDB import PDBParser, Polypeptide
from Bio.PDB.Polypeptide import PPBuilder

# 将三字母氨基酸转一字母，遇到非常规残基返回 'X'
def res_to_one(resname: str) -> str:
    resname = resname.strip().upper()
    try:
        return Polypeptide.three_to_one(resname)
    except KeyError:
        # 常见非常规氨基酸可在这里做些映射；默认给 X
        mapping = {
            "SEC": "U",  # Selenocysteine
            "PYL": "O",  # Pyrrolysine
        }
        return mapping.get(resname, "X")

def extract_sequence_from_pdb(pdb_path: Path) -> Optional[str]:
    """
    尝试从 peptide.pdb 中提取氨基酸序列。
    优先通过 PPBuilder 构建多肽；若失败则按残基顺序手动转换。
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("peptide", str(pdb_path))
    except Exception:
        return None

    # 1) 先尝试 PPBuilder（基于几何距离连续性）
    ppb = PPBuilder()
    seqs: List[str] = []
    try:
        for pp in ppb.build_peptides(structure, aa_only=False):
            seqs.append(str(pp.get_sequence()))
    except Exception:
        seqs = []

    # 2) 如果 PPBuilder 拿不到，就手动遍历残基
    if not seqs:
        seq_chars = []
        for model in structure:
            for chain in model:
                for res in chain:
                    # 过滤水和杂原子；Polypeptide.is_aa(res, standard=True) 只接收标准AA
                    if Polypeptide.is_aa(res, standard=False):
                        seq_chars.append(res_to_one(res.get_resname()))
        if seq_chars:
            seqs = ["".join(seq_chars)]

    if not seqs:
        return None

    # 多条多肽链的情况：通常 peptide.pdb 只有一条；若有多条，默认拼接
    return "".join(seqs)

def write_fasta(path: Path, seq_id: str, seq: str):
    path.write_text(f">{seq_id}\n{seq}\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="从每个样本目录的 peptide.pdb 提取多肽序列并写入 peptide.fa")
    ap.add_argument("--root",
                    default="/root/autodl-tmp/train_data_augmentation",
                    help="训练集根目录，形如根下有 1A1M_1/peptide.pdb")
    ap.add_argument("--pdb-name",
                    default="peptide.pdb",
                    help="多肽 PDB 文件名（默认 peptide.pdb）")
    ap.add_argument("--combined-out",
                    default="/root/autodl-tmp/Peptide_3D/data/train_augmentation_peptides.fasta",
                    help="可选：把所有样本汇总为一个 fasta（例如 /root/autodl-tmp/all_peptides.fa）")
    args = ap.parse_args()

    root = Path(args.root)
    assert root.is_dir(), f"root 不存在或不是目录: {root}"

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    total = 0
    ok = 0
    combined_records = []

    for d in subdirs:
        total += 1
        pdb_path = d / args.pdb_name
        if not pdb_path.exists():
            # 兼容大小写或备选文件名
            candidates = [
                d / "peptide.PDB",
                d / "PEPTIDE.pdb",
                d / "ligand.pdb",
            ]
            for c in candidates:
                if c.exists():
                    pdb_path = c
                    break
        if not pdb_path.exists():
            print(f"[WARN] 跳过 {d.name}: 找不到 {args.pdb_name}")
            continue

        seq = extract_sequence_from_pdb(pdb_path)
        if not seq or len(seq) == 0:
            print(f"[WARN] 跳过 {d.name}: 无法从 {pdb_path.name} 提取序列")
            continue

        # 写入当前目录的 peptide.fa
        fasta_path = d / "peptide.fa"
        write_fasta(fasta_path, d.name, seq)
        ok += 1

        if args.combined_out:
            combined_records.append((d.name, seq))

    # 写汇总
    if args.combined_out:
        outp = Path(args.combined_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as fh:
            for sid, s in combined_records:
                fh.write(f">{sid}\n{s}\n")
        print(f"[OK] 汇总写入: {outp} （{len(combined_records)} 条）")

    print(f"[DONE] 子目录总数={total}，成功提取={ok}，输出文件名=peptide.fa")

if __name__ == "__main__":
    main()

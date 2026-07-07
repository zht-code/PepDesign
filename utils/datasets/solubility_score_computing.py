#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import freesasa
from Bio.PDB import PDBParser, Polypeptide
import os
import json
import traceback
from collections import Counter

# ---------- 配置 ----------
DATASET_PATH = '/root/autodl-tmp/train_data_augmentation'
OUTPUT_FILE  = '/root/autodl-tmp/Peptide_3D/data/train_data_augmentation_solubility_scores.json'
BATCH_SIZE   = 20  # 每处理多少个样本落盘一次

# ---------- 工具与映射 ----------
ION_SOLVENT = {
    "HOH","WAT","DOD","NA","K","CL","CA","MG","MN","FE","ZN","CU","CO","NI","CD",
    "SO4","PO4","GOL","MPD","PEG","PG4","PGE","TRS","MES","HEP","ACT","EDO","FMT","TAR"
}
CAPPING = {"ACE","NME"}  # 端基

NONSTD_TO_ONE = {  # 常见非标->一字母
    "MSE":"M",  # Selenomethionine
    "SEC":"U",  # Selenocysteine
    "PYL":"O",  # Pyrrolysine
    "HYP":"P",  # Hydroxyproline（保守映射）
    "PCA":"Q",  # Pyroglutamate（保守映射）
}

# Kyte–Doolittle（按一字母）
KD = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,
    'E':-3.5,'Q':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
    'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,
    'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,
    'U':-1.0,'O':-1.0,'X':0.0  # 兜底
}

def clamp(v, lo=0.0, hi=1e9):
    return max(lo, min(hi, v))

# ---------- 解析 PDB -> 氨基酸一字母序列 ----------
def residue_is_peptidic(res) -> bool:
    """是否可视作多肽残基：具备主链原子(N/CA/C)至少两者，且不是溶剂/离子/端基。"""
    resname = res.get_resname().strip().upper()
    if resname in ION_SOLVENT or resname in CAPPING:
        return False
    atom_names = {a.get_name().strip().upper() for a in res.get_atoms()}
    return len(atom_names & {"N","CA","C"}) >= 2 or Polypeptide.is_aa(res, standard=False)

def three_to_one_relaxed(resname: str) -> str:
    r3 = resname.strip().upper()
    if r3 in NONSTD_TO_ONE:
        return NONSTD_TO_ONE[r3]
    try:
        return Polypeptide.three_to_one(r3)
    except Exception:
        return "X"

def get_one_letter_seq_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("peptide", pdb_file)
    seq = []
    # 取第一个MODEL
    model = next(structure.get_models())
    for chain in model:
        for residue in chain:
            if not residue_is_peptidic(residue):
                continue
            aa1 = three_to_one_relaxed(residue.get_resname())
            seq.append(aa1)
    return "".join(seq)

# ---------- FreeSASA ----------
def calculate_sasa_with_freesasa(pdb_file):
    structure = freesasa.Structure(pdb_file)
    result = freesasa.calc(structure)
    return float(result.totalArea())  # Å^2

# ---------- 疏水性 & 溶解性 ----------
def average_hydropathy(seq_one_letter: str) -> float:
    if not seq_one_letter:
        return 0.0
    vals = [KD.get(a, 0.0) for a in seq_one_letter]
    return sum(vals) / len(vals)

def predict_solubility(avg_hydropathy, total_sasa, L=None):
    raw = total_sasa - 100.0*avg_hydropathy
    # 简单缩放到[0,100]（经验参数，可按你的数据再调）
    score = 100.0 / (1.0 + pow(2.71828, -(raw-600.0)/150.0))
    return float(score)


# ---------- I/O ----------
def load_existing_results(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

def save_to_file(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved data to {filename}")

# ---------- 主流程（单层目录版本） ----------
def process_and_save_single_level(root, batch_size=BATCH_SIZE, output_file=OUTPUT_FILE):
    results = load_existing_results(output_file)
    processed_count = 0
    batch_number = 1
    skipped = []

    for entry in sorted(os.listdir(root)):
        id_dir = os.path.join(root, entry)
        if not os.path.isdir(id_dir):
            skipped.append(f"Skipped {entry}: Not a directory")
            continue

        # 只按你的结构找 peptide.pdb
        pdb_file = os.path.join(id_dir, "peptide.pdb")
        if not os.path.exists(pdb_file):
            skipped.append(f"Skipped {entry}: No peptide.pdb")
            continue

        if entry in results:
            # 已有记录则跳过（你可以注释掉以强制重算）
            skipped.append(f"Skipped {entry}: Already processed")
            continue

        try:
            seq = get_one_letter_seq_from_pdb(pdb_file)
            if not seq:
                raise ValueError("Empty peptide sequence parsed.")

            sasa = calculate_sasa_with_freesasa(pdb_file)
            avg_kd = average_hydropathy(seq)
            solubility_val = predict_solubility(avg_kd, sasa)

            # === 输出 JSON 结构：只保留 solubility_score，符合你的目标格式 ===
            results[entry] = {
                "solubility_score": float(solubility_val)
            }

            processed_count += 1
            print(f"Processed {entry}: len={len(seq)}, SASA={sasa:.2f}, KDavg={avg_kd:.3f}, score={solubility_val:.2f}")

            if processed_count % batch_size == 0:
                save_to_file(results, output_file)
                print(f"Completed and saved batch {batch_number}")
                batch_number += 1
                results = load_existing_results(output_file)

        except Exception as e:
            msg = f"Error processing {entry}: {str(e)}\n{traceback.format_exc()}"
            print(msg)
            skipped.append(msg)

    # 收尾保存
    if processed_count % batch_size != 0:
        save_to_file(results, output_file)
        print("Saved final batch")

    print("All data processed and saved.")
    print("\nSkipped items:")
    for s in skipped:
        print(s)

# ---------- 运行 ----------
if __name__ == "__main__":
    process_and_save_single_level(DATASET_PATH)

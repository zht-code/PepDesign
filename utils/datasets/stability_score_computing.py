#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Bio import PDB
from Bio.SeqUtils import ProtParam
from Bio import SeqIO
import os
import json
import math
import traceback

# ====== 配置 ======
DATASET_PATH = '/root/autodl-tmp/train_data_augmentation'
FASTA_FILE   = '/root/autodl-tmp/Peptide_3D/data/train_augmentation_peptides.fasta'
OUTPUT_FILE  = '/root/autodl-tmp/Peptide_3D/data/train_data_augmentation_stability_scores.json'
BATCH_SIZE   = 20

# ====== 非标残基与杂质过滤 ======
ION_SOLVENT = {
    "HOH","WAT","DOD","NA","K","CL","CA","MG","MN","FE","ZN","CU","CO","NI","CD",
    "SO4","PO4","GOL","MPD","PEG","PG4","PGE","TRS","MES","HEP","ACT","EDO","FMT","TAR"
}
CAPPING = {"ACE","NME"}
NONSTD_TO_ONE = {
    "MSE":"M",  # Selenomethionine
    "SEC":"C",  # Selenocysteine -> C（为适配 ProtParam 20AA）
    "PYL":"K",  # Pyrrolysine    -> K
    "HYP":"P",  # Hydroxyproline -> P
    "PCA":"Q",  # Pyroglutamate  -> Q
}

ALLOWED = set("ACDEFGHIKLMNPQRSTVWY")
SANITIZE_MAP = {
    "U":"C", "O":"K", "B":"D", "Z":"E", "J":"L", "X":"G"
}

def sanitize_seq(seq: str) -> str:
    s = []
    for ch in seq.upper():
        if ch in ALLOWED:
            s.append(ch)
        elif ch in SANITIZE_MAP:
            s.append(SANITIZE_MAP[ch])
        else:
            s.append("A")  # 兜底为 A，避免 ProtParam 报错
    return "".join(s)

# ====== 从 FASTA 获取序列 ======
def get_sequence_from_fasta(fasta_file: str, target_id: str):
    if not os.path.exists(fasta_file):
        return None
    for record in SeqIO.parse(fasta_file, "fasta"):
        if record.id == target_id:
            return str(record.seq)
    return None

# ====== 从 PDB 解析一字母序列（鲁棒，支持 HETATM 非标残基）======
def residue_is_peptidic(res) -> bool:
    resname = res.get_resname().strip().upper()
    if resname in ION_SOLVENT or resname in CAPPING:
        return False
    atom_names = {a.get_name().strip().upper() for a in res.get_atoms()}
    # 具备主链原子至少两者，基本可判断为多肽型残基
    return len(atom_names & {"N","CA","C"}) >= 2

def three_to_one_relaxed(resname: str) -> str:
    r3 = resname.strip().upper()
    if r3 in NONSTD_TO_ONE:
        return NONSTD_TO_ONE[r3]
    try:
        return PDB.Polypeptide.three_to_one(r3)
    except Exception:
        return "X"

def get_sequence_from_pdb(pdb_file: str) -> str:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pep", pdb_file)
    seq = []
    model = next(structure.get_models())  # 取第一个 model
    for chain in model:
        for res in chain:
            if not residue_is_peptidic(res):
                continue
            aa1 = three_to_one_relaxed(res.get_resname())
            seq.append(aa1)
    return "".join(seq)

# ====== 计算 Instability Index 与稳定性分数 ======
def calculate_instability_index(sequence: str) -> float:
    clean = sanitize_seq(sequence)
    if not clean:
        raise ValueError("Empty sequence after sanitization.")
    pa = ProtParam.ProteinAnalysis(clean)
    return float(pa.instability_index())

def instability_to_score(ii: float) -> float:
    """
    将 Instability Index（II）映射为 0–100 的稳定性分数（高=更稳定）。
    40 为经典阈值：II=40 -> 约 50 分。
    这个 logistic 的坡度可按需微调（分母里的“5”越小越陡）。
    """
    return float(100.0 / (1.0 + math.exp((ii - 40.0)/5.0)))

# ====== I/O ======
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

# ====== 主流程：单层目录遍历 ======
def process_and_save_single_level(folder_path: str, fasta_file: str, batch_size: int = BATCH_SIZE,
                                  output_file: str = OUTPUT_FILE):
    processed_count = 0
    batch_number = 1
    results = load_existing_results(output_file)
    skipped = []

    for sample_id in sorted(os.listdir(folder_path)):
        id_dir = os.path.join(folder_path, sample_id)
        if not os.path.isdir(id_dir):
            skipped.append(f"Skipped {sample_id}: Not a directory")
            continue

        if sample_id in results:
            skipped.append(f"Skipped {sample_id}: Already processed")
            continue

        pdb_file = os.path.join(id_dir, "peptide.pdb")
        if not os.path.exists(pdb_file):
            skipped.append(f"Skipped {sample_id}: No peptide.pdb")
            continue

        try:
            # 1) 先试 FASTA
            seq = get_sequence_from_fasta(fasta_file, sample_id)

            # 2) FASTA 没有就用 PDB 解析
            if not seq:
                seq = get_sequence_from_pdb(pdb_file)

            if not seq:
                raise ValueError("No sequence found in FASTA or parsed from PDB.")

            ii = calculate_instability_index(seq)
            score = instability_to_score(ii)

            # 如需直接输出 Instability Index 当作 stability_score（低=好），改成：
            # score = ii

            results[sample_id] = {
                "stability_score": float(round(score, 6))
            }

            processed_count += 1
            print(f"Processed {sample_id}: II={ii:.2f}, stability_score={score:.2f}")

            if processed_count % batch_size == 0:
                save_to_file(results, output_file)
                print(f"Completed and saved batch {batch_number}")
                batch_number += 1
                results = load_existing_results(output_file)

        except Exception as e:
            err = f"Error processing {sample_id}: {str(e)}\n{traceback.format_exc()}"
            print(err)
            skipped.append(err)

    if processed_count % batch_size != 0:
        save_to_file(results, output_file)
        print("Saved final batch")

    print("All data processed and saved.")
    print("\nSkipped items:")
    for s in skipped:
        print(s)

# ====== 运行 ======
if __name__ == "__main__":
    process_and_save_single_level(DATASET_PATH, FASTA_FILE, batch_size=BATCH_SIZE, output_file=OUTPUT_FILE)

# extract_peptide_fasta.py
import os
import sys
import warnings

# 可选进度条
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

warnings.filterwarnings("ignore")

# 加到你的项目根，确保能 import ProteinChain
sys.path.append("/root/autodl-tmp/Peptide_3D")

from model.esm.utils.structure.protein_chain import ProteinChain  # noqa: E402

def wrap_fasta(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))

def extract_sequence_from_pdb(pdb_path: str) -> str:
    chain = ProteinChain.from_pdb(pdb_path)
    # ProteinChain.sequence 已是单字母序列；未知残基通常会映射为 'X'
    return chain.sequence

def main():
    # ===== 根据需要修改根目录 =====
    train_root = "/root/autodl-tmp/train_data"

    # 枚举子目录
    subdirs = sorted([d for d in os.listdir(train_root)
                      if os.path.isdir(os.path.join(train_root, d))])

    if not subdirs:
        print(f"[WARN] No subdirectories found in {train_root}")
        return

    ok, skipped = 0, 0
    for sub in tqdm(subdirs, desc="Extracting FASTA", unit="dir"):
        dir_path = os.path.join(train_root, sub)
        pdb_path = os.path.join(dir_path, "peptide.pdb")
        # fasta_path = os.path.join(dir_path, f"{sub}.fasta")
        fasta_path = os.path.join(dir_path, f"peptide.fasta")

        if not os.path.exists(pdb_path):
            skipped += 1
            continue

        try:
            seq = extract_sequence_from_pdb(pdb_path)
            if not seq:
                skipped += 1
                continue
            with open(fasta_path, "w") as f:
                f.write(f">{sub}\n")
                f.write(wrap_fasta(seq) + "\n")
            ok += 1
        except Exception as e:
            skipped += 1
            print(f"[ERROR] {sub}: {e}")

    print(f"[DONE] Wrote {ok} FASTA files. Skipped {skipped}.")

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
'''
从RCSB下载的蛋白-多肽复合物PDB
只保留氨基酸的结构信息；
多肽与受体最小距离阈值小于5（Å），默认5.0；
蛋白70以上，多肽1-50之间.
'''
# import os
# import sys
# import argparse
# import shutil
# import warnings
# from typing import List, Tuple, Optional, Set

# try:
#     from Bio.PDB import PDBParser, PDBIO, Select
#     from Bio.PDB.Polypeptide import is_aa
# except Exception as e:
#     print("Biopython is required. Please ensure 'biopython' is installed in your environment.", file=sys.stderr)
#     raise

# # ----------------------------
# # Configuration / Constants
# # ----------------------------

# STANDARD_AA: Set[str] = {
#     'ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY',
#     'HIS','ILE','LEU','LYS','MET','PHE','PRO','SER',
#     'THR','TRP','TYR','VAL'
# }

# DEFAULT_IDEAL_PEPTIDE_BOND = 1.33  # Å, typical C-N peptide bond length
# DEFAULT_TOL = 0.5                  # Å tolerance

# # ----------------------------
# # Helpers
# # ----------------------------

# def is_standard_aa_residue(res) -> bool:
#     """Return True if residue is a standard amino acid (ATOM record, not hetero) with standard 20 AA name."""
#     hetflag = res.get_id()[0]
#     return hetflag == ' ' and res.get_resname() in STANDARD_AA

# def residue_list_from_chain(chain) -> List:
#     """List of standard AA residues (excludes hetero/water)."""
#     return [r for r in chain.get_residues() if is_standard_aa_residue(r)]

# def chain_has_only_standard_aas(chain) -> bool:
#     """Return True if the chain contains ONLY standard amino-acid residues (no hetero, no modified AAs)."""
#     # If the chain includes any residue that is not a standard AA (including HETATM like HOH, ions, capping groups), return False
#     for r in chain.get_residues():
#         hetflag = r.get_id()[0]
#         if hetflag != ' ':
#             return False
#         if r.get_resname() not in STANDARD_AA:
#             return False
#     return True

# def peptide_bonds_intact(chain, ideal: float = DEFAULT_IDEAL_PEPTIDE_BOND, tol: float = DEFAULT_TOL) -> bool:
#     """
#     Check whether the peptide chain is intact by verifying that each consecutive
#     residue pair has a carbonyl C (res i) to backbone N (res i+1) distance within [ideal - tol, ideal + tol].
#     """
#     residues = residue_list_from_chain(chain)
#     if len(residues) < 2:
#         return False
#     low, high = ideal - tol, ideal + tol
#     for i in range(len(residues) - 1):
#         ri = residues[i]
#         rj = residues[i + 1]
#         if 'C' not in ri or 'N' not in rj:
#             return False
#         try:
#             d = ri['C'] - rj['N']
#         except Exception:
#             return False
#         if not (low <= d <= high):
#             return False
#     return True

# def choose_peptide_chain(structure, min_len: int, max_len: int, ideal: float, tol: float):
#     """
#     From all chains across the first model, select a single peptide chain:
#       - ONLY standard amino acids (no modified residues, no hetero)
#       - length in [min_len, max_len]
#       - passes peptide_bonds_intact()
#     Strategy: choose the longest valid chain (tie-breaker: most atoms).
#     Return (model_id, chain_id) or (None, None) if none.
#     """
#     best = (None, None, -1, -1)  # (model, chain, length, atom_count)
#     for model in structure:
#         for chain in model:
#             if not chain_has_only_standard_aas(chain):
#                 continue
#             residues = residue_list_from_chain(chain)
#             L = len(residues)
#             if L < min_len or L > max_len:
#                 continue
#             if not peptide_bonds_intact(chain, ideal=ideal, tol=tol):
#                 continue
#             atom_count = sum(1 for _ in chain.get_atoms())
#             if L > best[2] or (L == best[2] and atom_count > best[3]):
#                 best = (model.id, chain.id, L, atom_count)
#     return best[0], best[1]

# def has_any_receptor_residue(structure, peptide_model_id, peptide_chain_id) -> bool:
#     """Check if there is at least one receptor residue (standard AA, excluding the peptide chain)."""
#     for model in structure:
#         for chain in model:
#             # exclude selected peptide chain (model + id must match)
#             if model.id == peptide_model_id and chain.id == peptide_chain_id:
#                 continue
#             for r in chain.get_residues():
#                 if is_standard_aa_residue(r):
#                     return True
#     return False

# # ----------------------------
# # PDB Selectors for writing
# # ----------------------------

# class BaseSelect(Select):
#     def accept_atom(self, atom):
#         # Handle alternate locations: keep ' ' or 'A'
#         altloc = atom.get_altloc()
#         if altloc not in (' ', 'A'):
#             return 0
#         return 1

# class PeptideSelect(BaseSelect):
#     def __init__(self, peptide_model_id, peptide_chain_id):
#         super().__init__()
#         self.pm = peptide_model_id
#         self.pc = peptide_chain_id

#     def accept_model(self, model):
#         return model.id == self.pm

#     def accept_chain(self, chain):
#         return chain.id == self.pc

#     def accept_residue(self, residue):
#         # Only standard amino-acid residues
#         return is_standard_aa_residue(residue)

# class ReceptorSelect(BaseSelect):
#     def __init__(self, peptide_model_id, peptide_chain_id):
#         super().__init__()
#         self.pm = peptide_model_id
#         self.pc = peptide_chain_id

#     def accept_chain(self, chain):
#         # allow all chains except the selected peptide
#         if chain.get_parent().id == self.pm and chain.id == self.pc:
#             return 0
#         return 1

#     def accept_residue(self, residue):
#         # Only keep standard protein residues (exclude hetero, waters, nucleic acids, etc.)
#         return is_standard_aa_residue(residue)

# # ----------------------------
# # Core processing
# # ----------------------------

# def process_pdb(pdb_path: str,
#                 min_len: int,
#                 max_len: int,
#                 ideal: float,
#                 tol: float,
#                 quiet: bool = True) -> Tuple[bool, Optional[str]]:
#     """
#     Process a single PDB file:
#       - identify valid peptide chain
#       - write peptide.pdb and receptor.pdb (protein residues only, no waters/hetero)
#       - if no valid peptide, delete the directory and return (False, reason)
#       - return (True, None) if successful
#     """
#     parser = PDBParser(PERMISSIVE=True, QUIET=quiet)
#     try:
#         structure = parser.get_structure("struct", pdb_path)
#     except Exception as e:
#         return (False, f"Parse error: {e}")

#     pm, pc = choose_peptide_chain(structure, min_len=min_len, max_len=max_len, ideal=ideal, tol=tol)
#     if pm is None or pc is None:
#         # delete directory
#         d = os.path.dirname(pdb_path)
#         try:
#             shutil.rmtree(d)
#             return (False, "No valid peptide found; directory removed.")
#         except Exception as e:
#             return (False, f"No valid peptide; failed to remove dir: {e}")

#     # Ensure receptor has at least one residue
#     if not has_any_receptor_residue(structure, pm, pc):
#         # no receptor -> not a complex as desired
#         d = os.path.dirname(pdb_path)
#         try:
#             shutil.rmtree(d)
#             return (False, "No receptor residues found after excluding peptide; directory removed.")
#         except Exception as e:
#             return (False, f"No receptor; failed to remove dir: {e}")

#     out_dir = os.path.dirname(pdb_path)
#     peptide_out = os.path.join(out_dir, "peptide.pdb")
#     receptor_out = os.path.join(out_dir, "receptor.pdb")

#     io = PDBIO()
#     # Write peptide
#     io.set_structure(structure)
#     try:
#         io.save(peptide_out, select=PeptideSelect(pm, pc))
#     except Exception as e:
#         # fail hard: delete dir
#         d = os.path.dirname(pdb_path)
#         try:
#             shutil.rmtree(d)
#             return (False, f"Failed to write peptide.pdb: {e}; directory removed.")
#         except Exception as e2:
#             return (False, f"Failed to write peptide.pdb: {e}; also failed to remove dir: {e2}")

#     # Write receptor
#     io.set_structure(structure)
#     try:
#         io.save(receptor_out, select=ReceptorSelect(pm, pc))
#     except Exception as e:
#         # fail hard: delete dir
#         d = os.path.dirname(pdb_path)
#         try:
#             shutil.rmtree(d)
#             return (False, f"Failed to write receptor.pdb: {e}; directory removed.")
#         except Exception as e2:
#             return (False, f"Failed to write receptor.pdb: {e}; also failed to remove dir: {e2}")

#     return (True, None)

# def find_pdb_files(base_dir: str) -> List[str]:
#     """
#     Find .pdb files under base_dir. Assumes each target directory contains a single X.pdb to process.
#     """
#     pdbs = []
#     for root, dirs, files in os.walk(base_dir):
#         for f in files:
#             if f.lower().endswith(".pdb"):
#                 pdbs.append(os.path.join(root, f))
#     return sorted(pdbs)

# # ----------------------------
# # CLI
# # ----------------------------

# def main():
#     parser = argparse.ArgumentParser(
#         description="Process protein-peptide complex PDB files: remove waters/hetero, extract receptor and clean peptide."
#     )
#     parser.add_argument("--base_dir", type=str, default="/root/autodl-tmp/Peptide_3D/data/train_data",
#                         help="Base directory containing subfolders with the raw PDB (e.g., /root/autodl-tmp/Peptide_3D/data/train_data)")
#     parser.add_argument("--min_peptide_len", type=int, default=2, help="Minimum peptide length (number of residues).")
#     parser.add_argument("--max_peptide_len", type=int, default=50, help="Maximum peptide length (number of residues).")
#     parser.add_argument("--ideal_bond", type=float, default=DEFAULT_IDEAL_PEPTIDE_BOND, help="Ideal peptide C-N bond length in Å.")
#     parser.add_argument("--tolerance", type=float, default=DEFAULT_TOL, help="± tolerance (Å) around the ideal bond length.")
#     parser.add_argument("--dry_run", action="store_true", help="List what would be processed without writing or deleting.")
#     args = parser.parse_args()

#     base = os.path.abspath(args.base_dir)
#     if not os.path.isdir(base):
#         print(f"ERROR: base_dir does not exist: {base}", file=sys.stderr)
#         sys.exit(2)

#     pdb_files = find_pdb_files(base)
#     if not pdb_files:
#         print("No PDB files found.")
#         sys.exit(0)

#     print(f"Found {len(pdb_files)} PDB files. Processing...")

#     valid_pairs = 0
#     removed_dirs = 0
#     failures = 0
#     details = []

#     for pdb_path in pdb_files:
#         # We assume each directory is to be considered one "case"
#         if args.dry_run:
#             print(f"[DRY-RUN] Would process: {pdb_path}")
#             continue

#         ok, msg = process_pdb(
#             pdb_path,
#             min_len=args.min_peptide_len,
#             max_len=args.max_peptide_len,
#             ideal=args.ideal_bond,
#             tol=args.tolerance,
#             quiet=True
#         )
#         case_id = os.path.basename(os.path.dirname(pdb_path))
#         if ok:
#             valid_pairs += 1
#             details.append((case_id, "OK", ""))
#             print(f"[OK] {case_id}: receptor.pdb & peptide.pdb saved.")
#         else:
#             # Directory may have been removed inside process_pdb
#             dir_path = os.path.dirname(pdb_path)
#             if not os.path.isdir(dir_path):
#                 removed_dirs += 1
#                 details.append((case_id, "REMOVED", msg or ""))
#                 print(f"[REMOVED] {case_id}: {msg}")
#             else:
#                 failures += 1
#                 details.append((case_id, "FAILED", msg or ""))
#                 print(f"[FAILED] {case_id}: {msg}")

#     # Summary
#     print("\n====== Summary ======")
#     print(f"有效蛋白-多肽对数量 (valid protein–peptide pairs): {valid_pairs}")
#     print(f"已删除目录 (removed dirs without valid peptide): {removed_dirs}")
#     print(f"失败数量 (failures): {failures}")

#     # Also write a CSV summary next to base_dir
#     try:
#         import csv
#         csv_path = os.path.join(base, "processing_summary.csv")
#         with open(csv_path, "w", newline="", encoding="utf-8") as fo:
#             writer = csv.writer(fo)
#             writer.writerow(["case_id", "status", "note"])
#             writer.writerows(details)
#         print(f"Summary CSV: {csv_path}")
#     except Exception as e:
#         print(f"Warning: failed to write summary CSV: {e}", file=sys.stderr)

# if __name__ == "__main__":
#     main()

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量处理 PDB：
1) 去除水分子与杂原子，只保留标准 20 种氨基酸的 ATOM 记录；
2) 自动判定受体链与多肽链（多肽默认长度 2-50）；
3) 仅保留与受体最小距离 <= 指定阈值(默认 5Å) 的最近一条多肽；
4) 保存 receptor.pdb 与 peptide.pdb；
5) 若无有效多肽，支持删除该目录；
6) 打印统计总共有多少对有效蛋白-多肽。
"""

# import os
# import argparse
# import shutil
# import math
# from pathlib import Path

# import numpy as np

# from Bio.PDB import PDBParser, PDBIO, Select

# # 可选：用 SciPy KDTree 更快（你的环境有 scipy==1.7.1）
# try:
#     from scipy.spatial import cKDTree as KDTree
#     HAVE_SCIPY = True
# except Exception:
#     HAVE_SCIPY = False

# STD_AA = {
#     "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
#     "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
# }

# class ChainsSelect(Select):
#     """
#     仅写出指定链；仅保留标准氨基酸（ATOM，非水、非杂原子）。
#     可选择去掉氢（默认保留）。
#     """
#     def __init__(self, chain_ids, drop_hydrogen=False):
#         super().__init__()
#         self.chain_ids = set(chain_ids)
#         self.drop_hydrogen = drop_hydrogen

#     def accept_chain(self, chain):
#         return chain.id in self.chain_ids

#     def accept_residue(self, residue):
#         # residue.id[0] == ' ' 表示标准多肽/蛋白残基（ATOM）
#         if residue.id[0] != ' ':
#             return 0
#         if residue.get_resname().strip() not in STD_AA:
#             return 0
#         return 1

#     def accept_atom(self, atom):
#         if self.drop_hydrogen and atom.element == 'H':
#             return 0
#         return 1


# def is_standard_protein_chain(chain, min_len=1, only_std=True):
#     """
#     判断链是否由标准氨基酸组成（ATOM，非杂原，非水），长度>=min_len
#     """
#     residues = [r for r in chain.get_residues() if r.id[0] == ' ']
#     if len(residues) < min_len:
#         return False

#     if only_std:
#         for r in residues:
#             if r.get_resname().strip() not in STD_AA:
#                 return False
#     return True


# def chain_length(chain):
#     """标准残基计数"""
#     return sum(1 for r in chain.get_residues() if r.id[0] == ' ' and r.get_resname().strip() in STD_AA)


# def get_chain_coords(chain, drop_hydrogen=True):
#     coords = []
#     for atom in chain.get_atoms():
#         # 排除非标准残基的原子
#         res = atom.get_parent()
#         if res.id[0] != ' ':
#             continue
#         if res.get_resname().strip() not in STD_AA:
#             continue
#         if drop_hydrogen and atom.element == 'H':
#             continue
#         coords.append(atom.get_coord())
#     if not coords:
#         return np.zeros((0,3), dtype=float)
#     return np.vstack(coords)


# def min_distance_between_coords(A, B):
#     """两个坐标集合之间的最小距离（Å）"""
#     if A.size == 0 or B.size == 0:
#         return math.inf

#     if HAVE_SCIPY:
#         tree = KDTree(A)
#         dists, _ = tree.query(B, k=1)
#         return float(np.min(dists))
#     else:
#         # 退化实现（可能较慢，但可用）
#         # 使用分块避免内存爆炸
#         min_d = math.inf
#         block = 2048
#         for i in range(0, B.shape[0], block):
#             Bb = B[i:i+block]
#             # (m,1,3) vs (1,n,3) -> (m,n,3)
#             diff = Bb[:, None, :] - A[None, :, :]
#             d2 = np.sum(diff*diff, axis=2)  # (m,n)
#             md = float(np.min(np.sqrt(d2)))
#             if md < min_d:
#                 min_d = md
#         return min_d


# def decide_receptor_and_candidates(model, peptide_max_len=50):
#     """
#     - receptor_chains: 选最长链 max_len 及与其同量级的长链（长度 >= max(max_len*0.5, peptide_max_len+1)）
#     - candidate_peptides: 2 <= len <= peptide_max_len，且全部是标准氨基酸
#     """
#     chains = [ch for ch in model.get_chains()]
#     # 仅保留“看起来是蛋白/肽的链”
#     protein_like = [ch for ch in chains if is_standard_protein_chain(ch, min_len=1, only_std=True)]
#     if not protein_like:
#         return [], []

#     lengths = {ch.id: chain_length(ch) for ch in protein_like}
#     # 最大长度
#     max_len = max(lengths.values()) if lengths else 0
#     # 划分阈值：大于等于 max(max_len*0.5, peptide_max_len+1) 视为受体链
#     rec_thresh = max(int(max_len * 0.5), peptide_max_len + 1)

#     receptor_chains = [ch for ch in protein_like if lengths[ch.id] >= rec_thresh]

#     # 如果误判导致 receptor 为空，则退而求其次：取最长链为受体
#     if not receptor_chains:
#         longest = max(protein_like, key=lambda c: lengths[c.id])
#         receptor_chains = [longest]

#     candidate_peptides = [
#         ch for ch in protein_like
#         if 2 <= lengths[ch.id] <= peptide_max_len and ch not in receptor_chains
#     ]

#     return receptor_chains, candidate_peptides


# def write_chains(structure, chains, out_path, drop_hydrogen=False):
#     io = PDBIO()
#     io.set_structure(structure)
#     sel = ChainsSelect([ch.id for ch in chains], drop_hydrogen=drop_hydrogen)
#     io.save(str(out_path), sel)


# def process_one_pdb(pdb_path, distance_cutoff=5.0, peptide_max_len=50, drop_hydrogen=False, verbose=True):
#     """
#     返回 (ok, receptor_path, peptide_path, reason)
#     ok=True 表示找到了有效多肽并写出文件
#     """
#     pdb_path = Path(pdb_path)
#     out_dir = pdb_path.parent
#     receptor_out = out_dir / "receptor.pdb"
#     peptide_out  = out_dir / "peptide.pdb"

#     parser = PDBParser(QUIET=True)
#     try:
#         structure = parser.get_structure(pdb_path.stem, str(pdb_path))
#     except Exception as e:
#         return False, None, None, f"解析失败: {e}"

#     model = structure[0]  # 取第一个 model

#     receptor_chains, candidate_peptides = decide_receptor_and_candidates(model, peptide_max_len=peptide_max_len)

#     if not receptor_chains:
#         return False, None, None, "未能确定受体链"

#     if not candidate_peptides:
#         return False, None, None, "未发现候选多肽（仅标准AA且长度在阈值内）"

#     # 受体坐标
#     rec_coords = np.vstack([get_chain_coords(ch, drop_hydrogen=drop_hydrogen) for ch in receptor_chains]) \
#                  if receptor_chains else np.zeros((0,3))

#     if rec_coords.size == 0:
#         return False, None, None, "受体坐标为空"

#     # 在候选中找距离最近且 <= cutoff 的一条
#     best = (math.inf, None)
#     for pep in candidate_peptides:
#         pep_coords = get_chain_coords(pep, drop_hydrogen=drop_hydrogen)
#         dmin = min_distance_between_coords(rec_coords, pep_coords)
#         if verbose:
#             print(f"  - 候选多肽链 {pep.id}: 残基数={chain_length(pep)}, 最小距离={dmin:.3f} Å")
#         if dmin < best[0]:
#             best = (dmin, pep)

#     if best[1] is None or best[0] > float(distance_cutoff):
#         return False, None, None, f"无多肽满足距离阈值 ≤ {distance_cutoff} Å（最小={best[0]:.3f} Å）"

#     peptide_chain = [best[1]]

#     # 写出文件（仅所选链；且仅标准残基，剔除水与杂原）
#     try:
#         write_chains(structure, receptor_chains, receptor_out, drop_hydrogen=drop_hydrogen)
#         write_chains(structure, peptide_chain,  peptide_out,  drop_hydrogen=drop_hydrogen)
#     except Exception as e:
#         return False, None, None, f"写文件失败: {e}"

#     return True, str(receptor_out), str(peptide_out), f"OK，多肽链={peptide_chain[0].id}，最小距离={best[0]:.3f} Å"


# def find_pdb_file_in_dir(d):
#     """
#     在目录 d 中寻找 .pdb 文件：优先使用与目录同名的 pdb，其次任意一个 pdb
#     """
#     d = Path(d)
#     if not d.is_dir():
#         return None
#     # 优先：dirname/dirname.pdb
#     pref = d / (d.name + ".pdb")
#     if pref.exists():
#         return str(pref)
#     # 其次：目录中的第一个 .pdb
#     for p in d.glob("*.pdb"):
#         return str(p)
#     return None


# def main():
#     ap = argparse.ArgumentParser(description="过滤蛋白-多肽PDB，导出receptor.pdb与peptide.pdb")
#     ap.add_argument("--base", default="/root/autodl-tmp/Peptide_3D/data/train_data", help="包含多个结构子目录的根目录，如 /root/autodl-tmp/Peptide_3D/data/train_data")
#     ap.add_argument("--distance", type=float, default=5.0, help="多肽与受体最小距离阈值（Å），默认5.0")
#     ap.add_argument("--peptide-max-len", type=int, default=50, help="多肽最大残基数，默认50")
#     ap.add_argument("--drop-hydrogen", action="store_true", help="写出PDB时去掉氢原子")
#     ap.add_argument("--delete-invalid", action="store_true", help="删除无有效多肽的结构目录（危险操作！请先试跑）")
#     args = ap.parse_args()

#     base = Path(args.base)
#     if not base.is_dir():
#         print(f"[错误] 基目录不存在：{base}")
#         return

#     subdirs = [p for p in base.iterdir() if p.is_dir()]
#     total = len(subdirs)
#     valid_cnt = 0
#     invalid_cnt = 0

#     print(f"开始处理：{base}（共 {total} 个子目录）")
#     for d in sorted(subdirs):
#         pdb_file = find_pdb_file_in_dir(d)
#         if pdb_file is None:
#             print(f"[跳过] {d.name}: 未找到PDB文件")
#             if args.delete_invalid:
#                 try:
#                     shutil.rmtree(d)
#                     print(f"  已删除目录（无PDB）：{d}")
#                 except Exception as e:
#                     print(f"  删除失败：{e}")
#             invalid_cnt += 1
#             continue

#         print(f"[处理] {d.name}: {pdb_file}")
#         ok, rec_out, pep_out, info = process_one_pdb(
#             pdb_file,
#             distance_cutoff=args.distance,
#             peptide_max_len=args.peptide_max_len,
#             drop_hydrogen=args.drop_hydrogen,
#             verbose=True,
#         )

#         if ok:
#             print(f"  -> 成功：{info}")
#             print(f"     receptor: {rec_out}")
#             print(f"     peptide : {pep_out}")
#             valid_cnt += 1
#         else:
#             print(f"  -> 无效：{info}")
#             invalid_cnt += 1
#             if args.delete_invalid:
#                 try:
#                     shutil.rmtree(d)
#                     print(f"  已删除目录：{d}")
#                 except Exception as e:
#                     print(f"  删除失败：{e}")

#     print("\n====== 汇总 ======")
#     print(f"总目录数：{total}")
#     print(f"有效蛋白-多肽对：{valid_cnt}")
#     print(f"无效/已删：{invalid_cnt}")
#     print("==================")

# if __name__ == "__main__":
#     main()



# -----------------------------------------------------------------
'''
统计所有不含receptor.pdb 和 peptide.pdb文件的数量，没有的删除子文件夹
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from pathlib import Path
# import shutil

# # 根目录
# BASE = Path("/root/autodl-tmp/Peptide_3D/data/train_data")
# # True = 立即删除；False = 仅预览将删除哪些目录
# DO_DELETE = True

# def main():
#     if not BASE.is_dir():
#         print(f"[错误] 目录不存在：{BASE}")
#         return

#     total = 0
#     kept  = 0
#     fixed = 0
#     to_delete = []

#     for d in sorted(p for p in BASE.iterdir() if p.is_dir()):
#         total += 1

#         r_ok = d / "receptor.pdb"
#         r_typo = d / "recpetor.pdb"   # 常见拼写误差
#         p_ok = d / "peptide.pdb"

#         # 若只有 recpetor.pdb，自动改名为 receptor.pdb
#         if r_typo.is_file() and not r_ok.is_file():
#             try:
#                 r_typo.rename(r_ok)
#                 fixed += 1
#                 print(f"[修复文件名] {d.name}/recpetor.pdb -> receptor.pdb")
#             except Exception as e:
#                 print(f"[修复失败] {d.name}: {e}")

#         receptor_exists = r_ok.is_file()
#         peptide_exists  = p_ok.is_file()

#         if receptor_exists and peptide_exists:
#             kept += 1
#         else:
#             to_delete.append(d)

#     # 删除
#     removed = 0
#     for d in to_delete:
#         if DO_DELETE:
#             try:
#                 shutil.rmtree(d)
#                 removed += 1
#                 print(f"[删除] {d}")
#             except Exception as e:
#                 print(f"[删除失败] {d}: {e}")
#         else:
#             print(f"[将删除] {d}")

#     # 汇总
#     print("\n==== 汇总 ====")
#     print(f"总子目录数：{total}")
#     print(f"已修复 receptor 文件名：{fixed}")
#     print(f"保留（已有 receptor.pdb + peptide.pdb）：{kept}")
#     print(f"{'已删除' if DO_DELETE else '待删除'} 的目录数：{removed if DO_DELETE else len(to_delete)}")

# if __name__ == "__main__":
#     main()


# -------------------------------------------------------------
'''
筛选出Propedia和PepBDB数据集中与RCSB数据集id不同的数据
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import argparse
# import re
# from pathlib import Path
# import shutil

# def extract_id(name: str):
#     """从名称开头提取4位PDB id；找不到返回None。"""
#     m = re.match(r'^([0-9A-Za-z]{4})', name)
#     return m.group(1).upper() if m else None

# def collect_train_ids(train_dir: Path):
#     """收集 train_data 下子文件夹名的 4位 id 集合（大写）。"""
#     ids = set()
#     for p in train_dir.iterdir():
#         if p.is_dir():
#             pid = extract_id(p.name)
#             if pid:
#                 ids.add(pid)
#     return ids

# def delete_target(p: Path, do_delete: bool):
#     """删除文件或文件夹；返回(bool成功, 原因或'OK')。"""
#     if not do_delete:
#         return True, "DRYRUN"
#     try:
#         if p.is_dir():
#             shutil.rmtree(p)
#         else:
#             p.unlink()
#         return True, "OK"
#     except Exception as e:
#         return False, str(e)

# def main():
#     ap = argparse.ArgumentParser(
#         description="删除 complex/ 和 pepbdb/ 中与 train_data 同 id 的文件/文件夹"
#     )
#     ap.add_argument("--train",  default="/root/autodl-tmp/Peptide_3D/data/train_data")
#     ap.add_argument("--complex", default="/root/autodl-tmp/Peptide_3D/data/complex")
#     ap.add_argument("--pepbdb",  default="/root/autodl-tmp/Peptide_3D/data/pepbdb")
#     ap.add_argument("--delete", action="store_true",
#                     help="执行真实删除（默认只预览）")
#     args = ap.parse_args()

#     train = Path(args.train)
#     complex_dir = Path(args.complex)
#     pepbdb_dir = Path(args.pepbdb)

#     if not train.is_dir():
#         print(f"[错误] train_data 不存在：{train}")
#         return

#     train_ids = collect_train_ids(train)
#     print(f"[信息] train_data 子目录数：{len(list(train.iterdir()))}")
#     print(f"[信息] 提取到的唯一 PDB id 数量：{len(train_ids)}")

#     # 处理 complex
#     c_total = 0
#     c_dup = 0
#     c_deleted = 0
#     if complex_dir.is_dir():
#         print(f"\n[扫描] complex: {complex_dir}")
#         for p in sorted(complex_dir.iterdir()):
#             c_total += 1
#             pid = extract_id(p.name)
#             if pid and pid in train_ids:
#                 c_dup += 1
#                 ok, msg = delete_target(p, args.delete)
#                 if ok:
#                     c_deleted += 1 if args.delete else 0
#                     print(f"  {'[删除]' if args.delete else '[将删]'} {p.name}  (id={pid})")
#                 else:
#                     print(f"  [失败] {p.name}  (id={pid})  原因：{msg}")
#     else:
#         print(f"[警告] complex 目录不存在：{complex_dir}")

#     # 处理 pepbdb
#     p_total = 0
#     p_dup = 0
#     p_deleted = 0
#     if pepbdb_dir.is_dir():
#         print(f"\n[扫描] pepbdb: {pepbdb_dir}")
#         for p in sorted(pepbdb_dir.iterdir()):
#             p_total += 1
#             pid = extract_id(p.name)
#             if pid and pid in train_ids:
#                 p_dup += 1
#                 ok, msg = delete_target(p, args.delete)
#                 if ok:
#                     p_deleted += 1 if args.delete else 0
#                     print(f"  {'[删除]' if args.delete else '[将删]'} {p.name}  (id={pid})")
#                 else:
#                     print(f"  [失败] {p.name}  (id={pid})  原因：{msg}")
#     else:
#         print(f"[警告] pepbdb 目录不存在：{pepbdb_dir}")

#     # 汇总
#     print("\n======== 汇总 ========")
#     print(f"complex 总条目：{c_total}，重复：{c_dup}，{'已删' if args.delete else '待删'}：{c_deleted if args.delete else c_dup}")
#     print(f"pepbdb  总条目：{p_total}，重复：{p_dup}，{'已删' if args.delete else '待删'}：{p_deleted if args.delete else p_dup}")
#     print(f"删除模式：{'执行删除' if args.delete else '预览(未删除)'}")
#     print("======================")

# if __name__ == "__main__":
#     main()
# -------------------------------------------------------------
'''
将Propedia数据集中的pdb格式转换成子文件夹/id.pdb格式(3304条)
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import argparse
# import re
# from pathlib import Path
# import shutil

# def pick_preferred(files):
#     """
#     在同一 ID 的多个文件中挑选要保留的那个：
#     1) 优先包含 '_C_A'（不区分大小写）
#     2) 否则按文件名（不区分大小写）字典序最小
#     """
#     files = sorted(files, key=lambda p: p.name.lower())
#     for f in files:
#         if "_C_A" in f.name.upper():
#             return f
#     return files[0]

# def main():
#     ap = argparse.ArgumentParser(description="重命名并归档 complex/XXXX_*.pdb -> complex/XXXX/XXXX.pdb")
#     ap.add_argument("--complex", default="/root/autodl-tmp/Peptide_3D/data/complex",
#                     help="complex 目录路径")
#     ap.add_argument("--apply", action="store_true",
#                     help="执行改动（默认仅预览）")
#     ap.add_argument("--overwrite", action="store_true",
#                     help="若目标 complex/XXXX/XXXX.pdb 已存在则覆盖（默认不覆盖）")
#     args = ap.parse_args()

#     base = Path(args.complex)
#     if not base.is_dir():
#         print(f"[错误] 目录不存在：{base}")
#         return

#     # 收集：按 4 位 ID 分组（从文件名开头提取）
#     groups = {}
#     for p in base.iterdir():
#         if not p.is_file():
#             continue
#         if p.suffix.lower() != ".pdb":
#             continue
#         m = re.match(r"^([0-9A-Za-z]{4})_(.+)\.pdb$", p.name)
#         if not m:
#             continue
#         pid = m.group(1).upper()
#         groups.setdefault(pid, []).append(p)

#     total_ids = len(groups)
#     moved_cnt = 0
#     deleted_cnt = 0
#     skipped_cnt = 0

#     print(f"[信息] 扫描到 {total_ids} 个 ID 组（形如 XXXX_*.pdb）")

#     for pid, files in sorted(groups.items()):
#         keeper = pick_preferred(files)
#         dest_dir = base / pid
#         dest_file = dest_dir / f"{pid}.pdb"

#         # 目标目录
#         if args.apply:
#             dest_dir.mkdir(parents=True, exist_ok=True)

#         # 处理保留文件
#         if dest_file.exists():
#             if args.overwrite:
#                 if args.apply:
#                     dest_file.unlink()
#                     print(f"[覆盖] 已删除旧文件：{dest_file}")
#                 else:
#                     print(f"[预览-覆盖] 将覆盖：{dest_file}")
#             else:
#                 print(f"[跳过移动] 目标已存在且未指定 --overwrite：{dest_file}")
#                 skipped_cnt += 1
#         else:
#             if args.apply:
#                 try:
#                     shutil.move(str(keeper), str(dest_file))
#                     moved_cnt += 1
#                     print(f"[移动] {keeper.name} -> {dest_file}")
#                 except Exception as e:
#                     print(f"[失败] 无法移动 {keeper} 到 {dest_file}：{e}")
#             else:
#                 print(f"[预览-移动] {keeper.name} -> {dest_file}")

#         # 删除其他同 ID 的文件
#         for f in files:
#             if f == keeper:
#                 # 如果上面没有移动（因为已存在且未覆盖），保留原文件不删
#                 if not args.apply or dest_file.exists() and not args.overwrite and keeper.exists():
#                     pass
#                 continue
#             if args.apply:
#                 try:
#                     f.unlink()
#                     deleted_cnt += 1
#                     print(f"[删除] {f.name}")
#                 except Exception as e:
#                     print(f"[失败] 无法删除 {f}: {e}")
#             else:
#                 print(f"[预览-删除] {f.name}")

#     print("\n====== 汇总 ======")
#     print(f"ID 组总数：{total_ids}")
#     print(f"已移动：{moved_cnt}  | 跳过移动：{skipped_cnt}  | 已删除多余文件：{deleted_cnt}")
#     print(f"模式：{'执行改动' if args.apply else '预览（未实际修改）'}")
#     print("==================")

# if __name__ == "__main__":
#     main()

# -------------------------------------------------------------
'''
将PepBDB数据集中的pdb格式转换成id/id.pdb格式(1488条)，并删除重复的id文件夹，只保留一个
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import argparse
# import re
# from pathlib import Path
# import shutil

# def extract_id(name: str):
#     """从目录名开头提取4位PDB id；找不到返回None。"""
#     m = re.match(r'^([0-9A-Za-z]{4})', name)
#     return m.group(1).upper() if m else None

# def has_receptor(dirpath: Path):
#     """该子目录是否包含 receptor.pdb（或常见误拼 recpetor.pdb）"""
#     r = dirpath / "receptor.pdb"
#     typo = dirpath / "recpetor.pdb"
#     if r.is_file():
#         return r
#     if typo.is_file():
#         return typo  # 也接受误拼
#     return None

# def pick_preferred(pid: str, entries: list[Path]) -> Path:
#     """在同一ID的多个子目录中选择要保留的那个。
#        1) 优先目录名恰好等于 'XXXX_A'（忽略大小写）
#        2) 否则按目录名（不区分大小写）字典序最小
#     """
#     # 过滤必须含 receptor 文件的目录
#     candidates = [d for d in entries if has_receptor(d)]
#     if not candidates:
#         return None
#     # 1) 优先 XXXX_A
#     for d in candidates:
#         if d.name.upper() == f"{pid}_A":
#             return d
#     # 2) 词典序最小
#     return sorted(candidates, key=lambda p: p.name.lower())[0]

# def safe_delete(p: Path, do_delete: bool):
#     if not do_delete:
#         print(f"[预览-删除] {p}")
#         return True
#     try:
#         if p.is_dir():
#             shutil.rmtree(p)
#         else:
#             p.unlink()
#         print(f"[删除] {p}")
#         return True
#     except Exception as e:
#         print(f"[删除失败] {p}: {e}")
#         return False

# def main():
#     ap = argparse.ArgumentParser(
#         description="将 pepbdb/XXXX_*/receptor.pdb 规范为 pepbdb/XXXX/XXXX.pdb，仅保留一个，删除其它重复。"
#     )
#     ap.add_argument("--pepbdb", default="/root/autodl-tmp/Peptide_3D/data/pepbdb",
#                     help="PepBDB 目录")
#     ap.add_argument("--apply", action="store_true",
#                     help="执行改动（默认仅预览）")
#     ap.add_argument("--overwrite", action="store_true",
#                     help="若目标 pepbdb/XXXX/XXXX.pdb 已存在则覆盖（默认不覆盖）")
#     ap.add_argument("--delete-extras", action="store_true",
#                     help="删除同ID下未被选中目录（默认不删除，只移动保留者）")
#     ap.add_argument("--no-prune", action="store_true",
#                     help="不要删除移动后变空的源目录（默认会尝试清理空目录）")
#     args = ap.parse_args()

#     base = Path(args.pepbdb)
#     if not base.is_dir():
#         print(f"[错误] 目录不存在：{base}")
#         return

#     # 分组：按开头4位ID分组
#     groups: dict[str, list[Path]] = {}
#     for d in base.iterdir():
#         if not d.is_dir():
#             continue
#         pid = extract_id(d.name)
#         if not pid:
#             continue
#         # 只关心包含 receptor 的目录（否则无需参与“保留/删除”逻辑）
#         if has_receptor(d):
#             groups.setdefault(pid, []).append(d)

#     print(f"[信息] 找到 {len(groups)} 个包含 receptor 的 ID 组。")

#     moved_cnt = 0
#     overwritten = 0
#     deleted_dirs = 0
#     skipped_exists = 0

#     for pid, dirs in sorted(groups.items()):
#         keeper = pick_preferred(pid, dirs)
#         if keeper is None:
#             continue

#         # 目标路径 pepbdb/XXXX/XXXX.pdb
#         dest_dir = base / pid
#         dest_file = dest_dir / f"{pid}.pdb"

#         # 源 receptor 文件（容忍误拼）
#         src = has_receptor(keeper)
#         assert src is not None

#         # 创建目标目录
#         if args.apply:
#             dest_dir.mkdir(parents=True, exist_ok=True)

#         # 移动或覆盖
#         if dest_file.exists():
#             if args.overwrite:
#                 if args.apply:
#                     try:
#                         dest_file.unlink()
#                         overwritten += 1
#                         print(f"[覆盖] 删除旧目标：{dest_file}")
#                     except Exception as e:
#                         print(f"[覆盖失败] 无法删除 {dest_file}: {e}")
#                         continue
#                 else:
#                     print(f"[预览-覆盖] 将覆盖：{dest_file}")
#             else:
#                 print(f"[跳过移动] 目标已存在且未指定 --overwrite：{dest_file}")
#                 skipped_exists += 1
#                 # 即便不移动，也继续处理“删除多余目录”的逻辑
#         else:
#             if args.apply:
#                 try:
#                     shutil.move(str(src), str(dest_file))
#                     moved_cnt += 1
#                     print(f"[移动] {src} -> {dest_file}")
#                 except Exception as e:
#                     print(f"[失败] 无法移动 {src} 到 {dest_file}: {e}")
#             else:
#                 print(f"[预览-移动] {src} -> {dest_file}")

#         # 可选：删除非保留的重复目录
#         for d in dirs:
#             if d == keeper:
#                 # 清理空目录（如果 receptor 被移动走且目录空了）
#                 if not args.no_prune:
#                     try:
#                         # 仅当目录为空时删除
#                         if args.apply:
#                             if d.exists() and not any(d.iterdir()):
#                                 d.rmdir()
#                                 print(f"[清理空目录] {d}")
#                         else:
#                             print(f"[预览-清理空目录] {d}（若变空）")
#                     except Exception as e:
#                         print(f"[清理失败] {d}: {e}")
#                 continue
#             if args.delete_extras:
#                 safe_delete(d, args.apply)
#                 if args.apply:
#                     deleted_dirs += 1
#             else:
#                 print(f"[保留重复目录] {d}（未加 --delete-extras）")

#     print("\n====== 汇总 ======")
#     print(f"移动完成：{moved_cnt} 个")
#     print(f"目标已存在而跳过：{skipped_exists} 个")
#     print(f"覆盖旧目标：{overwritten} 个")
#     print(f"删除重复目录：{deleted_dirs} 个（需要 --delete-extras）")
#     print(f"模式：{'执行改动' if args.apply else '预览（未实际修改）'}")
#     print("==================")

# if __name__ == "__main__":
#     main()
# -------------------------------------------------------------
'''
将三个数据集合成并训练集
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import argparse
# from pathlib import Path
# import shutil

# def unique_dest(base: Path, name: str, hint: str = None) -> Path:
#     """
#     在 base 下为 name 生成不冲突的新目录名：
#     先尝试 name；若存在，用 name__{hint}；若仍存在，追加 __2, __3...
#     """
#     # 第一次尝试：原名
#     cand = base / name
#     if not cand.exists():
#         return cand
#     # 第二次：带来源提示
#     suffix = f"__{hint}" if hint else "__dup"
#     cand = base / f"{name}{suffix}"
#     if not cand.exists():
#         return cand
#     # 递增后缀
#     i = 2
#     while True:
#         cand = base / f"{name}{suffix}__{i}"
#         if not cand.exists():
#             return cand
#         i += 1

# def move_or_copy(src: Path, dst: Path, mode: str, dry_run: bool):
#     if dry_run:
#         print(f"[预览-{mode}] {src}  ->  {dst}")
#         return True, None
#     try:
#         if mode == "move":
#             shutil.move(str(src), str(dst))
#         else:
#             shutil.copytree(str(src), str(dst))
#         return True, None
#     except Exception as e:
#         return False, e

# def main():
#     ap = argparse.ArgumentParser(
#         description="把多个目录的一级子文件夹合并到一个新目录中（支持移动/拷贝，冲突处理可选）"
#     )
#     ap.add_argument(
#         "--sources",
#         nargs="+",
#         default=[
#             "/root/autodl-tmp/Peptide_3D/data/RCSB",
#             "/root/autodl-tmp/Peptide_3D/data/complex",
#             "/root/autodl-tmp/Peptide_3D/data/pepbdb",
#         ],
#         help="来源目录列表（可传多个）"
#     )
#     ap.add_argument(
#         "--dest",
#         default="/root/autodl-tmp/Peptide_3D/data/train_data",
#         help="目标目录"
#     )
#     ap.add_argument(
#         "--mode",
#         choices=["move", "copy"],
#         default="move",
#         help="合并方式：move=移动（省空间），copy=拷贝（保留源）"
#     )
#     ap.add_argument(
#         "--on-conflict",
#         choices=["skip", "overwrite", "rename"],
#         default="rename",
#         help="当目标已存在同名子文件夹时的处理：skip=跳过，overwrite=先删后写，rename=自动改名（默认）"
#     )
#     ap.add_argument(
#         "--dry-run",
#         action="store_true",
#         default=True,
#         help="预览模式（默认开启）。加 --no-dry-run 才执行改动"
#     )
#     ap.add_argument(
#         "--no-dry-run",
#         dest="dry_run",
#         action="store_false",
#         help="关闭预览，执行实际改动"
#     )
#     args = ap.parse_args()

#     sources = [Path(p) for p in args.sources]
#     dest = Path(args.dest)
#     dry = args.dry_run

#     # 准备目标目录
#     if dry:
#         print(f"[预览] 将创建目标目录：{dest}")
#     else:
#         dest.mkdir(parents=True, exist_ok=True)

#     total_dirs = 0
#     moved = 0
#     skipped = 0
#     overwritten = 0
#     renamed = 0
#     errors = 0

#     for s in sources:
#         if not s.is_dir():
#             print(f"[警告] 来源目录不存在或不是目录：{s}")
#             continue

#         print(f"\n[扫描] 来源：{s}")
#         for sub in sorted(p for p in s.iterdir() if p.is_dir()):
#             total_dirs += 1
#             name = sub.name
#             target = dest / name

#             if target.exists():
#                 if args.on_conflict == "skip":
#                     print(f"[跳过] 已存在同名目录：{target}")
#                     skipped += 1
#                     continue
#                 elif args.on_conflict == "overwrite":
#                     if dry:
#                         print(f"[预览-覆盖] 将删除：{target}")
#                     else:
#                         try:
#                             shutil.rmtree(target)
#                             overwritten += 1
#                             print(f"[覆盖] 已删除旧目录：{target}")
#                         except Exception as e:
#                             print(f"[覆盖失败] {target}: {e}")
#                             errors += 1
#                             continue
#                 elif args.on_conflict == "rename":
#                     new_target = unique_dest(dest, name, hint=s.name)
#                     if new_target.name != name:
#                         renamed += 1
#                     target = new_target

#             ok, err = move_or_copy(sub, target, args.mode, dry)
#             if ok:
#                 moved += 1
#             else:
#                 errors += 1
#                 print(f"[失败] {args.mode} {sub} -> {target}: {err}")

#     print("\n====== 汇总 ======")
#     print(f"来源目录数：{len(sources)}")
#     print(f"扫描到子文件夹：{total_dirs}")
#     print(f"{'移动' if args.mode=='move' else '拷贝'}成功：{moved}")
#     print(f"跳过：{skipped}，覆盖：{overwritten}，改名：{renamed}，错误：{errors}")
#     print(f"目标目录：{dest}")
#     print(f"模式：{'预览（未实际修改）' if dry else '执行改动'}")
#     print("==================")

# if __name__ == "__main__":
#     main()

# -----------------------------------------------------------------
'''
使用mmseqs2对训练集进行去相似度高于50%的聚类，得到非冗余训练集
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path
import shutil

from Bio.PDB import PDBParser, PPBuilder

def extract_sequence_from_pdb(pdb_file: Path) -> str:
    """
    从单链PDB中提取氨基酸序列；若是多链，只取第一条链的连续多肽片段并拼接。
    要求：文件只包含标准AA（你之前的清洗流程即如此）。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file.stem, str(pdb_file))
    model = structure[0]
    ppb = PPBuilder()
    seq_parts = []

    # 通常 peptide.pdb 只有一条链；为稳妥遍历全部链并拼接
    for chain in model:
        peptides = ppb.build_peptides(chain)
        # 可能被断点分成多个片段，这里简单拼接
        for pep in peptides:
            seq_parts.append(str(pep.get_sequence()))
    seq = "".join(seq_parts)
    return seq

def run_cmd(cmd, cwd=None):
    print("[CMD]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"命令执行失败（返回码 {proc.returncode}）: {' '.join(cmd)}")
    return proc

def main():
    ap = argparse.ArgumentParser(description="用 MMseqs2 对 train_data 的序列按身份阈值去冗余，并删除非代表目录。")
    ap.add_argument("--base", default="/root/autodl-tmp/Peptide_3D/data/train_data",
                    help="train_data 根目录")
    ap.add_argument("--target", choices=["peptide", "receptor"], default="peptide",
                    help="对哪个PDB抽序列去冗余（默认 peptide）")
    ap.add_argument("--min-seq-id", type=float, default=0.5,
                    help="序列身份阈值（默认0.5=50%）")
    ap.add_argument("--coverage", type=float, default=0.8,
                    help="覆盖度阈值 -c（默认0.8，避免局部比对误聚类）")
    ap.add_argument("--cov-mode", type=int, default=None,
                    help="MMseqs --cov-mode（可不设；如需严格可用2或0，视任务而定）")
    ap.add_argument("--mmseqs", default="/root/autodl-fs/mmseqs/bin/mmseqs",
                    help="mmseqs 可执行文件路径（默认在PATH里）")
    ap.add_argument("--tmpdir", default="mmseqs_tmp",
                    help="MMseqs临时目录")
    ap.add_argument("--outprefix", default="train_nr50",
                    help="MMseqs输出前缀名（会生成 *_rep_seq.fasta 等）")
    ap.add_argument("--dry-run", action="store_true",
                    help="仅预览：不删除任何目录（默认不加此项则根据 --delete 执行）")
    ap.add_argument("--delete", action="store_true",
                    help="删除被判为冗余的子目录（危险操作，建议先配合 --dry-run 预览）")
    ap.add_argument("--list-dir", default=".",
                    help="将保留/删除清单写到该目录（默认当前工作目录）")
    args = ap.parse_args()

    base = Path(args.base)
    if not base.is_dir():
        print(f"[错误] 基目录不存在：{base}")
        sys.exit(1)

    # 1) 汇总需要抽序列的 PDB 路径
    pdb_name = "peptide.pdb" if args.target == "peptide" else "receptor.pdb"
    items = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        pdb_path = d / pdb_name
        if pdb_path.is_file():
            items.append((d.name, pdb_path))

    if not items:
        print(f"[错误] 未在 {base} 下找到任何 {pdb_name}")
        sys.exit(1)

    print(f"[信息] 待处理子目录数：{len(items)}（目标文件：{pdb_name}）")

    # 2) 抽取序列，写FASTA
    fasta_path = Path(args.list_dir) / f"{args.outprefix}_all.fasta"
    kept_for_fasta = []
    with fasta_path.open("w") as fw:
        for dir_name, pdb_path in items:
            try:
                seq = extract_sequence_from_pdb(pdb_path).strip()
            except Exception as e:
                print(f"[警告] 解析失败，跳过 {pdb_path}: {e}")
                continue
            if not seq:
                print(f"[警告] 空序列，跳过 {pdb_path}")
                continue
            # 用子目录名作为FASTA ID，后续能直接映射回去
            fw.write(f">{dir_name}\n{seq}\n")
            kept_for_fasta.append(dir_name)

    if not kept_for_fasta:
        print("[错误] 抽取序列为空，退出。")
        sys.exit(1)

    print(f"[信息] 已写入 FASTA：{fasta_path}（序列数：{len(kept_for_fasta)}）")

    # 3) 运行 MMseqs2 easy-linclust 做去冗余
    tmpdir = Path(args.tmpdir)
    outprefix = Path(args.list_dir) / args.outprefix
    cmd = [args.mmseqs, "easy-linclust", str(fasta_path), str(outprefix), str(tmpdir),
           "--min-seq-id", str(args.min-seq-id if hasattr(args, 'min-seq-id') else args.min_seq_id)]
    # 上面一行属性名带连字符不方便，修正：
    cmd = [args.mmseqs, "easy-linclust", str(fasta_path), str(outprefix), str(tmpdir),
           "--min-seq-id", str(args.min_seq_id), "-c", str(args.coverage)]
    if args.cov_mode is not None:
        cmd += ["--cov-mode", str(args.cov_mode)]

    try:
        run_cmd(cmd)
    except Exception as e:
        print(f"[错误] 运行 MMseqs 失败：{e}")
        sys.exit(1)

    rep_fasta = Path(f"{outprefix}_rep_seq.fasta")
    cluster_tsv = Path(f"{outprefix}_cluster.tsv")
    if not rep_fasta.is_file():
        print(f"[错误] 未找到代表序列文件：{rep_fasta}")
        sys.exit(1)
    if not cluster_tsv.is_file():
        print(f"[警告] 未找到聚类映射：{cluster_tsv}（easy-linclust通常会有）")

    # 4) 解析代表ID列表（就是保留的子目录名）
    keep_ids = []
    with rep_fasta.open() as fr:
        for line in fr:
            if line.startswith(">"):
                keep_ids.append(line[1:].strip())

    keep_set = set(keep_ids)
    all_dirs = set(d for d, _ in items)
    drop_set = all_dirs - keep_set

    # 写清单
    list_dir = Path(args.list_dir)
    list_dir.mkdir(parents=True, exist_ok=True)
    keep_txt = list_dir / f"{args.outprefix}_KEEP.txt"
    drop_txt = list_dir / f"{args.outprefix}_DROP.txt"
    keep_txt.write_text("\n".join(sorted(keep_set)) + "\n")
    drop_txt.write_text("\n".join(sorted(drop_set)) + "\n")

    print("\n====== 去冗余结果 ======")
    print(f"总序列数：{len(all_dirs)}")
    print(f"代表序列/保留：{len(keep_set)}")
    print(f"冗余/待删：{len(drop_set)}")
    print(f"清单：{keep_txt} / {drop_txt}")

    # 5) 根据清单删除目录（可选）
    if args.delete and not args.dry_run:
        removed = 0
        for name in sorted(drop_set):
            d = base / name
            if d.is_dir():
                try:
                    shutil.rmtree(d)
                    removed += 1
                    print(f"[删除] {d}")
                except Exception as e:
                    print(f"[删除失败] {d}: {e}")
        print(f"[完成] 已删除目录：{removed}")
    else:
        print("\n[提示] 当前未删除任何目录。若要执行删除，请添加 --delete（建议先配合 --dry-run 预览）。")

if __name__ == "__main__":
    main()


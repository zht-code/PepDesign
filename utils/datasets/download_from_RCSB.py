# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# RCSB 批量下载脚本
# - 从 ids.txt 读取 PDB ID（逗号/空格/换行/分号分隔均可）
# - 并发下载 .pdb；若 .pdb 不存在，可自动降级下载 .cif（mmCIF）
# - 每个 ID 单独文件夹：<out>/<ID>/<ID>.pdb 或 <ID>.cif
# """
import argparse, os, re, sys, time, shutil
from urllib import request, error
from concurrent.futures import ThreadPoolExecutor, as_completed

PDB_URL = "https://files.rcsb.org/download/{id}.pdb"
CIF_URL = "https://files.rcsb.org/download/{id}.cif"

def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    tokens = re.split(r"[\s,;]+", txt.strip())   # 逗号/空格/换行/分号都行
    pat = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
    ids = []
    for t in tokens:
        if not t:
            continue
        t = t.strip().upper()
        if pat.match(t):
            ids.append(t)
        else:
            # 例如 failed_ids.txt 里带原因说明，这里会自动忽略非 ID 内容
            print(f"[WARN] 跳过非法ID：{t}", file=sys.stderr)
    # 去重，保持顺序
    seen = set(); uniq = []
    for i in ids:
        if i not in seen:
            uniq.append(i); seen.add(i)
    return uniq

def http_download(url, dst, timeout=60, max_retries=3):
    tmp = dst + ".part"
    headers = {"User-Agent": "Mozilla/5.0 (RCSB bulk downloader)"}
    for attempt in range(1, max_retries + 1):
        try:
            req = request.Request(url, headers=headers, method="GET")
            with request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    raise error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(resp, f)
            os.replace(tmp, dst)
            return True, None
        except error.HTTPError as e:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
            # 404/410 等直接返回，不再重试
            return False, f"HTTP {e.code}"
        except Exception as e:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
                continue
            return False, f"{type(e).__name__}: {e}"
    return False, "Unknown error"

def download_one(pid, outdir, overwrite=False, fallback=None):
    """
    fallback: None / 'cif' / 'mmcif'（'cif' 与 'mmcif' 等效）
    返回 (pid, saved_path 或 None, 状态字符串)
    """
    want_cif = (fallback or "").lower() in ("cif", "mmcif")

    target_dir = os.path.join(outdir, pid)
    os.makedirs(target_dir, exist_ok=True)

    pdb_path = os.path.join(target_dir, f"{pid}.pdb")
    if os.path.exists(pdb_path) and not overwrite:
        return pid, pdb_path, "exists(.pdb)"

    ok, err = http_download(PDB_URL.format(id=pid), pdb_path)
    if ok:
        return pid, pdb_path, "ok(.pdb)"

    if want_cif:
        cif_path = os.path.join(target_dir, f"{pid}.cif")
        if os.path.exists(cif_path) and not overwrite:
            return pid, cif_path, "exists(.cif)"
        ok2, err2 = http_download(CIF_URL.format(id=pid), cif_path)
        if ok2:
            return pid, cif_path, "ok(.cif_fallback)"
        return pid, None, f"fail(pdb:{err}; cif:{err2})"

    return pid, None, f"fail(pdb:{err})"

def main():
    ap = argparse.ArgumentParser(description="Batch download PDB/mmCIF from RCSB")
    ap.add_argument("--ids", default="/root/autodl-tmp/Peptide_3D/data/ids.txt",
                    help="ids.txt 路径（逗号/空格/换行分隔）")
    ap.add_argument("--out", default="/root/autodl-tmp/Peptide_3D/data/train_data",
                    help="输出根目录（默认 downloads）")
    ap.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8）")
    ap.add_argument("--overwrite", action="store_true", help="已存在文件也重新下载")
    ap.add_argument("--fallback", choices=["cif", "mmcif"], default="mmcif",
                    help="当 .pdb 不存在时自动下载 mmCIF（.cif），两者等效")
    args = ap.parse_args()

    ids = load_ids(args.ids)
    if not ids:
        print("未解析到任何合法 PDB ID。请检查 ids.txt。", file=sys.stderr)
        sys.exit(1)

    print(f"将下载 {len(ids)} 个结构到：{os.path.abspath(args.out)}")
    os.makedirs(args.out, exist_ok=True)

    ok_count = 0; fail = []; used_cif = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(download_one, pid, args.out, args.overwrite, args.fallback): pid for pid in ids}
        for i, fut in enumerate(as_completed(futs), 1):
            pid, path, status = fut.result()
            if path:
                ok_count += 1
                if status.startswith("ok(.cif"):
                    used_cif.append(pid)
                print(f"[{i}/{len(ids)}] {pid} -> {path}  [{status}]")
            else:
                fail.append((pid, status))
                print(f"[{i}/{len(ids)}] {pid}  [{status}]", file=sys.stderr)

    # 写出失败清单与 cIF 回退清单
    if used_cif:
        with open(os.path.join(args.out, "used_cif_fallback.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(used_cif) + "\n")

    if fail:
        fail_file = os.path.join(args.out, "failed_ids.txt")
        with open(fail_file, "w", encoding="utf-8") as f:
            for pid, msg in fail:
                f.write(f"{pid}\t{msg}\n")
        print(f"\n完成：成功 {ok_count}，失败 {len(fail)}。"
              f"失败列表: {fail_file}；使用 mmCIF 回退: {len(used_cif)} 条（见 used_cif_fallback.txt）")
    else:
        print(f"\n全部完成：成功 {ok_count}，0 失败。使用 mmCIF 回退 {len(used_cif)} 条（见 used_cif_fallback.txt）。")

if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Pipeline: 下载并筛选 PDB -> 输出 peptide.pdb / receptor.pdb
# 条件：
# - 至少 3 条聚合物链
# - 条目不含核酸残基
# - 肽链：<=30 个标准氨基酸，且无修饰（仅 20 AA）
# - 蛋白链：>30 个标准氨基酸
# - 去水与所有异原子（仅保留标准 AA 的 ATOM）
# - 肽链 C–N 键不断（1.33 Å ± 0.5 Å）
# - 蛋白-肽最小重原子距离 <= 5.0 Å
# - 每个 PDB 选择最靠近的一对并输出 peptide.pdb / receptor.pdb
# """

# import os, re, sys, math, argparse, shutil
# import urllib.request, urllib.error
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from collections import defaultdict

# try:
#     import numpy as np
#     from Bio.PDB import PDBParser, PDBIO, Select
# except ImportError as e:
#     print("需要依赖：biopython, numpy\npip install biopython numpy", file=sys.stderr)
#     sys.exit(1)

# STD_AA = {
#     "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE",
#     "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
# }
# # 常见核酸残基（含 DNA/RNA）
# NA_RES = {
#     "A","C","G","U","I","DA","DC","DG","DT","DI","DU",
#     "ADE","CYT","GUA","URI","THY","PSU"
# }
# WATER_NAMES = {"HOH","WAT"}

# PDB_URL = "https://files.rcsb.org/download/{id}.pdb"

# def parse_ids(path):
#     with open(path, "r", encoding="utf-8") as f:
#         txt = f.read().strip()
#     tokens = re.split(r"[\s,;]+", txt)
#     ids = []
#     for t in tokens:
#         t = t.strip().upper()
#         if re.fullmatch(r"[0-9][A-Z0-9]{3}", t):
#             ids.append(t)
#     # 去重保持顺序
#     seen = set(); out = []
#     for x in ids:
#         if x not in seen:
#             seen.add(x); out.append(x)
#     return out

# def download_pdb(pid, out_dir, timeout=60):
#     """下载 PDB；成功返回路径，失败返回 None"""
#     os.makedirs(out_dir, exist_ok=True)
#     dst = os.path.join(out_dir, f"{pid}.pdb")
#     if os.path.exists(dst):
#         return dst
#     url = PDB_URL.format(id=pid)
#     try:
#         req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"}, method="GET")
#         with urllib.request.urlopen(req, timeout=timeout) as resp:
#             if resp.status != 200:
#                 return None
#             tmp = dst + ".part"
#             with open(tmp, "wb") as f:
#                 f.write(resp.read())
#             os.replace(tmp, dst)
#         return dst
#     except Exception:
#         if os.path.exists(dst + ".part"):
#             try: os.remove(dst + ".part")
#             except: pass
#         return None

# def has_nucleic_acid(structure):
#     for model in structure:
#         for chain in model:
#             for res in chain:
#                 if res.id[0] != " ":
#                     # HETATM 里的核酸少见，这里主要看 ATOM
#                     continue
#                 name = res.get_resname().strip().upper()
#                 if name in NA_RES:
#                     return True
#     return False

# def count_std_aa(chain):
#     """返回(标准AA个数, 总残基个数, 非标准AA个数) 仅统计 ATOM 残基"""
#     total = 0; aa = 0; nonstd = 0
#     for res in chain:
#         if res.id[0] != " ":
#             # 排除异原子（含配体、MSE等）
#             continue
#         total += 1
#         name = res.get_resname().strip().upper()
#         if name in STD_AA:
#             aa += 1
#         else:
#             nonstd += 1
#     return aa, total, nonstd

# def peptide_bond_ok(chain, ideal=1.33, tol=0.5):
#     """检查相邻残基 C-N 键长在 ideal±tol 范围；缺原子视为失败"""
#     # 取标准AA残基并按序号排序
#     residues = [r for r in chain if r.id[0]==" " and r.get_resname().strip().upper() in STD_AA]
#     residues.sort(key=lambda r: (r.id[1], r.id[2].strip() or " "))
#     for i in range(len(residues)-1):
#         r1, r2 = residues[i], residues[i+1]
#         try:
#             C = r1["C"].get_vector()
#             N = r2["N"].get_vector()
#         except KeyError:
#             return False
#         d = (C - N).norm()
#         if not (ideal - tol <= d <= ideal + tol):
#             return False
#     return True

# def chain_heavy_coords(chain):
#     coords = []
#     for res in chain:
#         if res.id[0] != " ":
#             continue
#         name = res.get_resname().strip().upper()
#         if name not in STD_AA:
#             continue
#         for atom in res:
#             aname = atom.get_name().upper()
#             # 粗略排除氢
#             if aname.startswith("H"):
#                 continue
#             coords.append(atom.get_coord())
#     if not coords:
#         return None
#     return np.asarray(coords, dtype=float)

# def min_interchain_distance(chainA, chainB):
#     A = chain_heavy_coords(chainA)
#     B = chain_heavy_coords(chainB)
#     if A is None or B is None:
#         return math.inf
#     # 向量化最小距离
#     # (m,3) 与 (n,3) -> (m,n)
#     diff = A[:, None, :] - B[None, :, :]
#     d2 = np.sum(diff*diff, axis=2)
#     return float(np.sqrt(np.min(d2)))

# class ChainSelect(Select):
#     """仅选择指定链，且只保留标准AA的ATOM；去水与异原子"""
#     def __init__(self, target_chain_id):
#         super().__init__()
#         self.cid = target_chain_id
#     def accept_model(self, model): return 1
#     def accept_chain(self, chain): return 1 if chain.id == self.cid else 0
#     def accept_residue(self, residue):
#         if residue.id[0] != " ":
#             return 0
#         rn = residue.get_resname().strip().upper()
#         if rn in WATER_NAMES: return 0
#         return 1 if rn in STD_AA else 0
#     def accept_atom(self, atom):
#         # 去氢
#         return 0 if atom.get_name().upper().startswith("H") else 1

# def process_one_id(pid, out_root, dist_cut=5.0, overwrite=False, verbose=False):
#     """返回 (pid, status, message)"""
#     id_dir = os.path.join(out_root, pid)
#     pdb_path = os.path.join(id_dir, f"{pid}.pdb")
#     if not os.path.exists(pdb_path):
#         ok = download_pdb(pid, id_dir)
#         if ok is None:
#             return pid, "skip", "no_pdb"

#     parser = PDBParser(QUIET=True)
#     try:
#         structure = parser.get_structure(pid, pdb_path)
#     except Exception as e:
#         return pid, "fail", f"parse_error:{type(e).__name__}:{e}"

#     # 排除含核酸
#     if has_nucleic_acid(structure):
#         return pid, "skip", "contains_nucleic_acid"

#     # 统计聚合物链（标准AA>=1 的链）
#     aa_chains = []
#     for model in structure:
#         for chain in model:
#             aa_cnt, total, nonstd = count_std_aa(chain)
#             if aa_cnt >= 1:
#                 aa_chains.append(chain)
#         break  # 只看第一个 model（常见结构只有 model 0）

#     if len(aa_chains) < 3:
#         return pid, "skip", f"too_few_chains:{len(aa_chains)}"

#     # 分类：肽链（<=30 且 100% 标准AA） vs 蛋白链（>30 标准AA）
#     peptides = []
#     proteins = []
#     for chain in aa_chains:
#         aa_cnt, total, nonstd = count_std_aa(chain)
#         # 只接受标准AA，不含非标准或修饰（包括 MSE）
#         if nonstd > 0:
#             continue
#         if aa_cnt <= 30:
#             if peptide_bond_ok(chain):
#                 peptides.append(chain)
#         elif aa_cnt > 30:
#             proteins.append(chain)

#     if not peptides or not proteins:
#         return pid, "skip", f"no_valid_pairs(peptides={len(peptides)}, proteins={len(proteins)})"

#     # 找到距离最小的肽-蛋白对（min heavy-atom distance）
#     best = (math.inf, None, None)
#     for p in peptides:
#         for r in proteins:
#             d = min_interchain_distance(p, r)
#             if d < best[0]:
#                 best = (d, p, r)

#     min_d, pep_chain, rec_chain = best
#     if not math.isfinite(min_d) or min_d > dist_cut:
#         return pid, "skip", f"no_contact(min_d={min_d:.2f}Å)"

#     # 写出 peptide.pdb / receptor.pdb
#     pair_dir = id_dir  # 按你的需求：直接放在 <ID>/ 下
#     os.makedirs(pair_dir, exist_ok=True)
#     pep_out = os.path.join(pair_dir, "peptide.pdb")
#     rec_out = os.path.join(pair_dir, "receptor.pdb")

#     if (not overwrite) and os.path.exists(pep_out) and os.path.exists(rec_out):
#         return pid, "ok", f"exists(min_d={min_d:.2f}Å, pep={pep_chain.id}, rec={rec_chain.id})"

#     io = PDBIO()
#     io.set_structure(structure)
#     io.save(pep_out, select=ChainSelect(pep_chain.id))
#     io.save(rec_out, select=ChainSelect(rec_chain.id))

#     # 记录元数据
#     with open(os.path.join(pair_dir, "selection.txt"), "w", encoding="utf-8") as f:
#         f.write(f"peptide_chain\t{pep_chain.id}\n")
#         f.write(f"receptor_chain\t{rec_chain.id}\n")
#         f.write(f"min_heavy_atom_distance_A\t{min_d:.3f}\n")
#         f.write(f"criteria\tlen(peptide)<=30 & all_stdAA & CN bond ok; len(receptor)>30; min_dist<=5.0Å; no NA; >=3 chains\n")

#     return pid, "ok", f"min_d={min_d:.2f}Å pep={pep_chain.id} rec={rec_chain.id}"

# def main():
#     ap = argparse.ArgumentParser(description="Build peptide–protein pairs from PDB IDs")
#     ap.add_argument("--ids", default="/root/autodl-tmp/Peptide_3D/data/ids.txt", help="ids.txt（逗号/空格/换行分隔）")
#     ap.add_argument("--out", default="/root/autodl-tmp/Peptide_3D/data/datasets", help="输出根目录（每个ID一个文件夹）")
#     ap.add_argument("--workers", type=int, default=8, help="并行线程数")
#     ap.add_argument("--overwrite", action="store_true", help="覆盖已生成的 peptide/receptor")
#     ap.add_argument("--dist-cut", type=float, default=5.0, help="肽-蛋白最小距离阈值(Å)")
#     args = ap.parse_args()

#     ids = parse_ids(args.ids)
#     if not ids:
#         print("ids.txt 中未解析出合法 PDB ID。", file=sys.stderr)
#         sys.exit(1)
#     os.makedirs(args.out, exist_ok=True)

#     ok, skip, fail = 0, 0, 0
#     skip_reasons = defaultdict(int)
#     no_pdb_ids = []

#     with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
#         futs = {ex.submit(process_one_id, pid, args.out, args.dist_cut, args.overwrite): pid for pid in ids}
#         for i, fut in enumerate(as_completed(futs), 1):
#             pid, status, msg = fut.result()
#             if status == "ok":
#                 ok += 1
#                 print(f"[{i}/{len(ids)}] {pid}: OK  ({msg})")
#             elif status == "skip" and msg == "no_pdb":
#                 skip += 1; skip_reasons[msg] += 1
#                 no_pdb_ids.append(pid)
#                 print(f"[{i}/{len(ids)}] {pid}: SKIP - {msg}")
#             elif status == "skip":
#                 skip += 1; skip_reasons[msg] += 1
#                 print(f"[{i}/{len(ids)}] {pid}: SKIP - {msg}")
#             else:
#                 fail += 1
#                 print(f"[{i}/{len(ids)}] {pid}: FAIL - {msg}", file=sys.stderr)

#     # 输出汇总
#     if no_pdb_ids:
#         with open(os.path.join(args.out, "no_pdb_ids.txt"), "w", encoding="utf-8") as f:
#             f.write("\n".join(no_pdb_ids) + "\n")
#     with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
#         f.write(f"OK\t{ok}\nSKIP\t{skip}\nFAIL\t{fail}\n")
#         for k,v in sorted(skip_reasons.items(), key=lambda x: (-x[1], x[0])):
#             f.write(f"SKIP_REASON\t{v}\t{k}\n")

# if __name__ == "__main__":
#     main()


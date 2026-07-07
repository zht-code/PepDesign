#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
这里根据关键结合残基计算蛋白质口袋的 OT 相似度，找出每个蛋白的 top-5 邻居。
"""

import os
import sys
import json
import math
import warnings
import subprocess
from collections import OrderedDict

import numpy as np
import torch

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(x, **kwargs):
        return x

# ========= 可根据自己环境修改的路径配置 =========
TRAIN_ROOT = "/root/autodl-tmp/train_data"               # 1A1M/receptor.pdb 的根目录
PT_ROOT    = "/root/autodl-tmp/Peptide_3D/data/train_data_pt"  # 每个蛋白的 .pt 文件目录
POCKET_JSON_PATH   = "/root/autodl-tmp/Peptide_3D/utils/Data_augmentation/pocket_indices.json"
NEIGHBORS_JSON_PATH = "/root/autodl-tmp/Peptide_3D/utils/Data_augmentation/ot_top5_neighbors.json"
FPOCKET_BIN = "/usr/local/bin/fpocket"      # 如果不在 PATH，可以写全路径，例如 "/usr/local/bin/fpocket"

HIDDEN_KEY = "embedding"     # .pt 里最后一层 hidden 的键名（你当前脚本里就是 embedding）
TOPK_NEIGHBORS = 5
CANDIDATE_K = 50             # 用平均向量预筛选的候选数量，再用 OT 精排，极大减少计算量
OT_REG = 0.05                # Sinkhorn 正则项
# ==================================================


# ---------- 工具函数 ----------

def find_protein_pdbs(root: str):
    """
    在 TRAIN_ROOT 下寻找受体 PDB：
    假设目录结构：
        root/
          1A1M/receptor.pdb
          2ABC/receptor.pdb
          ...
    返回列表: [(protein_id, pdb_path), ...]
    """
    pairs = []
    root = os.path.abspath(root)
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        rec_pdb = os.path.join(d, "receptor.pdb")
        if os.path.isfile(rec_pdb):
            pairs.append((name, rec_pdb))   # protein_id = 目录名
    if len(pairs) == 0:
        print(f"[WARN] 在 {root} 下没有找到任何 receptor.pdb，检查目录结构是否正确。")
    else:
        print(f"[INFO] 在 {root} 下找到 {len(pairs)} 个蛋白 PDB.")
    return pairs


def run_fpocket(pdb_path: str, fpocket_bin: str = "fpocket"):
    """
    调用 fpocket 计算口袋。
    默认会在 pdb 所在目录下生成 <basename>_out 目录。
    """
    cmd = [fpocket_bin, "-f", pdb_path]
    print(f"[FPOCKET] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[ERROR] fpocket 失败: {pdb_path}")
        print(res.stdout)
        print(res.stderr)
        return False
    return True


def parse_residues_from_pdb(pdb_path: str, chain_filter: str | None = None):
    """
    读取原始 receptor.pdb，构建残基有序列表：
      [(chain_id, res_seq, i_code), ...]
    若 chain_filter 不为 None，则只保留指定链（例如 'A'）。
    """
    residues = []
    index_map = {}
    seen = set()

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21].strip() or " "
            if chain_filter is not None and chain_id != chain_filter:
                continue
            res_seq = int(line[22:26])
            i_code = line[26].strip() or " "
            key = (chain_id, res_seq, i_code)
            if key not in seen:
                seen.add(key)
                idx = len(residues)
                residues.append(key)
                index_map[key] = idx
    return residues, index_map



CHAIN_FILTER = "A"   # 根据你 encode_protein_from_pdb 使用的链来设

def pocket_residue_indices_from_fpocket_output(pdb_path: str, chain_filter: str | None = CHAIN_FILTER):
    """
    只在指定链上做残基 index 映射，避免其他链残基导致索引 >> hidden 长度。
    """
    pdb_dir = os.path.dirname(pdb_path)
    pdb_base = os.path.basename(pdb_path)
    base_no_ext = os.path.splitext(pdb_base)[0]
    out_dir = os.path.join(pdb_dir, f"{base_no_ext}_out")

    pockets_dir = os.path.join(out_dir, "pockets")
    if not os.path.isdir(pockets_dir):
        print(f"[WARN] {pdb_path} 还没有 pockets 目录，可能 fpocket 没成功.")
        return []

    pocket_files = sorted(
        f for f in os.listdir(pockets_dir) if f.lower().endswith("_atm.pdb")
    )
    if not pocket_files:
        print(f"[WARN] {pdb_path} pockets 目录下没有 *_atm.pdb 文件。")
        return []

    best_pocket_pdb = os.path.join(pockets_dir, pocket_files[0])
    print(f"[INFO] 使用口袋文件: {best_pocket_pdb}")

    # 只为指定链构建 index_map（例如 A 链）
    _, index_map = parse_residues_from_pdb(pdb_path, chain_filter=chain_filter)

    pocket_res_indices = set()
    with open(best_pocket_pdb, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21].strip() or " "
            if chain_filter is not None and chain_id != chain_filter:
                continue
            res_seq = int(line[22:26])
            i_code = line[26].strip() or " "
            key = (chain_id, res_seq, i_code)
            if key in index_map:
                pocket_res_indices.add(index_map[key])

    if not pocket_res_indices:
        print(f"[WARN] {pdb_path} 在链 {chain_filter} 上解析不到任何 pocket 残基索引。")
        return []

    return sorted(pocket_res_indices)



# ---------- 1. 构建 pocket 残基索引 JSON ----------

def build_pocket_index_json(train_root: str, json_out: str, fpocket_bin: str = "fpocket"):
    """
    为 train_root 下的每个 receptor.pdb 跑 fpocket，并解析出“最佳口袋”的残基索引，保存到 json。
    JSON 结构示例：
    {
      "1A1M": [0, 5, 6, 7, ...],
      "2ABC": [3, 4, 10, ...],
      ...
    }
    """
    proteins = find_protein_pdbs(train_root)

    pocket_dict = {}
    iterator = tqdm(proteins, desc="[STEP1] fpocket + pocket indices") if TQDM_AVAILABLE else proteins
    for prot_id, pdb_path in iterator:
        # 1) 若已有 _out 目录且 pockets 存在，可跳过 fpocket
        pdb_dir = os.path.dirname(pdb_path)
        base_no_ext = os.path.splitext(os.path.basename(pdb_path))[0]
        out_dir = os.path.join(pdb_dir, f"{base_no_ext}_out")
        need_run_fpocket = not os.path.isdir(out_dir)

        if need_run_fpocket:
            ok = run_fpocket(pdb_path, fpocket_bin=fpocket_bin)
            if not ok:
                continue

        # 2) 解析 pocket 残基索引
        idxs = pocket_residue_indices_from_fpocket_output(pdb_path, chain_filter=CHAIN_FILTER)
        if not idxs:
            print(f"[WARN] {prot_id} 没有 pocket 残基索引，跳过。")
            continue

        pocket_dict[prot_id] = idxs

    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(pocket_dict, f, indent=2)

    print(f"[STEP1 DONE] 已写出口袋残基索引 JSON: {json_out}, 共 {len(pocket_dict)} 个蛋白。")


# ---------- 2. 从 .pt 读取 hidden，并按 pocket index 截取 ----------

def load_pocket_hidden_matrices(pt_root: str, pocket_json_path: str, hidden_key: str = "embedding"):
    """
    读取 pocket_json，针对每个 protein_id：
      - 加载 pt_root/{protein_id}.pt
      - 取 data[hidden_key]，假设形状 [1, L, D] 或 [L, D]
      - 按 pocket_indices 截取 -> [L_pocket, D]
    返回:
      pocket_feats: dict[str, np.ndarray], 每个 value 形状 [L_pocket, D]
    """
    with open(pocket_json_path, "r") as f:
        pocket_dict = json.load(f)

    pocket_feats = {}
    iterator = tqdm(pocket_dict.items(), desc="[STEP2] load pocket hidden") if TQDM_AVAILABLE else pocket_dict.items()
    for prot_id, idx_list in iterator:
        pt_path = os.path.join(pt_root, f"{prot_id}.pt")
        if not os.path.isfile(pt_path):
            print(f"[WARN] {pt_path} 不存在，跳过 {prot_id}")
            continue

        data = torch.load(pt_path, map_location="cpu")
        if hidden_key not in data:
            print(f"[WARN] {pt_path} 中没有键 '{hidden_key}'，可打印 data.keys() 检查。跳过。")
            continue

        h = data[hidden_key]   # [1, L, D] 或 [L, D]
        if isinstance(h, torch.Tensor):
            h = h.float()
            if h.dim() == 3 and h.size(0) == 1:
                h = h[0]       # -> [L, D]
        else:
            h = torch.tensor(h, dtype=torch.float32)

        L, D = h.shape
        idxs = [i for i in idx_list if 0 <= i < L]
        if not idxs:
            print(f"[WARN] {prot_id} 的 pocket 索引与 hidden 长度不匹配，原 idx={idx_list}，L={L}")
            continue

        pocket_h = h[idxs]     # [L_pocket, D]
        pocket_feats[prot_id] = pocket_h.numpy()

    print(f"[STEP2 DONE] 共加载 {len(pocket_feats)} 个蛋白的 pocket hidden 矩阵。")
    return pocket_feats


# ---------- 3. 使用 POT 计算 OT 距离 ----------

import ot

def ot_distance_pockets(X, Y, reg: float = OT_REG):
    """
    计算两个 pocket hidden 矩阵之间的 OT 距离：
      - X: [n_x, d]
      - Y: [n_y, d]
    使用：
      - 均匀分布权重
      - 欧氏距离 cost
      - Sinkhorn (ot.sinkhorn2) 做近似 OT，速度比精确 EMD 快很多
    返回:
      标量 float (越小越相似)
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    n, d = X.shape
    m, d2 = Y.shape
    assert d == d2, "X 和 Y 的特征维度不一致"

    if n == 0 or m == 0:
        return float("inf")

    # a = np.full(n, 1.0 / n, dtype=np.float32)
    # b = np.full(m, 1.0 / m, dtype=np.float32)

    a=np.mean(X, axis=1)
    a=a/np.sum(a)

    b=np.mean(Y, axis=1)
    b=b/np.sum(b)
    
    # cost 矩阵 [n, m]
    M = ot.dist(X, Y, metric="euclidean")
    if M.max() > 0:
        M = M / M.max()  # 归一化，数值更稳定

    # 兼容不同 POT 版本：有的返回标量，有的返回 (value, log)
    dist2 = ot.sinkhorn2(a, b, M, reg=reg, numItermax=200)
    if isinstance(dist2, (tuple, list, np.ndarray)):
        dist2 = dist2[0]

    return float(dist2)



# ---------- 4. 为每个蛋白找 OT 相似度下的 top-5 邻居 ----------

def compute_ot_topk_neighbors(pocket_feats: dict,
                              topk: int = TOPK_NEIGHBORS,
                              candidate_k: int = CANDIDATE_K,
                              reg: float = OT_REG):
    """
    pocket_feats: dict[prot_id -> np.ndarray[L_pocket, D]]
    步骤：
      1) 先对每个 pocket 做 mean pooling 得到 [D] 向量，用它做粗筛；
      2) 对于每个蛋白 i：
           - 用 mean 向量与所有蛋白的 mean 向量做 L2 距离；
           - 取距离最近的 candidate_k 个候选 j；
           - 在这些候选上再用 OT 精排，得到 topk 邻居；
      3) 返回 neighbors: dict[prot_id -> list[(neighbor_id, ot_distance), ...] ]
    """
    prot_ids = sorted(pocket_feats.keys())
    n = len(prot_ids)
    print(f"[STEP4] 共有 {n} 个蛋白参与 OT 邻居搜索。")

    # 1) 计算每个 pocket 的 mean 向量 [D]
    D = None
    mean_vecs = []
    for pid in prot_ids:
        X = pocket_feats[pid]   # [L_pocket, D]
        if D is None:
            D = X.shape[1]
        v = X.mean(axis=0)
        mean_vecs.append(v)
    mean_vecs = np.stack(mean_vecs, axis=0)  # [N, D]

    neighbors = {}

    outer_iter = tqdm(range(n), desc="[STEP4] OT top-k neighbors") if TQDM_AVAILABLE else range(n)
    for i in outer_iter:
        pid_i = prot_ids[i]
        Xi = pocket_feats[pid_i]
        vi = mean_vecs[i]  # [D]

        # 粗筛：mean 向量 L2 距离
        diff = mean_vecs - vi[None, :]
        l2_dists = np.linalg.norm(diff, axis=1)   # [N]

        # 排除自己
        l2_dists[i] = np.inf

        # 取最相似的 candidate_k 个索引
        cand_k = min(candidate_k, n - 1)
        cand_idx = np.argpartition(l2_dists, cand_k)[:cand_k]

        # 在候选上用 OT 小心精排
        ot_list = []
        for j in cand_idx:
            pid_j = prot_ids[j]
            Xj = pocket_feats[pid_j]
            d_ij = ot_distance_pockets(Xi, Xj, reg=reg)
            ot_list.append((pid_j, d_ij))

        # 按 OT 距离从小到大排序，取 topk
        ot_list.sort(key=lambda x: x[1])
        neighbors[pid_i] = ot_list[:topk]

    print(f"[STEP4 DONE] 已为所有蛋白找到 OT 意义下的 top-{topk} 邻居。")
    return neighbors


# ---------- 整体主流程 ----------

def main():
    # 1. 构建 pocket 残基索引 JSON（如果已经有，可以注释掉这一步）
    if not os.path.isfile(POCKET_JSON_PATH):
        build_pocket_index_json(TRAIN_ROOT, POCKET_JSON_PATH, fpocket_bin=FPOCKET_BIN)
    else:
        print(f"[INFO] 已存在 pocket JSON: {POCKET_JSON_PATH}，跳过 STEP1。")

    # 2. 从 .pt 里读取 hidden，并按 pocket index 截取
    pocket_feats = load_pocket_hidden_matrices(PT_ROOT, POCKET_JSON_PATH, hidden_key=HIDDEN_KEY)

    # 3 & 4. 用 OT 计算 pocket 相似度，为每个蛋白找 top-5 邻居
    neighbors = compute_ot_topk_neighbors(
        pocket_feats,
        topk=TOPK_NEIGHBORS,
        candidate_k=CANDIDATE_K,
        reg=OT_REG,
    )

    # 保存邻居结果到 JSON
    os.makedirs(os.path.dirname(NEIGHBORS_JSON_PATH), exist_ok=True)
    # 为了方便阅读，把 float 转成普通 Python 类型
    neighbors_serializable = {
        pid: [(nid, float(d)) for (nid, d) in lst]
        for pid, lst in neighbors.items()
    }
    with open(NEIGHBORS_JSON_PATH, "w") as f:
        json.dump(neighbors_serializable, f, indent=2)

    print(f"[ALL DONE] OT top-{TOPK_NEIGHBORS} 邻居已保存到: {NEIGHBORS_JSON_PATH}")


if __name__ == "__main__":
    main()

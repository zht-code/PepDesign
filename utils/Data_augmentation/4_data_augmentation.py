#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
根据合并的邻居 JSON 文件，构建数据增强后的训练集目录。
'''
import os
import json
import shutil

# ===== 路径配置，根据你自己的实际情况改 =====
TRAIN_ROOT = "/root/autodl-tmp/train_data"   # 原始训练集，形如 train_data/1A1M/receptor.pdb, peptide.pdb
AUG_ROOT   = "/root/autodl-tmp/train_data_augmentation"  # 新的数据增强根目录
NEIGHBORS_JSON_PATH = "/root/autodl-tmp/Peptide_3D/utils/Data_augmentation/ot_top10_neighbors_merged.json"
# ==========================================

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(x, **kwargs):
        return x


def strip_pt_suffix(name: str) -> str:
    """
    把 '1A1M.pt' -> '1A1M'
    如果本来就没有 .pt 后缀，则原样返回。
    """
    base, ext = os.path.splitext(name)
    if ext.lower() == ".pt":
        return base
    return name


def build_augmented_dataset(
    train_root: str,
    aug_root: str,
    neighbors_json: str,
    receptor_name: str = "receptor.pdb",
    peptide_name: str = "peptide.pdb",
):
    # 1. 读入邻居 JSON
    with open(neighbors_json, "r") as f:
        neighbors = json.load(f)

    os.makedirs(aug_root, exist_ok=True)

    keys = list(neighbors.keys())
    iterator = tqdm(keys, desc="构建增强数据集") if TQDM_AVAILABLE else keys

    for key_fname in iterator:
        # key 可能是 "1A1M.pt" 或 "1A1M"
        key_id = strip_pt_suffix(key_fname)
        neigh_list = neighbors[key_fname]

        # 原始受体路径
        receptor_src = os.path.join(train_root, key_id, receptor_name)
        if not os.path.isfile(receptor_src):
            print(f"[WARN] 受体文件不存在，跳过 {key_id}: {receptor_src}")
            continue

        if not neigh_list:
            # 没有邻居就跳过
            continue

        for idx, (neigh_fname, dist) in enumerate(neigh_list, start=1):
            neigh_id = strip_pt_suffix(neigh_fname)
            peptide_src = os.path.join(train_root, neigh_id, peptide_name)

            if not os.path.isfile(peptide_src):
                print(f"[WARN] 多肽文件不存在: {peptide_src} (key={key_id}, neighbor={neigh_id})")
                continue

            # 新目录名：例如 1A1M_1, 1A1M_2, ...
            out_dir = os.path.join(aug_root, f"{key_id}_{idx}")
            os.makedirs(out_dir, exist_ok=True)

            # 目标路径
            receptor_dst = os.path.join(out_dir, receptor_name)
            peptide_dst = os.path.join(out_dir, peptide_name)

            # 拷贝（存在的话覆盖一下没关系）
            shutil.copy2(receptor_src, receptor_dst)
            shutil.copy2(peptide_src, peptide_dst)

            # 你也可以顺便写个 meta.json 记录一下来源，可以按需开关
            # meta = {
            #     "receptor_id": key_id,
            #     "peptide_id": neigh_id,
            #     "distance": dist,
            # }
            # with open(os.path.join(out_dir, "meta.json"), "w") as mf:
            #     json.dump(meta, mf, indent=2)

    print(f"[DONE] 数据增强目录已构建在: {aug_root}")


if __name__ == "__main__":
    build_augmented_dataset(TRAIN_ROOT, AUG_ROOT, NEIGHBORS_JSON_PATH)

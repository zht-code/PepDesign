import os
import json

TRAIN_ROOT = "/root/autodl-tmp/train_data"
AUG_ROOT   = "/root/autodl-tmp/train_data_augmentation"
NEIGHBORS_JSON_PATH = "/root/autodl-tmp/Peptide_3D/utils/Data_augmentation/ot_top10_neighbors_merged.json"

# 1) 先看 JSON 里理论上有多少 pair
with open(NEIGHBORS_JSON_PATH, "r") as f:
    neighbors = json.load(f)

total_pairs = sum(len(v) for v in neighbors.values())
print("JSON 中理论上的增强 pair 数量:", total_pairs)

# 2) 再看真正生成了多少个增强样本目录
real_dirs = [
    d for d in os.listdir(AUG_ROOT)
    if os.path.isdir(os.path.join(AUG_ROOT, d))
]
print("实际生成的增强样本目录数:", len(real_dirs))

import json
import os

# 根目录
BASE_DIR = "/root/autodl-tmp/Peptide_3D/utils/Data_augmentation"

# 5 个分片 JSON 文件路径
INPUT_FILES = [
    os.path.join(BASE_DIR, f"ot_top10_neighbors_{i}.json")
    for i in range(1, 6)
]

# 合并后的输出文件
OUTPUT_FILE = os.path.join(BASE_DIR, "ot_top10_neighbors_merged.json")


def merge_neighbor_jsons(input_files, output_file, topk=10):
    merged = {}

    # 1. 逐个文件读入并合并
    for path in input_files:
        if not os.path.isfile(path):
            print(f"[WARN] 文件不存在，跳过: {path}")
            continue

        print(f"[INFO] 读取: {path}")
        with open(path, "r") as f:
            data = json.load(f)

        for prot, neighbors in data.items():
            # 跳过 value 为空列表的键
            if not neighbors:
                continue

            if prot not in merged:
                merged[prot] = []

            # neighbors 应该是形如 [["1A1O.pt", 0.19], ...]
            merged[prot].extend(neighbors)

    # 2. 对每个蛋白的邻居去重 + 排序 + 截断 topk
    cleaned = {}
    for prot, neighbors in merged.items():
        if not neighbors:
            continue

        # 去重：同一个 neighbor 只保留最小距离
        best = {}
        for item in neighbors:
            # 兼容两种写法：["1A1O.pt", 0.19] 或 ["1A1O.pt", "0.19"]
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            nb, dist = item
            try:
                dist = float(dist)
            except (TypeError, ValueError):
                continue

            if nb not in best or dist < best[nb]:
                best[nb] = dist

        if not best:
            continue

        # 排序并保留 topk
        nb_list = sorted(best.items(), key=lambda x: x[1])[:topk]
        cleaned[prot] = [[nb, float(d)] for nb, d in nb_list]

    # 3. 写出合并后的 JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(cleaned, f, indent=2)

    print(f"[DONE] 合并完成，共 {len(cleaned)} 个蛋白写入: {output_file}")


if __name__ == "__main__":
    merge_neighbor_jsons(INPUT_FILES, OUTPUT_FILE, topk=10)

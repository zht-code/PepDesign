import os
import json

# 输入文件
json_path = "/root/autodl-tmp/Peptide_3D/data/train_data_augmentation_stability_scores.json"
strong_dir = "/root/autodl-tmp/train_data_augmentation_strong"

# 输出文件
output_path = "/root/autodl-tmp/Peptide_3D/data/train_data_augmentation_stability_scores.json"


def main():
    # 读取原始 JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 收集 strong_dir 下所有子文件夹名，完整保留，如 1A1M_1
    strong_ids = set()
    for name in os.listdir(strong_dir):
        full_path = os.path.join(strong_dir, name)
        if os.path.isdir(full_path):
            strong_ids.add(name)

    print(f"train_data_augmentation_strong 中共有 {len(strong_ids)} 个目录ID")

    # 根据完整ID过滤
    if isinstance(data, dict):
        filtered_data = {k: v for k, v in data.items() if k in strong_ids}
        print(f"原始 JSON 中共有 {len(data)} 个条目")
        print(f"过滤后保留 {len(filtered_data)} 个条目")
    elif isinstance(data, list):
        # 如果你的 JSON 是列表结构，这里尝试按 item['id'] 过滤
        filtered_data = []
        for item in data:
            if isinstance(item, dict) and "id" in item and item["id"] in strong_ids:
                filtered_data.append(item)
        print(f"原始 JSON 中共有 {len(data)} 个条目")
        print(f"过滤后保留 {len(filtered_data)} 个条目")
    else:
        raise ValueError("JSON 数据既不是 dict 也不是 list，请检查文件结构。")

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"过滤后的 JSON 已保存到: {output_path}")


if __name__ == "__main__":
    main()
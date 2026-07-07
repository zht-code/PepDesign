#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

JSON_PATH = "/root/autodl-tmp/Peptide_3D/utils/Data_augmentation/train_data_augmentation_hdock_scores.json"
THRESH = -200.0

def main():
    if not os.path.isfile(JSON_PATH):
        raise FileNotFoundError(f"找不到文件：{JSON_PATH}")

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    total_entries = 0          # 总条目数（有无 score 都算）
    valid_scores = 0           # 有合法 score 的条目数
    below_thresh = 0           # score < 阈值 的条目数

    for rid, info in data.items():
        total_entries += 1
        # 有的可能没有 "score" 或者为 None
        score = info.get("score", None)
        if score is None:
            continue

        # 有些 JSON 里会把数字存成字符串，这里安全转换一下
        try:
            s = float(score)
        except (TypeError, ValueError):
            continue

        valid_scores += 1
        if s < THRESH:
            below_thresh += 1

    print(f"总条目数（包含无 score 的）：{total_entries}")
    print(f"有有效 score 的条目数：{valid_scores}")
    print(f"score < {THRESH} 的条目数：{below_thresh}")

    if valid_scores > 0:
        ratio_valid = below_thresh / valid_scores * 100
        print(f"在所有有 score 的条目中，score < {THRESH} 的占比：{ratio_valid:.2f}%")
    else:
        print("没有任何有效的 score，无法计算占比。")

if __name__ == "__main__":
    main()

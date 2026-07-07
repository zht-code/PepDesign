#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil

JSON_PATH = "/root/autodl-tmp/Peptide_3D/utils/Data_augmentation/train_data_augmentation_hdock_scores.json"

SRC_ROOT = "/root/autodl-tmp/train_data_augmentation"
DST_ROOT = "/root/autodl-tmp/train_data_augmentation_strong"

THRESH = -180.0


def main():
    if not os.path.isfile(JSON_PATH):
        raise FileNotFoundError(f"找不到文件：{JSON_PATH}")

    if not os.path.isdir(SRC_ROOT):
        raise FileNotFoundError(f"找不到源目录：{SRC_ROOT}")

    os.makedirs(DST_ROOT, exist_ok=True)

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    total_entries = 0
    valid_scores = 0
    below_thresh = 0
    copied = 0

    for rid, info in data.items():
        total_entries += 1

        score = info.get("score", None)
        if score is None:
            continue

        try:
            s = float(score)
        except (TypeError, ValueError):
            continue

        valid_scores += 1

        if s < THRESH:
            below_thresh += 1

            src_dir = os.path.join(SRC_ROOT, rid)
            dst_dir = os.path.join(DST_ROOT, rid)

            if not os.path.isdir(src_dir):
                print(f"[WARN] 找不到目录: {src_dir}")
                continue

            # 避免重复复制
            if os.path.exists(dst_dir):
                continue

            shutil.copytree(src_dir, dst_dir)
            copied += 1

    print("============ 统计结果 ============")
    print(f"总条目数（包含无 score 的）：{total_entries}")
    print(f"有有效 score 的条目数：{valid_scores}")
    print(f"score < {THRESH} 的条目数：{below_thresh}")
    print(f"成功复制的样本数：{copied}")

    if valid_scores > 0:
        ratio_valid = below_thresh / valid_scores * 100
        print(f"在所有有 score 的条目中，score < {THRESH} 的占比：{ratio_valid:.2f}%")
    else:
        print("没有任何有效的 score，无法计算占比。")


if __name__ == "__main__":
    main()
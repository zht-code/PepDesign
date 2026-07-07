import os
import glob
import math
import shutil

# 原始文件夹
SRC_DIR = "/root/autodl-tmp/Peptide_3D/data/train_data_pt"

# 目标前缀，最终会生成 train_data_pt_1 ... train_data_pt_10
DST_PREFIX = "/root/autodl-tmp/Peptide_3D/data/train_data_pt_"

NUM_BUCKETS = 10  # 想分成几个文件夹就改这里


def main():
    # 找到所有 .pt 文件
    pattern = os.path.join(SRC_DIR, "*.pt")
    files = sorted(glob.glob(pattern))
    n = len(files)
    print(f"总共找到 {n} 个 .pt 文件")

    if n == 0:
        print("没有找到 .pt 文件，检查路径是否正确")
        return

    # 创建目标文件夹
    dst_dirs = []
    for i in range(1, NUM_BUCKETS + 1):
        d = f"{DST_PREFIX}{i}"
        os.makedirs(d, exist_ok=True)
        dst_dirs.append(d)
        print(f"目标文件夹已创建/存在: {d}")

    # 轮流把文件分配到 10 个文件夹中（round-robin）
    # 这样就算不能整除，数量差最多 1
    for idx, src_path in enumerate(files):
        bucket_id = idx % NUM_BUCKETS  # 0 ~ 9
        dst_dir = dst_dirs[bucket_id]
        filename = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, filename)

        print(f"[{idx+1}/{n}] 移动 {src_path} -> {dst_path}")
        shutil.copy2(src_path, dst_path)  # 如果想保留原文件，用 shutil.copy2

    print("全部文件分配完成！")


if __name__ == "__main__":
    main()


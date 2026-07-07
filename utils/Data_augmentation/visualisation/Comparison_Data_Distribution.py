#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # 使用无界面后端，防止 plt.show 卡住
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style("whitegrid")
# sns.set_palette("deep")

# 让 PDF 中的文字尽量保持为可编辑文本（对 Adobe Illustrator 更友好）
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# ====== 文件路径，根据需要修改 ======
orig_path = "/root/autodl-tmp/Peptide_3D/data/hdock_scores.json"  # 9244 原始训练集
aug_path  = "/root/autodl-tmp/Peptide_3D/utils/Data_augmentation/train_data_augmentation_hdock_scores.json"  # 增强集

# 输出图像保存目录
fig_dir = "/root/autodl-tmp/Peptide_3D/utils/Data_augmentation/visualisation"
os.makedirs(fig_dir, exist_ok=True)

# 剔除极端异常值阈值：score < floor 的样本不参与统计/绘图
SCORE_FLOOR = -500.0

png_path = os.path.join(fig_dir, "hdock_score_dist.png")
pdf_path = os.path.join(fig_dir, "hdock_score_dist.pdf")
# svg_path = os.path.join(fig_dir, "hdock_score_dist.svg")


def load_scores(json_path):
    """从 {id: {score: ...}} 结构的 json 中读出所有 score（去掉 None）"""
    with open(json_path, "r") as f:
        data = json.load(f)

    scores = []
    for k, v in data.items():
        if isinstance(v, dict) and "score" in v and v["score"] is not None:
            s = float(v["score"])
            # Drop extreme low affinity outliers.
            if s < SCORE_FLOOR:
                continue
            scores.append(s)

    scores = np.array(scores, dtype=float)
    return scores


def print_stats(name, scores):
    """打印分布的一些统计量"""
    print(f"\n===== {name} =====")
    print(f"样本数: {len(scores)}")
    if len(scores) == 0:
        return
    print(f"均值: {scores.mean():.2f}")
    print(f"中位数: {np.median(scores):.2f}")
    for q in [10, 25, 50, 75, 90]:
        print(f"{q}% 分位数: {np.percentile(scores, q):.2f}")
    print(f"最小值: {scores.min():.2f}")
    print(f"最大值: {scores.max():.2f}")


def main():
    # 1. 读取分数
    orig_scores = load_scores(orig_path)
    aug_scores  = load_scores(aug_path)

    print_stats("原始训练集 (9244 对)", orig_scores)
    print_stats("数据增强集", aug_scores)

    if len(orig_scores) == 0 and len(aug_scores) == 0:
        print(f"\n过滤后没有任何分数数据可用于绘图（score < {SCORE_FLOOR} 已全部剔除）。")
        return

    # 2. 画直方图对比
    plt.figure(figsize=(10, 6))

    all_scores = np.concatenate([orig_scores, aug_scores])
    bins = 60
    vmin, vmax = np.percentile(all_scores, [1, 99])  # 去掉极端尾部一点点

    # seaborn deep palette 风格配色
    color_orig = "#4C72B0"   # 蓝色
    color_aug  = "#E45756"   # 红色

    plt.hist(
        orig_scores,
        bins=bins,
        range=(vmin, vmax),
        density=True,
        alpha=0.6,
        color=color_orig,
        edgecolor="white",
        linewidth=0.3,
        label="Original",
    ) if len(orig_scores) else None

    if len(aug_scores):
        plt.hist(
            aug_scores,
            bins=bins,
            range=(vmin, vmax),
            density=True,
            alpha=0.6,
            color=color_aug,
            edgecolor="white",
            linewidth=0.3,
            label="Augmented",
        )

    plt.xlabel("HDOCK score")
    plt.ylabel("Probability density")
    plt.title("Distribution of HDOCK scores for the original training set versus the data augmentation set")

    plt.legend(frameon=False)
    # plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    # ====== 保存文件 ======
    # 高分辨率 PNG（位图）
    plt.savefig(png_path, dpi=600, bbox_inches="tight")

    # AI 可编辑 PDF（矢量格式）
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

    # SVG 矢量格式（如有需要也可打开）
    # plt.savefig(svg_path, format="svg", bbox_inches="tight")

    print(f"\n图像已保存：")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    # print(f"  SVG: {svg_path}")

    orig_scores = np.array(orig_scores)
    aug_scores  = np.array(aug_scores)

    for name, arr in [("original", orig_scores), ("aug", aug_scores)]:
        print(name)
        print("  N:", len(arr))
        print("  mean:", arr.mean())
        print("  median:", np.median(arr))
        print("  25/75 percentile:", np.percentile(arr, [25, 75]))
        print("  % < -150:", (arr < -150).mean())
        print("  % < -300:", (arr < -300).mean())
        print()


if __name__ == "__main__":
    main()
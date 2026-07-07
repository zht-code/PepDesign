# Decoy 生成计划（占位）

当前主表中 **未出现 `group=decoy`** 条目。后续建议基于 **`group=generated`** 的序列集合构建阴性对照，用于稳健性评估与过拟合检验。

## 1. Shuffle decoy（序列打乱对照）

- **输入**：从 `Table_S1_master_sequence_table.csv` 筛选 `group=generated` 且 `usable_for_solubility_analysis=true` 的 `sequence`。
- **操作**：在**保留氨基酸组成**（ multiset 不变）的前提下，对每条序列随机打乱顺序（Fisher–Yates）；可固定随机种子以保证可复现。
- **约束**：避免产生与原始序列完全相同的排列；可对 Pro/Cys 等结构敏感残基施加局部约束（可选）。
- **输出**：新列 `group=decoy_shuffle`，`peptide_id` 加后缀 `_shuffle{k}`，`sequence_source_path` 标注 `synthetic:shuffle`。

## 2. Random matched decoy（组成匹配随机序列）

- **输入**：同上 generated 序列；对每条序列计算长度与氨基酸频率向量。
- **操作**：从预定义氨基酸池或背景分布中**随机抽样**生成同长度序列，使期望频率接近原序列（可用多项式采样或迭代拒绝采样）。
- **输出**：`group=decoy_random_matched`，`notes` 中记录采样版本与种子。

## 3. 与主表合并

- 将 decoy 行追加到主表副本（或单独 `Table_S1_decoys.csv`），并在后续结构/界面步骤中 **跳过** `free_structure_path` / `complex_structure_path`（除非另行建模）。

## 4. 质量检查

- 校验 decoy 与原始序列的编辑距离、疏水性（GRAVY）分布，避免生成极端不可理化序列。

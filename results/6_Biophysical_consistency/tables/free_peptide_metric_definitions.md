# Free peptide 结构指标定义（`free_peptide_metric_definitions.md`）

本文档说明 `02_analyze_free_peptides.py` 输出的各字段含义。**凡标注 heuristic / proxy 的指标均为可复现近似，不等同于实验或全原子物理严格值。**

## 输入与对象

- 输入 PDB：`Table_S1_master_sequence_table.csv` 中 `usable_for_free_structure_analysis=True` 且路径存在的 `free_structure_path`。
- 解析：Bio.PDB `PDBParser` / `MMCIFParser`；默认取**首条模型**中含标准氨基酸最多的链。

## A. 基础结构

| 字段 | 定义 |
|------|------|
| `residue_count` | 链上 `is_aa(standard=True)` 残基数。 |
| `atom_count` | 该链重原子数（排除氢）。 |
| `helix_frac` | **Heuristic**：φ/ψ 落入宽松 α 区残基比例（见代码 `heuristic_ss_fractions_from_phi_psi`）。 |
| `sheet_frac` | **Heuristic**：φ/ψ 落入宽松 β/extended 区比例。 |
| `coil_frac` | **Heuristic**：其余有 φ/ψ 的残基比例；`helix+sheet+coil≈1`。 |
| `n_classified_dihedrals` | 参与二级结构分类的 φ/ψ 对数量。 |

## B. 几何与冲突

| 字段 | 定义 |
|------|------|
| `clash_count` | **Heuristic**：重原子对距离 < 0.82×(r_vdw_i+r_vdw_j)（非键合、排除相邻主链肽段）。 |
| `severe_clash_count` | 同上，阈值系数 0.72。 |
| `approximate_backbone_strain_score` | **Heuristic**：各残基 N–CA–C 键角相对 110° 的均方偏差 /100。 |
| `torsion_proxy_score` | **Proxy**：φ/ψ 落入宽松「核心允许区」并集的比例。 |

## C. 分子内氢键（几何计数）

| 字段 | 定义 |
|------|------|
| `intrapeptide_hbond_count` | **Heuristic**：供体 N 与受体 O（重原子）距离 2.4–3.5 Å 的对数（去重）。 |
| `backbone_hbond_count` | 其中 N/O 均为主链原子（N, O, OXT）。 |
| `sidechain_hbond_count` | `intrapeptide_hbond_count - backbone_hbond_count`（近似侧链参与）。 |

## D. 疏水内聚（序列 + CA 几何）

| 字段 | 定义 |
|------|------|
| `hydrophobic_residue_count` | 序列中 `AILMFWYV` 计数。 |
| `hydrophobic_cluster_count` | **Heuristic**：疏水残基（`AILMFWY`）CA 6 Å 内建图后的连通分量数。 |
| `longest_hydrophobic_run` | 序列上连续疏水（`AILMFWYV`）最大长度。 |
| `buried_hydrophobic_proxy` | **Proxy**：疏水残基中，CA 8 Å 邻域内其它 CA 数 ≥6 的个数。 |
| `buried_hydrophobic_fraction` | `buried_hydrophobic_proxy / max(疏水残基数,1)`。 |
| `hydrophobic_cohesion_score` | **Heuristic**：0.55×`buried_hydrophobic_fraction` + 0.45×归一化最长疏水串。 |

## E. Cys / 二硫键

| 字段 | 定义 |
|------|------|
| `cysteine_count` | Cys 残基数（基于 one-letter）。 |
| `disulfide_candidate_count` | 不同 Cys 的 SG（或别名）对，距离 < 4.0 Å 的对数。 |
| `disulfide_feasible_flag` | 若存在 SG–SG 距离 ∈ [2.0, 2.8] Å 则为 True（宽松几何可行）。 |

## 状态列

| 字段 | 定义 |
|------|------|
| `analysis_status` | `ok` / `error` / `skipped`。 |
| `error_message` | 失败或跳过时简短原因。 |
| `pdb_sequence_inferred` | 从 PDB 推断的 one-letter 序列（与主表 `sequence` 可对照）。 |


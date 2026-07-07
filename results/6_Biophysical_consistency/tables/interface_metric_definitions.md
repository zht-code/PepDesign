# 界面指标定义（interface metrics）

本文档与 `Table_S8_interface_metrics.csv` 列一一对应，均为**几何 / 序列启发式 proxy**，非严格 PBSA / SASA。

## 通用设定

- **重原子**：氢原子一律忽略。
- **界面距离阈值** `interface_cutoff`（Å）：肽原子与靶原子欧氏距离 ≤ 该值记为**原子接触**。
- **残基接触**：若一对标准残基（肽–靶）间存在至少一对原子接触，则记为 1 对残基接触；`min_distance_A` 为该对残基间**最近原子距离**。
- **链识别**：Bio.PDB 多链时，**标准残基数最少**的链标为肽，**最多**的链标为靶；多于两条链时其余链忽略（仅记 note）。单链且含 `TER` 时按段拆分，**残基数较少段**为肽并临时重标链 `P`/`T`。
- **盐桥**：K/R 侧链正电原子（NZ、NH*、NE、ND1、NE2）与 D/E 羧基氧（OE*、OD*、OXT）距离 ≤ `salt_bridge_cutoff`（默认 4.0 Å），按**残基对**去重计数。
- **氢键 proxy**：重原子距离 ∈ [2.4, `hbond_cutoff`] Å，且元素为 N–O 或 O–N（主链/侧链均计入，**无角度筛选**）。

## A. 基础界面

| 列名 | 含义 |
|------|------|
| `residue_contact_count` | 界面**残基对**数目（肽残基–靶残基，无序对但方向固定为肽→靶）。 |
| `atomic_contact_count` | 界面**原子对**数目（距离 ≤ interface_cutoff）。 |
| `interface_residue_count_peptide` | 参与至少一对残基接触的**不同肽残基**个数。 |
| `interface_residue_count_target` | 参与至少一对残基接触的**不同靶残基**个数。 |

## B. Buried / packing / gap proxy

| 列名 | 含义 |
|------|------|
| `buried_sasa_proxy` | 对每个肽界面残基，取其重原子坐标均值为中心，统计半径 4 Å 内靶重原子数，除以 40 截断到 [0,1]，再对界面肽残基取平均。近似「界面埋藏」程度。 |
| `interface_packing_density_proxy` | `atomic_contact_count / (interface_residue_count_peptide + interface_residue_count_target)`。 |
| `interface_gap_proxy` | 所有残基对 `min_distance_A` 的均值，线性映射到 [0,1]：`(mean_min_d - 2.5) / (interface_cutoff - 2.5)`，小于 0 截断为 0。值越大表示平均距离越大（更「松」）。 |

## C. 疏水互补

| 列名 | 含义 |
|------|------|
| `hydrophobic_contact_count` | 原子接触对中，两残基 one-letter 均属于 {A,I,L,M,F,W,Y,V} 的对数。 |
| `hydrophobic_patch_overlap_score` | 肽界面残基中 KD>1.5 占比 `f_p` 与靶界面 `f_t`，`1 - abs(f_p - f_t)`，截断到 [0,1]。 |
| `hydrophobic_mismatch_penalty` | 每个残基对上两残基 Kyte–Doolittle 差绝对值，取平均后除以 8。 |

## D. 静电互补

| 列名 | 含义 |
|------|------|
| `opposite_charge_contact_count` | 原子接触对中，形式电荷（K/R=+1, D/E=-1, H=+0.5）乘积为负的对数。 |
| `same_charge_contact_count` | 形式电荷同号且均非零的原子接触对数。 |
| `electrostatic_complementarity_score` | `opposite / (opposite + same + 1e-6)`。 |
| `salt_bridge_count` | 见上文盐桥几何定义，**残基对**去重。 |
| `unsatisfied_buried_charge_proxy` | 肽界面残基：形式电荷 |q|≥1，且 4 Å 内靶重原子≥8（近似埋藏），但 4.5 Å 内无相反电荷靶原子的残基数。 |

## E. 极性相互作用

| 列名 | 含义 |
|------|------|
| `interface_hbond_count` | 满足氢键距离与 N/O 元素判据的原子接触对数（无角约束）。 |
| `polar_contact_count` | 原子接触对中至少一侧 one-letter 属于极性集合 `NQSTDEKRHYWCM` 的对数。 |

## 输出文件

- `intermediate/interface_contacts/{peptide_id}__{target_id}__r{rank}_residue_contacts.csv`：残基对汇总（每行一对肽–靶残基，`min_distance_A`、`n_atom_atom_pairs`）。
- `intermediate/interface_contacts/..._atomic_contacts.csv`：原子对明细；若行数超过脚本参数 `max_atomic_export` 会截断，并在 `Table_S3` 的 `atomic_export_truncated` 与 `processing_notes` 中注明。
- `intermediate/interface_hit_frequency_by_target/{target_id}_target_site_hit_frequency.csv`：每个靶一条表，列为靶残基标识、`hit_peptide_count`（至少一个界面接触的不同肽数）、`n_analyzed_complexes`、`hit_frequency`。

---
版本：稳定 proxy 实现；后续可替换为真实 SASA / 氢键角筛选 / APBS 等。

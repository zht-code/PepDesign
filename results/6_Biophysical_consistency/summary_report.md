# Biophysical summary report

## 数据来源与规模

- 主合并行数（Table_S1 键）：**2128**
- 含 Table_S2（游离肽结构）行数：**1865**
- 含 Table_S8（界面）行数：**480**

## 标准化与综合分定义

所有 `FCS_*`、`SCS_*`、`ICS_*` 中带 `_r_` 的列为 **rank-percentile**（Pandas `rank(pct=True)`），
在「全队列、同列非缺失值」上计算：值越大越好用 `ascending=True` 的排名分位；越小越好用 `ascending=False`。
缺失（例如无界面结果）不参与该列排名；综合分时对缺失分量取队列 **中位数** 填补后再加权（仅用于 `OBCS`，并在下表外注记）。

### Foldability Composite Score (FCS)

对 `s2_analysis_status == ok` 的样本，取以下六项分位的 **算术平均**（0–1，越大越好）：

1. `FCS_r_clash`：`clash_count` 越小越好  
2. `FCS_r_severe`：`severe_clash_count` 越小越好  
3. `FCS_r_strain`：`approximate_backbone_strain_score` 越小越好  
4. `FCS_r_torsion`：`torsion_proxy_score`（Ramachandran 允许区比例）越大越好  
5. `FCS_r_hbond_density`：`intrapeptide_hbond_count / residue_count` 越大越好  
6. `FCS_r_dihedral_coverage`：`n_classified_dihedrals / residue_count` 越大越好  

### Solubility Compatibility Score (SCS)

对 `s5_analysis_status == success` 的样本，四项分位平均：

1. `SCS_r_camsol_like`：`camsol_like_score` 越大越好（启发式，非官方 CamSol）  
2. `SCS_r_gravy`：`|gravy|` 越小越好  
3. `SCS_r_net_charge`：`|net_charge_ph74|` 越小越好  
4. `SCS_r_aromatic`：`aromatic_fraction` 越小越好（降低芳香暴露聚集倾向代理）

### Interface Complementarity Score (ICS)

对存在界面表（`residue_contact_count` 非空且 >0）的样本，六项分位平均：

1. `ICS_r_contacts`：`log1p(residue_contact_count)` 越大越好  
2. `ICS_r_hbond`：`interface_hbond_count / interface_residue_count_peptide` 越大越好  
3. `ICS_r_elec_comp`：`electrostatic_complementarity_score` 越大越好  
4. `ICS_r_hyd_match`：`hydrophobic_mismatch_penalty` 越小越好  
5. `ICS_r_patch_overlap`：`hydrophobic_patch_overlap_score` 越大越好  
6. `ICS_r_packing`：`interface_packing_density_proxy` 越大越好  

### Aggregation Liability Index (ALI)

直接采用 Table_S7 的 `aggregation_liability_index`（0–1，**越大聚集风险越高**）。
另设 `ALI_complement = 1 - ALI`（越大越好）用于 OBCS。

### Overall Biophysical Consistency Score (OBCS)

加权线性组合（权重来自 `config.yaml` → `thresholds.obcs_weights`，默认 `anti_agg=0.16, fcs=0.28, ics=0.28, scs=0.28`）：

`OBCS = w_fcs * FCS_fill + w_scs * SCS_fill + w_ics * ICS_fill + w_anti_agg * ALI_complement_fill`

其中 `*_fill` 为原列缺失时用该列**全局中位数**填补（便于批量完整输出；敏感分析可改为剔除）。

## 分组对比（peptide-level）

```
    group  n_peptides  mean_FCS  std_FCS  median_FCS  mean_SCS  std_SCS  median_SCS  mean_ICS  std_ICS  median_ICS  mean_ALI  std_ALI  median_ALI  mean_OBCS  std_OBCS  median_OBCS
generated        2128  0.500268 0.099573    0.511394  0.500268 0.158933    0.490966  0.501109 0.158408    0.515152  0.236819 0.138982    0.218641    0.54609  0.073915     0.549921
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `tables/Table_S4_foldability_summary.csv` | 折叠相关原始列 + FCS 分量与 FCS |
| `tables/Table_S11_biophysical_summary_scores.csv` | 肽级 FCS/SCS/ICS/ALI/OBCS 及全部分项 |
| `tables/Table_S12_target_level_summary.csv` | 按 `target_id` 聚合的均值/标准差及各 `group` 计数与分组均值 |

---
由 `05_build_summary_scores.py` 自动生成。

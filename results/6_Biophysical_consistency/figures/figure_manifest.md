# Figure manifest

所有图由 `06_make_figures.py` 从下列数据表生成。

## `Fig7a_foldability_comparison.png` / `.pdf`

- **Table_S11**（`FCS`）与 **Table_S4** 同键对齐。
- 由 ``source`` 解析 **Ours / RFdiffusion / ProteinGenerator / BindCraft**；SOTA 仅使用 ``all_samples:`` 全量指标行，避免 ``baseline_input_index:`` 重复与缺失界面。
- 按方法着色的 FCS 密度直方图。

## `Fig7b_clash_hbond_hydrophobicity.png` / `.pdf`

- **Table_S4**：`s2_clash_count`, `s2_intrapeptide_hbond_count`, `s2_hydrophobic_cohesion_score`（`s2_analysis_status==ok`），按方法着色。

## `Fig7c_solubility_hotspot_comparison.png` / `.pdf`

- **Table_S11**（`SCS`）与 **Table_S7**（`hotspot_burden`, `aggregation_liability_index`），按方法分色散点。

## `Fig7d_interface_vs_solubility_tradeoff.png` / `.pdf`

- **Table_S11**：`ICS` vs `SCS`，按方法分色。

## `Fig7e_interface_complementarity_comparison.png` / `.pdf`

- **Table_S11**（`ICS` 及 `ICS_r_*`）与 **Table_S1**（`length`）。
- 面板：按方法的 ICS 直方图+KDE；方法箱线；肽长度四分位；高样本靶标；ICS 六项子指标 **分组横向条（并排按方法）**。

## `Fig7f_contact_enrichment_heatmap.png` / `.pdf`

- **Table_S8**：按方法分面板（含 **Ours** 与三种 SOTA），各方法共有靶标子集上行内 z-score；指标为 residue / atomic contacts 与界面 H-bond。

## `Supplementary_target_level_heatmap.png` / `.pdf`

- **Table_S11** 按 `target_id`×`method` 聚合 `FCS/SCS/ICS/ALI/OBCS`，宽列 `分数_方法`，列 z-score。

# 溶解度与聚集热点指标定义（solubility & aggregation）

本文档说明 `Table_S5`、`Table_S6`、`Table_S7` 及 `intermediate/solubility_profiles/` 中各列含义。
除特别说明外均为**序列启发式**；可选结构仅用于 **CA 邻域暴露度 proxy**，非真实 SASA。

## 通用

- **标准氨基酸**：输入序列转为大写，非字母剔除。
- **疏水集合** `AILMFWYV`，**芳香** `FWY`，**极性未带电** `NQST`（用于 CamSol-like 奖励项）。
- **pH 7.4 形式电荷**：K、R → +1；D、E → −1；H → +0.5；其余 0。
- **Kyte–Doolittle** 标度 `hydrophobicity` / `gravy`（GRAVY = 序列平均 KD）。

## A. 全局指标（Table_S5）

| 列名 | 含义 |
|------|------|
| `length` | 序列长度。 |
| `gravy` | 平均 KD。 |
| `net_charge_ph74` | 形式电荷之和。 |
| `positive_residue_fraction` | K+R 占比。 |
| `negative_residue_fraction` | D+E 占比。 |
| `aromatic_fraction` | F+W+Y 占比。 |
| `hydrophobic_fraction` | 疏水集合占比。 |
| `charge_density` | `net_charge_ph74 / length`。 |
| `pI_proxy` | 启发式：`clip(7.0 + 2.8 * (正残基占比 − 负残基占比), 4, 10.5)`，**非实验 pI**。 |
| `camsol_like_score` | **CamSol-like heuristic**（非官方 CamSol）：`-1.15*gravy -0.09*|net_charge| -0.55*f_aromatic -0.42*f_hydrophobic +0.28*f_polar_uncharged +0.04*min(len,40)/40`。数值越大通常表示**更可溶倾向**（与真实 logS 刻度未校准）。 |

## B. 残基级指标（Table_S6 & solubility_profiles/*.csv）

窗口半宽 **2**（即窗口长度 **5**），与 `config.thresholds.aggregation_hotspot_window` 对齐（脚本取 `max(3, min(window, 11))` 的奇数窗口）。

| 列名 | 含义 |
|------|------|
| `residue_index` | 1-based 残基索引。 |
| `residue` | 单字母。 |
| `hydrophobicity` | KD 值。 |
| `charge_state` | 同上形式电荷。 |
| `local_hydrophobic_run` | 含该残基的最长连续疏水段长度 / 当前窗口长度。 |
| `local_charge_balance` | `1 - min(1, |窗口电荷和|/窗口长度)`，越接近 1 越「电荷均衡」。 |
| `camsol_like_local_score` | 局部 CamSol-like：**非** CamSol 分解，启发式为 `-0.12*窗口平均KD -0.06*|本残电荷| -0.04*|窗口净电荷|/窗长 +0.05*窗口内NQST占比`。 |
| `hotspot_score` | 可解释规则加权（0–约1.5 后截断到合理范围），见下节。 |
| `hotspot_class` | `none` / `mild` / `strong`，阈值 **mild≥0.38**, **strong≥0.62**。 |

### hotspot_score 规则（可解释）

加权求和（权重之和≈1，再截断）：

1. **低局部溶解性**：窗口平均 KD 归一化 `clip(mean_KD/4.5, 0, 1)` × **0.24**。
2. **连续疏水**：窗口疏水占比 × **0.22**；另加 `local_hydrophobic_run` 相对窗口大小的项 × **0.22*0.35**。
3. **芳香/疏水聚集倾向**：窗口芳香占比 × **0.18**。
4. **周边缺少带电中和**：`charge_neutralization_deficit` = 若窗口带电残基比例 < 0.35 则线性升高到 1 × **0.18**。
5. **结构（可选）**：若 `free_structure_path` 可读且序列与结构链一致，则 CA 10 Å 邻域内邻居越少越暴露；**暴露度 × 疏水残基** × **0.18** 计入（暴露 0–1）。

无结构或序列不匹配时第 5 项为 0，`structure_alignment_note` 记录原因。

## C. 聚集汇总（Table_S7）

| 列名 | 含义 |
|------|------|
| `hotspot_count` | `hotspot_class` 为 mild 或 strong 的残基数。 |
| `strong_hotspot_count` | strong 残基数。 |
| `hotspot_burden` | 所有残基 `hotspot_score` 之和 / `length`。 |
| `longest_hotspot_span` | 最长连续 mild/strong 片段长度。 |
| `aggregation_liability_index` | 0–1 综合：`0.22*clip(|gravy|/1.2)+0.26*clip(burden/0.85)+0.2*clip(strong/ max(3,0.12L))+0.17*clip(longest/max(5,0.35L))+0.15*clip(|net_charge|/12)`。 |

## 分组

对 `group` ∈ {generated, reference, decoy} 的行**使用同一套公式**计算；其它分组可跳过或原样保留（由脚本参数控制）。

## 每条肽的 profile 文件

`intermediate/solubility_profiles/{peptide_id}.csv`：与 Table_S6 相同的残基列，并前置 `target_id`、`peptide_id`、`group`。序列为空时写入**仅表头**的空文件，便于批量流水线对齐。

## Table_S7 附加列

- `analysis_status`：`success` / `failed`（如空序列）。
- `hotspot_window`：实际使用的奇数窗口长度。

---
版本：稳定启发式；接入真实 CamSol / 实验溶解度时可替换 `camsol_like_*` 列来源。

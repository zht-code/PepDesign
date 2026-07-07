# Master table report（Table S1）

## 规模

- **主表肽条目数（行）**：1995
- **all_samples 原始行数**：3065
- **索引的对接模型 key 数**：10069（来自 **30207** 个 `model_*.pdb` 文件）

## 按 group 统计

- **generated**：1995

## 可做「完整三项」分析的 target

定义：同一 `target_id` 下至少存在一条记录，同时满足游离结构、界面复合物、序列三项可用。

- **满足条件的 target 数**：**133**

示例 target_id（按字母序，最多列 80 个）：

`1cjr`, `1cka`, `1cvu`, `1d4t`, `1eg4`, `1h6w`, `1hc9`, `1jbu`, `1k5n`, `1mfg`, `1nln`, `1nq7`, `1ntv`, `1nx1`, `1oai`, `1oj5`, `1ou8`, `1ow6`, `1pzl`, `1qkz`, `1rst`, `1rxz`, `1sfi`, `1ssh`, `1t08`, `1t4f`, `1t7r`, `1tfc`, `1u00`, `1uj0`, `1x2r`, `1xoc`, `1ymt`, `1yuc`, `1ywo`, `2a25`, `2a3i`, `2aq9`, `2b9h`, `2bba`, `2cch`, `2ce8`, `2d0n`, `2drk`, `2dyp`, `2fff`, `2ffu`, `2fka`, `2fmf`, `2fts`, `2fvj`, `2ho2`, `2ht9`, `2o02`, `2o4j`, `2o9v`, `2oei`, `2p0w`, `2p1o`, `2p1t`, `2p54`, `2peh`, `2pux`, `2puy`, `2qbx`, `2qos`, `2qse`, `2r7g`, `2r9q`, `2v8y`, `2vkn`, `2vr3`, `2vwf`, `2w2u`, `2whx`, `2xrw`, `2xu7`, `2xvc`, `2zjd`, `3asl` …

## 关键文件缺失统计（按行）

- 缺游离结构路径或文件不存在：**263**
- 缺对接复合物或文件不存在：**1648**
- 缺序列（且未能由 clean_properties 补全）：**260**

## 说明

- 主表唯一入口：`Table_S1_master_sequence_table.csv` / `.json`。
- `baseline_input_index` 补充行通常 **无 `condition_tag`**，对接模型匹配较保守；详见各行列 `notes`。


# Case studies（自动筛选）

## 生成方式

在项目根目录执行：

```bash
python3 scripts/07_select_case_studies.py
# 或指定参考肽序列表（含 target_id, sequence）以启用 motif recovery 代理：
python3 scripts/07_select_case_studies.py --reference-sequences /path/to/reference_sequences.csv
```

输出：

- `selected_cases.json`：最终入选案例及路径、打分。
- `case_XX_<peptide_id>/`：每例图件、PyMOL/ChimeraX 素材、`case_summary.md`。

## 作图说明脚本

仓库内 **`scripts/07_select_case_studies.py`** 为唯一可重复入口（matplotlib 生成 2D 图；3D 见各目录下 `pymol_render_instructions.md`）。

也可使用本目录下的便捷包装：

```bash
bash case_studies/regenerate_case_studies.sh
```

## Motif recovery

若提供参考肽序列 CSV（列 `target_id`, `sequence`），脚本用 `difflib.SequenceMatcher` 与生成肽序列算全局相似度 **0–1**，并纳入筛选加权。当前仓库默认尝试 `results/4_ablation/plot/reference_sequences.csv`。

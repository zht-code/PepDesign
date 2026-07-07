# Inventory report（自动生成）

- 扫描时间（UTC）：`2026-04-18T04:38:17.250314+00:00`
- 扫描根：`['/root/autodl-tmp/Peptide_3D/results', '/root/autodl-tmp/Peptide_3D/results/6_Biophysical_consistency/tables', '/root/autodl-tmp/Peptide_3D/results/6_Biophysical_consistency/intermediate']`
- 记录文件数：**144860**

## 按类型统计

- **pdb**：119371
- **dock_out**：10070
- **log**：7512
- **cif**：6400
- **json**：878
- **fasta**：343
- **csv**：253
- **pt**：16
- **tsv**：10
- **npz**：7

## 按 group 统计

- **generated**：108127
- **unknown**：35618
- **reference**：1115

## 后续分析可用性（基于路径与命名规则）

### Free peptide structure analysis

- **优先**：`results/5_robustness/baseline/cache/clean_inputs/**.pdb`（生成/清洗肽复合物输入）。
- **辅助**：`raw_results/*/all_samples.csv` 中的 `pdb_path` 列可批量定位同一批肽结构。

### Complex interface analysis

- **优先**：`results/5_robustness/baseline/cache/hdock_work/**/model_*.pdb`（对接复合物模型）。

### Sequence-based solubility analysis

- **优先**：`all_samples.csv` 等表中的 `sequence_top1` / `sequence` 列。
- **补充**：`results/2_SOTA/**/generated_sequences.fasta`；`clean_properties/*.json` 中的序列字段。

## 本次扫描最重要的发现（摘要）

- **体量**：共索引 **144860** 个相关扩展名文件（不含图片等非目标类型）。
- **结构文件**：PDB/ENT **119371**，CIF/MCIF **6400**；其中 **clean_inputs** 路径约 **1732** 条，**hdock_work** 路径约 **69063** 条（对接复合物模型）。
- **序列与表**：FASTA **343**，CSV **253**，TSV **10**（含 `all_samples` / `samples_*` 等可解析序列列的候选表）。
- **关键主表**：`all_samples.csv` 示例：`/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline/raw_results/proteingenerator/all_samples.csv`
- **索引元数据**：`baseline_input_index.csv`：`/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline/tables/baseline_input_index.csv`
- **对接模型示例**：`/root/autodl-tmp/Peptide_3D/results/5_robustness/cache/hdock_work/structure_missing_lvl0p0_r2/1cjr/pep_01/model_1.pdb`
- **clean_inputs 结构示例**：`/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline/cache/clean_inputs/proteingenerator/1hc9/1hc9_000003.pdb`
- **无条件生成序列示例**：`/root/autodl-tmp/Peptide_3D/results/2_SOTA/unconditional/family_level_test/4XKH/generated_sequences.fasta`
- **数据根**：当前扫描以 `/root/autodl-tmp/Peptide_3D` 下 `results/` 为主；详细路径见 `file_manifest.csv` 与 `suggested_inputs.json`。
- **大规模 `unknown` group（约 3.6 万条路径）**：多为 `results/` 下未命中「生成/对接/扰动」等关键词的辅助文件（例如其它子课题输出、历史缓存、非标准命名）；**不代表不可用**，下游应结合 `baseline_input_index.csv`、`all_samples.csv` 与路径子串（`clean_inputs` / `hdock_work`）二次筛选。
- **二进制与检查点**：本次索引到 **NPZ 7**、**PT 16**、**PKL 0**（若后续出现 `.pkl` 将出现在 `file_manifest.csv` 的 `file_type=pkl` 行）；适合作为深度学习预测或特征缓存的接入点。

## suggested_inputs.json 摘要

- 已写入 `suggested_inputs.json`，其中 `example_paths` 为从本次 manifest 抽取的代表路径（每类最多若干条）。

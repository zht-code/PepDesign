# PD-L1 靶向多肽 + 虚拟细胞五层验证（原型）

在无「PD-L1 多肽直接处理」单细胞数据的前提下，使用 **anti-PD-L1 / anti-PD-1 / PD-L1–TGFβ 双抗（如 Bintrafusp alfa, GSE182004 类数据）** 等作为 **功能等效阳性扰动**，建立参考转录组签名，并结合候选多肽的 **结合 / 对接 / 理化** 特征完成分层评估与综合排序。

## 环境

- Python **≥ 3.10**
- 安装依赖：

```bash
cd PDL1_peptide_virtual_cell
pip install -r requirements.txt
```

若 `scanpy` 提示缺少 `igraph`：

```bash
pip install igraph leidenalg
```

若已按 `scripts/install_geneformer_pypi_mirror.sh` 安装 **Geneformer**，但 `import geneformer` 报错 `SpecialTokensMixin` / `transformers` 相关，请与官方仓库一致 **固定**：

```bash
python3 -m pip install "transformers==4.46"
```

（`scgpt` 等环境若也依赖其它版本的 `transformers`，可能冲突，可单独建 conda 环境专跑 Geneformer。）

## 数据准备

1. 将预处理好的 **`h5ad`**，或 **10x** `mtx` 目录，或 **细胞×基因 CSV** 放入工程内，并在 `config.yaml` 中设置：
   - `scrna_input_path`（h5ad 相对路径），或
   - `tenx_matrix_dir`，或
   - `scrna_csv_path`
2. 配置 `condition_column`、`control_label`、`treatment_label`、`celltype_column`、`species`、`target_celltypes`。
3. 若 `data/raw/` 为空且未配置有效路径，pipeline **自动生成 demo `data/processed/demo_scrna.h5ad`** 以跑通全流程（**非真实生物学结论**）。

候选多肽：编辑 `peptides/candidate_peptides.csv`。可选对接汇总 `structures/docking/docking_summary.tsv`（列含 `peptide_id` 等）。

## 运行

```bash
python src/run_pipeline.py
python src/run_pipeline.py --config config.yaml
```

输出：

- **表**：`results/tables/`（`layer1`–`layer5`、`final_candidate_ranking.csv`）
- **图**：`results/figures/`（每个图 **PDF + PNG**）
- **报告**：`results/reports/final_report.md`

## 五层含义（生物学）

| 层 | 内容 |
|----|------|
| 1 | 多肽序列理化性质 + 对接得分归一化 → **binding_score** |
| 2 | 单细胞中 PD-1/PD-L1 相关基因集评分，治疗 vs 对照 → **pathway blockade 参考** |
| 3 | 标准 scanpy 流程 + 差异表达 → **reference signature**（上下调基因列表） |
| 4 | 默认 `simple_signature`：将肽段扰动 proxy 与 blockade 参考对齐 → **blockade_similarity**、免疫激活预测等；**scvi/scGPT/Geneformer** 可预留扩展 |
| 5 | 毒性/增殖/炎症/EMT/stemness 基因集 + 肽段理化风险 proxy → **toxicity / safety** |

## 综合分（0–1）

```
final_score =
  0.25 * binding_score
+ 0.20 * pathway_blockade_score
+ 0.25 * blockade_similarity_score
+ 0.15 * immune_activation_score
+ 0.15 * safety_score
```

（各分量在汇总前于 `run_pipeline.py` 内做 **min–max 归一化**。）

**recommendation**：`final_score ≥ 0.75` 且 `toxicity_risk_score < 0.30` → *Strong candidate*；`≥ 0.60` 且毒性 `< 0.50` → *Moderate candidate*；否则 *Not recommended*。

## 为什么没有多肽处理组仍可用 checkpoint 阻断数据？

多肽药与小分子/抗体不同，公开 scRNA 中罕见「多肽处理」标签；但 **PD-L1 通路阻断的下游转录程序**（T 细胞激活、IFN-γ、细胞毒性程序等）在 **抗体或双抗阻断 PD-1/PD-L1（及 TGFβ 轴）** 时具有可比较的 **功能读出**。本原型将该类数据作为 **reference perturbation**，再通过第四层 **signature 类比** 映射到候选肽，属 **假设生成与优先级排序工具**，不能替代实验验证。

## 模块说明

| 文件 | 作用 |
|------|------|
| `src/download_data.py` | GEO 提示与 raw 目录检查 |
| `src/preprocess_scrna.py` | 读取 h5ad/10x/csv；**demo 数据** |
| `src/peptide_features.py` | 第一层 |
| `src/docking_parser.py` | 对接表解析 |
| `src/pathway_scoring.py` | 第二层 |
| `src/virtual_cell_signature.py` | 第三层 DEG / 签名 |
| `src/similarity_analysis.py` | 第四层 |
| `src/risk_scoring.py` | 第五层 |
| `src/visualization.py` | 全部图像 |
| `src/run_pipeline.py` | 调度与综合分 |

## 免责声明

本代码用于方法学原型与可复现分析框架；**demo 数据结果不具备临床或成药意义**。使用真实队列时需核对物种、基因命名、批次与注释。

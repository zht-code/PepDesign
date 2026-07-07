# Peptide biophysical consistency（工程骨架）

本目录为 **生成肽生物物理一致性** 分析的可维护工程骨架：将数据盘点、主表构建、结构可行性、复合物界面、溶解度/聚集热点、汇总打分与制图拆为独立脚本，统一由 `config.yaml` 管理路径与阈值，由 `utils/` 提供日志与路径解析。

## 科学目的

在计算肽设计流程中，除亲和力或对接分数外，需要系统评估生成肽是否具备合理的 **折叠几何**、与靶蛋白的 **界面互补性**，以及 **溶解与聚集风险**。本流水线旨在：

1. **可追溯**：每一步读写的输入/输出路径可配置、可记录日志。  
2. **可批处理**：以主表（`tables/master_table.csv`）为核心驱动 02–05。  
3. **可扩展**：阈值与外部工具（如 CamSol）在 `config.yaml` 中预留，便于后续接入。

## 流程步骤与输入输出

| 步骤 | 脚本 | 主要输入 | 主要输出 |
|------|------|----------|----------|
| 00 | `scripts/00_inventory_and_manifest.py` | `project_root`（只读扫描） | `data_inventory/` 清单与 manifest 草稿 |
| 01 | `scripts/01_build_master_table.py` | `data_inventory/` | `tables/master_table.csv`，`intermediate/01_*` 元数据 |
| 02 | `scripts/02_analyze_free_peptides.py` | `tables/master_table.csv` | `intermediate/02_free_peptide/`，`tables/free_peptide_metrics.csv` |
| 03 | `scripts/03_analyze_complex_interfaces.py` | `tables/master_table.csv` | `intermediate/03_interface/`，`tables/interface_metrics.csv` |
| 04 | `scripts/04_analyze_solubility_and_hotspots.py` | `tables/master_table.csv` | `intermediate/04_solubility/`，`tables/solubility_and_hotspots.csv` |
| 05 | `scripts/05_build_summary_scores.py` | `tables/*.csv`（多表合并） | `tables/summary_scores.csv`，`intermediate/05_*` |
| 06 | `scripts/06_make_figures.py` | `tables/` | `figures/`（PNG/PDF，后续实现） |

**个案深读**：`case_studies/` 预留用于存放子集配置、说明文档与手工挑选样本的补充材料。

## 目录结构（约定）

```
6_Biophysical_consistency/
├── README.md
├── config.yaml
├── requirements_biophysical.txt
├── run_all.sh
├── utils/                 # 路径解析、日志等公共模块
├── scripts/               # 00–06 流水线脚本
├── data_inventory/        # 数据盘点与 manifest
├── intermediate/          # 中间结果（逐样本 JSON 等）
├── tables/                # 汇总 CSV
├── figures/               # 图
├── case_studies/          # 个案与补充说明
└── logs/                  # 运行日志
```

### 与早期实验代码的关系

目录中可能仍存在历史布局（例如 `src/biophysical_consistency/`、`run_pipeline.py` 及 `00_discovery/` 等）。**本 README 描述的 canonical 流程以当前 `scripts/00–06` + `utils/` + 顶层 `config.yaml` 为准**；后续实现新功能时建议逐步迁移到该骨架，避免两套入口长期并行。

## 环境与依赖

```bash
cd /root/autodl-tmp/Peptide_3D/results/6_Biophysical_consistency
pip install -r requirements_biophysical.txt
```

只读数据根目录默认见 `config.yaml` 的 `project_root`；也可用环境变量 `PEPTIDE_3D_ROOT` 覆盖。

## 单步运行示例

```bash
export PYTHONPATH="$(pwd)"
python3 scripts/00_inventory_and_manifest.py --config ./config.yaml --log-dir ./logs
python3 scripts/01_build_master_table.py --config ./config.yaml --log-dir ./logs
```

## 一键运行

`run_all.sh` 当前为 **调用顺序框架**（各 `python3` 行默认注释）；确认参数后取消注释即可串联执行。

```bash
chmod +x run_all.sh
./run_all.sh
```

## 配置说明（`config.yaml`）

- `paths.*`：本工程内相对路径，解析为绝对路径后用于默认输入/输出。  
- `thresholds.*`：几何冲突、界面距离与接触数、聚集热点窗口、GRAVY/电荷警告、**CamSol 占位阈值**等。  
- `execution.*`：后续可接 `resume`、`max_samples` 等批处理开关。

当前各脚本为 **骨架实现**：会创建目录、写占位文件并打日志；具体分析逻辑在后续迭代中填充。

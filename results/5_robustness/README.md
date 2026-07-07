# 第五章：最终模型目标扰动鲁棒性（`results/5_robustness`）

本目录包含「仅针对最终模型、在正式测试集（PPDbench）上评估三类目标扰动」的**可复现实验流水线**、配置、日志、缓存、样本与聚合表、主图与图注草稿。所有本章新增产物均位于 `autodl-tmp/Peptide_3D/results/5_robustness` 下。

## 目的

- 在**不重新训练**的前提下，仅在测试/生成阶段对目标蛋白条件施加扰动，量化最终模型的性能退化轨迹、敏感性、容忍阈值（相对下降 10%/20% 对应的扰动强度）以及面积型指标 AUDC。
- **不做**多模型对比、消融或与 OT 模块的对比。

## 最终模型与 checkpoint（默认）

- **默认权重**：`/root/autodl-tmp/Peptide_3D/logs_Ranger_no_DPO/best_model_epoch_72_loss_2.0048.pth`  
  （Ranger 训练、无 DPO，验证集最优 epoch 72，loss 2.0048。）
- 若需改用其它 checkpoint（例如 DPO weighted-sum），请通过 `--ckpt` 显式覆盖，并在论文中注明。
- **推理与生成**：复用 `models_DPO.ProteinPeptideModel`、`results/3_Pareto_improved/ppdbench_generate_core.py` 中的 OpenMM 螺旋初始化、界面重打分与 PDB 写出逻辑。

## 测试集

- **PPDbench 根目录**（默认）：`/root/autodl-tmp/PPDbench`  
  与主评估脚本一致：每个靶点子目录含 `receptor.pdb`（及用于定义口袋的 `peptide.pdb`）。
- 子集调试：`--max-targets N`。

## 三类扰动定义

均在**测试时**施加；生成后对接与物化性质评估使用**原始** `receptor.pdb`，以保证亲和力比较的是「同一物理受体上的肽」，而扰动仅改变**模型见到的条件**（与肽放置用的临时受体几何，见下）。

### 1. 结构缺失（`structure_missing`）

- **强度**：0%、10%、20%、30%、40%（默认）。
- **实现（`encoder_mode=geometry`，推荐）**：从 `ProteinChain` 中随机选取相应比例残基，将其 `atom37` 掩蔽并坐标置为 NaN，再与序列一起构造 ESM3 的 `structure_coords`（N/CA/C）输入。
- **`encoder_mode=sequence_only`**：退化为对序列随机位置替换为 `#` 掩码字符（不改变坐标分支，因原 `encode_protein_from_pdb` 不传真实坐标）。

### 2. 口袋噪声（`pocket_noise`）

- **强度**：0、0.5、1.0、1.5、2.0 Å（默认），对口袋残基主链 N/CA/C 施加高斯噪声；**0 Å 等价于干净条件**。
- **口袋定义**：受体中任一重原子与参考 `peptide.pdb` 中任一 CA 距离 ≤ `pocket_radius_A`（默认 10 Å）的残基。
- **`encoder_mode=geometry`**：噪声写入编码用主链坐标，并写入用于刚体放置的临时受体 PDB。
- **`encoder_mode=sequence_only`**：噪声**不改变** ESM3 序列条件（与官方 `encode_protein_from_pdb` 行为一致）；仅影响放置用几何。若需与论文表述严格一致，请使用 **`geometry`**。

### 3. 目标序列截断（`sequence_trunc`）

- **强度**：0%、10%、…、40%（默认）：随机**连续片段**保留 `(1 - p)` 比例残基，并同步截取 `ProteinChain` 的序列与坐标；过短样本会跳过并记日志。

## 编码模式 `encoder_mode`

- **`geometry`（默认）**：在 `models_DPO` 基础上，对 ESM3 显式传入 PDB 主链 `structure_coords`（与序列 token 对齐，BOS/EOS 位置为 NaN），使**坐标级**扰动进入条件分布。
- **`sequence_only`**：严格调用与 `encode_protein_from_pdb` 相同的**仅序列 token** 路径（结构缺失/截断通过序列层面近似；口袋噪声对编码影响极弱）。

详见 `configs/pipeline_defaults.json` 中 `encoder_mode_note`。

## 评价指标

- **亲和力**：HDOCK 解析分（**越低越好**），复用 `results/3_Pareto_improved/compute_ppdbench_generated_affinity.py` 中 `run_hdock_pair`。
- **稳定性**：FoldX Stability 流程，复用 `compute_ppdbench_generated_stability.foldx_stability_score_single`。
- **溶解度**：Protein-Sol wrapper，复用 `compute_ppdbench_generated_solubility.solubility_score_from_seq_single`。
- **综合成功率（Success rate）**：当且仅当三项同时满足 `configs/thresholds.json` 中 `success_criteria` 时记 1，否则 0；默认阈值为占位，请按论文操作点校准。

### 衍生指标（聚合表 / `Table_5_robustness_summary.csv`）

- **Relative drop（%）**：先将亲和力转为「越高越好」`−HDOCK`，再  
  \(\mathrm{RD} = \frac{m_{\mathrm{clean}} - m_{\mathrm{pert}}}{\max(|m_{\mathrm{clean}}|,\epsilon)} \times 100\%\)  
  其它指标本身为越高越好。
- **AUDC**：以**归一化扰动强度**为横轴（结构/序列：\(x=\mathrm{pct}/40\)；口袋：\(x=\sigma/2\)），对相对下降百分比曲线做梯形积分 \(\int \mathrm{RD}(x)\,dx\)。
- **τ@10% / τ@20%**：相对下降首次达到 10% 或 20% 时的**物理扰动强度**（线性插值）；达不到则为空（图中显示 NA）。
- **Sensitivity slope**：指标（越高越好空间）对**未归一化物理强度**的一元线性回归斜率。

## 运行方式

主控脚本（一键全流程，可拆分子命令）：

```bash
cd /root/autodl-tmp/Peptide_3D/results/5_robustness/scripts

# 完整：生成 → 评价 → 聚合 → 作图（耗时长，需 GPU + HDOCK/FoldX/Protein-Sol）
python run_robustness_pipeline.py \
  --ckpt /root/autodl-tmp/Peptide_3D/logs_Ranger_no_DPO/best_model_epoch_72_loss_2.0048.pth \
  --bench-root /root/autodl-tmp/PPDbench \
  --encoder-mode geometry \
  --seed 42 \
  --n-repeats 3 \
  --skip-existing

# 仅调试单一扰动、单一强度、前两靶点
python run_robustness_pipeline.py \
  --perturbation-type structure_missing \
  --level 20 \
  --max-targets 2 \
  --repeat 0 \
  --n-repeats 1
  
# 已有 samples_*.csv，仅重新聚合与作图
python run_robustness_pipeline.py --only-aggregate
python plot_robustness_figure.py
```

常用参数：`--device`、`--gpu`、`--num-per-target`、`--top-k`、`--max-len`、`--temperature`、`--hdock-bin`、`--createpl-bin`、`--foldx-bin`、`--proteinsol-wrapper`、`--hdock-work-root`、`--foldx-work-root`、`--no-plot`、`--no-aggregate`。

**并行与断点续跑（默认开启 resume）**

- **`--eval-workers`**：每进程内对接/物化评测的并发线程数；不设时按 CPU 核数与 GPU 数自动折中（单卡约可吃满多核，与 GPU 生成流水线重叠）。
- **`--num-gpus`**：多进程多卡；`0` 表示使用当前可见的全部 GPU；与 **`--gpu`** 组合表示从第几块卡起连续占用多卡。各卡处理靶点轮转分片，写出 `samples_{tag}_partXX.csv`，结束后自动合并为 `samples_{tag}.csv`。
- **断点续跑**：默认 **`--no-resume` 未设置** 时会读取已有 `samples_*.csv`，跳过已成功行（无 `error` 列），每完成一个靶点即写回 CSV；带 `error` 的靶点会重跑。整条 `samples_{tag}.csv` 已覆盖全部靶点时会自动跳过该条件。
- **`--skip-existing`**：若该条件的合并结果文件已存在则整条件跳过（与「按靶点续跑」二选一逻辑：先判 `skip-existing`，再判全量完成）。

## 输出结构

```
5_robustness/
├── configs/           # thresholds.json, pipeline_defaults.json
├── scripts/
│   ├── run_robustness_pipeline.py
│   ├── plot_robustness_figure.py
│   └── robustness_lib/
├── logs/              # 时间戳日志
├── cache/
│   ├── peptides/      # 条件×靶点的 pep_01.. 输出
│   ├── hdock_work/    # HDOCK 工作目录
│   └── foldx_work/
├── tables/
│   ├── samples_<condition>.csv   # 样本级（每靶点 top1）
│   ├── robustness_all_samples_merged.csv
│   ├── robustness_aggregate_by_condition.csv
│   ├── robustness_aggregate_<pert>.csv
│   └── Table_5_robustness_summary.csv
├── metrics/
│   └── Table_5_robustness_summary.json
├── figures/
│   ├── Figure_5_robustness_main.pdf / .png
│   └── Figure_5_robustness_caption.txt
├── cases/
│   └── selected_cases.json   # 代表性靶点（结构缺失高噪声）
└── tmp/                   # 放置用临时受体 PDB
```

## 主图与主表

- **主图**：`figures/Figure_5_robustness_main.pdf`（8 子图 a–h），由 `plot_robustness_figure.py` 根据聚合表绘制。
- **图注草稿（英文）**：`figures/Figure_5_robustness_caption.txt`。
- **主表**：`tables/Table_5_robustness_summary.csv`（AUDC、τ@10%、τ@20%、斜率、`clean_mean`、`max_drop` 等）。

## 复现与缓存

- 每个条件写入独立 `samples_*.csv`；`--skip-existing` 跳过已完成条件。
- **随机性**：`numpy.random.Generator` 种子由 `--seed`、重复编号、扰动类型与强度共同派生，保证可复现。
- **默认重复**：`n_repeats=3`，聚合时对 `repeat_id` 先平均再报告。

## 与既有结果的复用关系

- **不直接读取**历史 `generated_dpo_weighted_sum` 或 `multi_cands` 目录作为本章输入；本章在 `cache/peptides/` 下**重新生成**扰动条件下的肽，以便与扰动严格对齐。
- **工具链**（HDOCK 解析、FoldX、Protein-Sol、PPDbench 生成核心）均**调用** `results/3_Pareto_improved` 与 `ppdbench_generate_core.py` 的现有实现。

## 依赖说明

使用项目已有 Python 栈（PyTorch、Biopython、numpy、pandas、matplotlib 等）及 ESM3 相关依赖（`ProteinChain.from_pdb` 需要 **biotite** 等，与仓库 `model/esm` 一致）。**未新增** seaborn 等绘图库。若缺少 `HDOCK` / `FoldX` / `Protein-Sol` 可执行文件，相应分数为 `None`，流水线仍会继续并在日志中记录。

命令行中的 `--batch-size`、`--num-workers` 为接口预留位：当前生成循环为**逐靶点**执行，与 `ppdbench_generate_core` 一致。

## 已知限制

- 三种扰动在 **level=0** 时均运行「干净」生成；若一次跑完全部扰动族，**0% 条件会重复多次**（便于每条曲线自带基准）；若需共享一次 clean 基线，可自行后处理合并样本表。
- 完整 133 靶点 × 5 强度 × 3 重复 × 3 类扰动 ×（生成+HDOCK+FoldX+溶解度）计算量极大，建议先用 `--max-targets` 与 `--level` 做烟测。

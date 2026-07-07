# 环境说明

本分析工程仅依赖 `requirements_biophysical.txt` 中的 Python 包（NumPy、Pandas、Matplotlib）。不调用外部可执行程序（如 Rosetta、HADDOCK CLI），PDB 解析与几何计算均为本仓库内 Python 实现。

## 安装

```bash
cd /root/autodl-tmp/Peptide_3D/results/6_Biophysical_consistency
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_biophysical.txt
```

## 数据路径

默认从环境变量 `PEPTIDE_3D_ROOT` 读取项目根目录；未设置时使用配置文件 `config/default_config.yaml` 中的 `project_root`（指向 `Peptide_3D` 上级或 `Peptide_3D` 本身，脚本会自动解析）。

所有写入仅限本目录 `6_Biophysical_consistency` 下。

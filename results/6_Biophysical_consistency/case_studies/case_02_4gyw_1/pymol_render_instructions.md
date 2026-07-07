# 3D 渲染说明（PyMOL / ChimeraX）

本环境未嵌入光线追踪 3D 引擎；**高质量发表图**请在 PyMOL 或 ChimeraX 中完成。

## 文件

| 文件 | 用途 |
|------|------|
| `pymol_interface_selections.pml` | 界面肽/靶 `select` 与着色示例 |
| `contact_residues_peptide.tsv` / `contact_residues_target.tsv` | 界面残基清单 |
| `chimerax_commands.cxc` | ChimeraX 打开与链选择起点 |
| `figure_overall_complex_pca.png` | CA 主链 PCA 二维投影（快速总览） |

## PyMOL 建议流程

1. `pymol /root/autodl-tmp/Peptide_3D/results/5_robustness/baseline/cache/hdock_work/rfdiffusion/structure_missing_lvl10p0_r0/4gyw/model_1.pdb`
2. `@pymol_interface_selections.pml`（或粘贴其中 `select` 命令）
3. `set ray_trace_mode, 1` → `png figure_ray.png, width=2400`

## 结构文件

- 复合物 PDB：`/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline/cache/hdock_work/rfdiffusion/structure_missing_lvl10p0_r0/4gyw/model_1.pdb`

链 ID（来自 Table_S8）：肽 **P**，靶 **T**。

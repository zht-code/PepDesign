"""
综合打分：基于全队列的 rank-percentile 标准化（0–1，越大越好），
例外 ALI（聚集风险）保持「越大越差」。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def rank01_high(s: pd.Series) -> pd.Series:
    """越大越好 → 分位 0–1。"""
    return s.rank(pct=True, method="average", ascending=True)


def rank01_low(s: pd.Series) -> pd.Series:
    """越小越好 → 分位 0–1。"""
    return s.rank(pct=True, method="average", ascending=False)


def _safe_mean_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    exist = [c for c in cols if c in df.columns]
    if not exist:
        return pd.Series(np.nan, index=df.index)
    return df[exist].mean(axis=1, skipna=True)


def _prefix_merge(df: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    rest = [c for c in df.columns if c not in keys]
    out = df[keys].copy()
    for c in rest:
        out[f"{prefix}{c}"] = df[c]
    return out


def load_and_merge(
    tables_dir: Path,
    s1_name: str = "Table_S1_master_sequence_table.csv",
) -> pd.DataFrame:
    s1 = pd.read_csv(tables_dir / s1_name)
    keys = ["target_id", "peptide_id", "group"]
    base = s1[keys + [c for c in ("rank", "source", "sequence", "length") if c in s1.columns]].copy()

    s2 = pd.read_csv(tables_dir / "Table_S2_free_peptide_structure_metrics.csv")
    drop_s2 = {"sequence_table", "length_table", "pdb_sequence_inferred"}
    s2_cols = [c for c in s2.columns if c not in keys and c != "s1_row_index" and c not in drop_s2]
    s2r = _prefix_merge(s2[keys + s2_cols], keys, "s2_")

    s5 = pd.read_csv(tables_dir / "Table_S5_solubility_global_metrics.csv")
    s5_cols = [c for c in s5.columns if c not in keys]
    s5r = _prefix_merge(s5[keys + s5_cols], keys, "s5_")

    s7 = pd.read_csv(tables_dir / "Table_S7_aggregation_hotspot_summary.csv")
    s7_cols = [c for c in s7.columns if c not in keys]
    s7r = _prefix_merge(s7[keys + s7_cols], keys, "s7_")

    s8 = pd.read_csv(tables_dir / "Table_S8_interface_metrics.csv")
    s8_cols = [c for c in s8.columns if c not in keys]
    s8r = _prefix_merge(s8[keys + s8_cols], keys, "s8_")

    m = base.merge(s2r, on=keys, how="left").merge(s5r, on=keys, how="left").merge(s7r, on=keys, how="left").merge(
        s8r, on=keys, how="left"
    )
    return m


def compute_foldability_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ok = out.get("s2_analysis_status", pd.Series("", index=out.index)).astype(str) == "ok"
    clash = pd.to_numeric(out["s2_clash_count"], errors="coerce").where(ok)
    sev = pd.to_numeric(out["s2_severe_clash_count"], errors="coerce").where(ok)
    strain = pd.to_numeric(out["s2_approximate_backbone_strain_score"], errors="coerce").where(ok)
    torsion = pd.to_numeric(out["s2_torsion_proxy_score"], errors="coerce").where(ok)
    hb = pd.to_numeric(out["s2_intrapeptide_hbond_count"], errors="coerce").where(ok)
    nres = pd.to_numeric(out["s2_residue_count"], errors="coerce").replace(0, np.nan)
    hb_per = (hb / nres).where(ok)
    dih_cov = (pd.to_numeric(out["s2_n_classified_dihedrals"], errors="coerce") / nres).where(ok)

    out["FCS_r_clash"] = rank01_low(clash)
    out["FCS_r_severe"] = rank01_low(sev)
    out["FCS_r_strain"] = rank01_low(strain)
    out["FCS_r_torsion"] = rank01_high(torsion)
    out["FCS_r_hbond_density"] = rank01_high(hb_per)
    out["FCS_r_dihedral_coverage"] = rank01_high(dih_cov)

    fcs_cols = ["FCS_r_clash", "FCS_r_severe", "FCS_r_strain", "FCS_r_torsion", "FCS_r_hbond_density", "FCS_r_dihedral_coverage"]
    out["FCS"] = _safe_mean_cols(out, fcs_cols)
    return out


def compute_solubility_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ok = out.get("s5_analysis_status", pd.Series("", index=out.index)).astype(str) == "success"
    cam = pd.to_numeric(out["s5_camsol_like_score"], errors="coerce").where(ok)
    gravy = pd.to_numeric(out["s5_gravy"], errors="coerce").where(ok)
    nc = pd.to_numeric(out["s5_net_charge_ph74"], errors="coerce").where(ok)
    ar = pd.to_numeric(out["s5_aromatic_fraction"], errors="coerce").where(ok)

    out["SCS_r_camsol_like"] = rank01_high(cam)
    out["SCS_r_gravy"] = rank01_low(gravy.abs())  # 绝对值越小越好
    out["SCS_r_net_charge"] = rank01_low(nc.abs())
    out["SCS_r_aromatic"] = rank01_low(ar)
    scs_cols = ["SCS_r_camsol_like", "SCS_r_gravy", "SCS_r_net_charge", "SCS_r_aromatic"]
    out["SCS"] = _safe_mean_cols(out, scs_cols)
    return out


def compute_interface_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    has = out["s8_residue_contact_count"].notna() & (pd.to_numeric(out["s8_residue_contact_count"], errors="coerce") > 0)
    rc = pd.to_numeric(out["s8_residue_contact_count"], errors="coerce")
    hb = pd.to_numeric(out["s8_interface_hbond_count"], errors="coerce")
    ipp = pd.to_numeric(out["s8_interface_residue_count_peptide"], errors="coerce").replace(0, np.nan)
    elec = pd.to_numeric(out["s8_electrostatic_complementarity_score"], errors="coerce")
    mm = pd.to_numeric(out["s8_hydrophobic_mismatch_penalty"], errors="coerce")
    ov = pd.to_numeric(out["s8_hydrophobic_patch_overlap_score"], errors="coerce")
    pack = pd.to_numeric(out["s8_interface_packing_density_proxy"], errors="coerce")

    rc = rc.where(has)
    hb_per = (hb / ipp).where(has)
    elec = elec.where(has)
    mm = mm.where(has)
    ov = ov.where(has)
    pack = pack.where(has)

    out["ICS_r_contacts"] = rank01_high(np.log1p(rc))
    out["ICS_r_hbond"] = rank01_high(hb_per)
    out["ICS_r_elec_comp"] = rank01_high(elec)
    out["ICS_r_hyd_match"] = rank01_low(mm)
    out["ICS_r_patch_overlap"] = rank01_high(ov)
    out["ICS_r_packing"] = rank01_high(pack)
    ics_cols = [
        "ICS_r_contacts",
        "ICS_r_hbond",
        "ICS_r_elec_comp",
        "ICS_r_hyd_match",
        "ICS_r_patch_overlap",
        "ICS_r_packing",
    ]
    out["ICS"] = _safe_mean_cols(out, ics_cols)
    return out


def attach_ali(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ok = out.get("s7_analysis_status", pd.Series("", index=out.index)).astype(str) == "success"
    ali = pd.to_numeric(out["s7_aggregation_liability_index"], errors="coerce").where(ok)
    out["ALI"] = ali
    out["ALI_complement"] = 1.0 - ali.clip(0, 1)
    return out


def compute_obcs(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    wf = weights.get("fcs", 0.28)
    ws = weights.get("scs", 0.28)
    wi = weights.get("ics", 0.28)
    wa = weights.get("anti_agg", 0.16)
    ics_fill = df["ICS"].copy()
    med = ics_fill.median(skipna=True)
    ics_fill = ics_fill.fillna(med)

    fcs_fill = df["FCS"].fillna(df["FCS"].median(skipna=True))
    scs_fill = df["SCS"].fillna(df["SCS"].median(skipna=True))
    ali_c = df["ALI_complement"].fillna(df["ALI_complement"].median(skipna=True))

    return wf * fcs_fill + ws * scs_fill + wi * ics_fill + wa * ali_c


def build_table_s4(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "target_id",
        "peptide_id",
        "group",
        "rank",
        "source",
        "s2_analysis_status",
        "s2_residue_count",
        "s2_atom_count",
        "s2_clash_count",
        "s2_severe_clash_count",
        "s2_approximate_backbone_strain_score",
        "s2_torsion_proxy_score",
        "s2_intrapeptide_hbond_count",
        "s2_n_classified_dihedrals",
        "s2_helix_frac",
        "s2_sheet_frac",
        "s2_coil_frac",
        "s2_hydrophobic_cohesion_score",
        "FCS_r_clash",
        "FCS_r_severe",
        "FCS_r_strain",
        "FCS_r_torsion",
        "FCS_r_hbond_density",
        "FCS_r_dihedral_coverage",
        "FCS",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def build_table_s11(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["OBCS"] = compute_obcs(out, weights)
    keep = [
        "target_id",
        "peptide_id",
        "group",
        "rank",
        "source",
        "FCS",
        "SCS",
        "ICS",
        "ALI",
        "ALI_complement",
        "OBCS",
        "FCS_r_clash",
        "FCS_r_severe",
        "FCS_r_strain",
        "FCS_r_torsion",
        "FCS_r_hbond_density",
        "FCS_r_dihedral_coverage",
        "SCS_r_camsol_like",
        "SCS_r_gravy",
        "SCS_r_net_charge",
        "SCS_r_aromatic",
        "ICS_r_contacts",
        "ICS_r_hbond",
        "ICS_r_elec_comp",
        "ICS_r_hyd_match",
        "ICS_r_patch_overlap",
        "ICS_r_packing",
    ]
    keep = [c for c in keep if c in out.columns]
    meta = []
    for c in keep:
        if c in out.columns:
            meta.append(c)
    return out[meta].copy()


def build_table_s12(df: pd.DataFrame) -> pd.DataFrame:
    scores = ["FCS", "SCS", "ICS", "ALI", "OBCS"]
    g_main = df.groupby("target_id")[scores].agg(["mean", "std"])
    g_main.columns = [f"{agg}_{sc}" for sc, agg in g_main.columns]
    n_all = df.groupby("target_id").size().rename("n_peptides")
    cnt = df.groupby(["target_id", "group"]).size().unstack(fill_value=0)
    cnt = cnt.rename(columns={c: f"n_{c}" for c in cnt.columns})
    parts = [n_all, g_main, cnt]
    for sc in scores:
        pt = df.pivot_table(index="target_id", columns="group", values=sc, aggfunc="mean")
        pt = pt.rename(columns={c: f"mean_{sc}_{c}" for c in pt.columns})
        parts.append(pt)
    out = pd.concat(parts, axis=1).reset_index()
    return out


def group_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    scores = ["FCS", "SCS", "ICS", "ALI", "OBCS"]
    rows = []
    for grp, g in df.groupby("group"):
        r: dict[str, Any] = {"group": grp, "n_peptides": len(g)}
        for sc in scores:
            r[f"mean_{sc}"] = g[sc].mean(skipna=True)
            r[f"std_{sc}"] = g[sc].std(skipna=True)
            r[f"median_{sc}"] = g[sc].median(skipna=True)
        rows.append(r)
    return pd.DataFrame(rows)


def write_summary_report(
    path: Path,
    *,
    n_rows: int,
    weights: dict[str, float],
    group_cmp: pd.DataFrame,
    n_s2: int,
    n_s8: int,
) -> None:
    wtxt = ", ".join(f"{k}={v}" for k, v in sorted(weights.items()))
    if len(group_cmp) == 0:
        gmd = "_无分组数据_"
    else:
        try:
            gmd = group_cmp.to_markdown(index=False)
        except ImportError:
            gmd = "```\n" + group_cmp.to_string(index=False) + "\n```"
    body = f"""# Biophysical summary report

## 数据来源与规模

- 主合并行数（Table_S1 键）：**{n_rows}**
- 含 Table_S2（游离肽结构）行数：**{n_s2}**
- 含 Table_S8（界面）行数：**{n_s8}**

## 标准化与综合分定义

所有 `FCS_*`、`SCS_*`、`ICS_*` 中带 `_r_` 的列为 **rank-percentile**（Pandas `rank(pct=True)`），
在「全队列、同列非缺失值」上计算：值越大越好用 `ascending=True` 的排名分位；越小越好用 `ascending=False`。
缺失（例如无界面结果）不参与该列排名；综合分时对缺失分量取队列 **中位数** 填补后再加权（仅用于 `OBCS`，并在下表外注记）。

### Foldability Composite Score (FCS)

对 `s2_analysis_status == ok` 的样本，取以下六项分位的 **算术平均**（0–1，越大越好）：

1. `FCS_r_clash`：`clash_count` 越小越好  
2. `FCS_r_severe`：`severe_clash_count` 越小越好  
3. `FCS_r_strain`：`approximate_backbone_strain_score` 越小越好  
4. `FCS_r_torsion`：`torsion_proxy_score`（Ramachandran 允许区比例）越大越好  
5. `FCS_r_hbond_density`：`intrapeptide_hbond_count / residue_count` 越大越好  
6. `FCS_r_dihedral_coverage`：`n_classified_dihedrals / residue_count` 越大越好  

### Solubility Compatibility Score (SCS)

对 `s5_analysis_status == success` 的样本，四项分位平均：

1. `SCS_r_camsol_like`：`camsol_like_score` 越大越好（启发式，非官方 CamSol）  
2. `SCS_r_gravy`：`|gravy|` 越小越好  
3. `SCS_r_net_charge`：`|net_charge_ph74|` 越小越好  
4. `SCS_r_aromatic`：`aromatic_fraction` 越小越好（降低芳香暴露聚集倾向代理）

### Interface Complementarity Score (ICS)

对存在界面表（`residue_contact_count` 非空且 >0）的样本，六项分位平均：

1. `ICS_r_contacts`：`log1p(residue_contact_count)` 越大越好  
2. `ICS_r_hbond`：`interface_hbond_count / interface_residue_count_peptide` 越大越好  
3. `ICS_r_elec_comp`：`electrostatic_complementarity_score` 越大越好  
4. `ICS_r_hyd_match`：`hydrophobic_mismatch_penalty` 越小越好  
5. `ICS_r_patch_overlap`：`hydrophobic_patch_overlap_score` 越大越好  
6. `ICS_r_packing`：`interface_packing_density_proxy` 越大越好  

### Aggregation Liability Index (ALI)

直接采用 Table_S7 的 `aggregation_liability_index`（0–1，**越大聚集风险越高**）。
另设 `ALI_complement = 1 - ALI`（越大越好）用于 OBCS。

### Overall Biophysical Consistency Score (OBCS)

加权线性组合（权重来自 `config.yaml` → `thresholds.obcs_weights`，默认 `{wtxt}`）：

`OBCS = w_fcs * FCS_fill + w_scs * SCS_fill + w_ics * ICS_fill + w_anti_agg * ALI_complement_fill`

其中 `*_fill` 为原列缺失时用该列**全局中位数**填补（便于批量完整输出；敏感分析可改为剔除）。

## 分组对比（peptide-level）

{gmd}

## 输出文件

| 文件 | 说明 |
|------|------|
| `tables/Table_S4_foldability_summary.csv` | 折叠相关原始列 + FCS 分量与 FCS |
| `tables/Table_S11_biophysical_summary_scores.csv` | 肽级 FCS/SCS/ICS/ALI/OBCS 及全部分项 |
| `tables/Table_S12_target_level_summary.csv` | 按 `target_id` 聚合的均值/标准差及各 `group` 计数与分组均值 |

---
由 `05_build_summary_scores.py` 自动生成。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")

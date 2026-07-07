from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

_RE_MODEL = re.compile(r"model_(\d+)\.pdb$", re.I)
_RE_PDBID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
GENERATED_METHODS = frozenset({"rfdiffusion", "bindcraft", "proteingenerator"})


def load_suggested_inputs(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def discover_all_samples(project_root: Path, suggested: dict[str, Any]) -> list[Path]:
    found: list[Path] = []
    for item in suggested.get("canonical_csv_checks", []) or []:
        p = Path(item.get("absolute_path", ""))
        if p.exists() and p.name == "all_samples.csv":
            found.append(p)
    base = project_root / "results/5_robustness/baseline/raw_results"
    if base.is_dir():
        found.extend(sorted(base.glob("*/all_samples.csv")))
    return list(dict.fromkeys(found))


def read_all_samples_tables(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        df = df.copy()
        df["__sequence_source_path"] = str(p.resolve())
        if "sequence_top1" not in df.columns and "sequence" in df.columns:
            df["sequence_top1"] = df["sequence"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_clean_properties_sequences(project_root: Path) -> pd.DataFrame:
    """method+target_id -> sequence from clean_properties JSON（补全空序列）。"""
    rows: list[dict[str, str]] = []
    for sub in (
        project_root / "results/5_robustness/baseline/raw_results/rfdiffusion/clean_properties",
        project_root / "results/5_robustness/baseline/raw_results/proteingenerator/clean_properties",
        project_root / "results/5_robustness/baseline/raw_results/bindcraft/clean_properties",
    ):
        if not sub.is_dir():
            continue
        for jp in sub.glob("*.json"):
            try:
                obj = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            seq = str(obj.get("sequence") or "").strip()
            if not seq:
                continue
            tid = str(obj.get("target_id") or jp.stem).strip()
            meth = str(obj.get("method") or "").strip().lower()
            if not meth:
                pl = [x.lower() for x in jp.parts]
                for cand in ("rfdiffusion", "bindcraft", "proteingenerator"):
                    if cand in pl:
                        meth = cand
                        break
            rows.append({"target_id": tid.lower(), "method": meth, "__seq_json": seq, "__json_path": str(jp)})
    if not rows:
        return pd.DataFrame(columns=["target_id", "method", "__seq_json", "__json_path"])
    return pd.DataFrame(rows)


def discover_complex_pdbs(project_root: Path) -> list[Path]:
    roots = [
        project_root / "results/5_robustness/baseline/cache/hdock_work",
        project_root / "results/5_robustness/cache/hdock_work",
    ]
    out: list[Path] = []
    for r in roots:
        if not r.is_dir():
            continue
        out.extend(p for p in r.rglob("model_*.pdb") if p.is_file())
    return out


def parse_hdock_model(p: Path) -> tuple[str, str, str, int] | None:
    """
    返回 (method_lower_or_empty, target_id_lower, condition_tag, rank)。
    支持：
    - baseline/.../hdock_work/<method>/<condition>/<target>/model_n.pdb
    - .../hdock_work/<condition>/<target>/pep_xx/model_n.pdb
    """
    parts = p.parts
    pl = [x.lower() for x in parts]
    if "hdock_work" not in pl:
        return None
    i = pl.index("hdock_work")
    m = _RE_MODEL.search(p.name)
    rank = int(m.group(1)) if m else 0
    if "baseline" in pl and i + 3 < len(parts):
        method = parts[i + 1].lower()
        condition = parts[i + 2]
        target = parts[i + 3]
        if _RE_PDBID.match(target):
            return method, target.lower(), condition, rank
        return None
    if i + 2 < len(parts):
        condition = parts[i + 1]
        target = parts[i + 2]
        if _RE_PDBID.match(target):
            return "", target.lower(), condition, rank
    return None


def build_complex_lookup(paths: list[Path]) -> dict[tuple[str, str, str], tuple[int, str]]:
    """key=(method, target, condition) -> (best_rank, path) 取 rank 最小。"""
    best: dict[tuple[str, str, str], tuple[int, str]] = {}
    for p in paths:
        parsed = parse_hdock_model(p)
        if not parsed:
            continue
        method, target, condition, rank = parsed
        key = (method, target, condition)
        cur = best.get(key)
        if cur is None or rank < cur[0] or (rank == cur[0] and str(p) < cur[1]):
            best[key] = (rank, str(p.resolve()))
    return best


def find_complex_path(
    lookup: dict[tuple[str, str, str], tuple[int, str]],
    method: str,
    target_id: str,
    condition_tag: str,
) -> tuple[str, int]:
    m = str(method or "").strip().lower()
    t = str(target_id or "").strip().lower()
    c = str(condition_tag or "").strip()
    keys = [(m, t, c), ("", t, c)]
    for k in keys:
        if k in lookup:
            rank, path = lookup[k]
            return path, int(rank)
    # 无 method 的目录：尝试任意 method 匹配同一 (t,c)
    hits = [
        (rk, pt)
        for (km, kt, kc), (rk, pt) in lookup.items()
        if kt == t and kc == c and (not m or km == m)
    ]
    if not hits:
        return "", 0
    hits.sort(key=lambda x: (x[0], x[1]))
    return hits[0][1], int(hits[0][0])


def classify_group_from_row(method: str, pdb_path: str, notes: str) -> str:
    text = f"{notes} {pdb_path}".lower()
    if "decoy" in text or "negative" in text:
        return "decoy"
    if any(k in text for k in ("reference_sequences", "/reference/", "_reference")):
        return "reference"
    if str(method).lower() in GENERATED_METHODS:
        return "generated"
    return "unknown"


def rows_from_all_samples(df: pd.DataFrame, lookup: dict[tuple[str, str, str], tuple[int, str]]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        method = str(r.get("method", "") or "").strip()
        target_id = str(r.get("target_id", "") or "").strip().lower()
        peptide_id = str(r.get("candidate_id", "") or "").strip()
        condition_tag = str(r.get("condition_tag", "") or "").strip()
        seq = str(r.get("sequence_top1", "") or "").strip()
        free_p = str(r.get("pdb_path", "") or "").strip()
        notes_csv = str(r.get("notes", "") or "").strip()
        err = str(r.get("error", "") or "").strip()
        src_path = str(r.get("__sequence_source_path", "") or "")

        cpx, rank = find_complex_path(lookup, method, target_id, condition_tag)
        group = classify_group_from_row(method, free_p, notes_csv)

        seq_src = f"{src_path}#column=sequence_top1" if src_path else ""
        source = f"all_samples:{method}:{src_path}" if src_path else f"all_samples:{method}"

        notes = []
        if notes_csv and notes_csv.lower() not in ("nan", "none"):
            notes.append(notes_csv)
        if err and err.lower() not in ("nan", "none", ""):
            notes.append(f"error={err}")
        if not cpx:
            notes.append("complex_not_matched_by_condition")

        row = {
            "target_id": target_id,
            "peptide_id": peptide_id,
            "group": group,
            "sequence": seq,
            "length": len(seq) if seq else 0,
            "sequence_source_path": seq_src,
            "free_structure_path": free_p,
            "complex_structure_path": cpx,
            "rank": int(rank) if rank else 0,
            "source": source,
            "usable_for_free_structure_analysis": bool(free_p and Path(free_p).exists()),
            "usable_for_interface_analysis": bool(cpx and Path(cpx).exists()),
            "usable_for_solubility_analysis": bool(seq),
            "notes": " | ".join(notes) if notes else "",
            "meta_method": method,
            "meta_condition_tag": condition_tag,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def supplement_sequence_from_json(master: pd.DataFrame, seq_df: pd.DataFrame) -> pd.DataFrame:
    if master.empty or seq_df.empty:
        return master
    m = master.copy()
    if "meta_method" not in m.columns:
        return m
    j = seq_df.rename(columns={"__seq_json": "_json_seq", "method": "meta_method_right"})
    merged = m.merge(
        j,
        left_on=["target_id", "meta_method"],
        right_on=["target_id", "meta_method_right"],
        how="left",
    )
    need = merged["sequence"].astype(str).str.len() == 0
    merged.loc[need, "sequence"] = merged.loc[need, "_json_seq"].fillna("")
    merged.loc[need, "length"] = merged.loc[need, "sequence"].astype(str).str.len()
    merged.loc[need, "sequence_source_path"] = merged.loc[need, "__json_path"].fillna(
        merged.loc[need, "sequence_source_path"]
    )
    merged.loc[need, "usable_for_solubility_analysis"] = merged.loc[need, "sequence"].astype(
        str
    ).str.len() > 0
    merged = merged.drop(
        columns=[c for c in ("_json_seq", "meta_method_right", "__json_path") if c in merged.columns],
        errors="ignore",
    )
    return merged


def rows_from_baseline_index(path: Path, existing_free: set[str], lookup: dict) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        idx = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    rows = []
    for _, r in idx.iterrows():
        pdb_path = str(r.get("pdb_path", "") or "").strip()
        if not pdb_path or pdb_path in existing_free:
            continue
        method = str(r.get("method", "") or "").strip()
        target_id = str(r.get("target_id", "") or "").strip().lower()
        peptide_id = str(r.get("candidate_id", "") or "").strip()
        group = classify_group_from_row(method, pdb_path, "")
        cpx, rank = find_complex_path(lookup, method, target_id, "")
        notes = ["from_baseline_input_index"]
        if not cpx:
            notes.append("complex_unmatched_no_condition_tag")
        row = {
            "target_id": target_id,
            "peptide_id": peptide_id,
            "group": group,
            "sequence": "",
            "length": 0,
            "sequence_source_path": str(path.resolve()) + "#column=(none)",
            "free_structure_path": pdb_path,
            "complex_structure_path": cpx,
            "rank": int(rank) if rank else 0,
            "source": f"baseline_input_index:{method}:{path}",
            "usable_for_free_structure_analysis": bool(pdb_path and Path(pdb_path).exists()),
            "usable_for_interface_analysis": bool(cpx and Path(cpx).exists()),
            "usable_for_solubility_analysis": False,
            "notes": " | ".join(notes),
            "meta_method": method,
            "meta_condition_tag": "",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def dedupe_master(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_free_ok"] = out["free_structure_path"].map(lambda p: Path(p).exists() if p else False)
    out["_cpx_ok"] = out["complex_structure_path"].map(lambda p: Path(p).exists() if p else False)
    out["_seq_ok"] = out["sequence"].astype(str).str.len() > 0
    out["_cred"] = (
        out["_seq_ok"].astype(int) * 5
        + out["_free_ok"].astype(int) * 3
        + out["_cpx_ok"].astype(int) * 4
        + out["length"].fillna(0).astype(int).clip(upper=200) / 200.0
    )
    out["__dedupe_key"] = (
        out["meta_method"].astype(str).str.lower()
        + "|"
        + out["target_id"].astype(str).str.lower()
        + "|"
        + out["peptide_id"].astype(str)
        + "|"
        + out["meta_condition_tag"].astype(str)
    )
    out = out.sort_values("_cred", ascending=False)
    out = out.drop_duplicates(subset=["__dedupe_key"], keep="first")
    out = out.drop(
        columns=[c for c in ("_cred", "_free_ok", "_cpx_ok", "_seq_ok", "__dedupe_key") if c in out.columns],
        errors="ignore",
    )
    # 次要：同一非空 free_structure_path 只保留一条（更可 cred 的在前）
    nonempty = out[out["free_structure_path"].astype(str).str.len() > 0].copy()
    empty = out[out["free_structure_path"].astype(str).str.len() == 0].copy()
    nonempty = nonempty.sort_values(
        ["usable_for_interface_analysis", "usable_for_solubility_analysis", "length"],
        ascending=False,
    ).drop_duplicates(subset=["free_structure_path"], keep="first")
    out = pd.concat([nonempty, empty], ignore_index=True)
    return out.reset_index(drop=True)


def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "target_id",
        "peptide_id",
        "group",
        "sequence",
        "length",
        "sequence_source_path",
        "free_structure_path",
        "complex_structure_path",
        "rank",
        "source",
        "usable_for_free_structure_analysis",
        "usable_for_interface_analysis",
        "usable_for_solubility_analysis",
        "notes",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c not in ("length", "rank") else 0
    if "usable_for_free_structure_analysis" in df.columns:
        for c in (
            "usable_for_free_structure_analysis",
            "usable_for_interface_analysis",
            "usable_for_solubility_analysis",
        ):
            df[c] = df[c].astype(bool)
    return df[cols]


def build_master_table(project_root: Path, suggested: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_paths = discover_all_samples(project_root, suggested)
    raw = read_all_samples_tables(all_paths)
    cpx_paths = discover_complex_pdbs(project_root)
    lookup = build_complex_lookup(cpx_paths)

    meta: dict[str, Any] = {
        "n_all_samples_files": len(all_paths),
        "n_all_samples_rows_raw": int(len(raw)),
        "n_complex_models_indexed": len(lookup),
        "n_complex_model_files": len(cpx_paths),
        "all_samples_paths": [str(p) for p in all_paths],
    }

    m1 = rows_from_all_samples(raw, lookup)
    existing_free = set(m1["free_structure_path"].astype(str)) if not m1.empty else set()
    idx_path = project_root / "results/5_robustness/baseline/tables/baseline_input_index.csv"
    m2 = rows_from_baseline_index(idx_path, existing_free, lookup)
    if not m2.empty:
        meta["n_index_only_rows"] = int(len(m2))
    m = pd.concat([m1, m2], ignore_index=True) if not m2.empty else m1
    if m.empty:
        return m, meta

    seq_df = load_clean_properties_sequences(project_root)
    m = supplement_sequence_from_json(m, seq_df)
    m = dedupe_master(m)
    m = m.drop(columns=[c for c in ("meta_method", "meta_condition_tag") if c in m.columns], errors="ignore")
    m = finalize_columns(m)
    meta["n_rows_final"] = int(len(m))
    meta["group_counts"] = m["group"].value_counts().to_dict()
    full_targets: list[str] = []
    for tid, g in m.groupby("target_id"):
        if (
            bool(g["usable_for_free_structure_analysis"].any())
            and bool(g["usable_for_interface_analysis"].any())
            and bool(g["usable_for_solubility_analysis"].any())
        ):
            full_targets.append(str(tid))
    meta["n_targets_full_analysis"] = int(len(full_targets))
    meta["targets_full_analysis"] = sorted(set(full_targets))
    meta["n_missing_free"] = int((~m["usable_for_free_structure_analysis"]).sum())
    meta["n_missing_complex"] = int((~m["usable_for_interface_analysis"]).sum())
    meta["n_missing_sequence"] = int((~m["usable_for_solubility_analysis"]).sum())
    return m, meta


def write_master_table_report(
    path: Path,
    df: pd.DataFrame,
    meta: dict[str, Any],
) -> None:
    lines = [
        "# Master table report（Table S1）",
        "",
        "## 规模",
        "",
        f"- **主表肽条目数（行）**：{len(df)}",
        f"- **all_samples 原始行数**：{meta.get('n_all_samples_rows_raw', 'N/A')}",
        f"- **索引的对接模型 key 数**：{meta.get('n_complex_models_indexed', 'N/A')}（来自 **{meta.get('n_complex_model_files', 'N/A')}** 个 `model_*.pdb` 文件）",
        "",
        "## 按 group 统计",
        "",
    ]
    gc = meta.get("group_counts") or {}
    for k, v in sorted(gc.items(), key=lambda x: -x[1]):
        lines.append(f"- **{k}**：{v}")
    lines += [
        "",
        "## 可做「完整三项」分析的 target",
        "",
        "定义：同一 `target_id` 下至少存在一条记录，同时满足游离结构、界面复合物、序列三项可用。",
        "",
        f"- **满足条件的 target 数**：**{meta.get('n_targets_full_analysis', 0)}**",
        "",
    ]
    tlist = meta.get("targets_full_analysis") or []
    if tlist:
        lines.append("示例 target_id（按字母序，最多列 80 个）：")
        lines.append("")
        lines.append("`" + "`, `".join(tlist[:80]) + "`" + (" …" if len(tlist) > 80 else ""))
        lines.append("")
    lines += [
        "## 关键文件缺失统计（按行）",
        "",
        f"- 缺游离结构路径或文件不存在：**{meta.get('n_missing_free', 0)}**",
        f"- 缺对接复合物或文件不存在：**{meta.get('n_missing_complex', 0)}**",
        f"- 缺序列（且未能由 clean_properties 补全）：**{meta.get('n_missing_sequence', 0)}**",
        "",
        "## 说明",
        "",
        "- 主表唯一入口：`Table_S1_master_sequence_table.csv` / `.json`。",
        "- `baseline_input_index` 补充行通常 **无 `condition_tag`**，对接模型匹配较保守；详见各行列 `notes`。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decoy_generation_plan(path: Path) -> None:
    text = """# Decoy 生成计划（占位）

当前主表中 **未出现 `group=decoy`** 条目。后续建议基于 **`group=generated`** 的序列集合构建阴性对照，用于稳健性评估与过拟合检验。

## 1. Shuffle decoy（序列打乱对照）

- **输入**：从 `Table_S1_master_sequence_table.csv` 筛选 `group=generated` 且 `usable_for_solubility_analysis=true` 的 `sequence`。
- **操作**：在**保留氨基酸组成**（ multiset 不变）的前提下，对每条序列随机打乱顺序（Fisher–Yates）；可固定随机种子以保证可复现。
- **约束**：避免产生与原始序列完全相同的排列；可对 Pro/Cys 等结构敏感残基施加局部约束（可选）。
- **输出**：新列 `group=decoy_shuffle`，`peptide_id` 加后缀 `_shuffle{k}`，`sequence_source_path` 标注 `synthetic:shuffle`。

## 2. Random matched decoy（组成匹配随机序列）

- **输入**：同上 generated 序列；对每条序列计算长度与氨基酸频率向量。
- **操作**：从预定义氨基酸池或背景分布中**随机抽样**生成同长度序列，使期望频率接近原序列（可用多项式采样或迭代拒绝采样）。
- **输出**：`group=decoy_random_matched`，`notes` 中记录采样版本与种子。

## 3. 与主表合并

- 将 decoy 行追加到主表副本（或单独 `Table_S1_decoys.csv`），并在后续结构/界面步骤中 **跳过** `free_structure_path` / `complex_structure_path`（除非另行建模）。

## 4. 质量检查

- 校验 decoy 与原始序列的编辑距离、疏水性（GRAVY）分布，避免生成极端不可理化序列。
"""
    path.write_text(text, encoding="utf-8")

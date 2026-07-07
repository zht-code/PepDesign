#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Affinity-related metrics: grouped bar charts (Top-1 / Top-3) for 4 methods.

================================================================================
指标定义（与实现严格一致）
================================================================================

记 HDOCK 原始分为 score_raw（越小表示对接越好）。定义正向亲和力：
    affinity_value = - score_raw
（越大越好。）

对每个 target、每种方法，设该方法在该 target 下全部合法候选的 affinity_value 集合为 {a_1,...,a_n}，
均值 mean_all = mean({a_i})。取排序后 a_(1) >= a_(2) >= ...（即最优在前）。

一、Mean affinity（按图 Top-1 / Top-3 不同）
  - Top-1 图：每个 target 用 a_(1)；全集上再对 target 取算术平均。
  - Top-3 图：每个 target 用 mean(a_(1), a_(2), a_(3))；若 n<3，用实际可用 top-n 平均（打印 warning）。
  - 最终 Mean affinity = 对所有参与统计的 target 的上述值取平均。

二、Hit rate@T（可配置阈值 T，默认 T=100）
  - 已转为越大越好的 affinity_value。
  - 每个 target：Top-1 图用该 target 的 top1 affinity_value；Top-3 图用 top3 平均 affinity_value。
  - 若 affinity_value >= T 则该 target 为 hit。
  - Hit rate@T = hit 的 target 数 / 参与统计的 target 数。

三、Specificity score（缺少 off-target 对接时的 target specificity proxy）
  - 衡量「最优候选」相对于「该 target 上全部候选平均表现」的突出程度（概率质量是否更集中于更优候选）。
  - Top-1 图：对每个 target
        spec_t = a_(1) / (mean_all + eps)
  - Top-3 图：对每个 target，top3_mean = mean(a_(1),a_(2),a_(3))（不足 3 同前）
        spec_t = top3_mean / (mean_all + eps)
  - eps = 1e-8
  - 最终 Specificity score = 对全部参与 target 的 spec_t 取平均。

注意：这不是严格生化 off-target 特异性，仅在有 on-target 多候选时的 proxy。

================================================================================
绘图说明
================================================================================
Mean affinity、Hit rate@T（0~1）、Specificity（常在 ~1 附近）量纲不同。若共用同一绝对纵轴，
Hit rate 会几乎看不见。因此柱高使用「每种指标在 4 种方法间 min-max 归一化到 0~100」仅用于视觉比例；
柱顶文字始终标注该指标的原始数值（3 位小数）。y 轴标签在图中说明此处理方式。

输出：300 dpi PNG + pdf.fonttype=42 矢量 PDF（Adobe Illustrator 友好）。

依赖：Python 3.9+，matplotlib，numpy（标准库 json / pathlib / typing）

运行：
    python plot_affinity_grouped_bar.py

输出目录（默认）：本脚本所在目录（results/4_ablation/plot）
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 全局可配置
# ---------------------------------------------------------------------------
EPS = 1e-8
DEFAULT_HIT_T = 100.0

# Nature 系期刊常用：低饱和、偏淡的区分色（易印刷、色盲友好）
METRIC_COLORS = [
    "#A8C5E2",  # mean affinity — 淡灰蓝（Nature 系 muted blue）
    "#F2C4B8",  # hit rate — 淡珊瑚 / 杏粉
    "#B8DCC6",  # specificity — 淡青绿 / 薄荷灰
]

METHOD_LABELS_X = ["Base", "Base+OT", "Base+DPO", "Full"]
METHOD_KEYS = ["base", "base_ot", "base_dpo", "full"]


# ---------------------------------------------------------------------------
# 1) scan_pdb_dir
# ---------------------------------------------------------------------------
def scan_pdb_dir(
    directory: Path,
    *,
    label: str = "",
    recursive: bool = False,
) -> List[Path]:
    """
    扫描目录下 .pdb 文件。目录不存在时打印 warning，返回空列表。
    """
    if not directory.exists():
        warnings.warn(f"[scan_pdb_dir] 目录不存在 ({label}): {directory}")
        return []
    if not directory.is_dir():
        warnings.warn(f"[scan_pdb_dir] 非目录 ({label}): {directory}")
        return []
    pattern = "**/*.pdb" if recursive else "*.pdb"
    files = sorted(directory.glob(pattern))
    tag = f"[{label}] " if label else ""
    print(f"{tag}PDB 数量: {len(files)}  @ {directory}")
    return files


def scan_bench_ablation_peptide_roots(
    bench_root: Path,
    subdir_suffix: str,
    *,
    label: str,
) -> None:
    """
    扫描 bench_root/*/generated_ablation_<suffix>/ 下每个靶点的 PDB 数量摘要。
    """
    if not bench_root.is_dir():
        warnings.warn(f"[scan_bench] bench_root 不存在: {bench_root}")
        return
    counts: List[int] = []
    for d in sorted(p for p in bench_root.iterdir() if p.is_dir()):
        g = d / f"generated_ablation_{subdir_suffix}"
        if not g.is_dir():
            continue
        n = len(list(g.glob("*.pdb")))
        if n:
            counts.append(n)
    if not counts:
        print(f"[{label}] 未在 {bench_root}/*/generated_ablation_{subdir_suffix}/ 找到 PDB")
        return
    arr = np.array(counts, dtype=np.int64)
    print(
        f"[{label}] 各靶点 PDB 数: min={arr.min()}, max={arr.max()}, "
        f"mean={arr.mean():.2f}, targets_with_pdb={len(counts)}"
    )


# ---------------------------------------------------------------------------
# 2) load_json
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"JSON 不存在: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {path}\n{e}") from e


# ---------------------------------------------------------------------------
# 辅助：从嵌套结构中取 score / target / candidate
# ---------------------------------------------------------------------------
_SCORE_KEYS = (
    "score",
    "hdock_score",
    "docking_score",
    "hdock",
    "affinity",
    "binding_score",
    "energy",
)
_TARGET_KEYS = ("target_id", "target", "protein", "receptor", "pdb_id", "prot_id")
_CAND_KEYS = ("candidate", "name", "pdb", "peptide", "sample_id", "id", "peptide_basename")


def _first_float(d: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return None


def _first_str(d: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            return str(d[k])
    return None


def _target_from_path_string(path_str: str, bench_root: Optional[Path]) -> Optional[str]:
    """从路径中推断 PPDbench/<target>/... 的 target。"""
    p = Path(path_str)
    parts = [x for x in p.parts if x]
    if bench_root is not None:
        try:
            rel = p.resolve().relative_to(bench_root.resolve())
            if rel.parts:
                return rel.parts[0]
        except Exception:
            pass
    # 启发式：常见目录名 multi_cands / generated_ablation_*
    for i, name in enumerate(parts):
        if name in ("multi_cands", "generated_ablation_base", "generated_ablation_base_ot", "generated_ablation_base_dpo"):
            if i > 0:
                return parts[i - 1]
    return None


def _record(
    target_id: str,
    candidate: str,
    score_raw: float,
) -> Dict[str, Any]:
    return {
        "candidate": candidate,
        "score_raw": float(score_raw),
        "affinity_value": float(-score_raw),
    }


# ---------------------------------------------------------------------------
# 3) parse_method_scores + 4) normalize_to_target_dict
# ---------------------------------------------------------------------------
def normalize_to_target_dict(
    records: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """合并为 target_id -> [ {candidate, score_raw, affinity_value}, ... ]"""
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        tid = r["target_id"]
        out[tid].append(
            {
                "candidate": r["candidate"],
                "score_raw": r["score_raw"],
                "affinity_value": r["affinity_value"],
            }
        )
    return dict(out)


def parse_method_scores(
    data: Any,
    *,
    source_path: Path,
    bench_root: Optional[Path] = None,
    hint: str = "auto",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    将任意常见 JSON 结构解析为统一 target -> candidates 格式。
    hint: 'auto' | 'ablation_dict' | 'path_to_float'
    """
    records: List[Dict[str, Any]] = []

    def try_add(tid: Optional[str], cand: str, score: Optional[float]) -> None:
        if tid is None or score is None:
            return
        if not tid.strip():
            return
        try:
            sf = float(score)
        except (TypeError, ValueError):
            return
        if not np.isfinite(sf):
            return
        records.append(
            {
                "target_id": tid.strip(),
                "candidate": cand or "unknown",
                "score_raw": sf,
                "affinity_value": -sf,
            }
        )

    # --- 显式 hint ---
    if hint == "path_to_float" and isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (int, float)):
                tid = _target_from_path_string(str(k), bench_root)
                cand = Path(str(k)).name
                try_add(tid, cand, float(v))
        if not records:
            raise ValueError(f"path_to_float 解析未得到任何记录: {source_path}")
        return normalize_to_target_dict(records)

    if hint == "ablation_dict" and isinstance(data, dict):
        for key, v in data.items():
            if not isinstance(v, dict):
                continue
            tid = _first_str(v, _TARGET_KEYS)
            if not tid and isinstance(key, str) and "/" in key:
                tid = key.split("/", 1)[0]
            sc = _first_float(v, _SCORE_KEYS)
            cand = _first_str(v, _CAND_KEYS) or (key if isinstance(key, str) else "unknown")
            try_add(tid, str(cand), sc)
        if not records:
            raise ValueError(f"ablation_dict 解析未得到任何记录: {source_path}")
        return normalize_to_target_dict(records)

    # --- auto ---
    if isinstance(data, dict):
        # 子类型 A: 键像路径、值为 float（Full multi_cands）
        float_vals = sum(1 for _k, v in data.items() if isinstance(v, (int, float)))
        if float_vals >= max(1, len(data) // 2):
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    tid = _target_from_path_string(str(k), bench_root)
                    cand = Path(str(k)).name
                    try_add(tid, cand, float(v))
            if records:
                return normalize_to_target_dict(records)
            records.clear()

        # 子类型 B: 键为 target，值为 list[dict]
        for tk, v in data.items():
            if isinstance(v, list):
                tid_guess = str(tk)
                for item in v:
                    if not isinstance(item, dict):
                        continue
                    tid = _first_str(item, _TARGET_KEYS) or tid_guess
                    sc = _first_float(item, _SCORE_KEYS)
                    cand = _first_str(item, _CAND_KEYS) or "unknown"
                    try_add(tid, cand, sc)
                continue
            if isinstance(v, dict):
                tid = _first_str(v, _TARGET_KEYS) or (str(tk) if tk else None)
                sc = _first_float(v, _SCORE_KEYS)
                cand = _first_str(v, _CAND_KEYS) or str(tk)
                try_add(tid, str(cand), sc)

        if records:
            return normalize_to_target_dict(records)
        records.clear()

        # 子类型 C: ablation 风格 顶 dict，键 target/pep
        for key, v in data.items():
            if isinstance(v, dict):
                tid = _first_str(v, _TARGET_KEYS)
                if not tid and isinstance(key, str) and "/" in key:
                    tid = key.split("/", 1)[0]
                sc = _first_float(v, _SCORE_KEYS)
                cand = _first_str(v, _CAND_KEYS) or (key if isinstance(key, str) else "unknown")
                try_add(tid, str(cand), sc)
        if records:
            return normalize_to_target_dict(records)
        raise ValueError(
            f"无法从 dict 结构解析出任何候选: {source_path}\n"
            f"（请检查 score/target 字段或路径键格式）"
        )

    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                tid = _first_str(item, _TARGET_KEYS)
                sc = _first_float(item, _SCORE_KEYS)
                cand = _first_str(item, _CAND_KEYS) or f"item_{i}"
                try_add(tid, str(cand), sc)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # (target, score) 等
                try:
                    tid = str(item[0])
                    sc = float(item[1])
                    try_add(tid, f"item_{i}", sc)
                except Exception:
                    continue
        if not records:
            raise ValueError(f"无法从 list 结构解析: {source_path}")
        return normalize_to_target_dict(records)

    raise TypeError(f"不支持的 JSON 根类型 {type(data)}: {source_path}")


def load_full_method_from_bench(
    bench_root: Path,
    multi_subdir: str = "multi_cands",
    scores_name: str = "cands_hdock_scores.json",
) -> Dict[str, List[Dict[str, Any]]]:
    """聚合每个靶点 <target>/multi_cands/cands_hdock_scores.json（路径→分）。"""
    merged: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not bench_root.is_dir():
        raise FileNotFoundError(f"PPDbench 根目录不存在: {bench_root}")
    n_files = 0
    for target_dir in sorted(p for p in bench_root.iterdir() if p.is_dir()):
        jpath = target_dir / multi_subdir / scores_name
        if not jpath.is_file():
            continue
        n_files += 1
        tid = target_dir.name
        data = load_json(jpath)
        if not isinstance(data, dict):
            print(f"[WARN][Full] 跳过非 dict: {jpath}")
            continue
        for k, v in data.items():
            if not isinstance(v, (int, float)):
                continue
            try:
                sf = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(sf):
                continue
            cand = Path(str(k)).name
            merged[tid].append(_record(tid, cand, sf))
    if n_files == 0:
        raise FileNotFoundError(
            f"未找到任何 {bench_root}/*/{multi_subdir}/{scores_name}"
        )
    print(f"[Full] 读取 {n_files} 个靶点的 {scores_name}")
    return dict(merged)


# ---------------------------------------------------------------------------
# 5) compute_target_metrics
# ---------------------------------------------------------------------------
def compute_target_metrics(
    affinities: Sequence[float],
    *,
    hit_threshold: float,
    top_k_mode: str,
) -> Optional[Dict[str, float]]:
    """
    top_k_mode: 'top1' | 'top3'
    返回单 target 的 mean_aff（该 target 聚合值）, hit (0/1), specificity
    """
    vals = sorted([float(x) for x in affinities if np.isfinite(x)], reverse=True)
    if not vals:
        return None
    mean_all = float(np.mean(vals))

    if top_k_mode == "top1":
        top_agg = vals[0]
    elif top_k_mode == "top3":
        k = min(3, len(vals))
        top_agg = float(np.mean(vals[:k]))
    else:
        raise ValueError(top_k_mode)

    spec = top_agg / (mean_all + EPS)
    hit = 1.0 if top_agg >= hit_threshold else 0.0
    return {
        "top_agg": top_agg,
        "mean_all": mean_all,
        "hit": hit,
        "specificity": spec,
        "n_cand": float(len(vals)),
    }


# ---------------------------------------------------------------------------
# 6) aggregate_metrics
# ---------------------------------------------------------------------------
def aggregate_metrics(
    method_data: Dict[str, List[Dict[str, Any]]],
    *,
    hit_threshold: float,
    top_k_mode: str,
    common_targets: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, float], List[str], Dict[str, Any]]:
    """
    返回 (metrics_dict, targets_used, debug_info)
    """
    if common_targets is not None:
        targets = [t for t in common_targets if t in method_data]
    else:
        targets = sorted(method_data.keys())

    per_target_top_agg: List[float] = []
    per_target_hit: List[float] = []
    per_target_spec: List[float] = []
    under3_warnings = 0

    for tid in targets:
        rows = method_data.get(tid) or []
        affs = [r["affinity_value"] for r in rows if "affinity_value" in r]
        if not affs:
            continue
        if top_k_mode == "top3" and len(affs) < 3:
            under3_warnings += 1
            print(
                f"[WARN][{top_k_mode}] target={tid} 候选数={len(affs)} < 3，"
                f"top3 退化为对现有 {len(affs)} 个取平均"
            )

        m = compute_target_metrics(affs, hit_threshold=hit_threshold, top_k_mode=top_k_mode)
        if m is None:
            continue
        per_target_top_agg.append(m["top_agg"])
        per_target_hit.append(m["hit"])
        per_target_spec.append(m["specificity"])

    if not per_target_top_agg:
        raise RuntimeError("没有可用的 target 指标（请检查数据与 common_targets）")

    out = {
        "mean_affinity": float(np.mean(per_target_top_agg)),
        "hit_rate_at_T": float(np.mean(per_target_hit)),
        "specificity_score": float(np.mean(per_target_spec)),
    }
    debug = {
        "n_targets": len(per_target_top_agg),
        "under3_count": under3_warnings,
    }
    return out, targets, debug


def intersection_targets(
    all_methods: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> List[str]:
    sets = []
    for _mk, data in all_methods.items():
        tset = {t for t, rows in data.items() if rows}
        sets.append(tset)
    if not sets:
        return []
    inter = set.intersection(*sets)
    return sorted(inter)


# ---------------------------------------------------------------------------
# 7) plot_grouped_bar
# ---------------------------------------------------------------------------
def _minmax_scale_to_100(values: Sequence[float]) -> np.ndarray:
    v = np.array(values, dtype=np.float64)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < EPS:
        return np.full_like(v, 50.0)
    return (v - lo) / (hi - lo + EPS) * 100.0


def plot_grouped_bar(
    metrics_by_method: Dict[str, Dict[str, float]],
    *,
    out_png: Path,
    out_pdf: Path,
    title: str,
    hit_threshold: float,
    dpi: int = 300,
) -> None:
    """
    metrics_by_method: method_key -> { mean_affinity, hit_rate_at_T, specificity_score }
    柱高：各指标在 4 方法间分别 min-max 到 0~100；柱顶显示原始值（3 位小数）。
    """
    methods_order = METHOD_KEYS
    metric_keys = ["mean_affinity", "hit_rate_at_T", "specificity_score"]
    metric_labels = [
        "Mean affinity",
        f"Hit rate@T (T={hit_threshold:g})",
        "Specificity score",
    ]

    n_m = len(methods_order)
    n_b = len(metric_keys)
    means = np.array([[metrics_by_method[mk][bk] for bk in metric_keys] for mk in methods_order])

    heights = np.zeros_like(means)
    for j in range(n_b):
        heights[:, j] = _minmax_scale_to_100(means[:, j])

    fig_w = max(8.0, 1.4 * n_m + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8), layout="tight")
    x = np.arange(n_m, dtype=np.float64)
    total_w = 0.72
    bar_w = total_w / n_b
    offsets = (np.arange(n_b) - (n_b - 1) / 2.0) * bar_w

    for j in range(n_b):
        xs = x + offsets[j]
        bars = ax.bar(
            xs,
            heights[:, j],
            width=bar_w * 0.92,
            color=METRIC_COLORS[j],
            edgecolor="#9AA5B1",
            linewidth=0.55,
            label=metric_labels[j],
        )
        for i, rect in enumerate(bars):
            raw = means[i, j]
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 1.2,
                f"{raw:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS_X)
    ax.set_ylabel("Score (0–100 within each metric; labels = raw)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.margins(x=0.02)
    ymax = float(np.max(heights)) if heights.size else 100.0
    ax.set_ylim(0, min(115.0, ymax * 1.15 + 18.0))

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8) main
# ---------------------------------------------------------------------------
def main() -> int:
    here = Path(__file__).resolve().parent
    ablation_dir = here.parent
    bench_root = Path("/root/autodl-tmp/PPDbench")

    paths_json = {
        "base": ablation_dir / "ppdbench_hdock_ablation_base.json",
        "base_ot": ablation_dir / "ppdbench_hdock_ablation_base_ot.json",
        "base_dpo": ablation_dir / "ppdbench_hdock_ablation_base_dpo.json",
    }

    # 示例 PDB 目录（用于统计；不用于算分）
    pdb_demo_dirs = {
        "base": bench_root / "1cjr" / "generated_ablation_base",
        "base_ot": bench_root / "1cjr" / "generated_ablation_base_ot",
        "base_dpo": bench_root / "1cjr" / "generated_ablation_base_dpo",
    }

    hit_T = DEFAULT_HIT_T
    out_png_top1 = here / "top1_grouped_bar.png"
    out_pdf_top1 = here / "top1_grouped_bar.pdf"
    out_png_top3 = here / "top3_grouped_bar.png"
    out_pdf_top3 = here / "top3_grouped_bar.pdf"
    csv_top1 = here / "top1_metrics.csv"
    csv_top3 = here / "top3_metrics.csv"

    print("=" * 72)
    print("配置: Hit 阈值 T =", hit_T, "| affinity_value = -hdock_raw")
    print("JSON:", paths_json)
    print("PPDbench:", bench_root)
    print("=" * 72)

    # --- PDB 扫描（示例目录 + 全局摘要）---
    for k, pdir in pdb_demo_dirs.items():
        scan_pdb_dir(pdir, label=f"PDB-demo-{k}")
    scan_bench_ablation_peptide_roots(bench_root, "base", label="Bench-Base")
    scan_bench_ablation_peptide_roots(bench_root, "base_ot", label="Bench-Base+OT")
    scan_bench_ablation_peptide_roots(bench_root, "base_dpo", label="Bench-Base+DPO")

    all_methods: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    # --- 加载三种 ablation JSON ---
    for mk, jpath in paths_json.items():
        try:
            raw = load_json(jpath)
            parsed = parse_method_scores(raw, source_path=jpath, bench_root=bench_root, hint="auto")
            all_methods[mk] = parsed
            print(f"[{mk}] 解析 target 数: {len(parsed)}  @ {jpath.name}")
            cand_counts = [len(v) for v in parsed.values()]
            if cand_counts:
                a = np.array(cand_counts)
                print(
                    f"    每 target 候选数: min={a.min()}, max={a.max()}, mean={a.mean():.2f}"
                )
        except Exception as e:
            print(f"[ERROR] 方法 {mk} JSON 失败: {e}", file=sys.stderr)
            return 1

    # --- Full：聚合各靶点 multi_cands ---
    try:
        all_methods["full"] = load_full_method_from_bench(bench_root)
        print(f"[full] 解析 target 数: {len(all_methods['full'])}")
        cc = [len(v) for v in all_methods["full"].values()]
        if cc:
            a = np.array(cc)
            print(f"    每 target 候选数: min={a.min()}, max={a.max()}, mean={a.mean():.2f}")
    except Exception as e:
        print(f"[ERROR] Full 方法加载失败: {e}", file=sys.stderr)
        return 1

    common = intersection_targets(all_methods)
    print(f"\n四种方法交集 target 数（用于横向对比）: {len(common)}")
    if len(common) < 1:
        print("[ERROR] 无公共 target，无法绘图", file=sys.stderr)
        return 1

    for mk in METHOD_KEYS:
        lt3 = sum(1 for t in common if 0 < len(all_methods[mk].get(t, [])) < 3)
        if lt3:
            print(
                f"[WARN][{mk}] 交集 target 中仅 {lt3} 个候选数 <3（Top-3 将用不足 3 条的平均）"
            )

    # --- 聚合指标 ---
    results_top1: Dict[str, Dict[str, float]] = {}
    results_top3: Dict[str, Dict[str, float]] = {}
    debug_top1: Dict[str, Any] = {}
    debug_top3: Dict[str, Any] = {}

    for mk in METHOD_KEYS:
        m1, _, d1 = aggregate_metrics(
            all_methods[mk],
            hit_threshold=hit_T,
            top_k_mode="top1",
            common_targets=common,
        )
        m3, _, d3 = aggregate_metrics(
            all_methods[mk],
            hit_threshold=hit_T,
            top_k_mode="top3",
            common_targets=common,
        )
        results_top1[mk] = m1
        results_top3[mk] = m3
        debug_top1[mk] = d1
        debug_top3[mk] = d3

    print("\n--- Top-1 指标（交集 target）---")
    for mk in METHOD_KEYS:
        r = results_top1[mk]
        print(
            f"  {mk:10s}  mean_aff={r['mean_affinity']:.4f}  "
            f"hit_rate={r['hit_rate_at_T']:.4f}  spec={r['specificity_score']:.4f}  "
            f"n_targets={debug_top1[mk]['n_targets']}"
        )
    print("\n--- Top-3 指标（交集 target）---")
    for mk in METHOD_KEYS:
        r = results_top3[mk]
        print(
            f"  {mk:10s}  mean_aff={r['mean_affinity']:.4f}  "
            f"hit_rate={r['hit_rate_at_T']:.4f}  spec={r['specificity_score']:.4f}  "
            f"n={debug_top3[mk]['n_targets']}  top3_under3_warns={debug_top3[mk]['under3_count']}"
        )

    # --- CSV ---
    def write_csv(path: Path, res: Dict[str, Dict[str, float]]) -> None:
        lines = ["method,mean_affinity,hit_rate_at_T,specificity_score\n"]
        order = ["base", "base_ot", "base_dpo", "full"]
        label_map = dict(zip(order, METHOD_LABELS_X))
        for mk in order:
            r = res[mk]
            lines.append(
                f"{label_map[mk]},{r['mean_affinity']:.6f},{r['hit_rate_at_T']:.6f},{r['specificity_score']:.6f}\n"
            )
        path.write_text("".join(lines), encoding="utf-8")

    write_csv(csv_top1, results_top1)
    write_csv(csv_top3, results_top3)

    # --- 绘图 ---
    plot_grouped_bar(
        results_top1,
        out_png=out_png_top1,
        out_pdf=out_pdf_top1,
        title="Affinity-related metrics (Top-1)",
        hit_threshold=hit_T,
        dpi=300,
    )
    plot_grouped_bar(
        results_top3,
        out_png=out_png_top3,
        out_pdf=out_pdf_top3,
        title="Affinity-related metrics (Top-3)",
        hit_threshold=hit_T,
        dpi=300,
    )

    print("\n输出文件:")
    print(" ", out_png_top1)
    print(" ", out_pdf_top1)
    print(" ", out_png_top3)
    print(" ", out_pdf_top3)
    print(" ", csv_top1)
    print(" ", csv_top3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 4 种方法（Base / Base+OT / Base+DPO / Full）生成多肽的 diversity 与 novelty 做比较，并绘制二维散点图。

-----------------------------------------------------------------------------
【我对目录结构的推断（会做兼容式扫描，不会写死过窄）】
- Base / Base+OT / Base+DPO 的候选 PDB 主要位于：
    /root/autodl-tmp/PPDbench/<target_id>/generated_ablation_base*/**/*.pdb
- Full（你的方法）候选 PDB 主要位于：
    /root/autodl-tmp/PPDbench/<target_id>/multi_cands/**/*.pdb
  且 Full 的 docking 分数优先读取：
    /root/autodl-tmp/PPDbench/1cjr/multi_cands/cands_hdock_scores.json（路径->分数）
- Ablation 的 docking 分数优先在：
    /root/autodl-tmp/Peptide_3D/results/4_ablation/ppdbench_hdock_ablation_base(_ot/_dpo).json

-----------------------------------------------------------------------------
【Top-1 / Top-3 统计口径（严格按你的要求实现，并在代码中明确）】
Top-1：
- 每个 target、每个方法：从所有 candidate 中选 1 个“最佳候选”
- “最佳候选”优先按 HDOCK score（越小越好）；若无分数则按文件名排序取第一个并 WARN
- Top-1 的 diversity 使用 across-target 定义：
    对该方法所有 target 的 top1 序列集合 S_top1，计算 Diversity(S_top1)
- Top-1 的 novelty：
    对该方法所有 target 的 top1 序列 s，计算 Novelty(s)，再取平均
=> Top-1 图：每个方法一个点（基于所有 target 的 top1 序列集合统计）

Top-3：
- 每个 target、每个方法：取最佳 3 个候选（不足 3 则按实际数量取，并 WARN）
- Top-3 的 diversity（target-level diversity 再 across-target average）：
    1) 对每个 target 的 top3 序列集合 S_t，算 Diversity(S_t)
    2) 再对所有 target 取平均
- Top-3 的 novelty（target-level mean 再 across-target average）：
    1) 对每个 target 的 top3 序列逐条算 Novelty(s)
    2) 先对该 target 取均值
    3) 再 across-target 取平均
=> Top-3 图：每个方法一个点（基于每个 target 的 top3 候选集合统计后再 across-target 平均）

-----------------------------------------------------------------------------
【指标定义（经典写法，便于写入论文）】
1) Sequence identity（全局比对）：
    identity(s, t) = matches / alignment_length
   其中 matches 为全局比对后对应位置残基相同的计数，alignment_length 为对齐长度（含 gap）。

2) Diversity（平均两两序列不相似度）：
    dissimilarity(s_i, s_j) = 1 - identity(s_i, s_j)
    Diversity(S) = mean_{i<j} dissimilarity(s_i, s_j)

3) Novelty（相对 reference set）：
    Novelty(s) = 1 - max_{r in reference_set} identity(s, r)

-----------------------------------------------------------------------------
【reference set 构建策略（自动化、鲁棒、打印日志，并落盘 reference_sequences.csv）】
优先级 1：
- 在 /root/autodl-tmp/PPDbench/ 下，尽量为每个 target 找到真实/参考多肽结构（优先 peptide.pdb）
- reference set = 所有 target 的 reference peptide 序列集合
- 若某 target 找不到 reference peptide，则该 target 的 novelty 记 NaN 并 WARN（并在汇总里说明）
注意：禁止把生成序列当作 reference；本脚本不会这么做。

-----------------------------------------------------------------------------
【工作目录要求】
- 所有临时目录、缓存、pip 下载缓存都放在 /tmp 下
- 输出图片与 CSV 写到本脚本所在目录（results/4_ablation/plot）

-----------------------------------------------------------------------------
运行：
    python plot_diversity_novelty_scatter.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# 全局配置：临时目录（严格 /tmp）
# -----------------------------
TOOLS_DIR = Path("/tmp/ablation_diversity_novelty_tools")
CACHE_DIR = Path("/tmp/ablation_diversity_novelty_cache")
PIP_CACHE_DIR = Path("/tmp/ablation_diversity_novelty_pip_cache")


# -----------------------------
# 方法配置与论文风格配色
# -----------------------------
METHOD_ORDER = ["Base", "Base+OT", "Base+DPO", "Full"]
METHOD_KEYS = ["base", "base_ot", "base_dpo", "full"]
METHOD_LABEL = {"base": "Base", "base_ot": "Base+OT", "base_dpo": "Base+DPO", "full": "Full"}

# 论文常用、低饱和配色（与该项目其它图保持相近风格）
METHOD_COLORS = {
    "Base": "#4C78A8",
    "Base+OT": "#59A14F",
    "Base+DPO": "#E15759",
    "Full": "#B07AA1",
}

# docking json 常见字段（按需求列出）
SCORE_KEYS = ("score", "hdock_score", "docking_score", "hdock", "affinity", "binding_score")
TARGET_KEYS = ("target", "target_id", "protein", "receptor")
CANDIDATE_KEYS = ("candidate", "candidate_id", "name", "pdb", "peptide", "sample_id", "peptide_basename")


# =============================================================================
# 1) ensure_dir
# =============================================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# 依赖：biopython（缺失则尽量自动安装；cache 放 /tmp）
# =============================================================================
def _ensure_biopython() -> None:
    try:
        import Bio  # noqa: F401

        return
    except Exception:
        pass

    print("[deps] 未检测到 biopython，尝试自动安装（pip，缓存写入 /tmp）...")
    ensure_dir(PIP_CACHE_DIR)
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(PIP_CACHE_DIR)
    # 避免在系统盘写大量缓存（尽量）
    env.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
    cmd = [sys.executable, "-m", "pip", "install", "biopython"]
    try:
        subprocess.check_call(cmd, env=env)
    except Exception as e:
        print("[ERROR] biopython 自动安装失败。请手动安装后重试：")
        print("  pip install biopython")
        raise SystemExit(2) from e


_ensure_biopython()

from Bio import BiopythonDeprecationWarning  # noqa: E402

# 全局屏蔽 pairwise2 弃用警告（其警告可能在 import 时触发）
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from Bio import pairwise2  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402
from Bio.SeqUtils import seq1  # noqa: E402


# =============================================================================
# 数据结构
# =============================================================================
@dataclass(frozen=True)
class CandidateRecord:
    method: str
    method_key: str
    target_id: str
    candidate_id: str
    pdb_path: Path
    sequence: str


# =============================================================================
# 2) scan_pdb_files
# =============================================================================
def scan_pdb_files(root: Path, *, recursive: bool = True) -> List[Path]:
    if not root.exists():
        warnings.warn(f"[scan_pdb_files] 路径不存在: {root}")
        return []
    if root.is_file() and root.suffix.lower() == ".pdb":
        return [root.resolve()]
    pat = "**/*.pdb" if recursive else "*.pdb"
    return sorted({p.resolve() for p in root.glob(pat)})


# =============================================================================
# 3) infer_target_and_candidate
# =============================================================================
def infer_target_and_candidate(pdb_path: Path, bench_root: Path, *, method_key: str) -> Tuple[str, str]:
    """
    兼容式推断：
    - 优先：若 pdb_path 在 bench_root 下，target_id 取相对路径第一级目录名
    - 回退：从路径中找 4 字符 PDB id（如 1cjr）
    - candidate_id：优先用文件名（不含后缀），并尽量保留 pep_01 这类信息
    """
    pdb_path = pdb_path.resolve()
    bench_root = bench_root.resolve()

    target_id = ""
    try:
        rel = pdb_path.relative_to(bench_root)
        if rel.parts:
            target_id = rel.parts[0]
    except Exception:
        pass

    if not target_id:
        # 从路径中抓 1cjr / 2abc 之类
        m = re.search(r"(?i)\b([0-9][0-9a-z]{3})\b", str(pdb_path))
        if m:
            target_id = m.group(1).lower()

    if not target_id:
        target_id = "unknown_target"
        print(f"[WARN] 无法推断 target_id ({method_key}): {pdb_path}")

    candidate_id = pdb_path.stem
    if not candidate_id:
        candidate_id = pdb_path.name

    return target_id, candidate_id


# =============================================================================
# 4) extract_sequence_from_pdb
# =============================================================================
def extract_sequence_from_pdb(pdb_path: Path) -> Optional[str]:
    """
    从 PDB 中提取“更像 peptide 的链”的序列。
    策略：
    - 用 Biopython PDBParser 解析
    - 对每条链提取标准氨基酸残基序列（尽量忽略非标准残基）
    - 选择长度最短但 >=2 的链作为 peptide（通常 peptide 更短）
    - 若只有一条链则直接使用
    """
    pdb_path = pdb_path.resolve()
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception as e:
        print(f"[WARN] PDB 解析失败，跳过: {pdb_path} ({e})")
        return None

    chain_seqs: List[Tuple[str, str]] = []

    # 只取第一个 model（多数 PDB 只有一个）
    try:
        model = next(structure.get_models())
    except StopIteration:
        print(f"[WARN] PDB 无模型，跳过: {pdb_path}")
        return None

    for chain in model.get_chains():
        aa3_list: List[str] = []
        for res in chain.get_residues():
            # 过滤水/配体等：res.id[0] == ' ' 通常为标准残基
            hetflag = res.id[0]
            if hetflag not in (" ", ""):
                continue
            resname = (res.get_resname() or "").strip()
            if not resname:
                continue
            aa3_list.append(resname)
        if not aa3_list:
            continue
        # 三字母 -> 一字母；未知残基用 'X'，后续再过滤
        seq = "".join(seq1(x, custom_map={"MSE": "M"}, undef_code="X") for x in aa3_list)
        # 去掉连续的 X（保守处理：若 X 比例过高就丢弃）
        if len(seq) < 2:
            continue
        x_frac = seq.count("X") / max(1, len(seq))
        if x_frac > 0.25:
            # 非标准残基过多，容易导致 identity/novelty 不稳
            continue
        seq = seq.replace("X", "")
        if len(seq) >= 2:
            chain_seqs.append((chain.id, seq))

    if not chain_seqs:
        print(f"[WARN] 未能提取到合法序列，跳过: {pdb_path}")
        return None

    # 如果只有一条链，直接用；否则选最短链（更像 peptide）
    if len(chain_seqs) == 1:
        return chain_seqs[0][1]

    chain_seqs.sort(key=lambda x: len(x[1]))
    return chain_seqs[0][1]


# =============================================================================
# 5) scan_and_extract_sequences
# =============================================================================
def scan_and_extract_sequences(
    method_key: str,
    method_label: str,
    roots: Sequence[Path],
    bench_root: Path,
) -> List[CandidateRecord]:
    all_pdbs: List[Path] = []
    for r in roots:
        all_pdbs.extend(scan_pdb_files(r, recursive=True))
    all_pdbs = sorted(set(all_pdbs))

    print(f"[{method_label}] 扫描到 PDB 文件: {len(all_pdbs)}")

    out: List[CandidateRecord] = []
    for p in all_pdbs:
        tid, cid = infer_target_and_candidate(p, bench_root, method_key=method_key)
        seq = extract_sequence_from_pdb(p)
        if not seq:
            continue
        out.append(
            CandidateRecord(
                method=method_label,
                method_key=method_key,
                target_id=tid,
                candidate_id=cid,
                pdb_path=p,
                sequence=seq,
            )
        )

    print(f"[{method_label}] 成功提取合法序列: {len(out)}")
    print(f"[{method_label}] 识别 target 数: {len({r.target_id for r in out})}")
    return out


# =============================================================================
# 6) load_json
# =============================================================================
def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"JSON 读取失败: {path} ({e})") from e


# =============================================================================
# 7) parse_hdock_scores（通用 parser：尽量自动识别）
# =============================================================================
def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _extract_score_from_obj(obj: Any) -> Optional[float]:
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, dict):
        for k in SCORE_KEYS:
            if k in obj:
                v = _as_float(obj.get(k))
                if v is not None:
                    return v
    return None


def _extract_target_from_obj(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for k in TARGET_KEYS:
            if k in obj and obj.get(k):
                return str(obj.get(k))
    return None


def _extract_candidate_from_obj(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for k in CANDIDATE_KEYS:
            if k in obj and obj.get(k):
                return str(obj.get(k))
    return None


def parse_hdock_scores(raw: Any, *, source_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    解析后统一为：
        { target_id: [ {"candidate": "...", "score_raw": float, "source": "..."} , ... ] }

    兼容常见结构：
    1) dict[key]=record，其中 key 可能是 "target/cand.pdb" 或路径，record 内含 target_id / peptide_basename / score
    2) dict[target]=list/record
    3) list[record]
    4) Full 的 dict[path]=score（路径->分）
    """
    out: Dict[str, List[Dict[str, Any]]] = {}

    def add(tid: Optional[str], cand: Optional[str], score: Optional[float], extra: Optional[Dict[str, Any]] = None) -> None:
        if tid is None or cand is None or score is None:
            return
        tid2 = str(tid).lower()
        cand2 = Path(str(cand)).name  # 统一成 basename 风格，便于匹配
        rec = {"candidate": cand2, "score_raw": float(score), "source": str(source_path)}
        if extra:
            rec.update(extra)
        out.setdefault(tid2, []).append(rec)

    # Case: Full 的 {".../pep_01.pdb": -123.4, ...}
    if isinstance(raw, dict) and raw and all(isinstance(v, (int, float)) for v in raw.values()):
        for k, v in raw.items():
            s = _as_float(v)
            if s is None:
                continue
            # 从 key 推 target；优先从路径中抓 1cjr
            m = re.search(r"(?i)\b([0-9][0-9a-z]{3})\b", str(k))
            tid = m.group(1).lower() if m else None
            add(tid, Path(str(k)).name, s, extra={"key": str(k)})
        if not out:
            raise RuntimeError(f"无法从 dict[path]=score 解析任何分数: {source_path}")
        return out

    # Case: dict
    if isinstance(raw, dict):
        for k, v in raw.items():
            # 1) k 可能是 "1cjr/pep_01.pdb"
            tid_from_key = None
            cand_from_key = None
            if isinstance(k, str):
                m = re.search(r"(?i)\b([0-9][0-9a-z]{3})\b", k)
                if m:
                    tid_from_key = m.group(1).lower()
                cand_from_key = Path(k).name

            if isinstance(v, dict):
                tid = _extract_target_from_obj(v) or tid_from_key
                cand = _extract_candidate_from_obj(v) or v.get("peptide_basename") or v.get("peptide_pdb") or cand_from_key
                score = _extract_score_from_obj(v)
                add(tid, cand, score, extra={"key": str(k)})
            elif isinstance(v, list):
                # 2) dict[target] = [record...]
                tid = str(k).lower()
                for it in v:
                    if isinstance(it, dict):
                        cand = _extract_candidate_from_obj(it) or it.get("peptide_basename") or it.get("pdb") or it.get("peptide_pdb")
                        score = _extract_score_from_obj(it)
                        add(tid, cand, score, extra={"key": str(k)})
            else:
                # 3) dict[key] = score（但不是全为数值的情况）
                score = _as_float(v)
                if score is not None and tid_from_key is not None:
                    add(tid_from_key, cand_from_key, score, extra={"key": str(k)})

        if out:
            return out

    # Case: list
    if isinstance(raw, list):
        for it in raw:
            if isinstance(it, dict):
                tid = _extract_target_from_obj(it)
                cand = _extract_candidate_from_obj(it) or it.get("peptide_basename") or it.get("pdb") or it.get("peptide_pdb")
                score = _extract_score_from_obj(it)
                add(tid, cand, score, extra=None)
        if out:
            return out

    raise RuntimeError(f"无法解析 docking json：{source_path}（top-level 类型={type(raw)}）")


# =============================================================================
# 8) build_candidate_score_mapping
# =============================================================================
def build_candidate_score_mapping(parsed: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """
    target_id -> {candidate_basename: score_raw}
    若同一 candidate 出现多次，取最小 score（越小越好）。
    """
    out: Dict[str, Dict[str, float]] = {}
    for tid, rows in parsed.items():
        for r in rows:
            cand = Path(str(r["candidate"])).name
            score = float(r["score_raw"])
            mp = out.setdefault(tid, {})
            if cand not in mp or score < mp[cand]:
                mp[cand] = score
    return out


# =============================================================================
# 9) group_candidates_by_target
# =============================================================================
def group_candidates_by_target(records: Sequence[CandidateRecord]) -> Dict[str, List[CandidateRecord]]:
    mp: Dict[str, List[CandidateRecord]] = {}
    for r in records:
        mp.setdefault(r.target_id.lower(), []).append(r)
    # 文件名稳定排序（fallback 用）
    for tid in mp:
        mp[tid].sort(key=lambda x: x.pdb_path.name)
    return mp


# =============================================================================
# 10) select_top1_candidates
# =============================================================================
def select_top1_candidates(
    grouped: Dict[str, List[CandidateRecord]],
    score_map: Optional[Dict[str, Dict[str, float]]],
    *,
    method_label: str,
) -> Dict[str, CandidateRecord]:
    """
    对每个 target 选 1 个候选：
    - 优先按 score_map（越小越好）
    - 若某 target 无分数/无匹配，fallback 为文件名排序第一个，并打印 warning
    """
    picked: Dict[str, CandidateRecord] = {}
    used_score = 0
    used_fallback = 0

    for tid, cands in grouped.items():
        if not cands:
            continue
        best: Optional[CandidateRecord] = None
        if score_map and tid in score_map and score_map[tid]:
            # candidate basename -> score
            sc = score_map[tid]
            # 为该 target 的候选找匹配分数；匹配逻辑：basename 直接匹配；否则用 stem.pdb
            scored: List[Tuple[float, CandidateRecord]] = []
            for c in cands:
                bn = c.pdb_path.name
                score = None
                if bn in sc:
                    score = sc[bn]
                else:
                    # 尝试 "pep_01.pdb" 风格
                    alt = f"{c.pdb_path.stem}.pdb"
                    if alt in sc:
                        score = sc[alt]
                if score is not None:
                    scored.append((float(score), c))
            if scored:
                scored.sort(key=lambda x: x[0])
                best = scored[0][1]
                used_score += 1

        if best is None:
            best = sorted(cands, key=lambda x: x.pdb_path.name)[0]
            used_fallback += 1
            print(f"[WARN] [{method_label}] target={tid} 无 docking 排序信息，fallback 为文件名排序 top1: {best.pdb_path.name}")

        picked[tid] = best

    if score_map:
        print(f"[{method_label}] Top-1 选择：{len(picked)} targets | score 排序={used_score} | fallback={used_fallback}")
    else:
        print(f"[WARN] [{method_label}] 未提供 docking score 映射；Top-1 全部 fallback（文件名排序）")
        print(f"[{method_label}] Top-1 选择：{len(picked)} targets | fallback={used_fallback}")
    return picked


# =============================================================================
# 11) select_top3_candidates
# =============================================================================
def select_top3_candidates(
    grouped: Dict[str, List[CandidateRecord]],
    score_map: Optional[Dict[str, Dict[str, float]]],
    *,
    method_label: str,
    k: int = 3,
) -> Dict[str, List[CandidateRecord]]:
    """
    对每个 target 选 top-k（默认 3）候选：
    - 优先按 docking score（越小越好）
    - 无分数则按文件名排序
    - 不足 k 则按实际数量取并 WARN
    """
    picked: Dict[str, List[CandidateRecord]] = {}
    used_score = 0
    used_fallback = 0
    insufficient = 0

    for tid, cands in grouped.items():
        if not cands:
            continue

        ordered: List[CandidateRecord] = []
        if score_map and tid in score_map and score_map[tid]:
            sc = score_map[tid]
            scored: List[Tuple[float, CandidateRecord]] = []
            unscored: List[CandidateRecord] = []
            for c in cands:
                bn = c.pdb_path.name
                score = None
                if bn in sc:
                    score = sc[bn]
                else:
                    alt = f"{c.pdb_path.stem}.pdb"
                    if alt in sc:
                        score = sc[alt]
                if score is None:
                    unscored.append(c)
                else:
                    scored.append((float(score), c))
            if scored:
                scored.sort(key=lambda x: x[0])
                ordered = [c for _, c in scored] + sorted(unscored, key=lambda x: x.pdb_path.name)
                used_score += 1
            else:
                ordered = sorted(cands, key=lambda x: x.pdb_path.name)
                used_fallback += 1
                print(f"[WARN] [{method_label}] target={tid} docking 存在但未匹配到候选名，fallback 文件名排序 top3")
        else:
            ordered = sorted(cands, key=lambda x: x.pdb_path.name)
            used_fallback += 1

        kk = min(k, len(ordered))
        if kk < k:
            insufficient += 1
            print(f"[WARN] [{method_label}] target={tid} 候选数不足 {k}，实际取 {kk}")

        picked[tid] = ordered[:kk]

    if score_map:
        print(
            f"[{method_label}] Top-3 选择：{len(picked)} targets | score 排序={used_score} | fallback={used_fallback} | 不足{k}={insufficient}"
        )
    else:
        print(f"[WARN] [{method_label}] 未提供 docking score 映射；Top-3 全部 fallback（文件名排序）")
        print(f"[{method_label}] Top-3 选择：{len(picked)} targets | fallback={used_fallback} | 不足{k}={insufficient}")
    return picked


# =============================================================================
# 12) find_reference_peptides
# =============================================================================
def find_reference_peptides(bench_root: Path, target_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """
    为每个 target 找 reference/native/gt peptide 序列（优先 peptide.pdb）。
    返回：target_id -> {"sequence": str, "source_pdb": Path, "status": str}
    """
    ref: Dict[str, Dict[str, Any]] = {}

    # 优先级 1：<target>/peptide.pdb
    for tid in sorted(set(t.lower() for t in target_ids)):
        tdir = bench_root / tid
        cand_paths: List[Tuple[str, Path]] = []
        if (tdir / "peptide.pdb").exists():
            cand_paths.append(("priority:peptide.pdb", tdir / "peptide.pdb"))

        # 若 peptide.pdb 不存在，尝试关键词匹配
        if not cand_paths and tdir.exists():
            keywords = re.compile(r"(?i)(native|ref|reference|gt|ground[_-]?truth|true|crystal|peptide)")
            for p in tdir.rglob("*.pdb"):
                if keywords.search(p.name):
                    cand_paths.append(("keyword_match", p))

        # 取第一条能成功提取序列的
        for status, p in cand_paths:
            seq = extract_sequence_from_pdb(p)
            if seq:
                ref[tid] = {"sequence": seq, "source_pdb": p.resolve(), "status": status}
                break

        if tid not in ref:
            print(f"[WARN] [reference] target={tid} 未找到可用 reference peptide（novelty 将记为 NaN）")

    return ref


# =============================================================================
# 13) compute_sequence_identity（经典全局比对 identity）
# =============================================================================
def compute_sequence_identity(seq_a: str, seq_b: str) -> float:
    """
    经典全局比对 identity：
    - 用 Biopython pairwise2 全局比对（globalms），给 gap 轻微惩罚以减少过度插 gap
    - identity = matches / alignment_length（alignment_length 含 gap）
    """
    if not seq_a or not seq_b:
        return float("nan")
    if seq_a == seq_b:
        return 1.0

    # 参数选择：match=1, mismatch=0, gapopen=-1, gapextend=-0.1（常见且稳定）
    # pairwise2 在新版本 Biopython 中已标记 deprecated，但仍是“经典可复现”的实现；
    # 这里仅屏蔽其弃用警告，避免终端输出被淹没。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonDeprecationWarning)
        aln = pairwise2.align.globalms(seq_a, seq_b, 1.0, 0.0, -1.0, -0.1, one_alignment_only=True)
    if not aln:
        return float("nan")
    a_aln, b_aln, _score, _begin, _end = aln[0]
    if not a_aln or not b_aln or len(a_aln) != len(b_aln):
        return float("nan")

    matches = 0
    aln_len = len(a_aln)
    for ca, cb in zip(a_aln, b_aln):
        if ca == cb and ca != "-" and cb != "-":
            matches += 1
        elif ca == cb and ca == "-":
            # gap-gap 理论上不应出现；忽略
            pass
    return matches / max(1, aln_len)


# =============================================================================
# 14) compute_set_diversity
# =============================================================================
def compute_set_diversity(seqs: Sequence[str]) -> float:
    """
    Diversity(S) = mean_{i<j} (1 - identity(s_i, s_j))
    - n<2 返回 NaN（无法定义）
    """
    uniq = [s for s in seqs if isinstance(s, str) and s]
    if len(uniq) < 2:
        return float("nan")
    diss: List[float] = []
    for a, b in combinations(uniq, 2):
        ident = compute_sequence_identity(a, b)
        if math.isnan(ident):
            continue
        diss.append(1.0 - ident)
    if not diss:
        return float("nan")
    return float(statistics.mean(diss))


# =============================================================================
# 15) compute_novelty_for_sequence
# =============================================================================
def compute_novelty_for_sequence(seq: str, reference_seqs: Sequence[str]) -> float:
    """
    Novelty(s) = 1 - max_{r in reference_set} identity(s, r)
    reference_set 为空则返回 NaN。
    """
    refs = [r for r in reference_seqs if isinstance(r, str) and r]
    if not seq or not refs:
        return float("nan")
    best = -1.0
    for r in refs:
        ident = compute_sequence_identity(seq, r)
        if math.isnan(ident):
            continue
        if ident > best:
            best = ident
    if best < 0:
        return float("nan")
    return 1.0 - best


# =============================================================================
# 16) aggregate_top1_diversity_novelty
# =============================================================================
def aggregate_top1_diversity_novelty(
    picked_top1: Dict[str, CandidateRecord],
    reference_by_target: Dict[str, Dict[str, Any]],
    *,
    method_label: str,
) -> Tuple[float, float, int, int, str, List[str]]:
    """
    Top-1 图：每个方法一个点，基于所有 target 的 top1 peptide 集合统计
    - diversity：across-target top1 序列集合的整体 Diversity
    - novelty：对每个 target 的 top1 序列算 novelty，再取平均
             若该 target 找不到 reference，则 novelty 为 NaN，并在汇总 notes 中体现
    """
    targets = sorted(picked_top1.keys())
    seqs = [picked_top1[t].sequence for t in targets]
    # Top-1 的 diversity 采用 across-target top1 集合的 Diversity。
    # 当 target 数 < 2 时，两两组合为空，严格定义下不可计算；
    # 为了让图上“每个方法一个点”且不引入虚构差异，这里将其置为 0.0，并在 notes 中标注。
    if len(seqs) < 2:
        diversity = 0.0
        diversity_note = "diversity_undefined_n_targets_lt2->set0"
    else:
        diversity = compute_set_diversity(seqs)
        diversity_note = ""

    ref_set = [v["sequence"] for v in reference_by_target.values() if v.get("sequence")]
    novelty_vals: List[float] = []
    missing_ref_targets: List[str] = []
    for t in targets:
        if t not in reference_by_target:
            missing_ref_targets.append(t)
            continue
        nv = compute_novelty_for_sequence(picked_top1[t].sequence, ref_set)
        if not math.isnan(nv):
            novelty_vals.append(nv)
    novelty = float(statistics.mean(novelty_vals)) if novelty_vals else float("nan")
    notes = []
    if diversity_note:
        notes.append(diversity_note)
    if missing_ref_targets:
        notes.append(f"missing_ref_targets={len(missing_ref_targets)}")
    note_str = ";".join(notes) if notes else ""
    return diversity, novelty, len(targets), len(seqs), note_str, missing_ref_targets


# =============================================================================
# 17) aggregate_top3_diversity_novelty
# =============================================================================
def aggregate_top3_diversity_novelty(
    picked_top3: Dict[str, List[CandidateRecord]],
    reference_by_target: Dict[str, Dict[str, Any]],
    *,
    method_label: str,
) -> Tuple[float, float, int, int, str, List[str]]:
    """
    Top-3 图：每个方法一个点，基于每个 target 的 top3 候选集合统计后再 across-target 平均
    - diversity：
        1) per-target：Diversity(S_t)（top3 集合内两两不相似度均值）
        2) across-target：对所有 target 的 per-target diversity 取平均
    - novelty：
        1) 对每个 target 的 top3 序列逐条算 novelty
        2) 先对该 target 内取均值
        3) 再 across-target 取均值
      若该 target 找不到 reference，则该 target novelty 记 NaN（并跳过平均），并记录日志
    """
    targets = sorted(picked_top3.keys())
    ref_set = [v["sequence"] for v in reference_by_target.values() if v.get("sequence")]

    div_per_t: List[float] = []
    nov_per_t: List[float] = []
    missing_ref_targets: List[str] = []

    n_seq_total = 0
    for t in targets:
        cands = picked_top3[t]
        seqs = [c.sequence for c in cands if c.sequence]
        n_seq_total += len(seqs)

        d = compute_set_diversity(seqs)
        if not math.isnan(d):
            div_per_t.append(d)

        if t not in reference_by_target:
            missing_ref_targets.append(t)
            continue
        # target-level novelty mean
        nvs = [compute_novelty_for_sequence(s, ref_set) for s in seqs]
        nvs = [x for x in nvs if not math.isnan(x)]
        if nvs:
            nov_per_t.append(float(statistics.mean(nvs)))

    diversity = float(statistics.mean(div_per_t)) if div_per_t else float("nan")
    novelty = float(statistics.mean(nov_per_t)) if nov_per_t else float("nan")

    notes = []
    if missing_ref_targets:
        notes.append(f"missing_ref_targets={len(missing_ref_targets)}")
    note_str = ";".join(notes) if notes else ""
    return diversity, novelty, len(targets), n_seq_total, note_str, missing_ref_targets


# =============================================================================
# 18) save_csv
# =============================================================================
def save_csv(path: Path, rows: Sequence[Dict[str, Any]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# =============================================================================
# 19) plot_scatter
# =============================================================================
def plot_scatter(
    results: pd.DataFrame,
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """
    matplotlib 论文风格散点图：
    - 每个方法一个点
    - 标注文本
    - PNG 300 dpi，PDF 矢量可编辑
    """
    # 论文风格：轻网格、无上右边框
    matplotlib.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,  # TrueType，AI 更友好
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)

    # 保持固定顺序
    for lab in METHOD_ORDER:
        r = results.loc[results["method"] == lab]
        if r.empty:
            continue
        x = float(r["diversity"].iloc[0])
        y = float(r["novelty"].iloc[0])
        if not (math.isfinite(x) and math.isfinite(y)):
            print(f"[WARN] plot_scatter: 跳过非有限坐标点 method={lab} (diversity={x}, novelty={y})")
            continue
        ax.scatter(
            [x],
            [y],
            s=120,
            color=METHOD_COLORS.get(lab, "#333333"),
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        # 轻微偏移避免遮挡
        ax.text(x + 0.003, y + 0.003, lab, fontsize=11, ha="left", va="bottom")

    ax.set_xlabel("Diversity")
    ax.set_ylabel("Novelty")
    ax.set_title(title)

    # 右上角小提示：越靠右上越好
    ax.text(
        0.98,
        0.98,
        "upper-right is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#444444",
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")  # 矢量 PDF
    plt.close(fig)


# =============================================================================
# 辅助：搜索 ablation docking json（按你给的目录与关键词）
# =============================================================================
def _auto_find_ablation_jsons(ablation_dir: Path) -> Dict[str, Optional[Path]]:
    """
    在 results/4_ablation 目录中自动找 base/base_ot/base_dpo 的 docking json。
    如果找不到，返回 None（后续会触发 fallback 文件名排序并 WARN）。
    """
    out: Dict[str, Optional[Path]] = {"base": None, "base_ot": None, "base_dpo": None}

    if not ablation_dir.exists():
        return out

    # 优先：项目中已存在的标准命名
    fixed = {
        "base": ablation_dir / "ppdbench_hdock_ablation_base.json",
        "base_ot": ablation_dir / "ppdbench_hdock_ablation_base_ot.json",
        "base_dpo": ablation_dir / "ppdbench_hdock_ablation_base_dpo.json",
    }
    for k, p in fixed.items():
        if p.exists():
            out[k] = p.resolve()

    # 若缺失则关键词模糊匹配
    for k in list(out.keys()):
        if out[k] is not None:
            continue
        key_pat = {"base": r"base\b", "base_ot": r"base[_-]?ot|ot", "base_dpo": r"base[_-]?dpo|dpo"}[k]
        best: Optional[Path] = None
        for p in ablation_dir.rglob("*.json"):
            name = p.name.lower()
            if "hdock" not in name and "docking" not in name and "score" not in name and "ablation" not in name:
                continue
            if re.search(key_pat, name) and ("ablation" in name or "hdock" in name):
                best = p
                break
        if best:
            out[k] = best.resolve()

    return out


def main() -> None:
    # -----------------------------
    # 路径配置（集中在 main）
    # -----------------------------
    plot_dir = Path("/root/autodl-tmp/Peptide_3D/results/4_ablation/plot").resolve()
    ablation_dir = Path("/root/autodl-tmp/Peptide_3D/results/4_ablation").resolve()
    bench_root = Path("/root/autodl-tmp/PPDbench").resolve()

    ensure_dir(plot_dir)
    ensure_dir(TOOLS_DIR)
    ensure_dir(CACHE_DIR)
    ensure_dir(PIP_CACHE_DIR)

    # 输入目录：直接使用你给的 4 个目录（递归扫描，结构不写死）
    method_roots: Dict[str, List[Path]] = {
        "base": [Path("/root/autodl-tmp/PPDbench/1cjr/generated_ablation_base")],
        "base_ot": [Path("/root/autodl-tmp/PPDbench/1cjr/generated_ablation_base_ot")],
        "base_dpo": [Path("/root/autodl-tmp/PPDbench/1cjr/generated_ablation_base_dpo")],
        "full": [Path("/root/autodl-tmp/PPDbench/1cjr/multi_cands")],
    }

    # -----------------------------
    # 扫描并提取候选序列
    # -----------------------------
    all_records_by_method: Dict[str, List[CandidateRecord]] = {}
    for mk in METHOD_KEYS:
        lab = METHOD_LABEL[mk]
        recs = scan_and_extract_sequences(mk, lab, method_roots[mk], bench_root)
        all_records_by_method[mk] = recs

    # 写 extracted_sequences.csv
    extracted_rows: List[Dict[str, Any]] = []
    for mk in METHOD_KEYS:
        for r in all_records_by_method[mk]:
            extracted_rows.append(
                {
                    "method": r.method,
                    "target_id": r.target_id,
                    "candidate_id": r.candidate_id,
                    "pdb_path": str(r.pdb_path),
                    "sequence": r.sequence,
                    "length": len(r.sequence),
                }
            )
    save_csv(
        plot_dir / "extracted_sequences.csv",
        extracted_rows,
        fieldnames=["method", "target_id", "candidate_id", "pdb_path", "sequence", "length"],
    )

    # -----------------------------
    # 读取 / 解析 docking json（HDOCK score，越小越好）
    # -----------------------------
    ablation_jsons = _auto_find_ablation_jsons(ablation_dir)
    hdock_maps: Dict[str, Optional[Dict[str, Dict[str, float]]]] = {"base": None, "base_ot": None, "base_dpo": None, "full": None}
    hdock_mapping_rows: List[Dict[str, Any]] = []

    # Base / OT / DPO
    for mk in ("base", "base_ot", "base_dpo"):
        jp = ablation_jsons.get(mk)
        if jp and jp.exists():
            try:
                raw = load_json(jp)
                parsed = parse_hdock_scores(raw, source_path=jp)
                score_map = build_candidate_score_mapping(parsed)
                hdock_maps[mk] = score_map
                print(f"[{METHOD_LABEL[mk]}] 成功读取 docking json: {jp}")
                # 记录映射
                for tid, mp in score_map.items():
                    for cand, sc in mp.items():
                        hdock_mapping_rows.append(
                            {"method": METHOD_LABEL[mk], "target_id": tid, "candidate": cand, "score_raw": sc, "source_json": str(jp)}
                        )
            except Exception as e:
                print(f"[ERROR] [{METHOD_LABEL[mk]}] docking json 解析失败: {jp} ({e})")
                hdock_maps[mk] = None
        else:
            print(f"[WARN] [{METHOD_LABEL[mk]}] 未找到 docking json（将 fallback 文件名排序）")
            hdock_maps[mk] = None

    # Full：优先读取你指定的 cands_hdock_scores.json（已确认存在于 1cjr）
    full_score_json = Path("/root/autodl-tmp/PPDbench/1cjr/multi_cands/cands_hdock_scores.json")
    if full_score_json.exists():
        try:
            raw = load_json(full_score_json)
            parsed = parse_hdock_scores(raw, source_path=full_score_json)
            score_map = build_candidate_score_mapping(parsed)
            hdock_maps["full"] = score_map
            print(f"[Full] 成功读取 docking json: {full_score_json}")
            for tid, mp in score_map.items():
                for cand, sc in mp.items():
                    hdock_mapping_rows.append(
                        {"method": "Full", "target_id": tid, "candidate": cand, "score_raw": sc, "source_json": str(full_score_json)}
                    )
        except Exception as e:
            print(f"[ERROR] [Full] docking json 解析失败: {full_score_json} ({e})")
            hdock_maps["full"] = None
    else:
        print(f"[WARN] [Full] 未找到 {full_score_json}（将 fallback 文件名排序）")
        hdock_maps["full"] = None

    save_csv(
        plot_dir / "hdock_score_mapping.csv",
        hdock_mapping_rows,
        fieldnames=["method", "target_id", "candidate", "score_raw", "source_json"],
    )

    # -----------------------------
    # 构建 reference set（从 PPDbench/<target>/peptide.pdb）
    # -----------------------------
    # 只以本次四种方法扫描到的 targets 为候选集合（更合理）
    all_targets: List[str] = []
    for mk in METHOD_KEYS:
        all_targets.extend([r.target_id.lower() for r in all_records_by_method[mk]])
    all_targets = sorted(set(all_targets))

    reference_by_target = find_reference_peptides(bench_root, all_targets)
    ref_rows: List[Dict[str, Any]] = []
    for tid in sorted(set(all_targets)):
        if tid in reference_by_target:
            ref_rows.append(
                {
                    "target_id": tid,
                    "source_pdb": str(reference_by_target[tid]["source_pdb"]),
                    "sequence": reference_by_target[tid]["sequence"],
                    "status": reference_by_target[tid]["status"],
                }
            )
        else:
            ref_rows.append({"target_id": tid, "source_pdb": "", "sequence": "", "status": "missing"})
    save_csv(plot_dir / "reference_sequences.csv", ref_rows, fieldnames=["target_id", "source_pdb", "sequence", "status"])
    print(f"[reference] reference set 序列数（有序列者）: {sum(1 for v in reference_by_target.values() if v.get('sequence'))}")

    # -----------------------------
    # 分组 + 选择 top1 / top3
    # -----------------------------
    grouped_by_method: Dict[str, Dict[str, List[CandidateRecord]]] = {}
    for mk in METHOD_KEYS:
        grouped_by_method[mk] = group_candidates_by_target(all_records_by_method[mk])

    top1_picks: Dict[str, Dict[str, CandidateRecord]] = {}
    top3_picks: Dict[str, Dict[str, List[CandidateRecord]]] = {}
    for mk in METHOD_KEYS:
        lab = METHOD_LABEL[mk]
        top1_picks[mk] = select_top1_candidates(grouped_by_method[mk], hdock_maps.get(mk), method_label=lab)
        top3_picks[mk] = select_top3_candidates(grouped_by_method[mk], hdock_maps.get(mk), method_label=lab, k=3)

    # candidate_selection_top1.csv / top3.csv
    sel1_rows: List[Dict[str, Any]] = []
    for mk in METHOD_KEYS:
        lab = METHOD_LABEL[mk]
        for tid, rec in sorted(top1_picks[mk].items()):
            sel1_rows.append(
                {
                    "method": lab,
                    "target_id": tid,
                    "candidate_id": rec.candidate_id,
                    "pdb_path": str(rec.pdb_path),
                    "sequence": rec.sequence,
                    "length": len(rec.sequence),
                }
            )
    save_csv(
        plot_dir / "candidate_selection_top1.csv",
        sel1_rows,
        fieldnames=["method", "target_id", "candidate_id", "pdb_path", "sequence", "length"],
    )

    sel3_rows: List[Dict[str, Any]] = []
    for mk in METHOD_KEYS:
        lab = METHOD_LABEL[mk]
        for tid, recs in sorted(top3_picks[mk].items()):
            for rank, rec in enumerate(recs, start=1):
                sel3_rows.append(
                    {
                        "method": lab,
                        "target_id": tid,
                        "rank": rank,
                        "candidate_id": rec.candidate_id,
                        "pdb_path": str(rec.pdb_path),
                        "sequence": rec.sequence,
                        "length": len(rec.sequence),
                    }
                )
    save_csv(
        plot_dir / "candidate_selection_top3.csv",
        sel3_rows,
        fieldnames=["method", "target_id", "rank", "candidate_id", "pdb_path", "sequence", "length"],
    )

    # -----------------------------
    # 聚合：Top-1 / Top-3 diversity & novelty
    # -----------------------------
    top1_rows_out: List[Dict[str, Any]] = []
    top3_rows_out: List[Dict[str, Any]] = []

    missing_ref_all_top1: Dict[str, List[str]] = {}
    missing_ref_all_top3: Dict[str, List[str]] = {}

    for mk in METHOD_KEYS:
        lab = METHOD_LABEL[mk]
        d1, n1, nt1, ns1, notes1, miss1 = aggregate_top1_diversity_novelty(top1_picks[mk], reference_by_target, method_label=lab)
        top1_rows_out.append(
            {
                "method": lab,
                "diversity": d1,
                "novelty": n1,
                "n_targets": nt1,
                "n_sequences_used": ns1,
                "notes": notes1,
            }
        )
        missing_ref_all_top1[lab] = miss1

        d3, n3, nt3, ns3, notes3, miss3 = aggregate_top3_diversity_novelty(top3_picks[mk], reference_by_target, method_label=lab)
        top3_rows_out.append(
            {
                "method": lab,
                "diversity": d3,
                "novelty": n3,
                "n_targets": nt3,
                "n_sequences_used": ns3,
                "notes": notes3,
            }
        )
        missing_ref_all_top3[lab] = miss3

    top1_df = pd.DataFrame(top1_rows_out)
    top3_df = pd.DataFrame(top3_rows_out)

    top1_df.to_csv(plot_dir / "top1_diversity_novelty.csv", index=False)
    top3_df.to_csv(plot_dir / "top3_diversity_novelty.csv", index=False)

    # -----------------------------
    # 终端打印汇总（按你的要求）
    # -----------------------------
    print("\n==================== 汇总（Top-1）====================")
    for _, r in top1_df.iterrows():
        print(
            f"{r['method']:8s}  diversity={float(r['diversity']):.6f}  novelty={float(r['novelty']):.6f}  "
            f"n_targets={int(r['n_targets'])}  n_seq={int(r['n_sequences_used'])}  notes={r.get('notes','')}"
        )
    print("\n==================== 汇总（Top-3）====================")
    for _, r in top3_df.iterrows():
        print(
            f"{r['method']:8s}  diversity={float(r['diversity']):.6f}  novelty={float(r['novelty']):.6f}  "
            f"n_targets={int(r['n_targets'])}  n_seq={int(r['n_sequences_used'])}  notes={r.get('notes','')}"
        )

    # 缺 reference 的 target
    miss_any = sorted({t for v in missing_ref_all_top1.values() for t in v} | {t for v in missing_ref_all_top3.values() for t in v})
    if miss_any:
        print(f"\n[reference] 缺少 reference 的 targets（novelty 跳过/记 NaN）: {len(miss_any)}")
        print("  " + ", ".join(miss_any[:50]) + (" ..." if len(miss_any) > 50 else ""))

    # docking 读取情况
    print("\n==================== Docking JSON 读取状态 ====================")
    for mk in ("base", "base_ot", "base_dpo"):
        p = ablation_jsons.get(mk)
        ok = p is not None and p.exists()
        print(f"{METHOD_LABEL[mk]:8s} docking_json={'OK' if ok else 'MISSING'}  path={str(p) if p else ''}")
    print(f"{'Full':8s} docking_json={'OK' if (hdock_maps['full'] is not None) else 'MISSING'}  path={str(full_score_json)}")

    # -----------------------------
    # 绘图
    # -----------------------------
    plot_scatter(
        top1_df,
        title="Diversity–Novelty trade-off (Top-1)",
        out_png=plot_dir / "top1_diversity_novelty_scatter.png",
        out_pdf=plot_dir / "top1_diversity_novelty_scatter.pdf",
    )
    plot_scatter(
        top3_df,
        title="Diversity–Novelty trade-off (Top-3)",
        out_png=plot_dir / "top3_diversity_novelty_scatter.png",
        out_pdf=plot_dir / "top3_diversity_novelty_scatter.pdf",
    )

    print("\n==================== 输出文件 ====================")
    print("CSV:")
    for fn in [
        "top1_diversity_novelty.csv",
        "top3_diversity_novelty.csv",
        "extracted_sequences.csv",
        "reference_sequences.csv",
        "hdock_score_mapping.csv",
        "candidate_selection_top1.csv",
        "candidate_selection_top3.csv",
    ]:
        print(" -", str((plot_dir / fn).resolve()))
    print("Figures:")
    for fn in [
        "top1_diversity_novelty_scatter.png",
        "top1_diversity_novelty_scatter.pdf",
        "top3_diversity_novelty_scatter.png",
        "top3_diversity_novelty_scatter.pdf",
    ]:
        print(" -", str((plot_dir / fn).resolve()))


if __name__ == "__main__":
    main()


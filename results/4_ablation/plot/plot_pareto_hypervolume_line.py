#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 总结（按你的“最后输出要求”）：
# 1) 输入文件来源（推断/自动发现）：
#    - Affinity(HDOCK): Base/Base+OT/Base+DPO 来自
#        /root/autodl-tmp/Peptide_3D/results/4_ablation/ppdbench_hdock_ablation_*.json
#      Full 优先自动扫描
#        /root/autodl-tmp/PPDbench/*/multi_cands/cands_hdock_scores.json
#    - pLDDT: 优先复用 /root/autodl-tmp/Peptide_3D/results/4_ablation/plot/esmfold_plddt_cache.json
#      （key=sha256(sequence)），再 fallback 从 PDB B-factor 求均值（若非零且非恒定）
#    - stability/solubility: 在指定目录内自动搜索（若不存在则不伪造，优雅降级）
# 2) 主实验二维 Pareto 目标（默认）：
#    - (affinity_norm, developability_norm)，其中 affinity_value = -hdock_score（越大越好）
#    - developability 由 pLDDT(+stability+solubility) 的“全局 min-max 标准化后均值”构造
# 3) candidate 截断策略：
#    - 每个 (method, target) 最多用前 MAX_CANDIDATES_PER_TARGET=10 个 candidate
#    - 优先按 score_raw(HDOCK, 越小越好) 排序；若缺失则按 candidate_id 排序并 warning
# 4) Hypervolume 计算层级：
#    - candidate-level 点集 -> target-level HV -> method-level across-target 汇总(mean/std/sem)
#
# 运行：
#   python plot_pareto_hypervolume_line.py
#
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# 可配置参数
# -----------------------------

EPS = 1e-8

# 主实验：二维 Pareto（默认）
USE_THREE_OBJECTIVES = False  # 可选扩展（默认关闭，不作为主图）

MAX_CANDIDATES_PER_TARGET = 10
METHOD_ORDER = ["Base", "Base+OT", "Base+DPO", "Full"]

# 参考点（目标已全局 min-max 到 [0,1]，且均为“越大越好”）
REF_POINT_2D = (0.0, 0.0)
REF_POINT_3D = (0.0, 0.0, 0.0)


# -----------------------------
# 输出文件名（固定）
# -----------------------------

OUT_PNG = "pareto_hypervolume_line.png"
OUT_PDF = "pareto_hypervolume_line.pdf"
OUT_PNG_TOP3 = "pareto_hypervolume_line_top3.png"
OUT_PDF_TOP3 = "pareto_hypervolume_line_top3.pdf"

CSV_PER_TARGET = "pareto_hypervolume_per_target.csv"
CSV_SUMMARY = "pareto_hypervolume_summary.csv"
CSV_PER_TARGET_TOP3 = "pareto_hypervolume_per_target_top3.csv"
CSV_SUMMARY_TOP3 = "pareto_hypervolume_summary_top3.csv"
CSV_MERGED = "pareto_candidate_metrics_merged.csv"
CSV_PARETO_POINTS = "pareto_front_points.csv"
CSV_NORM_STATS = "pareto_normalization_stats.csv"

# 可选补图（尽量生成，失败不影响主流程）
BOX_PNG = "pareto_hypervolume_boxplot.png"
BOX_PDF = "pareto_hypervolume_boxplot.pdf"


@dataclass
class DiscoveredFiles:
    plot_dir: Path
    ablation_dir: Path
    ppdbench_dir: Path

    # affinity sources
    ablation_hdock_base: Path
    ablation_hdock_base_ot: Path
    ablation_hdock_base_dpo: Path
    full_hdock_jsons: List[Path]

    # sequences / plddt cache
    extracted_sequences_csv: Optional[Path]
    esmfold_plddt_cache_json: Optional[Path]

    # optional metric files to reuse (not strictly required)
    top1_metrics_csv: Optional[Path]
    top3_metrics_csv: Optional[Path]

    # extra metric candidates
    stability_like: List[Path]
    solubility_like: List[Path]
    novelty_like: List[Path]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def discover_existing_files() -> DiscoveredFiles:
    """
    在指定目录内自动发现“现成结果文件”。
    注意：这里只做文件发现，不做深度全盘扫描，以避免性能问题。
    """
    plot_dir = Path("/root/autodl-tmp/Peptide_3D/results/4_ablation/plot")
    ablation_dir = Path("/root/autodl-tmp/Peptide_3D/results/4_ablation")
    ppdbench_dir = Path("/root/autodl-tmp/PPDbench")

    ablation_hdock_base = ablation_dir / "ppdbench_hdock_ablation_base.json"
    ablation_hdock_base_ot = ablation_dir / "ppdbench_hdock_ablation_base_ot.json"
    ablation_hdock_base_dpo = ablation_dir / "ppdbench_hdock_ablation_base_dpo.json"

    # Full: one per target
    full_hdock_jsons = sorted(ppdbench_dir.glob("*/multi_cands/cands_hdock_scores.json"))

    extracted_sequences_csv = plot_dir / "extracted_sequences.csv"
    if not extracted_sequences_csv.exists():
        extracted_sequences_csv = None

    esmfold_plddt_cache_json = plot_dir / "esmfold_plddt_cache.json"
    if not esmfold_plddt_cache_json.exists():
        esmfold_plddt_cache_json = None

    top1_metrics_csv = plot_dir / "top1_metrics.csv"
    if not top1_metrics_csv.exists():
        top1_metrics_csv = None
    top3_metrics_csv = plot_dir / "top3_metrics.csv"
    if not top3_metrics_csv.exists():
        top3_metrics_csv = None

    # “轻量”搜索 stability/solubility/novelty 相关文件（只在三个主目录内按文件名关键词匹配）
    def _keyword_hits(root: Path, keywords: Sequence[str], exts: Sequence[str] = (".csv", ".tsv", ".json")) -> List[Path]:
        hits: List[Path] = []
        if not root.exists():
            return hits
        # 限制扫描深度：用 rglob 但只收集一定数量，避免过大目录卡住
        cap = 3000
        for p in root.rglob("*"):
            if len(hits) >= cap:
                break
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            name = p.name.lower()
            if any(k in name for k in keywords):
                hits.append(p)
        return sorted(set(hits), key=lambda x: (len(str(x)), str(x)))

    stability_like = (
        _keyword_hits(plot_dir, ["stability", "stable"])
        + _keyword_hits(ablation_dir, ["stability", "stable"])
        + _keyword_hits(ppdbench_dir, ["stability", "stable"])
    )
    solubility_like = (
        _keyword_hits(plot_dir, ["solubility", "soluble"])
        + _keyword_hits(ablation_dir, ["solubility", "soluble"])
        + _keyword_hits(ppdbench_dir, ["solubility", "soluble"])
    )
    novelty_like = (
        _keyword_hits(plot_dir, ["novelty", "diversity"])
        + _keyword_hits(ablation_dir, ["novelty", "diversity"])
        + _keyword_hits(ppdbench_dir, ["novelty", "diversity"])
    )

    return DiscoveredFiles(
        plot_dir=plot_dir,
        ablation_dir=ablation_dir,
        ppdbench_dir=ppdbench_dir,
        ablation_hdock_base=ablation_hdock_base,
        ablation_hdock_base_ot=ablation_hdock_base_ot,
        ablation_hdock_base_dpo=ablation_hdock_base_dpo,
        full_hdock_jsons=full_hdock_jsons,
        extracted_sequences_csv=extracted_sequences_csv,
        esmfold_plddt_cache_json=esmfold_plddt_cache_json,
        top1_metrics_csv=top1_metrics_csv,
        top3_metrics_csv=top3_metrics_csv,
        stability_like=stability_like,
        solubility_like=solubility_like,
        novelty_like=novelty_like,
    )


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _stem_id(s: str) -> str:
    s = str(s).strip().replace("\\", "/")
    s = s.split("/")[-1]
    s = re.sub(r"\.(pdb|cif|json|csv|tsv)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return s.lower()


def infer_target_candidate_from_path(p: str) -> Tuple[str, str]:
    """
    从路径/键中推断 target_id 和 candidate_id。
    """
    s = str(p).replace("\\", "/")
    cand = Path(s).name
    target = "unknown"
    m = re.search(r"/PPDbench/([^/]+)/", s)
    if m:
        target = m.group(1)
    else:
        # fallback: "1abc/pep_01.pdb"
        if "/" in s:
            target = s.split("/")[0]
    return target, cand


def parse_hdock_scores(method: str, json_path: Path) -> pd.DataFrame:
    """
    解析 HDOCK 结果，统一为 candidate-level:
      method, target_id, candidate_id, score_raw, affinity_value, pdb_path, notes

    score_raw: 原始 HDOCK score（越小越好）
    affinity_value = -score_raw（越大越好）
    """
    if not json_path.exists():
        return pd.DataFrame(
            columns=["method", "target_id", "candidate_id", "score_raw", "affinity_value", "pdb_path", "notes"]
        )
    data = load_json(json_path)
    rows: List[Dict[str, Any]] = []

    # Case: pdb_path -> score map (Full)
    if isinstance(data, dict) and all(isinstance(k, str) for k in data.keys()) and all(
        isinstance(v, (int, float)) for v in data.values()
    ):
        for k, v in data.items():
            target_id, candidate_id = infer_target_candidate_from_path(k)
            score_raw = float(v)
            rows.append(
                {
                    "method": method,
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "score_raw": score_raw,
                    "affinity_value": -score_raw,
                    "pdb_path": str(k),
                    "notes": f"affinity_source={json_path.name}(map)",
                }
            )
        return pd.DataFrame(rows)

    # Case: dict-of-records (ablation)
    if isinstance(data, dict):
        for key, rec in data.items():
            if not isinstance(rec, dict):
                continue
            target_id = rec.get("target_id") or rec.get("target") or rec.get("protein") or rec.get("receptor")
            if target_id is None and isinstance(key, str) and "/" in key:
                target_id = key.split("/")[0]
            target_id = str(target_id) if target_id is not None else "unknown"

            candidate_id = rec.get("peptide_basename") or rec.get("candidate") or rec.get("name") or rec.get("pdb") or rec.get("peptide")
            if candidate_id is None and isinstance(key, str) and "/" in key:
                candidate_id = key.split("/")[-1]
            candidate_id = Path(str(candidate_id) if candidate_id is not None else str(key)).name

            score_fields = ["score", "hdock_score", "docking_score", "hdock", "affinity", "value"]
            score_raw = None
            for sf in score_fields:
                if sf in rec:
                    score_raw = _safe_float(rec.get(sf))
                    if score_raw is not None:
                        break
            if score_raw is None:
                continue

            pdb_path = rec.get("peptide_pdb") or rec.get("pdb_path") or rec.get("pdb") or rec.get("peptide")
            pdb_path = str(pdb_path) if pdb_path is not None else ""

            rows.append(
                {
                    "method": method,
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "score_raw": float(score_raw),
                    "affinity_value": -float(score_raw),
                    "pdb_path": pdb_path,
                    "notes": f"affinity_source={json_path.name}(records)",
                }
            )
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=["method", "target_id", "candidate_id", "score_raw", "affinity_value", "pdb_path", "notes"])


def scan_pdb_files(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []
    return sorted([p for p in root_dir.rglob("*.pdb") if p.is_file()])


def extract_plddt_from_pdb(pdb_path: Path) -> Optional[float]:
    """
    从 PDB 的 B-factor 提取 mean pLDDT（若 B-factor 全为 0 或恒定，则视为缺失）。
    """
    if not pdb_path.exists():
        return None
    bfs: List[float] = []
    try:
        with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    if len(line) >= 66:
                        bf = _safe_float(line[60:66])
                        if bf is not None:
                            bfs.append(float(bf))
    except Exception:
        return None
    if not bfs:
        return None
    arr = np.asarray(bfs, dtype=float)
    if not np.isfinite(arr).any():
        return None
    if float(np.nanstd(arr)) < 1e-6:
        if abs(float(np.nanmean(arr))) < 1e-6:
            return None
        return None
    return float(np.nanmean(arr))


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _try_install_pymoo(tmp_root: Path) -> bool:
    """
    优先使用 pymoo。若缺少则尝试 pip 安装。
    要求：所有缓存/临时目录都在 /tmp 下。
    """
    try:
        import pymoo  # type: ignore  # noqa: F401

        return True
    except Exception:
        pass

    print("[INFO] pymoo not found. Trying to install pymoo via pip...", flush=True)
    pip_cache = tmp_root / "pip-cache"
    ensure_dir(pip_cache)
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(pip_cache)
    env["TMPDIR"] = str(tmp_root)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymoo"], env=env)
    except Exception as e:
        print(f"[WARN] Failed to install pymoo: {e}", flush=True)
        return False

    try:
        import pymoo  # type: ignore  # noqa: F401

        print("[INFO] pymoo installed successfully.", flush=True)
        return True
    except Exception:
        return False


def load_or_compute_candidate_metrics(df_aff: pd.DataFrame, discovered: DiscoveredFiles) -> pd.DataFrame:
    """
    为每个 candidate 尽量补齐：
      - plddt: 优先来自 esmfold_plddt_cache.json(sha256(sequence)->plddt)
        若缺失则 fallback 从 pdb B-factor
      - stability / solubility / novelty: 尝试从发现到的 csv/json 里匹配（若无法匹配则缺失）
    """
    df = df_aff.copy()
    df["candidate_id"] = df["candidate_id"].map(lambda x: Path(str(x)).name)
    df["target_id"] = df["target_id"].astype(str)
    df["method"] = df["method"].astype(str)

    # 读取 extracted_sequences.csv（如果存在，可提供 sequence 与 pdb_path 的更可靠映射）
    seq_map = None
    if discovered.extracted_sequences_csv is not None and discovered.extracted_sequences_csv.exists():
        df_seq = load_csv(discovered.extracted_sequences_csv)
        # 标准化 candidate_id：文件里是 pep_01（无 .pdb），Affinity json 里通常是 pep_01.pdb
        df_seq = df_seq.copy()
        df_seq["candidate_id_norm"] = df_seq["candidate_id"].map(lambda x: _stem_id(x) + ".pdb")
        df_seq["candidate_id_norm2"] = df_seq["candidate_id"].map(lambda x: _stem_id(x))
        df_seq["method"] = df_seq["method"].astype(str)
        df_seq["target_id"] = df_seq["target_id"].astype(str)
        seq_map = df_seq
        print(f"[INFO] Reusing existing sequence file: {discovered.extracted_sequences_csv}", flush=True)
    else:
        print("[WARN] extracted_sequences.csv not found; pLDDT mapping may be incomplete.", flush=True)

    # pLDDT cache
    plddt_cache: Dict[str, float] = {}
    if discovered.esmfold_plddt_cache_json is not None and discovered.esmfold_plddt_cache_json.exists():
        try:
            obj = load_json(discovered.esmfold_plddt_cache_json)
            if isinstance(obj, dict):
                plddt_cache = {str(k): float(v) for k, v in obj.items() if isinstance(v, (int, float))}
            print(f"[INFO] Reusing pLDDT cache: {discovered.esmfold_plddt_cache_json} (n={len(plddt_cache)})", flush=True)
        except Exception as e:
            print(f"[WARN] Failed to read pLDDT cache: {e}", flush=True)
            plddt_cache = {}
    else:
        print("[WARN] esmfold_plddt_cache.json not found; will fallback to PDB B-factor.", flush=True)

    # Join sequence/pdb_path if available
    if seq_map is not None:
        tmp = df.copy()
        tmp["candidate_id_stem"] = tmp["candidate_id"].map(_stem_id)
        tmp["candidate_id_norm"] = tmp["candidate_id_stem"] + ".pdb"
        # method+target+cand join
        df = pd.merge(
            tmp,
            seq_map[["method", "target_id", "candidate_id_norm", "pdb_path", "sequence"]],
            left_on=["method", "target_id", "candidate_id_norm"],
            right_on=["method", "target_id", "candidate_id_norm"],
            how="left",
            suffixes=("", "_seq"),
        )
        # prefer pdb_path from seq file
        df["pdb_path"] = df["pdb_path_seq"].where(df["pdb_path_seq"].notna() & (df["pdb_path_seq"] != ""), df["pdb_path"])
        df = df.drop(columns=["pdb_path_seq"], errors="ignore")
    else:
        df["sequence"] = np.nan

    def _extract_sequence_from_pdb(pdb_path: Path) -> Optional[str]:
        """
        从 PDB 中提取序列（按残基顺序去重）。
        说明：这是“补算”路径，用于把 candidate-level 序列映射到 esmfold_plddt_cache（sha256）。
        """
        aa3_to_1 = {
            "ALA": "A",
            "CYS": "C",
            "ASP": "D",
            "GLU": "E",
            "PHE": "F",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LYS": "K",
            "LEU": "L",
            "MET": "M",
            "ASN": "N",
            "PRO": "P",
            "GLN": "Q",
            "ARG": "R",
            "SER": "S",
            "THR": "T",
            "VAL": "V",
            "TRP": "W",
            "TYR": "Y",
        }
        if not pdb_path.exists():
            return None
        seq: List[str] = []
        last_key = None
        try:
            with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.startswith("ATOM"):
                        continue
                    if len(line) < 26:
                        continue
                    resn = line[17:20].strip().upper()
                    chain = (line[21].strip() or "?")
                    try:
                        resi = int(line[22:26])
                    except Exception:
                        continue
                    key = (chain, resi)
                    if key == last_key:
                        continue
                    last_key = key
                    aa = aa3_to_1.get(resn)
                    if aa is None:
                        # 非标准残基：跳过（不伪造）
                        continue
                    seq.append(aa)
        except Exception:
            return None
        if not seq:
            return None
        return "".join(seq)

    # 若 sequence 缺失，尝试从 pdb_path 补算（这是低成本操作，比重新跑模型便宜很多）
    n_seq_from_pdb = 0
    seq_notes: List[str] = []
    seq_list: List[Any] = []
    for _, r in df.iterrows():
        seq = r.get("sequence")
        if isinstance(seq, str) and seq.strip():
            seq_list.append(seq.strip())
            seq_notes.append("sequence_source=existing_csv")
            continue
        pdb_path = str(r.get("pdb_path") or "")
        s2 = None
        if pdb_path and Path(pdb_path).exists():
            s2 = _extract_sequence_from_pdb(Path(pdb_path))
        if s2 is not None:
            seq_list.append(s2)
            seq_notes.append("sequence_source=pdb_parsed")
            n_seq_from_pdb += 1
        else:
            seq_list.append(np.nan)
            seq_notes.append("sequence_source=missing")
    df["sequence"] = seq_list
    df["sequence_notes"] = seq_notes
    if n_seq_from_pdb > 0:
        print(f"[INFO] Sequence补算：from_pdb={n_seq_from_pdb} (others from existing/missing)", flush=True)

    # pLDDT from cache or PDB
    plddt_vals: List[Optional[float]] = []
    plddt_notes: List[str] = []
    cache_hits = 0
    pdb_hits = 0
    for _, r in df.iterrows():
        seq = r.get("sequence")
        pdb_path = str(r.get("pdb_path") or "")
        v = None
        note = ""
        if isinstance(seq, str) and seq.strip():
            h = _sha256_hex(seq.strip())
            if h in plddt_cache:
                v = float(plddt_cache[h])
                note = "plddt_source=esmfold_cache"
                cache_hits += 1
        if v is None and pdb_path and Path(pdb_path).exists():
            v2 = extract_plddt_from_pdb(Path(pdb_path))
            if v2 is not None:
                v = float(v2)
                note = "plddt_source=pdb_bfactor"
                pdb_hits += 1
        plddt_vals.append(v)
        plddt_notes.append(note)

    df["plddt"] = pd.to_numeric(pd.Series(plddt_vals), errors="coerce")
    df["plddt_notes"] = plddt_notes
    print(f"[INFO] pLDDT mapping: cache_hits={cache_hits}, pdb_bfactor_hits={pdb_hits}, total={len(df)}", flush=True)

    # stability / solubility / novelty
    # 当前目录下未发现相关文件也要优雅处理：保持 NaN，并记录来源尝试信息
    df["stability"] = np.nan
    df["solubility"] = np.nan
    df["novelty"] = np.nan

    # 简化策略：如果未来你补充了 stability/solubility csv/json，这里会自动尝试加载“表格型数据”，
    # 并用 (target_id, candidate_id) 做模糊匹配（candidate_id stem）。
    def _try_map_metric(files: List[Path], metric_name: str, col_keywords: Sequence[str]) -> Tuple[pd.Series, str]:
        if not files:
            return pd.Series([np.nan] * len(df)), "missing"
        # candidate keys
        key_to_idx = {}
        for i, rr in enumerate(df.itertuples(index=False)):
            key = (_stem_id(getattr(rr, "target_id")), _stem_id(getattr(rr, "candidate_id")))
            key_to_idx[key] = i
        out = [np.nan] * len(df)
        used = ""
        for fp in files[:30]:
            try:
                if fp.suffix.lower() == ".json":
                    obj = load_json(fp)
                    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                        tdf = pd.DataFrame(obj)
                    elif isinstance(obj, dict) and obj and all(isinstance(v, dict) for v in obj.values()):
                        tdf = pd.DataFrame(list(obj.values()))
                        tdf["_key"] = list(obj.keys())
                    else:
                        continue
                elif fp.suffix.lower() in {".csv", ".tsv"}:
                    sep = "\t" if fp.suffix.lower() == ".tsv" else ","
                    tdf = pd.read_csv(fp, sep=sep)
                else:
                    continue
            except Exception:
                continue
            if tdf is None or tdf.empty:
                continue

            cols = {c.lower(): c for c in tdf.columns}
            metric_col = None
            for kw in col_keywords:
                if kw.lower() in cols:
                    metric_col = cols[kw.lower()]
                    break
            if metric_col is None:
                continue
            # find target/candidate columns
            tcol = None
            for cand in ["target_id", "target", "protein", "receptor"]:
                if cand in cols:
                    tcol = cols[cand]
                    break
            ccol = None
            for cand in ["candidate_id", "candidate", "name", "pdb", "peptide", "sample_id", "_key"]:
                if cand in cols:
                    ccol = cols[cand]
                    break
            if tcol is None or ccol is None:
                continue
            nmap = 0
            for _, rr in tdf.iterrows():
                t = rr.get(tcol)
                c = rr.get(ccol)
                if t is None or c is None:
                    continue
                val = _safe_float(rr.get(metric_col))
                if val is None:
                    continue
                key = (_stem_id(t), _stem_id(c))
                if key in key_to_idx:
                    out[key_to_idx[key]] = float(val)
                    nmap += 1
            if nmap > 10:
                used = f"{fp.name}:{metric_col}"
                break
        status = used if used else "not_matched"
        return pd.Series(out), status

    stab_series, stab_src = _try_map_metric(discovered.stability_like, "stability", ["stability", "stable", "score"])
    sol_series, sol_src = _try_map_metric(discovered.solubility_like, "solubility", ["solubility", "soluble", "score"])
    nov_series, nov_src = _try_map_metric(discovered.novelty_like, "novelty", ["novelty", "diversity", "score"])

    if stab_src not in {"missing", "not_matched"}:
        df["stability"] = pd.to_numeric(stab_series, errors="coerce")
        print(f"[INFO] stability mapped from: {stab_src}", flush=True)
    else:
        print("[INFO] stability not found (or not matched); will omit from developability.", flush=True)

    if sol_src not in {"missing", "not_matched"}:
        df["solubility"] = pd.to_numeric(sol_series, errors="coerce")
        print(f"[INFO] solubility mapped from: {sol_src}", flush=True)
    else:
        print("[INFO] solubility not found (or not matched); will omit from developability.", flush=True)

    if nov_src not in {"missing", "not_matched"}:
        df["novelty"] = pd.to_numeric(nov_series, errors="coerce")
        print(f"[INFO] novelty mapped from: {nov_src}", flush=True)
    else:
        print("[INFO] novelty not found (or not matched); USE_THREE_OBJECTIVES will stay disabled.", flush=True)

    return df


def normalize_metrics(df: pd.DataFrame, metrics: Sequence[str], eps: float = EPS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    在全体方法/target/candidate 上做全局 min-max normalization。
    若某指标全常数或无有效值：设为 0.5 并打印 warning（同时记录到 stats csv）。
    返回：归一化后的 df，以及 normalization stats 表。
    """
    out = df.copy()
    stats_rows: List[Dict[str, Any]] = []

    for m in metrics:
        raw = pd.to_numeric(out[m], errors="coerce")
        finite = raw[np.isfinite(raw)]
        norm_col = f"{m}_norm"
        if finite.shape[0] == 0:
            out[norm_col] = np.nan
            stats_rows.append({"metric": m, "min_value": np.nan, "max_value": np.nan, "note": "all NaN -> norm NaN"})
            print(f"[WARN] metric '{m}' has no valid values; normalized column will be NaN.", flush=True)
            continue
        vmin = float(finite.min())
        vmax = float(finite.max())
        if abs(vmax - vmin) < 1e-12:
            out[norm_col] = 0.5
            stats_rows.append({"metric": m, "min_value": vmin, "max_value": vmax, "note": "constant -> norm=0.5"})
            print(f"[WARN] metric '{m}' is constant (min=max={vmin}); set {norm_col}=0.5.", flush=True)
            continue
        out[norm_col] = (raw - vmin) / (vmax - vmin + eps)
        stats_rows.append({"metric": m, "min_value": vmin, "max_value": vmax, "note": "ok"})

    stats_df = pd.DataFrame(stats_rows)
    return out, stats_df


def build_developability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    developability 构造规则：
    - 优先用 plddt + stability + solubility
    - 对“可用项”的标准化值求均值：
        developability = mean(available normalized components)
    - 若 stability/solubility 缺失，则退化为 developability = plddt_norm
    """
    out = df.copy()
    components = []
    comp_notes = []

    for comp in ["plddt_norm", "stability_norm", "solubility_norm"]:
        if comp in out.columns and np.isfinite(pd.to_numeric(out[comp], errors="coerce")).any():
            components.append(comp)

    if not components:
        out["developability"] = np.nan
        out["developability_norm"] = np.nan
        out["notes"] = out.get("notes", "").astype(str) + "|developability=missing"
        print("[WARN] No developability components available (plddt/stability/solubility).", flush=True)
        return out

    mat = out[components].to_numpy(dtype=float)
    # row-wise mean of finite components（避免 empty-slice 警告）
    finite_mask = np.isfinite(mat)
    counts = finite_mask.sum(axis=1)
    sums = np.where(finite_mask, mat, 0.0).sum(axis=1)
    dev = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    out["developability"] = dev
    out["developability_notes"] = "developability_components=" + "+".join(components)
    # developability 本身已在 [0,1] 附近，但仍按要求再做一次全局 min-max 得到 developability_norm
    out, _ = normalize_metrics(out, ["developability"])
    out["notes"] = out.get("notes", "").astype(str) + f"|{out['developability_notes'].iloc[0]}"
    return out


def select_topk_candidates_per_target(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    对每个 (method,target) 选择最多 k 个 candidate。
    优先按 score_raw 升序（越小越好），否则按 candidate_id 排序。
    """
    parts: List[pd.DataFrame] = []
    for (method, target), g in df.groupby(["method", "target_id"], sort=False):
        g2 = g.copy()
        has_score = np.isfinite(pd.to_numeric(g2["score_raw"], errors="coerce")).any()
        if has_score:
            g2 = g2.sort_values("score_raw", ascending=True)
            note = ""
        else:
            g2 = g2.sort_values("candidate_id", ascending=True)
            note = "no_score_raw_sort"
            print(f"[WARN] {method} {target}: score_raw missing; fallback to filename sort.", flush=True)
        n_total = len(g2)
        g2["n_candidates_total"] = n_total
        g2["n_candidates_used"] = min(n_total, k)
        g2["selection_note"] = note
        if n_total > k:
            print(f"[INFO] {method} {target}: truncate candidates {n_total} -> {k}", flush=True)
        parts.append(g2.head(k))
    return pd.concat(parts, ignore_index=True) if parts else df.head(0)


def is_non_dominated(points: np.ndarray) -> np.ndarray:
    """
    points: NxD, maximize.
    返回 bool mask，True 表示非支配点（Pareto front）。
    """
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                keep[i] = False
                break
    return keep


def compute_pareto_front(df: pd.DataFrame, obj_cols: Sequence[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    返回带 is_pareto 列的 df（仅用于当前 target-method 的候选子集），以及 Pareto 前沿点集。
    """
    mat = df[list(obj_cols)].to_numpy(dtype=float)
    mask = np.all(np.isfinite(mat), axis=1)
    df2 = df.copy()
    df2["valid_for_pareto"] = mask
    df2["is_pareto"] = False
    if mask.sum() == 0:
        return df2, np.zeros((0, len(obj_cols)), dtype=float)
    mat2 = mat[mask]
    nd = is_non_dominated(mat2)
    idx_valid = np.where(mask)[0]
    pareto_idx = idx_valid[nd]
    df2.loc[df2.index[pareto_idx], "is_pareto"] = True
    return df2, mat2[nd]


def compute_2d_hypervolume(points: np.ndarray, ref: Tuple[float, float] = REF_POINT_2D) -> float:
    """
    手写二维 hypervolume（即使没有 pymoo 也必须可运行）。
    points 必须是非支配点，且越大越好，reference=(0,0)。
    """
    if points.size == 0:
        return 0.0
    pts = np.clip(points.copy(), 0.0, 1.0)
    pts = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])]
    if pts.shape[0] == 0:
        return 0.0
    # x 降序扫描，累积矩形面积
    pts = pts[np.argsort(-pts[:, 0])]
    hv = 0.0
    y_max = ref[1]
    for x, y in pts:
        if y <= y_max:
            continue
        hv += (x - ref[0]) * (y - y_max)
        y_max = y
    return float(hv)


def _compute_hv(points: np.ndarray, use_pymoo: bool, ref: Sequence[float]) -> float:
    if points.size == 0:
        return 0.0
    d = points.shape[1]
    if d == 2:
        if use_pymoo:
            try:
                from pymoo.indicators.hv import HV  # type: ignore

                # pymoo 默认最小化：转换为 f = 1 - x
                F = 1.0 - np.clip(points, 0.0, 1.0)
                ref_min = 1.0 - np.asarray(ref, dtype=float)
                return float(HV(ref_point=ref_min)(F))
            except Exception:
                pass
        return compute_2d_hypervolume(points, ref=(float(ref[0]), float(ref[1])))
    # 3D 可选：仅在 pymoo 可用时启用；否则跳过（返回 NaN）
    if d == 3 and use_pymoo:
        try:
            from pymoo.indicators.hv import HV  # type: ignore

            F = 1.0 - np.clip(points, 0.0, 1.0)
            ref_min = 1.0 - np.asarray(ref, dtype=float)
            return float(HV(ref_point=ref_min)(F))
        except Exception:
            return float("nan")
    return float("nan")


def compute_target_hypervolume(
    df_used: pd.DataFrame,
    obj_cols: Sequence[str],
    use_pymoo: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对每个 (method,target) 计算 HV，并输出：
    - per_target_hv_df
    - pareto_points_df（包含 is_pareto 与 objective 值）
    """
    hv_rows: List[Dict[str, Any]] = []
    pareto_rows: List[Dict[str, Any]] = []

    ref = REF_POINT_3D if len(obj_cols) == 3 else REF_POINT_2D

    for (method, target), g in df_used.groupby(["method", "target_id"], sort=False):
        # 必须字段检查：affinity_norm + developability_norm
        mat = g[list(obj_cols)].to_numpy(dtype=float)
        valid = np.all(np.isfinite(mat), axis=1)
        if valid.sum() == 0:
            print(f"[WARN] Skip {method} {target}: missing required objectives for all candidates.", flush=True)
            continue

        g_valid = g.loc[g.index[valid]].copy()
        g_marked, front = compute_pareto_front(g_valid, obj_cols=obj_cols)
        hv = _compute_hv(front, use_pymoo=use_pymoo, ref=ref)

        hv_rows.append(
            {
                "method": method,
                "target_id": target,
                "n_candidates_total": int(g.get("n_candidates_total", len(g)).iloc[0]) if len(g) else 0,
                "n_candidates_used": int(len(g_valid)),
                "n_pareto_points": int(front.shape[0]),
                "hypervolume": float(hv),
            }
        )

        for _, rr in g_marked.iterrows():
            pareto_rows.append(
                {
                    "method": method,
                    "target_id": target,
                    "candidate_id": rr["candidate_id"],
                    "objective_1": float(rr[obj_cols[0]]),
                    "objective_2": float(rr[obj_cols[1]]),
                    "objective_3": float(rr[obj_cols[2]]) if len(obj_cols) == 3 else np.nan,
                    "is_pareto": bool(rr["is_pareto"]),
                }
            )

        print(
            f"[INFO] {method} {target}: used={len(g_valid)} pareto={front.shape[0]} hv={hv:.4f}",
            flush=True,
        )

    per_target_df = pd.DataFrame(hv_rows)
    pareto_points_df = pd.DataFrame(pareto_rows)
    return per_target_df, pareto_points_df


def summarize_hypervolumes(per_target_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method, g in per_target_df.groupby("method", sort=False):
        vals = pd.to_numeric(g["hypervolume"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        n = int(vals.shape[0])
        if n == 0:
            rows.append(
                {
                    "method": method,
                    "n_targets": 0,
                    "mean_hv": np.nan,
                    "std_hv": np.nan,
                    "sem_hv": np.nan,
                    "median_hv": np.nan,
                }
            )
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n >= 2 else 0.0
        sem = float(std / math.sqrt(n)) if n >= 2 else 0.0
        med = float(np.median(vals))
        rows.append(
            {
                "method": method,
                "n_targets": n,
                "mean_hv": mean,
                "std_hv": std,
                "sem_hv": sem,
                "median_hv": med,
            }
        )
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    out = out.sort_values("method")
    return out


def save_csv_outputs(
    out_dir: Path,
    merged_df: pd.DataFrame,
    per_target_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pareto_points_df: pd.DataFrame,
    norm_stats_df: pd.DataFrame,
    *,
    suffix: str = "",
) -> None:
    merged_path = out_dir / (CSV_MERGED if suffix == "" else f"pareto_candidate_metrics_merged{suffix}.csv")
    per_target_path = out_dir / (CSV_PER_TARGET if suffix == "" else f"pareto_hypervolume_per_target{suffix}.csv")
    summary_path = out_dir / (CSV_SUMMARY if suffix == "" else f"pareto_hypervolume_summary{suffix}.csv")
    pareto_path = out_dir / (CSV_PARETO_POINTS if suffix == "" else f"pareto_front_points{suffix}.csv")
    stats_path = out_dir / (CSV_NORM_STATS if suffix == "" else f"pareto_normalization_stats{suffix}.csv")

    merged_df.to_csv(merged_path, index=False)
    per_target_df.to_csv(per_target_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pareto_points_df.to_csv(pareto_path, index=False)
    norm_stats_df.to_csv(stats_path, index=False)

    print(f"[INFO] Saved: {merged_path}", flush=True)
    print(f"[INFO] Saved: {per_target_path}", flush=True)
    print(f"[INFO] Saved: {summary_path}", flush=True)
    print(f"[INFO] Saved: {pareto_path}", flush=True)
    print(f"[INFO] Saved: {stats_path}", flush=True)


def _set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "savefig.bbox": "tight",
            # Illustrator：TrueType 文字（避免 Type 3 字体无法编辑）
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _apply_illustrator_pdf_rc() -> None:
    """保存 PDF 前再设一次，避免被其它代码覆盖 rcParams。"""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def _save_fig_png_pdf(fig: mpl.figure.Figure, png_path: Path, pdf_path: Path, *, dpi: int = 300) -> None:
    """PNG + Adobe Illustrator 可编辑 PDF（fonttype 42，全图不强制栅格化）。"""
    _apply_illustrator_pdf_rc()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    for artist in fig.findobj(include_self=False):
        if hasattr(artist, "set_rasterized"):
            try:
                artist.set_rasterized(False)
            except Exception:
                pass
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", format="pdf")


def plot_hypervolume_line(
    out_dir: Path,
    summary_df: pd.DataFrame,
    *,
    err_mode: str = "sem",
    title: str = "Pareto hypervolume across targets",
    out_png: str = OUT_PNG,
    out_pdf: str = OUT_PDF,
) -> Tuple[Path, Path]:
    """
    主图：折线图（Base -> Base+OT -> Base+DPO -> Full）
    - 点上方标注数值（3 位小数）
    - 误差线：默认 SEM，可选 STD
    """
    df = summary_df.set_index("method").reindex(METHOD_ORDER).reset_index()
    y = df["mean_hv"].to_numpy(dtype=float)
    if err_mode.lower() == "std":
        yerr = df["std_hv"].to_numpy(dtype=float)
    else:
        yerr = df["sem_hv"].to_numpy(dtype=float)

    x = np.arange(len(METHOD_ORDER))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, y, marker="o", linewidth=2.2, color="black")
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER)
    ax.set_ylabel("Mean hypervolume")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)

    # annotate
    for xi, yi in zip(x, y):
        if np.isfinite(yi):
            ax.text(xi, yi + 0.02, f"{yi:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    png_path = out_dir / out_png
    pdf_path = out_dir / out_pdf
    _save_fig_png_pdf(fig, png_path, pdf_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved figure: {png_path}", flush=True)
    print(f"[INFO] Saved figure: {pdf_path}", flush=True)
    return png_path, pdf_path


def plot_hypervolume_boxplot(out_dir: Path, per_target_df: pd.DataFrame) -> Optional[Tuple[Path, Path]]:
    """
    补图：boxplot 展示各 target 的 HV 分布（非必须）。
    """
    try:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        data = []
        labels = []
        for m in METHOD_ORDER:
            vals = pd.to_numeric(per_target_df.loc[per_target_df["method"] == m, "hypervolume"], errors="coerce")
            vals = vals[np.isfinite(vals)]
            data.append(vals.to_numpy(dtype=float))
            labels.append(m)
        ax.boxplot(
            data,
            labels=labels,
            showmeans=True,
            meanline=False,
            widths=0.6,
        )
        ax.set_ylabel("Hypervolume")
        ax.set_title("Hypervolume distribution across targets")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
        fig.tight_layout()
        png_path = out_dir / BOX_PNG
        pdf_path = out_dir / BOX_PDF
        _save_fig_png_pdf(fig, png_path, pdf_path, dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved boxplot: {png_path}", flush=True)
        print(f"[INFO] Saved boxplot: {pdf_path}", flush=True)
        return png_path, pdf_path
    except Exception as e:
        print(f"[WARN] Failed to generate boxplot: {e}", flush=True)
        return None


def main() -> None:
    out_dir = Path("/root/autodl-tmp/Peptide_3D/results/4_ablation/plot")
    ensure_dir(out_dir)
    _set_paper_style()

    # 所有临时目录统一放 /tmp
    tmp_root = Path(tempfile.mkdtemp(prefix="pareto-hv-line-", dir="/tmp"))
    print(f"[INFO] tmp_root = {tmp_root}", flush=True)

    discovered = discover_existing_files()

    # 打印发现到的“现成结果文件”
    print("[INFO] Discovered existing files:", flush=True)
    print(f"  - plot_dir: {discovered.plot_dir}", flush=True)
    print(f"  - ablation_dir: {discovered.ablation_dir}", flush=True)
    print(f"  - ppdbench_dir: {discovered.ppdbench_dir}", flush=True)
    for p in [discovered.ablation_hdock_base, discovered.ablation_hdock_base_ot, discovered.ablation_hdock_base_dpo]:
        print(f"  - hdock_ablation_json: {p} (exists={p.exists()})", flush=True)
    print(f"  - full cands_hdock_scores.json: found {len(discovered.full_hdock_jsons)} files", flush=True)
    if discovered.extracted_sequences_csv is not None:
        print(f"  - extracted_sequences.csv: {discovered.extracted_sequences_csv}", flush=True)
    if discovered.esmfold_plddt_cache_json is not None:
        print(f"  - esmfold_plddt_cache.json: {discovered.esmfold_plddt_cache_json}", flush=True)
    if discovered.top1_metrics_csv is not None:
        print(f"  - top1_metrics.csv (method-level, not for Pareto): {discovered.top1_metrics_csv}", flush=True)
    if discovered.top3_metrics_csv is not None:
        print(f"  - top3_metrics.csv (method-level, not for Pareto): {discovered.top3_metrics_csv}", flush=True)

    # hypervolume backend
    use_pymoo = _try_install_pymoo(tmp_root)
    if use_pymoo:
        print("[INFO] Hypervolume backend: pymoo (preferred).", flush=True)
    else:
        print("[INFO] Hypervolume backend: manual 2D HV fallback.", flush=True)

    # 1) 读取 affinity（candidate-level）
    aff_dfs: List[pd.DataFrame] = []

    aff_dfs.append(parse_hdock_scores("Base", discovered.ablation_hdock_base))
    aff_dfs.append(parse_hdock_scores("Base+OT", discovered.ablation_hdock_base_ot))
    aff_dfs.append(parse_hdock_scores("Base+DPO", discovered.ablation_hdock_base_dpo))

    # Full: 合并所有 targets 的 json
    full_parts: List[pd.DataFrame] = []
    for fp in discovered.full_hdock_jsons:
        df_f = parse_hdock_scores("Full", fp)
        if not df_f.empty:
            full_parts.append(df_f)
    df_full = pd.concat(full_parts, ignore_index=True) if full_parts else pd.DataFrame()
    aff_dfs.append(df_full)

    df_aff = pd.concat(aff_dfs, ignore_index=True)
    df_aff["score_raw"] = pd.to_numeric(df_aff["score_raw"], errors="coerce")
    df_aff["affinity_value"] = pd.to_numeric(df_aff["affinity_value"], errors="coerce")

    # 基本统计打印
    for m in METHOD_ORDER:
        dm = df_aff[df_aff["method"] == m]
        print(f"[INFO] {m}: targets={dm['target_id'].nunique()} candidates={len(dm)}", flush=True)

    # 2) 补齐 pLDDT/stability/solubility/novelty 等
    df_metrics = load_or_compute_candidate_metrics(df_aff, discovered)

    # 3) 全局标准化（为构造 developability 做准备）
    #    注意：affinity_value 已是“越大越好”（=-score_raw）
    df_norm1, stats1 = normalize_metrics(df_metrics, ["affinity_value", "plddt", "stability", "solubility", "novelty"])

    # 4) 构造 developability（基于标准化 component 的均值），并再做 developability_norm
    df_dev = build_developability_score(df_norm1)

    # 5) 生成 Pareto 目标的最终 normalization stats
    #    - 主实验目标：affinity_norm, developability_norm
    #    - 这里 affinity_norm 来自 affinity_value_norm；为清晰起见重命名
    df_dev = df_dev.copy()
    df_dev["affinity_norm"] = df_dev["affinity_value_norm"]

    # 可选：三目标扩展（affinity_norm + plddt_norm + novelty_norm 或 solubility_norm）
    obj_cols = ["affinity_norm", "developability_norm"]
    if USE_THREE_OBJECTIVES:
        # 仅当数据足够完整才启用
        if np.isfinite(pd.to_numeric(df_dev.get("novelty_norm", np.nan), errors="coerce")).any():
            obj_cols = ["affinity_norm", "plddt_norm", "novelty_norm"]
            print("[INFO] USE_THREE_OBJECTIVES enabled: objectives=(affinity, pLDDT, novelty)", flush=True)
        elif np.isfinite(pd.to_numeric(df_dev.get("solubility_norm", np.nan), errors="coerce")).any():
            obj_cols = ["affinity_norm", "plddt_norm", "solubility_norm"]
            print("[INFO] USE_THREE_OBJECTIVES enabled: objectives=(affinity, pLDDT, solubility)", flush=True)
        else:
            print("[WARN] USE_THREE_OBJECTIVES requested but insufficient metrics; fallback to 2D.", flush=True)
            obj_cols = ["affinity_norm", "developability_norm"]

    # 汇总 normalization stats（包含 developability）
    df_dev2, stats2 = normalize_metrics(df_dev, ["developability"])
    df_dev2["developability_norm"] = df_dev2["developability_norm"]  # ensure present

    norm_stats = pd.concat(
        [
            stats1,
            pd.DataFrame([{"metric": "developability", "min_value": float(stats2.loc[stats2["metric"] == "developability", "min_value"].iloc[0]),
                           "max_value": float(stats2.loc[stats2["metric"] == "developability", "max_value"].iloc[0]),
                           "note": "developability built from normalized components then min-max again"}]),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["metric"], keep="first")

    # 6) 保存 candidate-level 合并表（一次即可，后续 Top-K 复用它）
    merged_cols = [
        "method",
        "target_id",
        "candidate_id",
        "score_raw",
        "affinity_value",
        "affinity_norm",
        "plddt",
        "plddt_norm",
        "stability",
        "stability_norm",
        "solubility",
        "solubility_norm",
        "developability",
        "developability_norm",
        "pdb_path",
        "notes",
    ]
    for c in merged_cols:
        if c not in df_dev2.columns:
            df_dev2[c] = np.nan
    merged_df_out = df_dev2[merged_cols].copy()
    merged_df_out.to_csv(out_dir / CSV_MERGED, index=False)
    norm_stats.to_csv(out_dir / CSV_NORM_STATS, index=False)
    print(f"[INFO] Saved: {out_dir / CSV_MERGED}", flush=True)
    print(f"[INFO] Saved: {out_dir / CSV_NORM_STATS}", flush=True)

    # 7) 计算两套：Top-K=MAX（主结果）以及 Top-3（你追加需求）
    def _run_one(topk: int, *, suffix: str, fig_title: str, out_png: str, out_pdf: str) -> None:
        print(f"[INFO] ===== Computing hypervolume with Top-{topk} candidates per target =====", flush=True)
        df_used = select_topk_candidates_per_target(df_dev2, topk)
        needed = obj_cols
        valid_mask = np.all(np.isfinite(df_used[needed].to_numpy(dtype=float)), axis=1)
        df_used_valid = df_used.loc[df_used.index[valid_mask]].copy()

        for m in METHOD_ORDER:
            dm = df_used_valid[df_used_valid["method"] == m]
            print(f"[INFO] Top-{topk} {m}: used candidates = {len(dm)} (targets={dm['target_id'].nunique()})", flush=True)

        per_target_df, pareto_points_df = compute_target_hypervolume(df_used_valid, obj_cols=obj_cols, use_pymoo=use_pymoo)
        summary_df = summarize_hypervolumes(per_target_df)
        for _, r in summary_df.iterrows():
            print(
                f"[INFO] Top-{topk} {r['method']}: n_targets={int(r['n_targets'])} mean={r['mean_hv']:.4f} std={r['std_hv']:.4f} sem={r['sem_hv']:.4f}",
                flush=True,
            )

        # 保存 per-target / summary / pareto-front 点（suffix 区分）
        save_csv_outputs(
            out_dir=out_dir,
            merged_df=merged_df_out,
            per_target_df=per_target_df,
            summary_df=summary_df,
            pareto_points_df=pareto_points_df,
            norm_stats_df=norm_stats,
            suffix=suffix,
        )

        # 出图（line + boxplot）
        plot_hypervolume_line(out_dir, summary_df, err_mode="sem", title=fig_title, out_png=out_png, out_pdf=out_pdf)
        plot_hypervolume_boxplot(out_dir, per_target_df)

    # 主结果：Top-MAX
    _run_one(
        MAX_CANDIDATES_PER_TARGET,
        suffix="",
        fig_title=f"Pareto hypervolume across targets (Top-{MAX_CANDIDATES_PER_TARGET})",
        out_png=OUT_PNG,
        out_pdf=OUT_PDF,
    )
    # 追加：Top-3
    _run_one(
        3,
        suffix="_top3",
        fig_title="Pareto hypervolume across targets (Top-3)",
        out_png=OUT_PNG_TOP3,
        out_pdf=OUT_PDF_TOP3,
    )

    print(f"[INFO] All outputs saved to: {out_dir}", flush=True)
    print(f"[INFO] Temp directory kept at: {tmp_root} (you can delete it safely)", flush=True)


if __name__ == "__main__":
    main()


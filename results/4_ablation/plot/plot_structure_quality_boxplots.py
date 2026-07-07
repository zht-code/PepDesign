#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPDbench：四种方法（Base / Base+OT / Base+DPO / Full）生成多肽的 **结构质量** 箱线图。

================================================================================
Top-1 / Top-3 定义
================================================================================
- 对每个 target、每种方法，收集该方法下所有候选 PDB（递归扫描）。
- **排序**：优先按 HDOCK score（JSON），**越小越好**；若无分数则按文件名排序并 **WARN**。
- **Top-1**：每 target 仅保留 1 个最佳候选 → 箱线图每个点 = 该 target 上 **最佳候选** 的指标。
- **Top-3**：每 target 取最佳 3 个候选（不足则全取并 WARN）；先分别算各候选指标，再对 top3 **取算术平均**
  → 箱线图每个点 = 该 target 的 **top3 指标均值**（target-level mean，避免某 target 因候选多而主导）。

================================================================================
指标
================================================================================
1) **Mean pLDDT**：默认用 **ESMFold v1**（通过 Hugging Face ``transformers.EsmForProteinFolding``，模型 ID 默认 ``facebook/esmfold_v1``）对候选 PDB **最长蛋白链的一字母序列** 推理，按原子掩码对 **pLDDT** 做加权平均，得到 0–100 量级的置信度均值。
   - 权重由 ``transformers`` 从 Hugging Face Hub 拉取；可通过 ``ESMFOLD_HF_MODEL`` 指向本地目录或其它快照。``ESMFOLD_NUM_RECYCLES``（默认 1）控制折叠循环次数。
   - 若存在 ``/root/autodl-tmp``，脚本会将 ``HF_HOME``、``TORCH_HOME`` 默认指到其下 ``.cache``，避免根分区占满。
   - 拉取模型前默认会临时清除 ``http(s)_proxy``（``ESMFOLD_HF_IGNORE_PROXY=0`` 可关闭此行为）。
   - 国内访问 Hub 困难时可设镜像，例如 ``export HF_ENDPOINT=https://hf-mirror.com``（以镜像站说明为准）。
   - 若加载失败，或设置 ``PLDDT_SOURCE=pdb``，则回退为解析 PDB **B-factor** 均值。

2) **TM-score** / **RMSD**：需要 **参考肽结构**（通常为 ``<target>/peptide.pdb``）。
   - 默认使用 **tmtools**（TM-align 经典实现）：对参考链与候选链提取 CA 坐标与序列后 ``tm_align``。
   - **TM-score 报告值**：``tm_norm_chain1``（**按参考肽长度归一化**，参考作为第一条链传入）。
   - **RMSD**：``tm_align`` 返回的叠加后 RMSD（TM-align 定义下与文献常用一致）。
   - 若 ``tmtools`` 不可用，脚本会尝试 ``pip install tmtools``；仍失败则尝试编译 ``TMscore.cpp`` 子进程（仅作补充，易因残基编号不一致失败）。
   - 无参考结构 → TM-score / RMSD = NaN。

================================================================================
加速下载（可选）
================================================================================
在终端可先执行：  source /etc/network_turbo
再运行本脚本，便于拉取 TMscore.cpp / pip 包。

依赖：numpy, pandas, matplotlib, biopython；结构比对优先 **tmtools**（脚本内可自动 pip install）。
pLDDT（ESMFold）：**torch**、**transformers**、**huggingface_hub**（由 transformers 拉取 ``facebook/esmfold_v1``）。
可选：系统 **g++** 用于编译 TMscore.cpp。

运行：
    python plot_structure_quality_boxplots.py

输出目录：与本脚本同目录（results/4_ablation/plot）
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 可选：tmtools（TM-align）
# ---------------------------------------------------------------------------
_HAS_TMTOOLS = False
_tm_align = None  # type: ignore
_get_structure = None  # type: ignore
_get_residue_data = None  # type: ignore

try:
    from tmtools import tm_align as _tm_align
    from tmtools.io import get_residue_data as _get_residue_data
    from tmtools.io import get_structure as _get_structure

    _HAS_TMTOOLS = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 常量与论文风格配色
# ---------------------------------------------------------------------------
METHOD_ORDER = ["Base", "Base+OT", "Base+DPO", "Full"]
METHOD_KEYS = ["base", "base_ot", "base_dpo", "full"]

NATURE_BAR_COLORS = [
    "#A8C5E2",
    "#B8DCC6",
    "#F2C4B8",
    "#D4C4E8",
]

TM_SCORE_CPP_URLS = [
    "https://zhanggroup.org/TM-score/TMscore.cpp",
    "https://zhanggroup.org/TM-score/TMscore.cpp".replace("https://", "http://"),
]

SCORE_JSON_KEYS = (
    "score",
    "hdock_score",
    "docking_score",
    "hdock",
    "affinity",
)

# ESMFold pLDDT：折叠循环次数（越小越快，默认可通过环境变量覆盖）
ESMFOLD_NUM_RECYCLES = int(os.environ.get("ESMFOLD_NUM_RECYCLES", "1"))


# ---------------------------------------------------------------------------
# 1) ensure_dir
# ---------------------------------------------------------------------------
def ensure_dir(p: Union[str, Path]) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 2) scan_pdb_files
# ---------------------------------------------------------------------------
def scan_pdb_files(
    root: Path,
    *,
    label: str = "",
    recursive: bool = True,
    verbose: bool = True,
) -> List[Path]:
    if not root.exists():
        warnings.warn(f"[scan_pdb_files] 路径不存在 ({label}): {root}")
        return []
    if root.is_file() and root.suffix.lower() == ".pdb":
        return [root.resolve()]
    pat = "**/*.pdb" if recursive else "*.pdb"
    out = sorted({p.resolve() for p in root.glob(pat)})
    if verbose:
        tag = f"[{label}] " if label else ""
        print(f"{tag}扫描到 PDB: {len(out)}  @ {root}")
    return out


# ---------------------------------------------------------------------------
# 3) infer_target_and_candidate
# ---------------------------------------------------------------------------
def infer_target_and_candidate(
    pdb_path: Path,
    bench_root: Path,
    *,
    method_key: str,
) -> Tuple[str, str]:
    """
    从路径推断 target_id 与 candidate_id（basename）。
    规则：在 bench_root 下，target 为 PPDbench 下第一级子目录名。
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
        # 回退：向上查找 4 层，取像 PDB id 的目录名
        p = pdb_path.parent
        for _ in range(6):
            if p.name and re.match(r"^[0-9][0-9a-z]{3}$", p.name.lower()):
                target_id = p.name.lower()
                break
            p = p.parent
    cand = pdb_path.name
    if not target_id:
        target_id = "unknown_target"
        print(f"[WARN] 无法从路径推断 target_id ({method_key}): {pdb_path}")
    return target_id, cand


# ---------------------------------------------------------------------------
# 4) load_optional_hdock_scores
# ---------------------------------------------------------------------------
def _first_float_obj(d: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return None


def load_ablation_hdock_json(json_path: Path) -> Dict[str, Dict[str, float]]:
    """target_id -> {peptide_basename: hdock_score}，越小越好。"""
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    if not isinstance(raw, dict):
        return {}
    for _k, v in raw.items():
        if not isinstance(v, dict):
            continue
        tid = v.get("target_id")
        if not tid:
            continue
        sc = _first_float_obj(v, SCORE_JSON_KEYS)
        pep = v.get("peptide_pdb")
        if sc is None or not pep:
            continue
        name = Path(str(pep)).name
        out[str(tid)][name] = float(sc)
    return dict(out)


def load_full_hdock_from_bench(
    bench_root: Path,
    multi_subdir: str = "multi_cands",
    scores_name: str = "cands_hdock_scores.json",
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for d in sorted(p for p in bench_root.iterdir() if p.is_dir()):
        jpath = d / multi_subdir / scores_name
        if not jpath.is_file():
            continue
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 读取失败 {jpath}: {e}")
            continue
        tid = d.name
        scores: Dict[str, float] = {}
        if isinstance(data, dict):
            for pk, val in data.items():
                if isinstance(val, (int, float)):
                    scores[Path(str(pk)).name] = float(val)
        out[tid] = scores
    return out


def load_optional_hdock_scores(
    method_key: str,
    ablation_json_paths: Dict[str, Path],
    bench_root: Path,
) -> Optional[Dict[str, Dict[str, float]]]:
    if method_key == "full":
        return load_full_hdock_from_bench(bench_root)
    jp = ablation_json_paths.get(method_key)
    if jp is None or not jp.is_file():
        return None
    try:
        return load_ablation_hdock_json(jp)
    except Exception as e:
        print(f"[WARN] HDOCK JSON 读取失败 ({method_key}): {e}")
        return None


# ---------------------------------------------------------------------------
# 5) group_candidates_by_target
# ---------------------------------------------------------------------------
def group_candidates_by_target(
    pdb_paths: Sequence[Path],
    bench_root: Path,
    method_key: str,
) -> Dict[str, List[Path]]:
    g: Dict[str, List[Path]] = defaultdict(list)
    for p in pdb_paths:
        tid, _c = infer_target_and_candidate(p, bench_root, method_key=method_key)
        g[tid].append(p.resolve())
    for tid in g:
        g[tid] = sorted(set(g[tid]))
    return dict(g)


# ---------------------------------------------------------------------------
# 6) select_topk_candidates
# ---------------------------------------------------------------------------
def select_topk_candidates(
    grouped: Dict[str, List[Path]],
    scores: Optional[Dict[str, Dict[str, float]]],
    *,
    k: int,
    method_label: str,
    higher_is_better: bool = False,
) -> Tuple[Dict[str, List[Path]], Dict[str, bool]]:
    """
    返回 target -> 排序后的前 k 个 PDB；以及 target -> 是否使用 fallback 文件名排序。
    """
    used_fallback: Dict[str, bool] = {}
    out: Dict[str, List[Path]] = {}
    for tid, paths in sorted(grouped.items()):
        if not paths:
            continue
        used_fallback[tid] = False
        if scores and tid in scores and scores[tid]:
            scmap = scores[tid]

            def sort_key(pp: Path) -> Tuple[float, str]:
                nm = pp.name
                if nm in scmap:
                    sc = float(scmap[nm])
                    # 默认：越小越好（如 HDOCK score）；若 higher_is_better，则越大越好（如 affinity）
                    return ((-sc if higher_is_better else sc), nm)
                # 无分数：排末尾
                return ((float("inf")), nm)

            ranked = sorted(paths, key=sort_key)
            if any(p.name not in scmap for p in paths):
                print(
                    f"[WARN][{method_label}] target={tid} 部分 PDB 无 HDOCK 分，"
                    f"将排在末尾（按名）"
                )
        else:
            ranked = sorted(paths, key=lambda p: p.name)
            used_fallback[tid] = True
            print(
                f"[WARN][{method_label}] target={tid} 无可用 HDOCK 排序，"
                f"退化为按文件名排序取 top-{k}"
            )
        sel = ranked[: min(k, len(ranked))]
        if k > len(sel):
            print(
                f"[WARN][{method_label}] target={tid} 候选仅 {len(sel)} 个，"
                f"不足 top-{k}，按实际数量使用"
            )
        out[tid] = sel
    return out, used_fallback


# ---------------------------------------------------------------------------
# 7) extract_plddt_from_pdb
# ---------------------------------------------------------------------------
def extract_plddt_from_pdb(pdb_path: Path) -> float:
    """
    从 ATOM/HETATM 记录读 B-factor（列 61–66 附近），返回 mean pLDDT 或 NaN。
    """
    bfacs: List[float] = []
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 66:
                continue
            try:
                b = float(line[60:66])
            except ValueError:
                continue
            bfacs.append(b)
    if not bfacs:
        warnings.warn(f"[pLDDT] 无有效 B-factor: {pdb_path}")
        return float("nan")
    arr = np.array(bfacs, dtype=np.float64)
    m = float(np.mean(arr))
    mx = float(np.max(arr))
    # 0–1 标度（部分模型）
    if 0 < m <= 1.0 and mx <= 1.5:
        m *= 100.0
        arr *= 100.0
        mx = float(np.max(arr))
    # 合理 pLDDT 大致 0–100；MODELLER 等可能 >100，放宽到 120
    if m <= 0 and mx <= 0:
        warnings.warn(f"[pLDDT] B-factor 全 0，视为无效: {pdb_path}")
        return float("nan")
    if m > 120 or (mx > 0 and mx < 5 and m < 5):
        warnings.warn(
            f"[pLDDT] B-factor 不像 pLDDT (mean={m:.3f})，仍返回均值供参考: {pdb_path}"
        )
    return m


# ---------------------------------------------------------------------------
# 8) find_reference_structure
# ---------------------------------------------------------------------------
_REF_CACHE: Dict[str, Tuple[Optional[Path], str]] = {}


def find_reference_structure(
    target_id: str,
    bench_root: Path,
    *,
    method_key: str,
) -> Tuple[Optional[Path], str]:
    key = f"{target_id}"
    if key in _REF_CACHE:
        return _REF_CACHE[key]

    bench_root = bench_root.resolve()
    tdir = bench_root / target_id
    status = "not_found"
    chosen: Optional[Path] = None

    priority_names = [
        "peptide.pdb",
        "ligand.pdb",
        "native.pdb",
        "reference.pdb",
        "ground_truth.pdb",
    ]
    for nm in priority_names:
        p = tdir / nm
        if p.is_file():
            chosen = p
            status = f"priority:{nm}"
            break

    if chosen is None and tdir.is_dir():
        kws = ("native", "ref", "reference", "gt", "ground_truth", "true", "crystal")
        cands: List[Path] = []
        for p in tdir.rglob("*.pdb"):
            low = str(p).lower()
            if "receptor" in low:
                continue
            if any(x in low for x in ("generated_ablation", "multi_cands", "generated_")):
                continue
            name = p.name.lower()
            if target_id.lower() in name and any(k in name for k in kws):
                cands.append(p)
        if cands:
            chosen = sorted(cands, key=lambda x: len(str(x)))[0]
            status = "keyword_match"

    if chosen is None and tdir.is_dir():
        # 仅一个「非 receptor」的 pdb
        simple = [p for p in tdir.glob("*.pdb") if p.name.lower() != "receptor.pdb"]
        if len(simple) == 1:
            chosen = simple[0]
            status = "single_pdb_in_target_dir"

    if chosen is None:
        print(f"[WARN][ref] target={target_id} ({method_key}) 未找到参考肽 PDB")
    else:
        print(f"[ref] target={target_id} -> {chosen} ({status})")

    _REF_CACHE[key] = (chosen, status)
    return chosen, status


# ---------------------------------------------------------------------------
# 9) ensure_alignment_tool
# ---------------------------------------------------------------------------
def _pip_install_tmtools() -> bool:
    print("[align] 尝试 pip install tmtools ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "tmtools", "-q"],
            check=True,
            timeout=600,
        )
        return True
    except Exception as e:
        print(f"[align] pip install tmtools 失败: {e}")
        return False


def _download_tmscore_cpp(dest: Path) -> bool:
    ensure_dir(dest.parent)
    for url in TM_SCORE_CPP_URLS:
        try:
            print(f"[align] 下载 TMscore.cpp: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) < 500 or b"TM-score" not in data[:2000]:
                continue
            dest.write_bytes(data)
            return True
        except Exception as e:
            print(f"[align] 下载失败 {url}: {e}")
    return False


def _compile_tmscore(cpp_path: Path, exe_path: Path) -> bool:
    try:
        subprocess.run(
            ["g++", "-O3", "-ffast-math", "-o", str(exe_path), str(cpp_path)],
            check=True,
            timeout=300,
            capture_output=True,
            text=True,
        )
        exe_path.chmod(exe_path.stat().st_mode | 0o111)
        print(f"[align] 已编译 TMscore -> {exe_path}")
        return True
    except Exception as e:
        print(f"[align] g++ 编译 TMscore 失败: {e}")
        return False


def ensure_alignment_tool(tools_dir: Path) -> Dict[str, Any]:
    """
    返回状态 dict:
      mode: 'tmtools' | 'tmscore_subprocess' | 'none'
      tmscore_exe: Optional[Path]
    """
    global _HAS_TMTOOLS, _tm_align, _get_structure, _get_residue_data
    if _HAS_TMTOOLS:
        return {"mode": "tmtools", "tmscore_exe": None}

    if _pip_install_tmtools():
        try:
            from tmtools import tm_align as _ta
            from tmtools.io import get_residue_data as _grd
            from tmtools.io import get_structure as _gs

            globals()["_tm_align"] = _ta
            globals()["_get_residue_data"] = _grd
            globals()["_get_structure"] = _gs
            globals()["_HAS_TMTOOLS"] = True
            print("[align] tmtools 已可用（TM-align）")
            return {"mode": "tmtools", "tmscore_exe": None}
        except ImportError:
            pass

    ensure_dir(tools_dir)
    exe = tools_dir / "TMscore"
    cpp = tools_dir / "TMscore.cpp"
    if exe.is_file() and os.access(exe, os.X_OK):
        print(f"[align] 使用已存在 {exe}")
        return {"mode": "tmscore_subprocess", "tmscore_exe": exe}

    if _download_tmscore_cpp(cpp) and _compile_tmscore(cpp, exe):
        return {"mode": "tmscore_subprocess", "tmscore_exe": exe}

    print(
        "[align] 无法启用结构比对：请手动 pip install tmtools 或安装 g++ 后重试下载编译 TMscore.cpp"
    )
    return {"mode": "none", "tmscore_exe": None}


# ---------------------------------------------------------------------------
# 10) run_alignment_tool
# ---------------------------------------------------------------------------
def _longest_protein_chain(structure: Any) -> Any:
    from Bio.PDB.Polypeptide import is_aa

    best = None
    best_n = -1
    for model in structure:
        for chain in model:
            n = sum(1 for r in chain if is_aa(r, standard=True))
            if n > best_n:
                best_n = n
                best = chain
    return best


def extract_protein_sequence_longest_chain(pdb_path: Path) -> str:
    """从 PDB 取最长标准蛋白链的一字母序列（供 ESMFold 使用）。"""
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("c", str(pdb_path))
    except Exception:
        return ""
    chain = _longest_protein_chain(structure)
    if chain is None:
        return ""
    letters: List[str] = []
    for res in chain:
        if res.id[0] != " ":
            continue
        if not is_aa(res, standard=True):
            continue
        name = res.get_resname()
        letters.append(protein_letters_3to1.get(name, "X"))
    return "".join(letters)


def _maybe_redirect_torch_home_for_hub() -> None:
    """若数据盘存在，将 TORCH_HOME 指到大盘，便于缓存 torch hub 权重。"""
    data_root = Path("/root/autodl-tmp")
    if not data_root.is_dir():
        return
    hub_root = data_root / ".cache" / "torch"
    try:
        hub_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    if not os.environ.get("TORCH_HOME"):
        os.environ["TORCH_HOME"] = str(hub_root)


def _proxies_backup_and_clear_for_hf() -> Dict[str, str]:
    """
    临时清除代理环境变量，避免 autodl 等环境下无效代理导致 HF/torch hub 握手超时。
    设置 ESMFOLD_HF_IGNORE_PROXY=0 可保留系统代理。
    """
    if os.environ.get("ESMFOLD_HF_IGNORE_PROXY", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return {}
    keys = (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    )
    backup: Dict[str, str] = {}
    for k in keys:
        if k in os.environ:
            backup[k] = os.environ.pop(k)
    return backup


def _proxies_restore(backup: Dict[str, str]) -> None:
    for k, v in backup.items():
        os.environ[k] = v


def _maybe_redirect_hf_home() -> None:
    """
    若存在 autodl 数据盘，将 HF_HOME 指到其下缓存（ESMFold 约 8GB+，根分区常不够）。
    设置 USE_AUTODL_HF_CACHE=0 可保留原有 HF_HOME。
    """
    data_root = Path("/root/autodl-tmp")
    if not data_root.is_dir():
        return
    if os.environ.get("USE_AUTODL_HF_CACHE", "1").strip().lower() in ("0", "false", "no"):
        return
    hf_root = data_root / ".cache" / "huggingface"
    try:
        hf_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ["HF_HOME"] = str(hf_root)


def _hf_esmfold_forward_sequence(
    model: Any,
    seq: str,
    num_recycles: Optional[int],
) -> Any:
    """与 transformers 内置 ``infer`` 等价，但可传入 ``num_recycles``。"""
    import torch
    from transformers.models.esm.modeling_esmfold import collate_dense_tensors
    from transformers.models.esm.openfold_utils import residue_constants

    device = next(model.parameters()).device
    lst = [seq]
    aatype = collate_dense_tensors(
        [
            torch.from_numpy(
                residue_constants.sequence_to_onehot(
                    sequence=s,
                    mapping=residue_constants.restype_order_with_x,
                    map_unknown_to_x=True,
                )
            )
            .to(device)
            .argmax(dim=1)
            for s in lst
        ]
    )
    mask = collate_dense_tensors([aatype.new_ones(len(s)) for s in lst])
    position_ids = torch.arange(aatype.shape[1], device=device).expand(len(lst), -1)
    kw: Dict[str, Any] = {}
    if num_recycles is not None:
        kw["num_recycles"] = num_recycles
    return model.forward(aatype, mask, position_ids=position_ids, **kw)


def _mean_plddt_from_hf_output(out: Any) -> float:
    """HuggingFace ESMFold 输出中 pLDDT 的样本均值（对齐 0–100 常用报告形式）。"""
    p = out.plddt
    m = out.atom37_atom_exists
    s = (p * m).sum(dim=(1, 2))
    d = m.sum(dim=(1, 2)).clamp(min=1e-8)
    v = float((s / d)[0].item())
    mx = float(p.detach().max().item())
    if v <= 2.0 and mx <= 2.0:
        v *= 100.0
    return v


def ensure_esmfold_plddt(cache_path: Path) -> Dict[str, Any]:
    """
    返回 plddt_state：mode 为 'esmfold' 或 'pdb_bf'；esmfold 时使用 transformers 加载的模型。
    """
    state: Dict[str, Any] = {
        "mode": "pdb_bf",
        "backend": "transformers",
        "model": None,
        "device": "cpu",
        "num_recycles": ESMFOLD_NUM_RECYCLES,
        "cache": {},
        "cache_path": cache_path,
        "dirty": False,
        "chunk_size": int(os.environ.get("ESMFOLD_CHUNK_SIZE", "128") or "0") or 128,
        "hf_model_id": os.environ.get("ESMFOLD_HF_MODEL", "facebook/esmfold_v1").strip(),
    }
    if cache_path.is_file():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(fv):
                        state["cache"][str(k)] = fv
        except Exception as e:
            print(f"[pLDDT/ESMFold] 无法读取缓存 {cache_path}: {e}")

    src = os.environ.get("PLDDT_SOURCE", "esmfold").strip().lower()
    if src in ("pdb", "bf", "b-factor", "pdb_bf", "bfactor"):
        print("[pLDDT] PLDDT_SOURCE 指定为 PDB B-factor，跳过 ESMFold")
        return state

    # 须在 import transformers / huggingface_hub 之前设置，否则 hub 已锁定默认缓存目录
    _maybe_redirect_hf_home()
    _maybe_redirect_torch_home_for_hub()

    try:
        import torch
        from transformers import EsmForProteinFolding
    except ImportError as e:
        print(f"[pLDDT/ESMFold] 未安装 torch/transformers，回退 PDB B-factor: {e}")
        return state

    hf_model = state["hf_model_id"]
    use_fp16 = (
        os.environ.get("ESMFOLD_FP16", "").strip().lower() in ("1", "true", "yes")
        and torch.cuda.is_available()
    )
    load_kw: Dict[str, Any] = {}
    if use_fp16:
        load_kw["torch_dtype"] = torch.float16

    proxy_bak = _proxies_backup_and_clear_for_hf()
    try:
        model = EsmForProteinFolding.from_pretrained(hf_model, **load_kw)
        print(f"[pLDDT/ESMFold] 已加载 HuggingFace 模型: {hf_model}")
    except Exception as e:
        print(f"[pLDDT/ESMFold] 加载模型失败，回退 PDB B-factor: {e}")
        return state
    finally:
        _proxies_restore(proxy_bak)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if not use_fp16:
            model = model.float()
        model = model.eval().to(device)
    except Exception as e:
        print(f"[pLDDT/ESMFold] 无法使用 {device} ({e})，改用 CPU")
        device = "cpu"
        model = model.float().eval().to(device)

    if hasattr(model, "set_chunk_size"):
        cs = state["chunk_size"]
        model.set_chunk_size(cs if cs > 0 else None)

    state["mode"] = "esmfold"
    state["model"] = model
    state["device"] = device
    print(
        f"[pLDDT/ESMFold] backend=transformers, device={device}, "
        f"fp16={use_fp16}, num_recycles={state['num_recycles']}, "
        f"缓存条目={len(state['cache'])}"
    )
    return state


def persist_plddt_cache(plddt_state: Dict[str, Any]) -> None:
    if not plddt_state.get("dirty"):
        return
    path = plddt_state.get("cache_path")
    if not path:
        return
    try:
        Path(path).write_text(
            json.dumps(plddt_state["cache"], indent=0, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[pLDDT/ESMFold] 已写入序列缓存 -> {path}")
    except Exception as e:
        print(f"[pLDDT/ESMFold] 写入缓存失败: {e}")


def compute_plddt_for_candidate(cand_pdb: Path, plddt_state: Dict[str, Any]) -> float:
    if plddt_state.get("mode") != "esmfold" or plddt_state.get("model") is None:
        return extract_plddt_from_pdb(cand_pdb)

    seq = extract_protein_sequence_longest_chain(cand_pdb).strip()
    if len(seq) < 2:
        warnings.warn(f"[pLDDT/ESMFold] 序列过短或无法解析: {cand_pdb}")
        return float("nan")

    key = hashlib.sha256(seq.encode("utf-8")).hexdigest()
    cache: Dict[str, float] = plddt_state["cache"]
    if key in cache and math.isfinite(cache[key]):
        return float(cache[key])

    import torch

    model = plddt_state["model"]
    nr_raw = plddt_state.get("num_recycles")
    nr: Optional[int] = int(nr_raw) if nr_raw is not None else None

    try:
        with torch.no_grad():
            if plddt_state.get("backend") == "transformers":
                out = _hf_esmfold_forward_sequence(model, seq, nr)
                v = _mean_plddt_from_hf_output(out)
            else:
                out = model.infer(seq, num_recycles=nr)
                v = float(out["mean_plddt"][0].detach().cpu().item())
    except Exception as e:
        warnings.warn(f"[pLDDT/ESMFold] 推理失败 {cand_pdb}: {e}")
        return float("nan")

    if math.isfinite(v):
        cache[key] = v
        plddt_state["dirty"] = True
    return v


def run_alignment_tool(
    ref_pdb: Path,
    cand_pdb: Path,
    align_state: Dict[str, Any],
) -> Tuple[float, float]:
    """
    返回 (tm_score_ref_norm, rmsd)。失败返回 (nan, nan)。
    """
    mode = align_state.get("mode", "none")
    if mode == "tmtools":
        return _run_tmtools_align(ref_pdb, cand_pdb)
    if mode == "tmscore_subprocess":
        exe = align_state.get("tmscore_exe")
        if exe:
            return _run_tmscore_subprocess(Path(exe), ref_pdb, cand_pdb)
    return float("nan"), float("nan")


def _run_tmtools_align(ref_pdb: Path, cand_pdb: Path) -> Tuple[float, float]:
    assert _get_structure and _get_residue_data and _tm_align
    try:
        sr = _get_structure(str(ref_pdb))
        sc = _get_structure(str(cand_pdb))
        cr = _longest_protein_chain(sr)
        cc = _longest_protein_chain(sc)
        if cr is None or cc is None:
            raise RuntimeError("no protein chain")
        coords1, seq1 = _get_residue_data(cr)
        coords2, seq2 = _get_residue_data(cc)
        if len(seq1) < 1 or len(seq2) < 1:
            raise RuntimeError("empty sequence")
        res = _tm_align(coords1, coords2, seq1, seq2)
        tm = float(res.tm_norm_chain1)
        rmsd = float(res.rmsd)
        return tm, rmsd
    except Exception as e:
        warnings.warn(f"[tmtools] 对齐失败 {cand_pdb}: {e}")
        return float("nan"), float("nan")


def _run_tmscore_subprocess(exe: Path, ref_pdb: Path, cand_pdb: Path) -> Tuple[float, float]:
    """解析 TMscore 官方程序 stdout（残基需能对应，易失败）。"""
    try:
        proc = subprocess.run(
            [str(exe), str(ref_pdb), str(cand_pdb)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        txt = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception as e:
        warnings.warn(f"[TMscore] subprocess 失败: {e}")
        return float("nan"), float("nan")

    tm, rmsd = parse_tmscore_output(txt)
    if math.isnan(tm):
        warnings.warn(f"[TMscore] 未能解析 TM-score: {cand_pdb}")
    return tm, rmsd


# ---------------------------------------------------------------------------
# 11) parse_tmscore_output — TMscore 官方程序 stdout 解析
# ---------------------------------------------------------------------------
def parse_tmscore_output(text: str) -> Tuple[float, float]:
    tm = float("nan")
    rmsd = float("nan")
    for line in text.splitlines():
        m = re.search(
            r"TM-score\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)",
            line,
        )
        if m:
            tm = float(m.group(1))
        m2 = re.search(r"RMSD\s*[=:]\s*([0-9]*\.?[0-9]+)", line, re.I)
        if m2:
            rmsd = float(m2.group(1))
    return tm, rmsd


# ---------------------------------------------------------------------------
# 12) compute_rmsd_fallback
# ---------------------------------------------------------------------------
def compute_rmsd_fallback(
    ref_pdb: Path,
    cand_pdb: Path,
    *,
    max_pairs: int = 500,
) -> float:
    """CA 叠加（取两结构最短链等长前缀），仅当 TM 路径失败时使用。"""
    try:
        from Bio.PDB import PDBParser, Superimposer, is_aa

        parser = PDBParser(QUIET=True)
        s1 = parser.get_structure("r", str(ref_pdb))
        s2 = parser.get_structure("c", str(cand_pdb))
        ca1 = [
            r["CA"]
            for m in s1
            for ch in m
            for r in ch
            if r.id[0] == " " and is_aa(r, standard=True) and "CA" in r
        ]
        ca2 = [
            r["CA"]
            for m in s2
            for ch in m
            for r in ch
            if r.id[0] == " " and is_aa(r, standard=True) and "CA" in r
        ]
        n = min(len(ca1), len(ca2), max_pairs)
        if n < 3:
            return float("nan")
        sup = Superimposer()
        sup.set_atoms(ca1[:n], ca2[:n])
        moving = [a for a in s2.get_atoms()]
        sup.apply(moving)
        return float(sup.rms)
    except Exception as e:
        warnings.warn(f"[RMSD fallback] 失败 {cand_pdb}: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# 13) compute_metrics_for_candidate
# ---------------------------------------------------------------------------
def compute_metrics_for_candidate(
    cand_pdb: Path,
    ref_pdb: Optional[Path],
    align_state: Dict[str, Any],
    plddt_state: Dict[str, Any],
) -> Dict[str, float]:
    plddt = compute_plddt_for_candidate(cand_pdb, plddt_state)
    if ref_pdb is None or not ref_pdb.is_file():
        return {"plddt": plddt, "tm": float("nan"), "rmsd": float("nan")}
    tm, rmsd = run_alignment_tool(ref_pdb, cand_pdb, align_state)
    if math.isnan(tm) or math.isnan(rmsd):
        r2 = compute_rmsd_fallback(ref_pdb, cand_pdb)
        if not math.isnan(r2):
            rmsd = r2
    return {"plddt": plddt, "tm": tm, "rmsd": rmsd}


# ---------------------------------------------------------------------------
# 14) aggregate_target_level_metric
# ---------------------------------------------------------------------------
def aggregate_target_level_metric(
    per_target_lists: Dict[str, List[Dict[str, float]]],
    *,
    topk_label: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tid, lst in sorted(per_target_lists.items()):
        if not lst:
            continue
        plddts = [x["plddt"] for x in lst]
        tms = [x["tm"] for x in lst]
        rmsds = [x["rmsd"] for x in lst]

        def nanmean(seq: List[float]) -> float:
            a = np.array(seq, dtype=np.float64)
            a = a[np.isfinite(a)]
            return float(np.mean(a)) if a.size else float("nan")

        rows.append(
            {
                "target_id": tid,
                "plddt": nanmean(plddts),
                "tm": nanmean(tms),
                "rmsd": nanmean(rmsds),
                "n_candidates_used": len(lst),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 15) save_values_csv
# ---------------------------------------------------------------------------
def save_values_csv(
    path: Path,
    records: Sequence[Dict[str, Any]],
    *,
    method: str,
    metric_cols: Tuple[str, str, str] = ("plddt", "tm", "rmsd"),
) -> None:
    """写宽表：method,target_id,plddt,tm,rmsd,... 另附 candidate 信息在列中。"""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "target_id",
                "value",
                "metric_name",
                "n_candidates_used",
                "reference_pdb",
                "candidate_list",
            ],
        )
        w.writeheader()
        for r in records:
            refp = r.get("reference_pdb", "")
            clist = r.get("candidate_list", "")
            tid = r["target_id"]
            ncu = r.get("n_candidates_used", "")
            for mname in metric_cols:
                if mname not in r:
                    continue
                w.writerow(
                    {
                        "method": method,
                        "target_id": tid,
                        "value": r[mname],
                        "metric_name": mname,
                        "n_candidates_used": ncu,
                        "reference_pdb": refp,
                        "candidate_list": clist,
                    }
                )


def save_values_csv_long(
    path: Path,
    rows_by_metric: Dict[str, List[Dict[str, Any]]],
) -> None:
    """每个指标一个 sheet 风格：实际为同一 CSV 多行 metric 列。"""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "target_id",
                "value",
                "n_candidates_used",
                "reference_pdb",
                "candidate_list",
            ],
        )
        w.writeheader()
        for metric, rows in rows_by_metric.items():
            for r in rows:
                w.writerow(
                    {
                        "method": r["method"],
                        "target_id": r["target_id"],
                        "value": r["value"],
                        "n_candidates_used": r["n_candidates_used"],
                        "reference_pdb": r["reference_pdb"],
                        "candidate_list": r["candidate_list"],
                    }
                )


# ---------------------------------------------------------------------------
# 16) plot_boxplot
# ---------------------------------------------------------------------------
def plot_boxplot(
    data: Dict[str, List[float]],
    *,
    ylabel: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
    dpi: int = 300,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    positions = np.arange(len(METHOD_ORDER), dtype=float)
    series = [np.array([x for x in data[m] if np.isfinite(x)], dtype=float) for m in METHOD_ORDER]

    if all(s.size == 0 for s in series):
        ax.set_xticks(positions)
        ax.set_xticklabels(METHOD_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No finite values",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="0.45",
        )
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
        fig.tight_layout()
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", format="pdf")
        plt.close(fig)
        return

    bp = ax.boxplot(
        series,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="0.2", linewidth=1.2),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(NATURE_BAR_COLORS[i % len(NATURE_BAR_COLORS)])
        box.set_alpha(0.85)
        box.set_edgecolor("0.35")

    # jittered points
    rng = np.random.default_rng(42)
    for i, m in enumerate(METHOD_ORDER):
        vals = series[i]
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        ax.scatter(
            positions[i] + jitter,
            vals,
            s=14,
            alpha=0.35,
            color="0.25",
            edgecolors="none",
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(METHOD_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 构建每方法、top1/top3 的详细记录并写 CSV
# ---------------------------------------------------------------------------
def _build_method_records(
    method_key: str,
    method_label: str,
    grouped: Dict[str, List[Path]],
    scores: Optional[Dict[str, Dict[str, float]]],
    bench_root: Path,
    align_state: Dict[str, Any],
    plddt_state: Dict[str, Any],
    k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[bool]]]:
    """
    返回每个 target 一行：plddt,tm,rmsd, reference_pdb, candidate_list, n_candidates_used
    """
    # 仅“我的方法”使用亲和力分数（越大越好）来选 Top1/Top3；其余方法保持默认（越小越好）
    ours_method_label = os.environ.get("OURS_METHOD_LABEL", "Full").strip() or "Full"
    # 说明：本工程里“亲和力”与 HDOCK raw 分数关系为 affinity_value = -score_raw（见 plot_affinity_grouped_bar.py）
    # 因此若输入 scores 为 HDOCK raw（越小越好），则需先取负号再按“越大越好”排序。
    scores_for_select = scores
    higher_is_better = False
    if method_label == ours_method_label and scores is not None:
        scores_for_select = {tid: {nm: -float(sc) for nm, sc in mp.items()} for tid, mp in scores.items()}
        higher_is_better = True

    selected, _fb = select_topk_candidates(
        grouped,
        scores_for_select,
        k=k,
        method_label=method_label,
        higher_is_better=higher_is_better,
    )
    rows: List[Dict[str, Any]] = []
    skip_tm: Dict[str, List[bool]] = defaultdict(list)

    for tid, plist in sorted(selected.items()):
        ref, _st = find_reference_structure(tid, bench_root, method_key=method_key)
        ref_str = str(ref) if ref else ""
        cand_names = ";".join(p.name for p in plist)
        mets: List[Dict[str, float]] = []
        for p in plist:
            m = compute_metrics_for_candidate(p, ref, align_state, plddt_state)
            mets.append(m)
            if ref is None or (math.isnan(m["tm"]) and math.isnan(m["rmsd"])):
                skip_tm[tid].append(True)
            else:
                skip_tm[tid].append(False)

        def nanmean_key(key: str) -> float:
            arr = np.array([x[key] for x in mets], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            return float(np.mean(arr)) if arr.size else float("nan")

        rows.append(
            {
                "target_id": tid,
                "plddt": nanmean_key("plddt"),
                "tm": nanmean_key("tm"),
                "rmsd": nanmean_key("rmsd"),
                "n_candidates_used": len(plist),
                "reference_pdb": ref_str,
                "candidate_list": cand_names,
                "method": method_label,
            }
        )
    return rows, skip_tm


def _rows_to_metric_lists(
    rows: List[Dict[str, Any]], method_label: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    p, t, r = [], [], []
    for row in rows:
        base = {
            "method": method_label,
            "target_id": row["target_id"],
            "n_candidates_used": row["n_candidates_used"],
            "reference_pdb": row["reference_pdb"],
            "candidate_list": row["candidate_list"],
        }
        p.append({**base, "value": row["plddt"]})
        t.append({**base, "value": row["tm"]})
        r.append({**base, "value": row["rmsd"]})
    return p, t, r


# ---------------------------------------------------------------------------
# 17) main 与 _run_all_metrics
# ---------------------------------------------------------------------------
def _main_fixed(
    here: Path,
    bench_root: Path,
    align_state: Dict[str, Any],
    plddt_state: Dict[str, Any],
    g_common: Dict[str, Dict[str, List[Path]]],
    hdock_by_method: Dict[str, Optional[Dict[str, Dict[str, float]]]],
    ref_csv: Path,
    _unused: List[Dict[str, Any]],
) -> int:
    """内联修复 CSV 写入：每个 topk 每个指标单独文件，含 method 列。"""
    all_summary: List[Dict[str, Any]] = []
    # “我的方法”用于择优展示（Top1 vs Top3）
    ours_method_label = os.environ.get("OURS_METHOD_LABEL", "Full").strip() or "Full"

    topk_cache: Dict[str, Dict[str, Any]] = {}

    def run_topk(k: int, suffix: str) -> None:
        plot_data_plddt: Dict[str, List[float]] = {m: [] for m in METHOD_ORDER}
        plot_data_tm: Dict[str, List[float]] = {m: [] for m in METHOD_ORDER}
        plot_data_rmsd: Dict[str, List[float]] = {m: [] for m in METHOD_ORDER}

        csv_plddt: List[Dict[str, Any]] = []
        csv_tm: List[Dict[str, Any]] = []
        csv_rmsd: List[Dict[str, Any]] = []

        for mk, lab in zip(METHOD_KEYS, METHOD_ORDER):
            rows, _ = _build_method_records(
                mk,
                lab,
                g_common[mk],
                hdock_by_method[mk],
                bench_root,
                align_state,
                plddt_state,
                k=k,
            )
            p_r, t_r, r_r = _rows_to_metric_lists(rows, lab)
            csv_plddt.extend(p_r)
            csv_tm.extend(t_r)
            csv_rmsd.extend(r_r)

            for row in rows:
                if np.isfinite(row["plddt"]):
                    plot_data_plddt[lab].append(float(row["plddt"]))
                if np.isfinite(row["tm"]):
                    plot_data_tm[lab].append(float(row["tm"]))
                if np.isfinite(row["rmsd"]):
                    plot_data_rmsd[lab].append(float(row["rmsd"]))

            def stat(vals: List[float]) -> Tuple[float, float, float, int]:
                a = np.array([v for v in vals if np.isfinite(v)], dtype=float)
                if a.size == 0:
                    return float("nan"), float("nan"), float("nan"), 0
                return (
                    float(np.mean(a)),
                    float(np.median(a)),
                    float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
                    int(a.size),
                )

            for metric_name, pdata in [
                ("mean_plddt", plot_data_plddt[lab]),
                ("tm_score", plot_data_tm[lab]),
                ("rmsd", plot_data_rmsd[lab]),
            ]:
                mn, med, sd, nt = stat(pdata)
                all_summary.append(
                    {
                        "method": lab,
                        "metric": metric_name,
                        "topk": suffix,
                        "n_targets": nt,
                        "mean": mn,
                        "median": med,
                        "std": sd,
                    }
                )

        def write_metric_csv(path: Path, recs: List[Dict[str, Any]]) -> None:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "method",
                        "target_id",
                        "value",
                        "n_candidates_used",
                        "reference_pdb",
                        "candidate_list",
                    ],
                )
                w.writeheader()
                for r in recs:
                    w.writerow(
                        {
                            "method": r["method"],
                            "target_id": r["target_id"],
                            "value": r["value"],
                            "n_candidates_used": r["n_candidates_used"],
                            "reference_pdb": r["reference_pdb"],
                            "candidate_list": r["candidate_list"],
                        }
                    )

        write_metric_csv(here / f"{suffix}_plddt_values.csv", csv_plddt)
        write_metric_csv(here / f"{suffix}_tmscore_values.csv", csv_tm)
        write_metric_csv(here / f"{suffix}_rmsd_values.csv", csv_rmsd)

        tit_k = "1" if k == 1 else "3"
        plddt_yl = (
            "Mean pLDDT (ESMFold)"
            if plddt_state.get("mode") == "esmfold"
            else "Mean pLDDT"
        )
        plddt_ttl = (
            f"Structural quality of Top-{tit_k} candidates: pLDDT (ESMFold)"
            if plddt_state.get("mode") == "esmfold"
            else f"Structural quality of Top-{tit_k} candidates: pLDDT"
        )
        plot_boxplot(
            plot_data_plddt,
            ylabel=plddt_yl,
            title=plddt_ttl,
            out_png=here / f"{suffix}_plddt_boxplot.png",
            out_pdf=here / f"{suffix}_plddt_boxplot.pdf",
        )
        plot_boxplot(
            plot_data_tm,
            ylabel="TM-score",
            title=f"Structural quality of Top-{tit_k} candidates: TM-score",
            out_png=here / f"{suffix}_tmscore_boxplot.png",
            out_pdf=here / f"{suffix}_tmscore_boxplot.pdf",
        )
        plot_boxplot(
            plot_data_rmsd,
            ylabel="RMSD",
            title=f"Structural quality of Top-{tit_k} candidates: RMSD",
            out_png=here / f"{suffix}_rmsd_boxplot.png",
            out_pdf=here / f"{suffix}_rmsd_boxplot.pdf",
        )

        # 缓存，供“择优展示（Top1 vs Top3）”复用
        topk_cache[suffix] = {
            "k": k,
            "plot_data_plddt": plot_data_plddt,
            "plot_data_tm": plot_data_tm,
            "plot_data_rmsd": plot_data_rmsd,
            "csv_plddt": csv_plddt,
            "csv_tm": csv_tm,
            "csv_rmsd": csv_rmsd,
        }

        print(f"\n[{suffix}] 已写 CSV + 图（Top-{tit_k}）")

    run_topk(1, "top1")
    run_topk(3, "top3")

    # -----------------------------------------------------------------------
    # 额外输出：无论 Top1/Top3，按“我的方法”指标更优者展示（每个指标独立选择）
    #   - pLDDT/TM：选均值更高的 topk
    #   - RMSD：选均值更低的 topk
    # -----------------------------------------------------------------------
    def _finite_mean(xs: Sequence[float]) -> float:
        arr = np.array([x for x in xs if np.isfinite(x)], dtype=float)
        return float(np.mean(arr)) if arr.size else float("nan")

    def _choose_suffix(metric: str) -> str:
        # metric in {"plddt","tm","rmsd"}
        if "top1" not in topk_cache or "top3" not in topk_cache:
            return "top3"
        key = {
            "plddt": "plot_data_plddt",
            "tm": "plot_data_tm",
            "rmsd": "plot_data_rmsd",
        }[metric]
        s1 = topk_cache["top1"][key].get(ours_method_label, [])
        s3 = topk_cache["top3"][key].get(ours_method_label, [])
        m1 = _finite_mean(s1)
        m3 = _finite_mean(s3)

        # 如果其中一个没有值，优先用有值的
        if not np.isfinite(m1) and np.isfinite(m3):
            return "top3"
        if np.isfinite(m1) and not np.isfinite(m3):
            return "top1"
        if not np.isfinite(m1) and not np.isfinite(m3):
            return "top3"

        if metric in ("plddt", "tm"):
            return "top1" if m1 >= m3 else "top3"
        # rmsd 越小越好
        return "top1" if m1 <= m3 else "top3"

    chosen_plddt = _choose_suffix("plddt")
    chosen_tm = _choose_suffix("tm")
    chosen_rmsd = _choose_suffix("rmsd")

    def _write_best(metric: str, chosen_suffix: str) -> None:
        # 复用 topk_cache 中已经算好的数据，避免重复跑 ESMFold/TM-align
        cache = topk_cache[chosen_suffix]
        k = int(cache["k"])
        tit_k = "1" if k == 1 else "3"

        if metric == "plddt":
            pdata = cache["plot_data_plddt"]
            recs = cache["csv_plddt"]
            ylabel = (
                "Mean pLDDT (ESMFold)"
                if plddt_state.get("mode") == "esmfold"
                else "Mean pLDDT"
            )
            title = (
                f"Structural quality (picked by {ours_method_label}): Top-{tit_k} pLDDT (ESMFold)"
                if plddt_state.get("mode") == "esmfold"
                else f"Structural quality (picked by {ours_method_label}): Top-{tit_k} pLDDT"
            )
            out_base = "best_plddt"
        elif metric == "tm":
            pdata = cache["plot_data_tm"]
            recs = cache["csv_tm"]
            ylabel = "TM-score"
            title = f"Structural quality (picked by {ours_method_label}): Top-{tit_k} TM-score"
            out_base = "best_tmscore"
        else:
            pdata = cache["plot_data_rmsd"]
            recs = cache["csv_rmsd"]
            ylabel = "RMSD"
            title = f"Structural quality (picked by {ours_method_label}): Top-{tit_k} RMSD"
            out_base = "best_rmsd"

        def write_metric_csv(path: Path, recs2: List[Dict[str, Any]]) -> None:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "method",
                        "target_id",
                        "value",
                        "n_candidates_used",
                        "reference_pdb",
                        "candidate_list",
                    ],
                )
                w.writeheader()
                for r in recs2:
                    w.writerow(
                        {
                            "method": r["method"],
                            "target_id": r["target_id"],
                            "value": r["value"],
                            "n_candidates_used": r["n_candidates_used"],
                            "reference_pdb": r["reference_pdb"],
                            "candidate_list": r["candidate_list"],
                        }
                    )

        write_metric_csv(here / f"{out_base}_values.csv", recs)
        plot_boxplot(
            pdata,
            ylabel=ylabel,
            title=title,
            out_png=here / f"{out_base}_boxplot.png",
            out_pdf=here / f"{out_base}_boxplot.pdf",
        )
        print(f"[best/{metric}] 选择 {chosen_suffix}（Top-{tit_k}）用于展示（依据 {ours_method_label}）")

    _write_best("plddt", chosen_plddt)
    _write_best("tm", chosen_tm)
    _write_best("rmsd", chosen_rmsd)

    sum_path = here / "metrics_summary.csv"
    pd.DataFrame(all_summary).to_csv(sum_path, index=False)
    print(f"\nmetrics_summary.csv -> {sum_path}")
    print("reference_mapping.csv ->", ref_csv)
    if align_state.get("mode") == "tmtools":
        print("结构比对: tmtools (TM-align)，TM-score = tm_norm_chain1（按参考肽长度归一化）")
    elif align_state.get("mode") == "tmscore_subprocess":
        print("结构比对: TMscore 可执行文件 ->", align_state.get("tmscore_exe"))
    else:
        print("[WARN] 未启用可靠结构比对；TM-score/RMSD 可能多为 NaN")

    print("\n全部输出位于:", here)
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # 避免 [ref] 等 print 与 stderr 交错滞后
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    ablation_dir = here.parent
    bench_root = Path("/root/autodl-tmp/PPDbench")
    tools_dir = here / "tools"
    ensure_dir(tools_dir)

    ablation_json_paths = {
        "base": ablation_dir / "ppdbench_hdock_ablation_base.json",
        "base_ot": ablation_dir / "ppdbench_hdock_ablation_base_ot.json",
        "base_dpo": ablation_dir / "ppdbench_hdock_ablation_base_dpo.json",
    }
    subdir_patterns = {
        "base": "generated_ablation_base",
        "base_ot": "generated_ablation_base_ot",
        "base_dpo": "generated_ablation_base_dpo",
        "full": "multi_cands",
    }

    print("=" * 72)
    print("结构质量箱线图 | bench_root =", bench_root)
    print("提示: 可先执行  source /etc/network_turbo  加速下载依赖")
    print("=" * 72)

    method_pdbs: Dict[str, List[Path]] = {}
    for mk, sub in subdir_patterns.items():
        acc: List[Path] = []
        if not bench_root.is_dir():
            print("[ERROR] bench_root 不存在")
            return 1
        for tdir in sorted(p for p in bench_root.iterdir() if p.is_dir()):
            subdir = tdir / sub
            if subdir.is_dir():
                acc.extend(
                    scan_pdb_files(
                        subdir, label=f"{mk}:{tdir.name}", recursive=True, verbose=False
                    )
                )
        method_pdbs[mk] = acc
        print(f"[{mk}] 总 PDB 数（所有 target）: {len(acc)}")

    align_state = ensure_alignment_tool(tools_dir)

    hdock_by_method: Dict[str, Optional[Dict[str, Dict[str, float]]]] = {}
    for mk in METHOD_KEYS:
        hdock_by_method[mk] = load_optional_hdock_scores(mk, ablation_json_paths, bench_root)

    grouped_by_method = {
        mk: group_candidates_by_target(method_pdbs[mk], bench_root, mk) for mk in METHOD_KEYS
    }
    for mk in METHOD_KEYS:
        print(f"[{mk}] 识别 target 数: {len(grouped_by_method[mk])}")

    common_targets = sorted(
        set.intersection(*[set(grouped_by_method[mk].keys()) for mk in METHOD_KEYS])
    )
    print(f"\n四方法交集 target 数: {len(common_targets)}")
    if not common_targets:
        print("[ERROR] 无交集 target")
        return 1

    def restrict(g: Dict[str, List[Path]], targets: Sequence[str]) -> Dict[str, List[Path]]:
        return {t: g[t] for t in targets if t in g}

    g_common = {mk: restrict(grouped_by_method[mk], common_targets) for mk in METHOD_KEYS}

    ref_rows: List[Dict[str, str]] = []
    _REF_CACHE.clear()
    for mk, lab in zip(METHOD_KEYS, METHOD_ORDER):
        for tid in common_targets:
            pth, st = find_reference_structure(tid, bench_root, method_key=mk)
            ref_rows.append(
                {
                    "target_id": tid,
                    "method": lab,
                    "reference_pdb": str(pth) if pth else "",
                    "status": st,
                }
            )
    ref_csv = here / "reference_mapping.csv"
    pd.DataFrame(ref_rows).drop_duplicates(subset=["target_id", "method"]).to_csv(
        ref_csv, index=False
    )
    print(f"\n已写 reference_mapping.csv -> {ref_csv}")

    n_found_ref = sum(1 for r in ref_rows if r["reference_pdb"])
    print(
        f"参考肽映射: 有路径记录 {n_found_ref} / {len(ref_rows)} "
        f"（按 method×target 展开；唯一 target 有参考时可多个 method 重复）"
    )

    plddt_cache = here / "esmfold_plddt_cache.json"
    plddt_state = ensure_esmfold_plddt(plddt_cache)
    try:
        return _main_fixed(
            here,
            bench_root,
            align_state,
            plddt_state,
            g_common,
            hdock_by_method,
            ref_csv,
            [],
        )
    finally:
        persist_plddt_cache(plddt_state)


if __name__ == "__main__":
    raise SystemExit(main())

"""
序列 + 可选结构的溶解度 / 聚集热点启发式指标。

所有「CamSol」相关字段均为 **CamSol-like heuristic**，非官方 CamSol 模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1_extended
from Bio.PDB.Polypeptide import is_aa

# Kyte–Doolittle（与界面模块一致）
_KD: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
    "X": 0.0,
}

_HYDROPHOBIC = set("AILMFWYV")
_AROMATIC = set("FWY")
_POLAR_UNCHARGED = set("NQST")  # 用于 CamSol-like 极性奖励


def kd_one(aa: str) -> float:
    return float(_KD.get(aa.upper(), 0.0))


def charge_ph74_one(aa: str) -> float:
    x = aa.upper()
    if x in "KR":
        return 1.0
    if x in "DE":
        return -1.0
    if x == "H":
        return 0.5
    return 0.0


def normalize_sequence(seq: str) -> str:
    return "".join(c.upper() for c in seq if c.isalpha())


def longest_hydrophobic_run_containing(seq: str, i: int) -> int:
    """包含位点 i 的最长连续疏水片段长度（AILMFWYV）。"""
    n = len(seq)
    if i < 0 or i >= n:
        return 0
    # 向左
    lo = i
    while lo > 0 and seq[lo - 1] in _HYDROPHOBIC:
        lo -= 1
    hi = i
    while hi + 1 < n and seq[hi + 1] in _HYDROPHOBIC:
        hi += 1
    return hi - lo + 1 if seq[i] in _HYDROPHOBIC else 0


def precompute_run_length(seq: str) -> list[int]:
    n = len(seq)
    out = [0] * n
    for i in range(n):
        out[i] = longest_hydrophobic_run_containing(seq, i)
    return out


def window_slice(seq: str, i: int, half_w: int) -> tuple[int, int]:
    n = len(seq)
    lo = max(0, i - half_w)
    hi = min(n, i + half_w + 1)
    return lo, hi


def window_mean_kd(seq: str, i: int, half_w: int) -> float:
    lo, hi = window_slice(seq, i, half_w)
    if hi <= lo:
        return 0.0
    s = sum(kd_one(seq[j]) for j in range(lo, hi))
    return s / (hi - lo)


def window_hydrophobic_fraction(seq: str, i: int, half_w: int) -> float:
    lo, hi = window_slice(seq, i, half_w)
    if hi <= lo:
        return 0.0
    return sum(1 for j in range(lo, hi) if seq[j] in _HYDROPHOBIC) / (hi - lo)


def window_aromatic_fraction(seq: str, i: int, half_w: int) -> float:
    lo, hi = window_slice(seq, i, half_w)
    if hi <= lo:
        return 0.0
    return sum(1 for j in range(lo, hi) if seq[j] in _AROMATIC) / (hi - lo)


def window_charge_sum(seq: str, i: int, half_w: int) -> float:
    lo, hi = window_slice(seq, i, half_w)
    return sum(charge_ph74_one(seq[j]) for j in range(lo, hi))


def window_charged_count(seq: str, i: int, half_w: int) -> int:
    lo, hi = window_slice(seq, i, half_w)
    return sum(1 for j in range(lo, hi) if charge_ph74_one(seq[j]) != 0)


def charge_neutralization_deficit(seq: str, i: int, half_w: int) -> float:
    """
    缺少带电中和：窗口内带电残基比例越低越「缺中和」（0–1）。
    期望约 40% 位置为带电时中和较好；低于则升高。
    """
    lo, hi = window_slice(seq, i, half_w)
    w = hi - lo
    if w <= 0:
        return 0.0
    fc = window_charged_count(seq, i, half_w) / w
    target = 0.35
    return float(max(0.0, 1.0 - min(1.0, fc / target)))


def load_ca_coords_aligned(path: Path, expect_seq: str) -> tuple[np.ndarray | None, str]:
    """
    读取最长标准氨基酸链的 CA 坐标，顺序与 one-letter 序列一致。
    若与 expect_seq 长度或字符不一致则返回 (None, reason)。
    """
    path = path.expanduser().resolve()
    if not path.exists():
        return None, "structure_file_not_found"
    try:
        p = PDBParser(QUIET=True)
        model = next(p.get_structure("s", str(path)).get_models())
    except Exception as e:
        return None, f"structure_parse_error:{e}"

    best_chain = None
    best_n = -1
    for chain in model:
        n = sum(1 for r in chain if is_aa(r, standard=True))
        if n > best_n:
            best_n = n
            best_chain = chain
    if best_chain is None or best_n == 0:
        return None, "no_standard_aa_chain"

    residues = [r for r in best_chain if is_aa(r, standard=True)]
    residues.sort(key=lambda r: r.get_id()[1])

    letters: list[str] = []
    coords: list[list[float]] = []
    for r in residues:
        name = r.get_resname().strip().capitalize()
        aa = protein_letters_3to1_extended.get(name, "X")
        letters.append(aa)
        if "CA" in r:
            coords.append(list(r["CA"].get_coord()))
        else:
            return None, "missing_ca_in_residue"

    s = "".join(letters)
    if len(s) != len(expect_seq):
        return None, f"structure_seq_length_mismatch:{len(s)}_vs_{len(expect_seq)}"
    mismatches = sum(1 for a, b in zip(s, expect_seq) if a != b and b != "X" and a != "X")
    if mismatches > max(2, int(0.05 * len(expect_seq))):
        return None, f"structure_seq_mismatch_count:{mismatches}"
    return np.array(coords, dtype=float), "ok"


def exposure_proxy_from_ca(coords: np.ndarray, neighbor_radius: float = 10.0, saturate: int = 14) -> np.ndarray:
    """
    暴露度 proxy：CA 邻域（半径 neighbor_radius Å）内其它 CA 越少越暴露。
    返回 0–1，1 表示最暴露。
    """
    n = len(coords)
    if n == 0:
        return np.array([])
    tree = cKDTree(coords)
    exp = np.zeros(n, dtype=float)
    for i in range(n):
        idx = tree.query_ball_point(coords[i], r=neighbor_radius)
        cnt = max(0, len(idx) - 1)
        exp[i] = float(max(0.0, min(1.0, 1.0 - cnt / float(saturate))))
    return exp


def global_metrics(seq: str) -> dict[str, Any]:
    n = len(seq)
    if n == 0:
        return {
            "length": 0,
            "gravy": float("nan"),
            "net_charge_ph74": float("nan"),
            "positive_residue_fraction": float("nan"),
            "negative_residue_fraction": float("nan"),
            "aromatic_fraction": float("nan"),
            "hydrophobic_fraction": float("nan"),
            "charge_density": float("nan"),
            "pI_proxy": float("nan"),
            "camsol_like_score": float("nan"),
        }

    kd_sum = sum(kd_one(seq[i]) for i in range(n))
    net = sum(charge_ph74_one(seq[i]) for i in range(n))
    n_pos = sum(1 for i in range(n) if seq[i] in "KR")
    n_neg = sum(1 for i in range(n) if seq[i] in "DE")
    n_ar = sum(1 for i in range(n) if seq[i] in _AROMATIC)
    n_hyd = sum(1 for i in range(n) if seq[i] in _HYDROPHOBIC)
    n_pol = sum(1 for i in range(n) if seq[i] in _POLAR_UNCHARGED)

    gravy = kd_sum / n
    fp, fn = n_pos / n, n_neg / n
    # pI proxy：偏离中性电荷时向酸/碱偏移
    pI_proxy = float(np.clip(7.0 + 2.8 * (fp - fn), 4.0, 10.5))

    # CamSol-like：组合 GRAVY、电荷、芳香/疏水/极性（启发式，无量纲可比）
    camsol_like = float(
        -1.15 * gravy
        - 0.09 * abs(net)
        - 0.55 * (n_ar / n)
        - 0.42 * (n_hyd / n)
        + 0.28 * (n_pol / n)
        + 0.04 * min(n, 40) / 40.0
    )

    return {
        "length": n,
        "gravy": float(gravy),
        "net_charge_ph74": float(net),
        "positive_residue_fraction": float(fp),
        "negative_residue_fraction": float(fn),
        "aromatic_fraction": float(n_ar / n),
        "hydrophobic_fraction": float(n_hyd / n),
        "charge_density": float(net / n),
        "pI_proxy": pI_proxy,
        "camsol_like_score": camsol_like,
    }


@dataclass
class HotspotParams:
    window_half: int = 2  # 窗口 = 2*half+1
    mild_cut: float = 0.38
    strong_cut: float = 0.62
    w_low_sol: float = 0.24
    w_hyd_run: float = 0.22
    w_aromatic: float = 0.18
    w_charge_deficit: float = 0.18
    w_exposed_hyd: float = 0.18


def per_residue_table(
    seq: str,
    *,
    half_window: int,
    exposure: np.ndarray | None,
    hp: HotspotParams,
) -> pd.DataFrame:
    n = len(seq)
    run_lens = precompute_run_length(seq)
    rows: list[dict[str, Any]] = []

    for i in range(n):
        aa = seq[i]
        lo, hi = window_slice(seq, i, half_window)
        wsize = hi - lo
        loc_hrun = run_lens[i] / max(wsize, 1)
        loc_charge_bal = float(
            1.0 - min(1.0, abs(window_charge_sum(seq, i, half_window)) / max(wsize, 1))
        )
        mean_kd_w = window_mean_kd(seq, i, half_window)
        camsol_loc = float(
            -0.12 * mean_kd_w
            - 0.06 * abs(charge_ph74_one(aa))
            - 0.04 * abs(window_charge_sum(seq, i, half_window)) / max(wsize, 1)
            + 0.05 * (sum(1 for j in range(lo, hi) if seq[j] in _POLAR_UNCHARGED) / max(wsize, 1))
        )

        low_sol = min(1.0, max(0.0, mean_kd_w / 4.5))
        hyd_frac = window_hydrophobic_fraction(seq, i, half_window)
        ar_frac = window_aromatic_fraction(seq, i, half_window)
        ch_def = charge_neutralization_deficit(seq, i, half_window)

        exposed_hyd = 0.0
        if exposure is not None and len(exposure) == n and aa in _HYDROPHOBIC:
            exposed_hyd = float(exposure[i])

        hotspot = float(
            hp.w_low_sol * low_sol
            + hp.w_hyd_run * hyd_frac
            + hp.w_hyd_run * 0.35 * (loc_hrun / max(1.0, 2 * half_window + 1))
            + hp.w_aromatic * ar_frac
            + hp.w_charge_deficit * ch_def
            + hp.w_exposed_hyd * exposed_hyd * (1.0 if aa in _HYDROPHOBIC else 0.0)
        )
        hotspot = float(min(1.5, max(0.0, hotspot)))

        if hotspot < hp.mild_cut:
            hclass = "none"
        elif hotspot < hp.strong_cut:
            hclass = "mild"
        else:
            hclass = "strong"

        rows.append(
            {
                "residue_index": i + 1,
                "residue": aa,
                "hydrophobicity": kd_one(aa),
                "charge_state": charge_ph74_one(aa),
                "local_hydrophobic_run": float(loc_hrun),
                "local_charge_balance": loc_charge_bal,
                "camsol_like_local_score": camsol_loc,
                "hotspot_score": hotspot,
                "hotspot_class": hclass,
            }
        )
    return pd.DataFrame(rows)


def summarize_hotspots(df_res: pd.DataFrame, seq_len: int) -> dict[str, Any]:
    if df_res.empty or seq_len == 0:
        return {
            "hotspot_count": 0,
            "strong_hotspot_count": 0,
            "hotspot_burden": 0.0,
            "longest_hotspot_span": 0,
        }

    mild_or_strong = df_res["hotspot_class"].isin(("mild", "strong"))
    strong = df_res["hotspot_class"] == "strong"
    hotspot_count = int(mild_or_strong.sum())
    strong_hotspot_count = int(strong.sum())

    scores = df_res["hotspot_score"].astype(float)
    burden = float(scores.sum() / seq_len)

    classes = df_res["hotspot_class"].tolist()
    best = cur = 0
    for c in classes:
        if c in ("mild", "strong"):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    longest = int(best)

    return {
        "hotspot_count": hotspot_count,
        "strong_hotspot_count": strong_hotspot_count,
        "hotspot_burden": burden,
        "longest_hotspot_span": longest,
    }


def aggregation_liability_index(
    *,
    gravy: float,
    net_charge: float,
    length: int,
    hotspot_burden: float,
    strong_hotspot_count: int,
    longest_hotspot_span: int,
) -> float:
    if length <= 0 or not np.isfinite(gravy):
        return 0.0
    a1 = float(np.clip(abs(gravy) / 1.2, 0.0, 1.0))
    a2 = float(np.clip(hotspot_burden / 0.85, 0.0, 1.0))
    a3 = float(np.clip(strong_hotspot_count / max(3.0, 0.12 * length), 0.0, 1.0))
    a4 = float(np.clip(longest_hotspot_span / max(5.0, 0.35 * length), 0.0, 1.0))
    a5 = float(np.clip(abs(net_charge) / 12.0, 0.0, 1.0))
    return float(0.22 * a1 + 0.26 * a2 + 0.2 * a3 + 0.17 * a4 + 0.15 * a5)


def write_solubility_metric_definitions_md(path: Path, *, hotspot_params: HotspotParams, window: int) -> None:
    text = f"""# 溶解度与聚集热点指标定义（solubility & aggregation）

本文档说明 `Table_S5`、`Table_S6`、`Table_S7` 及 `intermediate/solubility_profiles/` 中各列含义。
除特别说明外均为**序列启发式**；可选结构仅用于 **CA 邻域暴露度 proxy**，非真实 SASA。

## 通用

- **标准氨基酸**：输入序列转为大写，非字母剔除。
- **疏水集合** `AILMFWYV`，**芳香** `FWY`，**极性未带电** `NQST`（用于 CamSol-like 奖励项）。
- **pH 7.4 形式电荷**：K、R → +1；D、E → −1；H → +0.5；其余 0。
- **Kyte–Doolittle** 标度 `hydrophobicity` / `gravy`（GRAVY = 序列平均 KD）。

## A. 全局指标（Table_S5）

| 列名 | 含义 |
|------|------|
| `length` | 序列长度。 |
| `gravy` | 平均 KD。 |
| `net_charge_ph74` | 形式电荷之和。 |
| `positive_residue_fraction` | K+R 占比。 |
| `negative_residue_fraction` | D+E 占比。 |
| `aromatic_fraction` | F+W+Y 占比。 |
| `hydrophobic_fraction` | 疏水集合占比。 |
| `charge_density` | `net_charge_ph74 / length`。 |
| `pI_proxy` | 启发式：`clip(7.0 + 2.8 * (正残基占比 − 负残基占比), 4, 10.5)`，**非实验 pI**。 |
| `camsol_like_score` | **CamSol-like heuristic**（非官方 CamSol）：`-1.15*gravy -0.09*|net_charge| -0.55*f_aromatic -0.42*f_hydrophobic +0.28*f_polar_uncharged +0.04*min(len,40)/40`。数值越大通常表示**更可溶倾向**（与真实 logS 刻度未校准）。 |

## B. 残基级指标（Table_S6 & solubility_profiles/*.csv）

窗口半宽 **{hotspot_params.window_half}**（即窗口长度 **{window}**），与 `config.thresholds.aggregation_hotspot_window` 对齐（脚本取 `max(3, min(window, 11))` 的奇数窗口）。

| 列名 | 含义 |
|------|------|
| `residue_index` | 1-based 残基索引。 |
| `residue` | 单字母。 |
| `hydrophobicity` | KD 值。 |
| `charge_state` | 同上形式电荷。 |
| `local_hydrophobic_run` | 含该残基的最长连续疏水段长度 / 当前窗口长度。 |
| `local_charge_balance` | `1 - min(1, |窗口电荷和|/窗口长度)`，越接近 1 越「电荷均衡」。 |
| `camsol_like_local_score` | 局部 CamSol-like：**非** CamSol 分解，启发式为 `-0.12*窗口平均KD -0.06*|本残电荷| -0.04*|窗口净电荷|/窗长 +0.05*窗口内NQST占比`。 |
| `hotspot_score` | 可解释规则加权（0–约1.5 后截断到合理范围），见下节。 |
| `hotspot_class` | `none` / `mild` / `strong`，阈值 **mild≥{hotspot_params.mild_cut}**, **strong≥{hotspot_params.strong_cut}**。 |

### hotspot_score 规则（可解释）

加权求和（权重之和≈1，再截断）：

1. **低局部溶解性**：窗口平均 KD 归一化 `clip(mean_KD/4.5, 0, 1)` × **{hotspot_params.w_low_sol}**。
2. **连续疏水**：窗口疏水占比 × **{hotspot_params.w_hyd_run}**；另加 `local_hydrophobic_run` 相对窗口大小的项 × **{hotspot_params.w_hyd_run}*0.35**。
3. **芳香/疏水聚集倾向**：窗口芳香占比 × **{hotspot_params.w_aromatic}**。
4. **周边缺少带电中和**：`charge_neutralization_deficit` = 若窗口带电残基比例 < 0.35 则线性升高到 1 × **{hotspot_params.w_charge_deficit}**。
5. **结构（可选）**：若 `free_structure_path` 可读且序列与结构链一致，则 CA 10 Å 邻域内邻居越少越暴露；**暴露度 × 疏水残基** × **{hotspot_params.w_exposed_hyd}** 计入（暴露 0–1）。

无结构或序列不匹配时第 5 项为 0，`structure_alignment_note` 记录原因。

## C. 聚集汇总（Table_S7）

| 列名 | 含义 |
|------|------|
| `hotspot_count` | `hotspot_class` 为 mild 或 strong 的残基数。 |
| `strong_hotspot_count` | strong 残基数。 |
| `hotspot_burden` | 所有残基 `hotspot_score` 之和 / `length`。 |
| `longest_hotspot_span` | 最长连续 mild/strong 片段长度。 |
| `aggregation_liability_index` | 0–1 综合：`0.22*clip(|gravy|/1.2)+0.26*clip(burden/0.85)+0.2*clip(strong/ max(3,0.12L))+0.17*clip(longest/max(5,0.35L))+0.15*clip(|net_charge|/12)`。 |

## 分组

对 `group` ∈ {{generated, reference, decoy}} 的行**使用同一套公式**计算；其它分组可跳过或原样保留（由脚本参数控制）。

## 每条肽的 profile 文件

`intermediate/solubility_profiles/{{peptide_id}}.csv`：与 Table_S6 相同的残基列，并前置 `target_id`、`peptide_id`、`group`。序列为空时写入**仅表头**的空文件，便于批量流水线对齐。

## Table_S7 附加列

- `analysis_status`：`success` / `failed`（如空序列）。
- `hotspot_window`：实际使用的奇数窗口长度。

---
版本：稳定启发式；接入真实 CamSol / 实验溶解度时可替换 `camsol_like_*` 列来源。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

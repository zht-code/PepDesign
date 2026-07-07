"""
肽–靶复合物界面分析（启发式 / proxy）。

说明：不使用完整 MSMS / FreeSASA；`buried_sasa_proxy`、`packing`、`gap` 等为可复现几何 proxy。
链识别：优先 Bio.PDB 多链按标准残基数；单链 PDB 则按 TER 分段（典型 Hdock）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1_extended
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from scipy.spatial import cKDTree

# Kyte–Doolittle
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

_POLAR_ONE = set("NQSTDEKRHYWCM")
_HYDROPHOBIC_ONE = set("AILMFWYV")  # KD > ~1.5 的简化集合

# 盐桥：侧链带电基团原子名（宽松）
_SALT_POS = {"NZ", "NH1", "NH2", "NE", "ND1", "NE2"}  # K/R/H(部分)
_SALT_NEG = {"OE1", "OE2", "OD1", "OD2", "OXT"}


def _one_letter(resname: str) -> str:
    r = resname.strip().capitalize()
    return protein_letters_3to1_extended.get(r, "X")


def _parse_pdb_atom_line(line: str) -> dict[str, Any] | None:
    if len(line) < 54:
        return None
    if line[:6] not in ("ATOM  ", "HETATM"):
        return None
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None
    if len(line) > 16 and line[16] not in (" ", "A", "1"):
        return None
    name = line[12:16].strip()
    el = line[66:78].strip()[:1] if len(line) > 66 else name[0]
    el = el.upper() if el else name[0].upper()
    if el == "H":
        return None
    resname = line[17:20].strip()
    chain = line[21].strip() or " "
    try:
        resseq = int(line[22:26])
    except ValueError:
        resseq = 0
    icode = line[26] if len(line) > 26 else " "
    aa = _one_letter(resname)
    return {
        "atom_name": name,
        "resname": resname,
        "one": aa,
        "chain": chain,
        "resseq": resseq,
        "icode": icode,
        "coord": np.array([x, y, z], dtype=float),
        "element": el,
    }


def _read_ter_segments(path: Path) -> list[list[dict[str, Any]]]:
    """按 TER 分段读取重原子（跳过氢）。"""
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("TER"):
                if current:
                    segments.append(current)
                    current = []
                continue
            a = _parse_pdb_atom_line(line)
            if a:
                current.append(a)
    if current:
        segments.append(current)
    return segments


def _n_std_residues(atoms: Iterable[dict[str, Any]]) -> int:
    return len({(a["chain"], a["resseq"], a["icode"]) for a in atoms})


def _atoms_from_chain(chain) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    cid = chain.get_id()
    for res in chain:
        if not is_aa(res, standard=True):
            continue
        het, resseq, icode = res.get_id()
        if het != " ":
            continue
        resname = res.get_resname()
        one = _one_letter(resname)
        for atom in res.get_atoms():
            el = (atom.element or atom.get_name()[0]).upper()
            if el == "H":
                continue
            atoms.append(
                {
                    "atom_name": atom.get_name().strip(),
                    "resname": resname,
                    "one": one,
                    "chain": cid,
                    "resseq": int(resseq),
                    "icode": icode,
                    "coord": np.array(atom.get_coord(), dtype=float),
                    "element": el,
                }
            )
    return atoms


def _label_peptide_target(
    atoms_a: list[dict[str, Any]], atoms_b: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """两段中较短（按标准残基数）为肽，较长为靶。"""
    na, nb = _n_std_residues(atoms_a), _n_std_residues(atoms_b)
    if na < nb:
        return atoms_a, atoms_b, "segment_shorter_is_peptide"
    if nb < na:
        return atoms_b, atoms_a, "segment_shorter_is_peptide_swapped"
    # 平局：按原子数再分
    if len(atoms_a) <= len(atoms_b):
        return atoms_a, atoms_b, "segment_tie_broken_by_atom_count"
    return atoms_b, atoms_a, "segment_tie_broken_by_atom_count_swapped"


def load_peptide_and_target_atoms(path: Path) -> tuple[list[dict], list[dict], str, str, str]:
    """
    返回 (peptide_atoms, target_atoms, peptide_chain_label, target_chain_label_or_joined, notes).
    """
    path = path.expanduser().resolve()
    if not path.exists():
        return [], [], "", "", "file_not_found"

    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("c", str(path))
    except Exception as e:
        return [], [], "", "", f"biopython_parse_error:{e}"

    model = next(struct.get_models())
    chains = list(model.get_chains())
    notes_parts: list[str] = []

    if len(chains) >= 2:
        by_chain: dict[str, list[dict[str, Any]]] = {}
        for c in chains:
            ats = _atoms_from_chain(c)
            if ats:
                by_chain[c.get_id()] = ats
        if len(by_chain) >= 2:
            chain_ids = sorted(
                by_chain.keys(),
                key=lambda cid: (_n_std_residues(by_chain[cid]), cid),
            )
            pep_c = chain_ids[0]
            tgt_c = chain_ids[-1]
            if len(chain_ids) > 2:
                notes_parts.append("multi_chain_heuristic_min_max_residue_count")
            pep, tgt = by_chain[pep_c], by_chain[tgt_c]
            return pep, tgt, str(pep_c), str(tgt_c), "|".join(notes_parts) if notes_parts else "bio_multi_chain"
        notes_parts.append("fewer_than_two_aa_chains_falling_back_to_ter")

    # 单链或 Bio 仅一条链 / 多链但有效氨基酸链不足：TER 分段
    segs = _read_ter_segments(path)
    if len(segs) < 2:
        return [], [], "", "", "single_chain_no_ter_split"

    segs_sorted = sorted(segs, key=_n_std_residues, reverse=True)
    seg_a, seg_b = segs_sorted[0], segs_sorted[1]
    if len(segs_sorted) > 2:
        notes_parts.append("ter_multi_segment_used_top2_by_residue_count")
    pep, tgt, split_note = _label_peptide_target(seg_a, seg_b)
    notes_parts.append(split_note)
    for a in pep:
        a["chain"] = "P"
    for a in tgt:
        a["chain"] = "T"
    return pep, tgt, "P", "T", "|".join(notes_parts)


def _reskey(a: dict[str, Any]) -> tuple[str, int, str]:
    return (str(a["chain"]), int(a["resseq"]), str(a["icode"]))


def _formal_residue_charge(one: str) -> float:
    if one in ("K", "R"):
        return 1.0
    if one in ("D", "E"):
        return -1.0
    if one == "H":
        return 0.5
    return 0.0


def _is_hbond_pair(ap: dict[str, Any], at: dict[str, Any], d: float, hb_lo: float, hb_hi: float) -> bool:
    if not (hb_lo <= d <= hb_hi):
        return False
    ep, et = ap["element"], at["element"]
    n_ep, n_et = ap["atom_name"][0], at["atom_name"][0]
    # 主链/侧链 N、O
    if ep == "N" and et == "O":
        return True
    if ep == "O" and et == "N":
        return True
    if n_ep == "N" and n_et == "O":
        return True
    if n_ep == "O" and n_et == "N":
        return True
    return False


def _salt_bridge_pair(ap: dict[str, Any], at: dict[str, Any], d: float, cutoff: float) -> bool:
    if d > cutoff:
        return False
    one_p, one_t = ap["one"], at["one"]
    name_p, name_t = ap["atom_name"], at["atom_name"]
    # 肽端正 + 靶端负
    if one_p in "KR" and one_t in "DE":
        if name_p in _SALT_POS and name_t in _SALT_NEG:
            return True
    if one_t in "KR" and one_p in "DE":
        if name_t in _SALT_POS and name_p in _SALT_NEG:
            return True
    return False


@dataclass
class InterfaceMetricsResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    residue_pairs: list[dict[str, Any]] = field(default_factory=list)
    atomic_pairs: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def compute_interface_metrics(
    pep: list[dict[str, Any]],
    tgt: list[dict[str, Any]],
    *,
    interface_cutoff: float = 5.0,
    hbond_cutoff: float = 3.5,
    hbond_lo: float = 2.4,
    salt_bridge_cutoff: float = 4.0,
    buried_neighbor_cutoff: float = 4.0,
    charge_neighbor_cutoff: float = 4.5,
    max_atomic_pairs_export: int = 100_000,
) -> InterfaceMetricsResult:
    out = InterfaceMetricsResult()
    if not pep or not tgt:
        out.notes.append("empty_peptide_or_target_atoms")
        return out

    P = np.array([a["coord"] for a in pep], dtype=float)
    T = np.array([a["coord"] for a in tgt], dtype=float)
    tree_t = cKDTree(T)

    # 原子对（肽 i — 靶 j）
    pairs: list[tuple[int, int, float]] = []
    for i, p in enumerate(P):
        idxs = tree_t.query_ball_point(p, r=interface_cutoff)
        for j in idxs:
            d = float(np.linalg.norm(p - T[j]))
            if d <= interface_cutoff:
                pairs.append((i, j, d))

    n_atomic = len(pairs)
    res_min_d: dict[tuple[tuple[str, int, str], tuple[str, int, str]], float] = {}
    res_atom_counts: dict[tuple[tuple[str, int, str], tuple[str, int, str]], int] = {}
    for i, j, d in pairs:
        kp, kt = _reskey(pep[i]), _reskey(tgt[j])
        key = (kp, kt)
        res_min_d[key] = min(res_min_d.get(key, 1e9), d)
        res_atom_counts[key] = res_atom_counts.get(key, 0) + 1

    n_res_contact = len(res_min_d)
    iface_pep_keys = {k[0] for k in res_min_d}
    iface_tgt_keys = {k[1] for k in res_min_d}
    n_iface_pep = len(iface_pep_keys)
    n_iface_tgt = len(iface_tgt_keys)

    # A. 基础
    out.metrics["residue_contact_count"] = n_res_contact
    out.metrics["atomic_contact_count"] = n_atomic
    out.metrics["interface_residue_count_peptide"] = n_iface_pep
    out.metrics["interface_residue_count_target"] = n_iface_tgt

    # 疏水原子对
    hyd_pairs = 0
    opp_q = 0
    same_q = 0
    polar_atom_pairs = 0
    hbonds = 0
    salt_keys: set[tuple[tuple[str, int, str], tuple[str, int, str]]] = set()

    for i, j, d in pairs:
        opi, otj = pep[i]["one"], tgt[j]["one"]
        if opi in _HYDROPHOBIC_ONE and otj in _HYDROPHOBIC_ONE:
            hyd_pairs += 1
        qi = _formal_residue_charge(opi)
        qj = _formal_residue_charge(otj)
        if qi != 0 and qj != 0:
            if qi * qj < 0:
                opp_q += 1
            elif qi * qj > 0:
                same_q += 1
        if opi in _POLAR_ONE or otj in _POLAR_ONE:
            polar_atom_pairs += 1
        if _is_hbond_pair(pep[i], tgt[j], d, hbond_lo, hbond_cutoff):
            hbonds += 1
        if _salt_bridge_pair(pep[i], tgt[j], d, salt_bridge_cutoff):
            salt_keys.add((_reskey(pep[i]), _reskey(tgt[j])))

    out.metrics["hydrophobic_contact_count"] = hyd_pairs

    def _one_for_key(atoms: list[dict], k: tuple[str, int, str]) -> str:
        for a in atoms:
            if _reskey(a) == k:
                return a["one"]
        return "X"

    # 疏水 patch：界面残基中疏水占比差异
    f_hyd_p = sum(1 for k in iface_pep_keys if _KD.get(_one_for_key(pep, k), 0) > 1.5) / max(n_iface_pep, 1)
    f_hyd_t = sum(1 for k in iface_tgt_keys if _KD.get(_one_for_key(tgt, k), 0) > 1.5) / max(n_iface_tgt, 1)
    out.metrics["hydrophobic_patch_overlap_score"] = float(max(0.0, 1.0 - abs(f_hyd_p - f_hyd_t)))

    mismatch_terms = [
        abs(_KD.get(_one_for_key(pep, kp), 0) - _KD.get(_one_for_key(tgt, kt), 0)) for kp, kt in res_min_d
    ]
    out.metrics["hydrophobic_mismatch_penalty"] = float(
        sum(mismatch_terms) / max(len(mismatch_terms), 1) / 8.0
    )

    out.metrics["opposite_charge_contact_count"] = opp_q
    out.metrics["same_charge_contact_count"] = same_q
    out.metrics["electrostatic_complementarity_score"] = float(
        opp_q / (opp_q + same_q + 1e-6)
    )
    out.metrics["salt_bridge_count"] = len(salt_keys)
    out.metrics["interface_hbond_count"] = hbonds
    out.metrics["polar_contact_count"] = polar_atom_pairs

    # B. buried / packing / gap proxy
    tree_t_b = cKDTree(T)
    buried_scores: list[float] = []
    for kp in iface_pep_keys:
        # 该残基 CA 或质心近似：用所有原子平均坐标
        coords = [a["coord"] for a in pep if _reskey(a) == kp]
        if not coords:
            continue
        c = np.mean(np.stack(coords, axis=0), axis=0)
        n_nei = len(tree_t_b.query_ball_point(c, r=buried_neighbor_cutoff))
        # 归一化：饱和约 40 个重原子球壳
        buried_scores.append(min(1.0, n_nei / 40.0))
    out.metrics["buried_sasa_proxy"] = float(np.mean(buried_scores)) if buried_scores else 0.0

    out.metrics["interface_packing_density_proxy"] = float(
        n_atomic / (n_iface_pep + n_iface_tgt + 1e-6)
    )
    mean_min_d = float(np.mean(list(res_min_d.values()))) if res_min_d else interface_cutoff
    out.metrics["interface_gap_proxy"] = float(
        max(0.0, (mean_min_d - 2.5) / max(interface_cutoff - 2.5, 1e-6))
    )

    # D. unsatisfied buried charge（肽界面带电残基）
    unsat = 0
    for kp in iface_pep_keys:
        one = _one_for_key(pep, kp)
        qc = _formal_residue_charge(one)
        if abs(qc) < 0.99:
            continue
        coords = [a["coord"] for a in pep if _reskey(a) == kp]
        if not coords:
            continue
        c = np.mean(np.stack(coords, axis=0), axis=0)
        n_nei = len(tree_t_b.query_ball_point(c, r=buried_neighbor_cutoff))
        if n_nei < 8:
            continue
        # 邻近是否有相反电荷靶原子
        close = tree_t_b.query_ball_point(c, r=charge_neighbor_cutoff)
        ok = False
        for j in close:
            qj = _formal_residue_charge(tgt[j]["one"])
            if qc * qj < 0:
                ok = True
                break
        if not ok:
            unsat += 1
    out.metrics["unsatisfied_buried_charge_proxy"] = int(unsat)

    # residue pair 明细
    def _pair_sort_key(item: tuple) -> tuple:
        (kp, kt), _mind = item
        return (kp[0], kp[1], kp[2], kt[0], kt[1], kt[2])

    for (kp, kt), mind in sorted(res_min_d.items(), key=_pair_sort_key):
        out.residue_pairs.append(
            {
                "peptide_chain": kp[0],
                "peptide_resseq": kp[1],
                "peptide_icode": kp[2],
                "peptide_resname": next((a["resname"] for a in pep if _reskey(a) == kp), ""),
                "peptide_one": _one_for_key(pep, kp),
                "target_chain": kt[0],
                "target_resseq": kt[1],
                "target_icode": kt[2],
                "target_resname": next((a["resname"] for a in tgt if _reskey(a) == kt), ""),
                "target_one": _one_for_key(tgt, kt),
                "min_distance_A": mind,
                "n_atom_atom_pairs": res_atom_counts.get((kp, kt), 0),
            }
        )

    # 原子对明细（截断）
    truncated = False
    for idx, (i, j, d) in enumerate(pairs):
        if idx >= max_atomic_pairs_export:
            truncated = True
            break
        ap, at = pep[i], tgt[j]
        out.atomic_pairs.append(
            {
                "peptide_chain": ap["chain"],
                "peptide_resseq": ap["resseq"],
                "peptide_icode": ap["icode"],
                "peptide_resname": ap["resname"],
                "peptide_atom": ap["atom_name"],
                "target_chain": at["chain"],
                "target_resseq": at["resseq"],
                "target_icode": at["icode"],
                "target_resname": at["resname"],
                "target_atom": at["atom_name"],
                "distance_A": d,
                "is_hydrophobic_pair": int(
                    ap["one"] in _HYDROPHOBIC_ONE and at["one"] in _HYDROPHOBIC_ONE
                ),
                "is_polar_contact": int(ap["one"] in _POLAR_ONE or at["one"] in _POLAR_ONE),
                "is_hbond": int(_is_hbond_pair(ap, at, d, hbond_lo, hbond_cutoff)),
                "is_opposite_charge": int(
                    _formal_residue_charge(ap["one"]) * _formal_residue_charge(at["one"]) < 0
                ),
                "is_salt_bridge": int(_salt_bridge_pair(ap, at, d, salt_bridge_cutoff)),
            }
        )
    if truncated:
        out.notes.append(f"atomic_pairs_truncated_at_{max_atomic_pairs_export}")

    return out


def write_interface_metric_definitions_md(path: Path) -> None:
    text = """# 界面指标定义（interface metrics）

本文档与 `Table_S8_interface_metrics.csv` 列一一对应，均为**几何 / 序列启发式 proxy**，非严格 PBSA / SASA。

## 通用设定

- **重原子**：氢原子一律忽略。
- **界面距离阈值** `interface_cutoff`（Å）：肽原子与靶原子欧氏距离 ≤ 该值记为**原子接触**。
- **残基接触**：若一对标准残基（肽–靶）间存在至少一对原子接触，则记为 1 对残基接触；`min_distance_A` 为该对残基间**最近原子距离**。
- **链识别**：Bio.PDB 多链时，**标准残基数最少**的链标为肽，**最多**的链标为靶；多于两条链时其余链忽略（仅记 note）。单链且含 `TER` 时按段拆分，**残基数较少段**为肽并临时重标链 `P`/`T`。
- **盐桥**：K/R 侧链正电原子（NZ、NH*、NE、ND1、NE2）与 D/E 羧基氧（OE*、OD*、OXT）距离 ≤ `salt_bridge_cutoff`（默认 4.0 Å），按**残基对**去重计数。
- **氢键 proxy**：重原子距离 ∈ [2.4, `hbond_cutoff`] Å，且元素为 N–O 或 O–N（主链/侧链均计入，**无角度筛选**）。

## A. 基础界面

| 列名 | 含义 |
|------|------|
| `residue_contact_count` | 界面**残基对**数目（肽残基–靶残基，无序对但方向固定为肽→靶）。 |
| `atomic_contact_count` | 界面**原子对**数目（距离 ≤ interface_cutoff）。 |
| `interface_residue_count_peptide` | 参与至少一对残基接触的**不同肽残基**个数。 |
| `interface_residue_count_target` | 参与至少一对残基接触的**不同靶残基**个数。 |

## B. Buried / packing / gap proxy

| 列名 | 含义 |
|------|------|
| `buried_sasa_proxy` | 对每个肽界面残基，取其重原子坐标均值为中心，统计半径 4 Å 内靶重原子数，除以 40 截断到 [0,1]，再对界面肽残基取平均。近似「界面埋藏」程度。 |
| `interface_packing_density_proxy` | `atomic_contact_count / (interface_residue_count_peptide + interface_residue_count_target)`。 |
| `interface_gap_proxy` | 所有残基对 `min_distance_A` 的均值，线性映射到 [0,1]：`(mean_min_d - 2.5) / (interface_cutoff - 2.5)`，小于 0 截断为 0。值越大表示平均距离越大（更「松」）。 |

## C. 疏水互补

| 列名 | 含义 |
|------|------|
| `hydrophobic_contact_count` | 原子接触对中，两残基 one-letter 均属于 {A,I,L,M,F,W,Y,V} 的对数。 |
| `hydrophobic_patch_overlap_score` | 肽界面残基中 KD>1.5 占比 `f_p` 与靶界面 `f_t`，`1 - abs(f_p - f_t)`，截断到 [0,1]。 |
| `hydrophobic_mismatch_penalty` | 每个残基对上两残基 Kyte–Doolittle 差绝对值，取平均后除以 8。 |

## D. 静电互补

| 列名 | 含义 |
|------|------|
| `opposite_charge_contact_count` | 原子接触对中，形式电荷（K/R=+1, D/E=-1, H=+0.5）乘积为负的对数。 |
| `same_charge_contact_count` | 形式电荷同号且均非零的原子接触对数。 |
| `electrostatic_complementarity_score` | `opposite / (opposite + same + 1e-6)`。 |
| `salt_bridge_count` | 见上文盐桥几何定义，**残基对**去重。 |
| `unsatisfied_buried_charge_proxy` | 肽界面残基：形式电荷 |q|≥1，且 4 Å 内靶重原子≥8（近似埋藏），但 4.5 Å 内无相反电荷靶原子的残基数。 |

## E. 极性相互作用

| 列名 | 含义 |
|------|------|
| `interface_hbond_count` | 满足氢键距离与 N/O 元素判据的原子接触对数（无角约束）。 |
| `polar_contact_count` | 原子接触对中至少一侧 one-letter 属于极性集合 `NQSTDEKRHYWCM` 的对数。 |

## 输出文件

- `intermediate/interface_contacts/{peptide_id}__{target_id}__r{rank}_residue_contacts.csv`：残基对汇总（每行一对肽–靶残基，`min_distance_A`、`n_atom_atom_pairs`）。
- `intermediate/interface_contacts/..._atomic_contacts.csv`：原子对明细；若行数超过脚本参数 `max_atomic_export` 会截断，并在 `Table_S3` 的 `atomic_export_truncated` 与 `processing_notes` 中注明。
- `intermediate/interface_hit_frequency_by_target/{target_id}_target_site_hit_frequency.csv`：每个靶一条表，列为靶残基标识、`hit_peptide_count`（至少一个界面接触的不同肽数）、`n_analyzed_complexes`、`hit_frequency`。

---
版本：稳定 proxy 实现；后续可替换为真实 SASA / 氢键角筛选 / APBS 等。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

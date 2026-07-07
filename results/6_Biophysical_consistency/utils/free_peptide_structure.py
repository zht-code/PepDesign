"""
游离肽 PDB 结构指标（启发式 + Bio.PDB）。

说明：部分指标（二级结构分数、埋藏疏水等）为**可复现的启发式代理**，
非 DSSP / Rosetta 等严格物理模型；函数名与 docstring 已标明 heuristic / proxy。
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, PPBuilder
from Bio.Data.IUPACData import protein_letters_3to1_extended
from Bio.PDB.Polypeptide import is_aa

# 常见重原子范德华半径（Å）— 用于 clash 启发式
_VDW: dict[str, float] = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
    "P": 1.8,
    "SE": 1.9,
    "H": 1.2,
}

_HYDROPHOBIC_ONE = set("AILMFWYV")
_KD_HYDROPHOBIC = set("AILMFWY")  # 用于疏水簇（略保守）


def _element(atom) -> str:
    e = (atom.element or "").strip().upper()
    if len(e) >= 1:
        return e[0]
    n = atom.get_name().strip()
    return n[0] if n else "C"


def _vdw_radius(atom) -> float:
    return _VDW.get(_element(atom), 1.7)


def _sanitize_filename(s: str, max_len: int = 100) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s[:max_len] or "row"


def load_structure(path: Path):
    """使用 Bio.PDB 读取 PDB 或 mmCIF（首条模型）。"""
    path = path.expanduser().resolve()
    suf = path.suffix.lower()
    if suf in (".pdb", ".ent"):
        p = PDBParser(QUIET=True)
        return p.get_structure("s", str(path)), None
    if suf in (".cif", ".mcif"):
        p = MMCIFParser(QUIET=True)
        return p.get_structure("s", str(path)), None
    return None, f"unsupported_extension:{suf}"


def pick_longest_protein_chain(model) -> Any | None:
    """选择含标准氨基酸残基数最多的链。"""
    best = None
    best_n = -1
    for chain in model:
        n = sum(1 for r in chain if is_aa(r, standard=True))
        if n > best_n:
            best_n = n
            best = chain
    return best


def residues_in_order(chain) -> list:
    """按残基序号排序的标准氨基酸残基列表。"""
    rs = [r for r in chain if is_aa(r, standard=True)]
    rs.sort(key=lambda r: r.get_id()[1])
    return rs


def one_letter(res) -> str | None:
    name = res.get_resname().strip().capitalize()
    return protein_letters_3to1_extended.get(name, "X")


def heuristic_ss_fractions_from_phi_psi(
    phi_psi: list[tuple[float | None, float | None]],
) -> tuple[float, float, float, int]:
    """
    **Heuristic** 二级结构比例：基于主链 φ/ψ 的宽松 Ramachandran 分区（非 DSSP）。
    返回 (helix_frac, sheet_frac, coil_frac, n_classified)
    """
    helix = sheet = coil = 0
    n = 0
    for phi, psi in phi_psi:
        if phi is None or psi is None or math.isnan(phi) or math.isnan(psi):
            continue
        n += 1
        # 宽松 α
        if -140 <= phi <= -30 and -80 <= psi <= 50:
            helix += 1
        # 宽松 β / extended
        elif (-180 <= phi <= -40 and 60 <= psi <= 180) or (-180 <= phi <= -100 and 40 <= psi <= 180):
            sheet += 1
        else:
            coil += 1
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    return helix / n, sheet / n, coil / n, n


def get_phi_psi_list(model, chain) -> list[tuple[float | None, float | None]]:
    """Bio.PDB PPBuilder：仅合并完全位于指定链上的肽段 φ/ψ。"""
    ppb = PPBuilder()
    out: list[tuple[float | None, float | None]] = []
    cid = chain.get_id()
    for pp in ppb.build_peptides(model):
        parents = {r.get_parent().get_id() for r in pp}
        if parents != {cid}:
            continue
        out.extend(pp.get_phi_psi_list())
    return out


def count_heavy_atoms(chain) -> int:
    n = 0
    for atom in chain.get_atoms():
        if _element(atom) == "H":
            continue
        n += 1
    return n


def heuristic_clash_counts(
    chain,
    scale_mild: float = 0.82,
    scale_severe: float = 0.72,
) -> tuple[int, int]:
    """
    **Heuristic clash**：非键合重原子对，若 dist < scale * (r_i+r_j) 记一次冲突。
    跳过同一残基内原子、序列相邻残基（|Δseq|<=1）及常见共价键对（N–CA、CA–C、C–N+1）。
    """
    atoms = [a for a in chain.get_atoms() if _element(a) != "H"]
    clash = severe = 0
    for i, a in enumerate(atoms):
        ra = _vdw_radius(a)
        res_a = a.get_parent()
        id_a = res_a.get_id()[1]
        for j in range(i + 1, len(atoms)):
            b = atoms[j]
            rb = _vdw_radius(b)
            res_b = b.get_parent()
            id_b = res_b.get_id()[1]
            if res_a is res_b:
                continue
            if abs(id_a - id_b) <= 1:
                # 相邻残基仍可能侧链冲突，但骨架键连对跳过
                na, nb = a.get_name().strip(), b.get_name().strip()
                if abs(id_a - id_b) == 0:
                    continue
                if abs(id_a - id_b) == 1:
                    backbone = {"N", "CA", "C"}
                    if na in backbone and nb in backbone:
                        continue
            d = float(a - b)
            cutoff_m = scale_mild * (ra + rb)
            cutoff_s = scale_severe * (ra + rb)
            if d < cutoff_s:
                severe += 1
                clash += 1
            elif d < cutoff_m:
                clash += 1
    return clash, severe


def backbone_n_ca_c_angle_deg(res) -> float | None:
    """单残基 N–CA–C 键角（度）；缺失返回 None。"""
    try:
        n = res["N"].get_coord()
        ca = res["CA"].get_coord()
        c = res["C"].get_coord()
    except Exception:
        return None
    v1 = n - ca
    v2 = c - ca
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def approximate_backbone_strain_score_heuristic(residues: list) -> float:
    """
    **Heuristic backbone strain**：|N–CA–C| 与理想 110° 的均方偏差（度²），再除以 100 缩放为 ~O(1)。
    越大表示局部几何越异常。
    """
    angles: list[float] = []
    for r in residues:
        ang = backbone_n_ca_c_angle_deg(r)
        if ang is not None:
            angles.append(ang)
    if not angles:
        return 0.0
    dev = np.array(angles, dtype=float) - 110.0
    return float(np.mean(dev**2) / 100.0)


def torsion_proxy_score_allowed_ramachandran(phi_psi: list[tuple[float | None, float | None]]) -> float:
    """
    **Proxy**：φ/ψ 落在「核心允许区」的比例（宽松并集，非严格 Lovell 等高线）。
    """
    ok = 0
    n = 0
    for phi, psi in phi_psi:
        if phi is None or psi is None or math.isnan(phi) or math.isnan(psi):
            continue
        n += 1
        # 核心：α + β + 左螺旋附近 general allowed 大块
        core = (
            (-135 <= phi <= -45 and -60 <= psi <= 30)
            or (-160 <= phi <= -40 and 90 <= psi <= 180)
            or (-100 <= phi <= 30 and -60 <= psi <= 120)
        )
        if core:
            ok += 1
    return ok / n if n else 0.0


def count_hbonds_heuristic(chain) -> tuple[int, int, int]:
    """
    **Heuristic H-bonds**：N 供体、O 受体，距离 2.4–3.5 Å；同一残基跳过；每对无序只计一次。
    - backbone：两端原子名均属于 {N, CA?, 不 — 仅 N 与 O 名 N,O,OXT}
    """
    atoms = [a for a in chain.get_atoms() if _element(a) != "H"]
    bb_names = {"N", "O", "OXT"}

    def is_bb_atom(atom) -> bool:
        return atom.get_name().strip() in bb_names

    def is_donor(atom) -> bool:
        el = _element(atom)
        n = atom.get_name().upper()
        return el == "N" or n.startswith("NH")

    def is_acceptor(atom) -> bool:
        el = _element(atom)
        n = atom.get_name().upper()
        return el == "O" or n.startswith("OD") or n.startswith("OE") or n.startswith("OG") or n.startswith("OH")

    pairs: set[tuple[int, int]] = set()
    bb = sc = 0
    donors = [a for a in atoms if is_donor(a)]
    acceptors = [a for a in atoms if is_acceptor(a)]
    for a in donors:
        ra = a.get_parent()
        for b in acceptors:
            rb = b.get_parent()
            if ra is rb:
                continue
            d = float(a - b)
            if not (2.4 <= d <= 3.5):
                continue
            ia, ib = id(a), id(b)
            key = (ia, ib) if ia < ib else (ib, ia)
            if key in pairs:
                continue
            pairs.add(key)
            a_bb, b_bb = is_bb_atom(a), is_bb_atom(b)
            if a_bb and b_bb:
                bb += 1
            else:
                sc += 1
    intra = len(pairs)
    return intra, bb, sc


def ca_coords(residues: list) -> np.ndarray | None:
    xs = []
    for r in residues:
        if "CA" in r:
            xs.append(r["CA"].get_coord())
    if not xs:
        return None
    return np.array(xs, dtype=float)


def hydrophobic_cluster_count_heuristic(coords: np.ndarray, seq: str, cutoff: float = 6.0) -> int:
    """
    **Heuristic**：疏水残基 CA 在 cutoff 内连边，统计连通分量数。
    """
    hyd = [i for i, aa in enumerate(seq) if aa in _KD_HYDROPHOBIC and i < len(coords)]
    if not hyd:
        return 0
    parent = {i: i for i in hyd}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a in range(len(hyd)):
        for b in range(a + 1, len(hyd)):
            i, j = hyd[a], hyd[b]
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                union(i, j)
    roots = {find(i) for i in hyd}
    return len(roots)


def longest_hydrophobic_run_in_sequence(seq: str) -> int:
    best = cur = 0
    for ch in seq:
        if ch in _HYDROPHOBIC_ONE:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def buried_hydrophobic_proxy_heuristic(coords: np.ndarray, seq: str, radius: float = 8.0, min_neighbors: int = 6) -> tuple[int, float]:
    """
    **Proxy buried**：疏水残基若 CA 邻域（其它 CA 距离<=radius）数量 >= min_neighbors，则计为 buried-like。
    返回 (buried_count, buried_fraction_of_hydrophobic)
    """
    if coords is None or len(seq) != len(coords):
        return 0, 0.0
    hyd_idx = [i for i, aa in enumerate(seq) if aa in _KD_HYDROPHOBIC]
    if not hyd_idx:
        return 0, 0.0
    buried = 0
    n = len(coords)
    for i in hyd_idx:
        cnt = 0
        for j in range(n):
            if i == j:
                continue
            if np.linalg.norm(coords[i] - coords[j]) <= radius:
                cnt += 1
        if cnt >= min_neighbors:
            buried += 1
    return buried, buried / max(len(hyd_idx), 1)


def hydrophobic_cohesion_score_heuristic(
    longest_run: int,
    n_res: int,
    buried_frac: float,
) -> float:
    """0–1 启发式：长疏水串 + 埋藏比例。"""
    if n_res <= 0:
        return 0.0
    run_part = min(1.0, longest_run / max(8.0, n_res * 0.25))
    return float(0.55 * buried_frac + 0.45 * run_part)


def cysteine_metrics_heuristic(residues: list) -> tuple[int, int, bool]:
    """
    Cys 相关：**SG–SG** 距离 < 4.0 Å 记候选对；若存在一对在 [2.0, 2.8] Å 则 feasible_flag=True（几何上可成二硫键的宽松判据）。
    """
    sg_atoms = []
    for r in residues:
        if one_letter(r) != "C":
            continue
        for name in ("SG", "SG1", "SG2"):
            if name in r:
                sg_atoms.append(r[name])
                break
    n_cys = len(sg_atoms)
    cand = 0
    feasible = False
    for i in range(n_cys):
        for j in range(i + 1, n_cys):
            d = float(sg_atoms[i] - sg_atoms[j])
            if d < 4.0:
                cand += 1
            if 2.0 <= d <= 2.8:
                feasible = True
    return n_cys, cand, feasible


def sequence_from_residues(residues: list) -> str:
    s = []
    for r in residues:
        o = one_letter(r)
        if o:
            s.append(o)
    return "".join(s)


@dataclass
class FreePeptideMetrics:
    target_id: str
    peptide_id: str
    free_structure_path: str
    s1_row_index: int
    residue_count: int
    atom_count: int
    helix_frac: float
    sheet_frac: float
    coil_frac: float
    n_classified_dihedrals: int
    clash_count: int
    severe_clash_count: int
    approximate_backbone_strain_score: float
    torsion_proxy_score: float
    intrapeptide_hbond_count: int
    backbone_hbond_count: int
    sidechain_hbond_count: int
    hydrophobic_residue_count: int
    hydrophobic_cluster_count: int
    longest_hydrophobic_run: int
    buried_hydrophobic_proxy: int
    buried_hydrophobic_fraction: float
    hydrophobic_cohesion_score: float
    cysteine_count: int
    disulfide_candidate_count: int
    disulfide_feasible_flag: bool
    analysis_status: str
    error_message: str
    pdb_sequence_inferred: str

    def as_flat_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def analyze_free_peptide_pdb(
    path: Path,
    target_id: str,
    peptide_id: str,
    s1_row_index: int,
    table_sequence: str,
) -> FreePeptideMetrics:
    err = ""
    status = "ok"
    path_s = str(path)
    empty = FreePeptideMetrics(
        target_id=target_id,
        peptide_id=peptide_id,
        free_structure_path=path_s,
        s1_row_index=s1_row_index,
        residue_count=0,
        atom_count=0,
        helix_frac=0.0,
        sheet_frac=0.0,
        coil_frac=0.0,
        n_classified_dihedrals=0,
        clash_count=0,
        severe_clash_count=0,
        approximate_backbone_strain_score=0.0,
        torsion_proxy_score=0.0,
        intrapeptide_hbond_count=0,
        backbone_hbond_count=0,
        sidechain_hbond_count=0,
        hydrophobic_residue_count=0,
        hydrophobic_cluster_count=0,
        longest_hydrophobic_run=0,
        buried_hydrophobic_proxy=0,
        buried_hydrophobic_fraction=0.0,
        hydrophobic_cohesion_score=0.0,
        cysteine_count=0,
        disulfide_candidate_count=0,
        disulfide_feasible_flag=False,
        analysis_status="error",
        error_message="",
        pdb_sequence_inferred="",
    )

    if not path.exists():
        empty.error_message = "file_not_found"
        return empty

    struct, parse_err = load_structure(path)
    if struct is None:
        empty.error_message = parse_err or "parse_failed"
        return empty

    model = next(struct.get_models())
    chain = pick_longest_protein_chain(model)
    if chain is None:
        empty.error_message = "no_protein_chain"
        return empty

    residues = residues_in_order(chain)
    if not residues:
        empty.error_message = "no_standard_residues"
        return empty

    seq = sequence_from_residues(residues)
    phi_psi = get_phi_psi_list(model, chain)
    # 对齐长度：以残基数为主，phi_psi 可能略短 — 截断到 min
    h, sh, co, ncls = heuristic_ss_fractions_from_phi_psi(phi_psi)

    clash, sev = heuristic_clash_counts(chain)
    strain = approximate_backbone_strain_score_heuristic(residues)
    torsion_proxy = torsion_proxy_score_allowed_ramachandran(phi_psi)

    hb_intra, hb_bb, hb_sc = count_hbonds_heuristic(chain)

    hyd_count = sum(1 for aa in seq if aa in _HYDROPHOBIC_ONE)
    coords = ca_coords(residues)
    if coords is not None and len(seq) == len(coords):
        clusters = hydrophobic_cluster_count_heuristic(coords, seq)
        buried_n, buried_f = buried_hydrophobic_proxy_heuristic(coords, seq)
        longest_run = longest_hydrophobic_run_in_sequence(seq)
        cohesion = hydrophobic_cohesion_score_heuristic(longest_run, len(seq), buried_f)
    else:
        clusters = 0
        buried_n, buried_f = 0, 0.0
        longest_run = longest_hydrophobic_run_in_sequence(seq)
        cohesion = hydrophobic_cohesion_score_heuristic(longest_run, len(seq), 0.0)

    cys_n, cys_cand, cys_ok = cysteine_metrics_heuristic(residues)

    return FreePeptideMetrics(
        target_id=target_id,
        peptide_id=peptide_id,
        free_structure_path=path_s,
        s1_row_index=s1_row_index,
        residue_count=len(residues),
        atom_count=count_heavy_atoms(chain),
        helix_frac=float(h),
        sheet_frac=float(sh),
        coil_frac=float(co),
        n_classified_dihedrals=int(ncls),
        clash_count=int(clash),
        severe_clash_count=int(sev),
        approximate_backbone_strain_score=float(strain),
        torsion_proxy_score=float(torsion_proxy),
        intrapeptide_hbond_count=int(hb_intra),
        backbone_hbond_count=int(hb_bb),
        sidechain_hbond_count=int(hb_sc),
        hydrophobic_residue_count=int(hyd_count),
        hydrophobic_cluster_count=int(clusters),
        longest_hydrophobic_run=int(longest_run),
        buried_hydrophobic_proxy=int(buried_n),
        buried_hydrophobic_fraction=float(buried_f),
        hydrophobic_cohesion_score=float(cohesion),
        cysteine_count=int(cys_n),
        disulfide_candidate_count=int(cys_cand),
        disulfide_feasible_flag=bool(cys_ok),
        analysis_status="ok",
        error_message="",
        pdb_sequence_inferred=seq,
    )


def write_metric_definitions(path: Path) -> None:
    text = """# Free peptide 结构指标定义（`free_peptide_metric_definitions.md`）

本文档说明 `02_analyze_free_peptides.py` 输出的各字段含义。**凡标注 heuristic / proxy 的指标均为可复现近似，不等同于实验或全原子物理严格值。**

## 输入与对象

- 输入 PDB：`Table_S1_master_sequence_table.csv` 中 `usable_for_free_structure_analysis=True` 且路径存在的 `free_structure_path`。
- 解析：Bio.PDB `PDBParser` / `MMCIFParser`；默认取**首条模型**中含标准氨基酸最多的链。

## A. 基础结构

| 字段 | 定义 |
|------|------|
| `residue_count` | 链上 `is_aa(standard=True)` 残基数。 |
| `atom_count` | 该链重原子数（排除氢）。 |
| `helix_frac` | **Heuristic**：φ/ψ 落入宽松 α 区残基比例（见代码 `heuristic_ss_fractions_from_phi_psi`）。 |
| `sheet_frac` | **Heuristic**：φ/ψ 落入宽松 β/extended 区比例。 |
| `coil_frac` | **Heuristic**：其余有 φ/ψ 的残基比例；`helix+sheet+coil≈1`。 |
| `n_classified_dihedrals` | 参与二级结构分类的 φ/ψ 对数量。 |

## B. 几何与冲突

| 字段 | 定义 |
|------|------|
| `clash_count` | **Heuristic**：重原子对距离 < 0.82×(r_vdw_i+r_vdw_j)（非键合、排除相邻主链肽段）。 |
| `severe_clash_count` | 同上，阈值系数 0.72。 |
| `approximate_backbone_strain_score` | **Heuristic**：各残基 N–CA–C 键角相对 110° 的均方偏差 /100。 |
| `torsion_proxy_score` | **Proxy**：φ/ψ 落入宽松「核心允许区」并集的比例。 |

## C. 分子内氢键（几何计数）

| 字段 | 定义 |
|------|------|
| `intrapeptide_hbond_count` | **Heuristic**：供体 N 与受体 O（重原子）距离 2.4–3.5 Å 的对数（去重）。 |
| `backbone_hbond_count` | 其中 N/O 均为主链原子（N, O, OXT）。 |
| `sidechain_hbond_count` | `intrapeptide_hbond_count - backbone_hbond_count`（近似侧链参与）。 |

## D. 疏水内聚（序列 + CA 几何）

| 字段 | 定义 |
|------|------|
| `hydrophobic_residue_count` | 序列中 `AILMFWYV` 计数。 |
| `hydrophobic_cluster_count` | **Heuristic**：疏水残基（`AILMFWY`）CA 6 Å 内建图后的连通分量数。 |
| `longest_hydrophobic_run` | 序列上连续疏水（`AILMFWYV`）最大长度。 |
| `buried_hydrophobic_proxy` | **Proxy**：疏水残基中，CA 8 Å 邻域内其它 CA 数 ≥6 的个数。 |
| `buried_hydrophobic_fraction` | `buried_hydrophobic_proxy / max(疏水残基数,1)`。 |
| `hydrophobic_cohesion_score` | **Heuristic**：0.55×`buried_hydrophobic_fraction` + 0.45×归一化最长疏水串。 |

## E. Cys / 二硫键

| 字段 | 定义 |
|------|------|
| `cysteine_count` | Cys 残基数（基于 one-letter）。 |
| `disulfide_candidate_count` | 不同 Cys 的 SG（或别名）对，距离 < 4.0 Å 的对数。 |
| `disulfide_feasible_flag` | 若存在 SG–SG 距离 ∈ [2.0, 2.8] Å 则为 True（宽松几何可行）。 |

## 状态列

| 字段 | 定义 |
|------|------|
| `analysis_status` | `ok` / `error` / `skipped`。 |
| `error_message` | 失败或跳过时简短原因。 |
| `pdb_sequence_inferred` | 从 PDB 推断的 one-letter 序列（与主表 `sequence` 可对照）。 |

"""
    path.write_text(text, encoding="utf-8")

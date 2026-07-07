#!/usr/bin/env python3
"""
07 — 自动筛选 case studies（2–3 例）并生成说明图、PyMOL 素材与 selected_cases.json。

筛选依据 Table_S11 + S1 + S8；可选 reference_sequences.csv 计算 motif recovery 代理。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logging_utils import setup_run_logger
from utils.nature_style import (
    ACCENT_LINE,
    ACCENT_MARK,
    MUTED_LINE,
    NATURE_SEQUENTIAL,
    NATURE_SEQUENTIAL_WARM,
    NPG,
    apply_nature_style,
)
from utils.paths import ProjectPaths, load_config


def _slug(s: str, max_len: int = 120) -> str:
    t = re.sub(r"[^\w.\-]+", "_", str(s).strip())
    return t[:max_len] if len(t) > max_len else t


def _contact_basename(peptide_id: str, target_id: str, rank: Any) -> str:
    return _slug(f"{peptide_id}__{target_id}__r{rank}")


def load_reference_sequences(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    if "target_id" not in df.columns or "sequence" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        tid = str(r["target_id"]).strip().lower()
        seq = str(r["sequence"]).strip()
        if tid and seq and seq != "nan":
            out[tid] = seq.upper()
    return out


def motif_recovery_ratio(gen: str, ref: str | None) -> float:
    if not ref or not gen:
        return 0.0
    g, r = gen.upper(), ref.upper()
    return float(SequenceMatcher(None, g, r).ratio())


def read_ca_coords_ter_segments(pdb_path: Path) -> list[np.ndarray]:
    """按 TER 分段收集 CA 坐标（与界面脚本一致，处理单链 Hdock 复合物）。"""
    segs: list[list[np.ndarray]] = []
    cur: list[np.ndarray] = []
    pdb_path = pdb_path.expanduser().resolve()
    if not pdb_path.is_file():
        return []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("TER"):
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            if not line.startswith("ATOM"):
                continue
            if len(line) < 54:
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            cur.append(np.array([x, y, z], dtype=float))
    if cur:
        segs.append(cur)
    return [np.stack(s, 0) for s in segs if len(s) >= 2]


def ca_coords_by_chain(pdb_path: Path, chain_pep: str, chain_tgt: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """返回 (pep_CA Nx3, tgt_CA Mx3, notes)。"""
    pdb_path = pdb_path.expanduser().resolve()
    if not pdb_path.exists():
        return np.zeros((0, 3)), np.zeros((0, 3)), ["pdb_missing"]
    try:
        model = next(PDBParser(QUIET=True).get_structure("x", str(pdb_path)).get_models())
    except Exception as e:
        return np.zeros((0, 3)), np.zeros((0, 3)), [f"parse_error:{e}"]

    def collect(cid: str) -> list[np.ndarray]:
        pts: list[np.ndarray] = []
        ch = None
        for c in model:
            if str(c.get_id()) == str(cid):
                ch = c
                break
        if ch is None:
            return pts
        for res in ch:
            if res.get_id()[0] != " ":
                continue
            if "CA" in res:
                pts.append(np.array(res["CA"].get_coord(), dtype=float))
        return pts

    def two_longest_chains() -> tuple[np.ndarray, np.ndarray, str]:
        scored: list[tuple[int, np.ndarray]] = []
        for c in model:
            pts: list[np.ndarray] = []
            for res in c:
                if res.get_id()[0] != " ":
                    continue
                if "CA" in res:
                    pts.append(np.array(res["CA"].get_coord(), dtype=float))
            if len(pts) >= 2:
                scored.append((len(pts), np.stack(pts, 0)))
        scored.sort(key=lambda x: -x[0])
        if len(scored) < 2:
            return np.zeros((0, 3)), np.zeros((0, 3)), "fallback_insufficient_chains"
        a, b = scored[0][1], scored[1][1]
        if len(a) > len(b):
            a, b = b, a
        return a, b, "fallback_two_longest_chains_shorter_is_peptide"

    p = collect(chain_pep)
    t = collect(chain_tgt)
    if len(p) and len(t):
        return np.stack(p, 0), np.stack(t, 0), ["ok"]
    a, b, note = two_longest_chains()
    if len(a) and len(b):
        return a, b, [f"chain_label_fallback:{note}"]
    ter_segs = read_ca_coords_ter_segments(pdb_path)
    ter_segs.sort(key=lambda x: -len(x))
    if len(ter_segs) >= 2:
        long, short = ter_segs[0], ter_segs[1]
        t, p = long, short
        if len(p) > len(t):
            p, t = t, p
        return p, t, ["ter_segment_split_shorter_is_peptide"]
    if len(ter_segs) == 1 and len(ter_segs[0]) >= 4:
        n = len(ter_segs[0]) // 2
        return ter_segs[0][:n], ter_segs[0][n:], ["ter_single_segment_split_half"]
    return np.zeros((0, 3)), np.zeros((0, 3)), ["no_ca_atoms"]


def pca2d(X: np.ndarray) -> np.ndarray:
    if len(X) < 2:
        return np.zeros((len(X), 2))
    Xc = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def plot_overall_pca(pep: np.ndarray, tgt: np.ndarray, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if len(pep) and len(tgt):
        xy = pca2d(np.vstack([pep, tgt]))
        n = len(pep)
        ax.scatter(xy[:n, 0], xy[:n, 1], s=22, alpha=0.75, label="Peptide CA", c=NPG[2])
        ax.scatter(xy[n:, 0], xy[n:, 1], s=10, alpha=0.45, label="Target CA", c=NPG[0])
    elif len(pep):
        xy = pca2d(pep)
        ax.scatter(xy[:, 0], xy[:, 1], s=22, alpha=0.75, label="Peptide CA", c=NPG[2])
    elif len(tgt):
        xy = pca2d(tgt)
        ax.scatter(xy[:, 0], xy[:, 1], s=10, alpha=0.45, label="Target CA", c=NPG[0])
    ax.set_title("Overall complex — CA PCA projection (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="best")
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_contact_map(rc: pd.DataFrame, out_base: Path) -> None:
    if rc.empty:
        return
    pr = pd.to_numeric(rc["peptide_resseq"], errors="coerce")
    tr = pd.to_numeric(rc["target_resseq"], errors="coerce")
    val = pd.to_numeric(rc["n_atom_atom_pairs"], errors="coerce")
    mat = rc.assign(_pr=pr, _tr=tr, _v=val).groupby(["_pr", "_tr"], as_index=False)["_v"].max()
    pivot = mat.pivot(index="_pr", columns="_tr", values="_v").fillna(0)
    if pivot.shape[0] > 55 or pivot.shape[1] > 55:
        pivot = pivot.iloc[:: max(1, pivot.shape[0] // 55), :: max(1, pivot.shape[1] // 55)]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap=NATURE_SEQUENTIAL_WARM, origin="lower")
    ax.set_xlabel("Target residue index (contact partner)")
    ax.set_ylabel("Peptide residue index")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns], rotation=90, fontsize=6)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(x)) for x in pivot.index], fontsize=6)
    ax.set_title("Contact map (max atom–atom pairs per residue pair)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pairs")
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def kd(aa: str) -> float:
    tab = {
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
    }
    return tab.get(str(aa).upper(), 0.0)


def plot_hydrophobic_complementarity(rc: pd.DataFrame, out_base: Path) -> None:
    if rc.empty:
        return
    po = rc["peptide_one"].astype(str).str.upper()
    to = rc["target_one"].astype(str).str.upper()
    x = [kd(a) for a in po]
    y = [kd(a) for a in to]
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    w = pd.to_numeric(rc["n_atom_atom_pairs"], errors="coerce").fillna(1)
    ax.scatter(x, y, s=np.clip(w * 2, 8, 120), alpha=0.45, c=np.log1p(w), cmap=NATURE_SEQUENTIAL)
    ax.set_xlabel("Peptide residue KD")
    ax.set_ylabel("Target residue KD")
    ax.set_title("Hydrophobic complementarity (interface residue pairs)")
    ax.axhline(0, color=MUTED_LINE, lw=0.5, alpha=0.55)
    ax.axvline(0, color=MUTED_LINE, lw=0.5, alpha=0.55)
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_electrostatic_proxy(s8_row: pd.Series, out_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    names = ["Opposite\ncharge contacts", "Same\ncharge contacts", "Salt\nbridges"]
    vals = [
        float(s8_row.get("opposite_charge_contact_count", 0) or 0),
        float(s8_row.get("same_charge_contact_count", 0) or 0),
        float(s8_row.get("salt_bridge_count", 0) or 0),
    ]
    axes[0].bar(names, vals, color=[NPG[6], NPG[7], NPG[3]], edgecolor=MUTED_LINE, linewidth=0.6)
    axes[0].set_ylabel("Count (atom / bridge proxy)")
    axes[0].set_title("Electrostatic interaction counts")
    ec = float(s8_row.get("electrostatic_complementarity_score", 0) or 0)
    unsat = float(s8_row.get("unsatisfied_buried_charge_proxy", 0) or 0)
    axes[1].bar(
        ["Elec. complement.\n(proxy)", "Unsatisfied buried\ncharge proxy"],
        [ec, unsat],
        color=[NPG[1], NPG[4]],
        edgecolor=MUTED_LINE,
        linewidth=0.6,
    )
    axes[1].set_ylim(0, max(1.0, ec, unsat) * 1.15)
    axes[1].set_title("Electrostatic complementarity metrics")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_solubility_profile(prof: pd.DataFrame, out_base: Path) -> None:
    if prof.empty or "residue_index" not in prof.columns:
        return
    fig, ax1 = plt.subplots(figsize=(8.5, 3.8))
    x = prof["residue_index"].values
    ax1.plot(x, prof["hotspot_score"].values, color=ACCENT_MARK, lw=1.2, label="Hotspot score")
    ax1.set_xlabel("Residue index")
    ax1.set_ylabel("Hotspot score", color=ACCENT_MARK)
    ax1.tick_params(axis="y", labelcolor=ACCENT_MARK)
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        prof["camsol_like_local_score"].values,
        color=ACCENT_LINE,
        lw=1.0,
        alpha=0.88,
        label="CamSol-like local",
    )
    ax2.set_ylabel("CamSol-like local", color=ACCENT_LINE)
    ax2.tick_params(axis="y", labelcolor=ACCENT_LINE)
    ax1.set_title("Residue-level solubility / hotspot profile")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_pymol_assets(
    case_dir: Path,
    pdb_path: Path,
    chain_pep: str,
    chain_tgt: str,
    rc: pd.DataFrame,
) -> None:
    pep_res = sorted(set(zip(rc["peptide_chain"], rc["peptide_resseq"], rc["peptide_icode"])))
    tgt_res = sorted(set(zip(rc["target_chain"], rc["target_resseq"], rc["target_icode"])))

    def pymol_sel(name: str, pairs: list[tuple]) -> str:
        parts = []
        for t in pairs:
            ch, rs, ic = str(t[0]), int(t[1]), str(t[2])
            ic = ic if ic.strip() else " "
            if ic.strip() == "" or ic == " ":
                parts.append(f"(chain {ch!r} and resi {rs})")
            else:
                parts.append(f"(chain {ch!r} and resi {rs} and icode {ic!r})")
        if not parts:
            return f"# select {name} — empty"
        return f"select {name}, " + " or ".join(parts)

    lines = [
        "# PyMOL session snippet — load complex then run:",
        f"# load {pdb_path}, complex",
        "",
        pymol_sel("iface_peptide", pep_res),
        pymol_sel("iface_target", tgt_res),
        "select iface_union, iface_peptide or iface_target",
        "color tv_blue, iface_peptide",
        "color tv_orange, iface_target",
        "show sticks, iface_union",
        "hide lines, all",
        "zoom iface_union",
        "",
    ]
    (case_dir / "pymol_interface_selections.pml").write_text("\n".join(lines), encoding="utf-8")

    # 去重写 peptide 界面残基
    seen = set()
    pr_lines = ["chain\tresseq\ticode\tresname\tone_letter"]
    for _, r in rc.sort_values(["peptide_resseq", "peptide_icode"]).iterrows():
        key = (r["peptide_chain"], r["peptide_resseq"], r["peptide_icode"])
        if key in seen:
            continue
        seen.add(key)
        pr_lines.append(
            f"{r['peptide_chain']}\t{r['peptide_resseq']}\t{r['peptide_icode']}\t{r['peptide_resname']}\t{r['peptide_one']}"
        )
    (case_dir / "contact_residues_peptide.tsv").write_text("\n".join(pr_lines), encoding="utf-8")

    seen_t = set()
    tr_lines = ["chain\tresseq\ticode\tresname\tone_letter"]
    for _, r in rc.sort_values(["target_resseq", "target_icode"]).iterrows():
        key = (r["target_chain"], r["target_resseq"], r["target_icode"])
        if key in seen_t:
            continue
        seen_t.add(key)
        tr_lines.append(
            f"{r['target_chain']}\t{r['target_resseq']}\t{r['target_icode']}\t{r['target_resname']}\t{r['target_one']}"
        )
    (case_dir / "contact_residues_target.tsv").write_text("\n".join(tr_lines), encoding="utf-8")

    chimerax = [
        "# ChimeraX — open structure then in command line:",
        f"open {pdb_path}",
        f"select peptide :{chain_pep}",
        f"select target :{chain_tgt}",
        "# 手动按 TSV 高亮界面残基，或使用 log 中 residue 列表",
    ]
    (case_dir / "chimerax_commands.cxc").write_text("\n".join(chimerax), encoding="utf-8")

    instr = f"""# 3D 渲染说明（PyMOL / ChimeraX）

本环境未嵌入光线追踪 3D 引擎；**高质量发表图**请在 PyMOL 或 ChimeraX 中完成。

## 文件

| 文件 | 用途 |
|------|------|
| `pymol_interface_selections.pml` | 界面肽/靶 `select` 与着色示例 |
| `contact_residues_peptide.tsv` / `contact_residues_target.tsv` | 界面残基清单 |
| `chimerax_commands.cxc` | ChimeraX 打开与链选择起点 |
| `figure_overall_complex_pca.png` | CA 主链 PCA 二维投影（快速总览） |

## PyMOL 建议流程

1. `pymol {pdb_path}`
2. `@pymol_interface_selections.pml`（或粘贴其中 `select` 命令）
3. `set ray_trace_mode, 1` → `png figure_ray.png, width=2400`

## 结构文件

- 复合物 PDB：`{pdb_path}`

链 ID（来自 Table_S8）：肽 **{chain_pep}**，靶 **{chain_tgt}**。
"""
    (case_dir / "pymol_render_instructions.md").write_text(instr, encoding="utf-8")


def write_case_summary(
    case_dir: Path,
    meta: dict[str, Any],
) -> None:
    m = "\n".join(f"- **{k}**: {v}" for k, v in meta.items() if not k.startswith("_"))
    body = f"""# Case study: {meta.get("peptide_id", "")}

## 筛选与综合分

{m}

## 生成图件

- `figure_overall_complex_pca.png` — 复合物 CA 的二维 PCA 投影
- `figure_electrostatic_complementarity.png` — 静电相关计数与 complementarity 代理
- `figure_hydrophobic_complementarity.png` — 界面残基 KD 散点（权重为原子对数）
- `figure_contact_map.png` — 肽–靶残基接触热图
- `figure_solubility_hotspot_profile.png` — 序列溶解度 / 热点曲线

## PyMOL / ChimeraX

见 `pymol_render_instructions.md` 与 `pymol_interface_selections.pml`。
"""
    (case_dir / "case_summary.md").write_text(body, encoding="utf-8")


def select_cases(
    df: pd.DataFrame,
    ref_map: dict[str, str],
    n_select: int = 3,
) -> pd.DataFrame:
    """df 已含 ICS, SCS, FCS, motif_ratio, has_files 等列。"""
    df = df.copy()
    df["selection_score"] = (
        0.38 * df["ICS"].fillna(0)
        + 0.24 * df["SCS"].fillna(0)
        + 0.18 * df["FCS"].fillna(0)
        + 0.12 * df["motif_ratio"].fillna(0)
        + 0.08 * df["has_files"].astype(float)
    )
    df = df.sort_values("selection_score", ascending=False)
    picked_idx: list[Any] = []
    used_targets: set[str] = set()
    for idx, row in df.iterrows():
        if len(picked_idx) >= n_select:
            break
        tid = str(row["target_id"])
        if tid in used_targets:
            continue
        used_targets.add(tid)
        picked_idx.append(idx)
    if len(picked_idx) < n_select:
        for idx, row in df.iterrows():
            if len(picked_idx) >= n_select:
                break
            if idx in picked_idx:
                continue
            picked_idx.append(idx)
    return df.loc[picked_idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--n-cases", type=int, default=3, help="2–3 例")
    p.add_argument(
        "--reference-sequences",
        type=Path,
        default=None,
        help="含 target_id, sequence 的 CSV；默认尝试 project_root 下 results/4_ablation/plot/reference_sequences.csv",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    n_cases = max(2, min(int(args.n_cases), 3))
    cfg = load_config(args.config)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_dirs()
    case_root = paths.case_studies
    case_root.mkdir(parents=True, exist_ok=True)
    apply_nature_style()

    log = setup_run_logger(paths.logs, "07_select_case_studies")

    proj = Path(str(cfg.get("project_root", ""))).expanduser()
    ref_path = args.reference_sequences
    if ref_path is None:
        candidates = [
            proj / "results" / "4_ablation" / "plot" / "reference_sequences.csv",
            ROOT / "data_inventory" / "reference_sequences.csv",
        ]
        ref_path = next((p for p in candidates if p.exists()), candidates[0])
    ref_map = load_reference_sequences(ref_path if ref_path.exists() else None)
    log.info("Reference sequences loaded: %s targets", len(ref_map))

    s11 = pd.read_csv(paths.tables / "Table_S11_biophysical_summary_scores.csv")
    s1 = pd.read_csv(paths.tables / "Table_S1_master_sequence_table.csv")
    s8 = pd.read_csv(paths.tables / "Table_S8_interface_metrics.csv")

    # 复合物路径以 Table_S8 为准，避免与 S1 同名列合并成 _x/_y 后丢失
    s1_use = s1[
        ["target_id", "peptide_id", "group", "sequence", "length", "free_structure_path"]
    ].copy()
    s8_use = s8.drop(columns=[c for c in ("rank", "source") if c in s8.columns], errors="ignore")

    m = s11.merge(s1_use, on=["target_id", "peptide_id", "group"], how="left")
    m = m.merge(s8_use, on=["target_id", "peptide_id", "group"], how="inner")

    m["motif_ratio"] = m.apply(
        lambda r: motif_recovery_ratio(
            str(r.get("sequence", "") or ""),
            ref_map.get(str(r["target_id"]).strip().lower()),
        ),
        axis=1,
    )

    ic_dir = paths.intermediate / "interface_contacts"
    sol_dir = paths.intermediate / "solubility_profiles"

    def has_files(row) -> bool:
        base = _contact_basename(row["peptide_id"], row["target_id"], row.get("rank", 1))
        rc = ic_dir / f"{base}_residue_contacts.csv"
        sol = sol_dir / f"{_slug(row['peptide_id'])}.csv"
        pdb = Path(str(row.get("complex_structure_path", "") or "")).expanduser()
        return bool(rc.exists() and sol.exists() and pdb.is_file())

    m["has_files"] = m.apply(has_files, axis=1)
    m = m[m["ICS"].notna() & (m["ICS"] > 0)].copy()
    m = m[m["has_files"]].copy()
    if len(m) < n_cases:
        log.warning("Few rows with full files (%s); relaxing has_files filter", len(m))
        m = s11.merge(s1_use, on=["target_id", "peptide_id", "group"], how="left")
        m = m.merge(s8_use, on=["target_id", "peptide_id", "group"], how="inner")
        m["motif_ratio"] = m.apply(
            lambda r: motif_recovery_ratio(
                str(r.get("sequence", "") or ""),
                ref_map.get(str(r["target_id"]).strip().lower()),
            ),
            axis=1,
        ).fillna(0.0)
        m["has_files"] = m.apply(has_files, axis=1)

    sel = select_cases(m, ref_map, n_select=n_cases)
    log.info("Selected %s cases", len(sel))

    selected_json: list[dict[str, Any]] = []

    for i, (_, row) in enumerate(sel.iterrows(), start=1):
        tid, pid = row["target_id"], row["peptide_id"]
        rank = row.get("rank", 1)
        slug = _slug(f"case_{i:02d}_{pid}")
        cdir = case_root / slug
        cdir.mkdir(parents=True, exist_ok=True)

        base = _contact_basename(pid, tid, rank)
        rc_path = ic_dir / f"{base}_residue_contacts.csv"
        rc = pd.read_csv(rc_path) if rc_path.exists() else pd.DataFrame()

        prof_path = sol_dir / f"{_slug(pid)}.csv"
        prof = pd.read_csv(prof_path) if prof_path.exists() else pd.DataFrame()

        pdb_path = Path(str(row.get("complex_structure_path", "") or ""))
        cpep = str(row.get("chain_peptide", "P"))
        ctgt = str(row.get("chain_target", "T"))

        pep_ca, tgt_ca, notes = ca_coords_by_chain(pdb_path, cpep, ctgt)
        plot_overall_pca(pep_ca, tgt_ca, cdir / "figure_overall_complex_pca")
        plot_contact_map(rc, cdir / "figure_contact_map")
        plot_hydrophobic_complementarity(rc, cdir / "figure_hydrophobic_complementarity")
        plot_electrostatic_proxy(row, cdir / "figure_electrostatic_complementarity")
        plot_solubility_profile(prof, cdir / "figure_solubility_hotspot_profile")

        write_pymol_assets(cdir, pdb_path, cpep, ctgt, rc)

        meta = {
            "case_id": slug,
            "target_id": tid,
            "peptide_id": pid,
            "rank": rank,
            "chain_peptide": str(row.get("chain_peptide", "")),
            "chain_target": str(row.get("chain_target", "")),
            "ICS": float(row["ICS"]),
            "SCS": float(row["SCS"]),
            "FCS": float(row["FCS"]),
            "ALI": float(row["ALI"]) if pd.notna(row.get("ALI")) else None,
            "OBCS": float(row["OBCS"]) if pd.notna(row.get("OBCS")) else None,
            "motif_recovery_sequence_ratio": float(row["motif_ratio"]) if ref_map else None,
            "selection_score": float(row["selection_score"]),
            "complex_structure_path": str(pdb_path),
            "reference_sequence_path": str(ref_path) if ref_path and ref_path.exists() else None,
            "pca_notes": ",".join(notes),
        }
        write_case_summary(cdir, meta)

        rec = {**meta, "case_directory": str(cdir.relative_to(paths.root))}
        selected_json.append(rec)

    out_json = case_root / "selected_cases.json"
    out_json.write_text(json.dumps({"n_selected": len(selected_json), "cases": selected_json}, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

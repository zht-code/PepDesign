#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze OT self-distillation augmentation for a directory structure like:

Original data root (e.g. /root/autodl-tmp/train_data)
└── 1A1M/
    ├── peptide.fa
    ├── peptide.pdb
    ├── receptor.pdb
    └── cands/
        ├── cands_hdock_scores.json
        ├── cands_solubility_scores.json
        └── cands_stability_scores.json

Augmented data root (e.g. /root/autodl-tmp/train_data_augmentation)
└── 1A1M_1/
    ├── peptide.fa
    ├── peptide.pdb
    └── receptor.pdb
└── 1A1M_2/
    ├── peptide.fa
    ├── peptide.pdb
    └── receptor.pdb
...

This script produces:
- dataset scale statistics
- peptide length distribution comparison
- amino-acid composition comparison
- peptide secondary structure composition comparison (if computable)
- affinity / stability / solubility distribution comparison
- report.md + analysis_results.json + PNG figures + editable PDF figures

Example
-------
python ot_augmentation_distribution_analysis.py \
  --original_root /root/autodl-tmp/train_data \
  --augmented_root /root/autodl-tmp/train_data_augmentation \
  --outdir /root/autodl-tmp/ot_aug_analysis
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Nature-like plotting style
# -----------------------------
COLOR_ORIG = "#4C72B0"   # muted blue
COLOR_AUG = "#DD8452"    # muted red
COLOR_TEXT = "#333333"
COLOR_SPINE = "#333333"
COLOR_BG = "#FFFFFF"

PNG_DPI = 300


def set_nature_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": COLOR_BG,
        "axes.facecolor": COLOR_BG,
        "savefig.facecolor": COLOR_BG,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.edgecolor": COLOR_SPINE,
        "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
        "text.color": COLOR_TEXT,
        "axes.linewidth": 1.0,
        "legend.frameon": False,
        "legend.borderaxespad": 0.4,
        "pdf.fonttype": 42,   # keep text editable in Illustrator
        "ps.fonttype": 42,
        "font.family": "sans-serif",
    })


AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}
SS3 = ["H", "E", "C"]


# -----------------------------
# Basic utilities
# -----------------------------
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fasta_sequence(path: Path) -> str:
    if not path.exists():
        return ""
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seqs.append(line)
    seq = "".join(seqs).strip().upper()
    seq = "".join([aa for aa in seq if aa.isalpha()])
    return seq


def load_pdb_sequence(path: Path) -> str:
    """Fallback sequence loader from ATOM residue names in a PDB."""
    if not path.exists():
        return ""
    seen = set()
    seq = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            aa = AA3_TO_1.get(resname, "X")
            seq.append(aa)
    return "".join([x for x in seq if x in AA20])


def get_sequence(fa_path: Path, pdb_path: Path) -> str:
    seq = load_fasta_sequence(fa_path)
    if seq:
        return seq
    return load_pdb_sequence(pdb_path)


def extract_target_and_aug_idx(folder_name: str) -> Tuple[Optional[str], Optional[int]]:
    m = re.match(r"^(.+?)_(\d+)$", folder_name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def save_figure_both(fig: plt.Figure, outpath_base: str | Path, png_dpi: int = PNG_DPI) -> None:
    """
    Save figure as both:
    - PNG (300 dpi bitmap)
    - PDF (vector, AI/Illustrator editable in most cases)
    outpath_base should be path WITHOUT suffix.
    """
    outpath_base = Path(outpath_base)
    ensure_dir(outpath_base.parent)

    png_path = outpath_base.with_suffix(".png")
    pdf_path = outpath_base.with_suffix(".pdf")

    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())


# -----------------------------
# Secondary structure estimation
# -----------------------------
def ss3_from_mdtraj(pdb_path: Path) -> Optional[pd.Series]:
    try:
        import mdtraj as md  # type: ignore
    except Exception:
        return None

    try:
        traj = md.load(str(pdb_path))
        dssp = md.compute_dssp(traj, simplified=True)
        if dssp.size == 0:
            return None
        chars = list("".join(dssp[0].tolist()))
        counts = pd.Series(0.0, index=SS3)
        for ch in chars:
            if ch in counts.index:
                counts[ch] += 1
        total = counts.sum()
        return counts / total if total > 0 else None
    except Exception:
        return None


def ss3_from_pdb_records(pdb_path: Path) -> Optional[pd.Series]:
    """
    Parse HELIX / SHEET records if present. Everything else is treated as coil.
    """
    if not pdb_path.exists():
        return None

    residues = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    seen = set()
    for line in lines:
        if line.startswith("ATOM"):
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key not in seen:
                seen.add(key)
                residues.append(key)

    if not residues:
        return None

    ss_map = {r: "C" for r in residues}

    for line in lines:
        rec = line[:6].strip()
        if rec == "HELIX":
            schain = line[19].strip()
            sseq = line[21:25].strip()
            sich = line[25].strip()
            echain = line[31].strip()
            eseq = line[33:37].strip()
            eich = line[37].strip()
            active = False
            for r in residues:
                if r == (schain, sseq, sich):
                    active = True
                if active:
                    ss_map[r] = "H"
                if r == (echain, eseq, eich):
                    active = False
        elif rec == "SHEET":
            schain = line[21].strip()
            sseq = line[22:26].strip()
            sich = line[26].strip()
            echain = line[32].strip()
            eseq = line[33:37].strip()
            eich = line[37].strip()
            active = False
            for r in residues:
                if r == (schain, sseq, sich):
                    active = True
                if active and ss_map.get(r) != "H":
                    ss_map[r] = "E"
                if r == (echain, eseq, eich):
                    active = False

    counts = pd.Series(0.0, index=SS3)
    for r in residues:
        counts[ss_map.get(r, "C")] += 1
    total = counts.sum()
    return counts / total if total > 0 else None


def get_ss3_composition(pdb_path: Path) -> Optional[pd.Series]:
    ss = ss3_from_mdtraj(pdb_path)
    if ss is not None:
        return ss
    ss = ss3_from_pdb_records(pdb_path)
    return ss


# -----------------------------
# Parsing original and augmented datasets
# -----------------------------
def load_original_sample(sample_dir: Path) -> Optional[Dict]:
    if not sample_dir.is_dir():
        return None

    target_id = sample_dir.name
    peptide_fa = sample_dir / "peptide.fa"
    peptide_pdb = sample_dir / "peptide.pdb"
    cands_dir = sample_dir / "cands"
    hdock_json = cands_dir / "cands_hdock_scores.json"
    sol_json = cands_dir / "cands_solubility_scores.json"
    stab_json = cands_dir / "cands_stability_scores.json"

    if not peptide_fa.exists() and not peptide_pdb.exists():
        return None
    if not hdock_json.exists() or not sol_json.exists() or not stab_json.exists():
        return None

    seq = get_sequence(peptide_fa, peptide_pdb)
    if not seq:
        return None

    hdock = read_json(hdock_json)
    sol = read_json(sol_json)
    stab = read_json(stab_json)

    affinity = hdock.get(str(cands_dir / "peptide.pdb"), hdock.get("peptide.pdb"))
    solubility = sol.get(str(cands_dir / "peptide.pdb"), sol.get("peptide.pdb"))
    stability = stab.get(str(cands_dir / "peptide.pdb"), stab.get("peptide.pdb"))

    row = {
        "pair_id": target_id,
        "target_id": target_id,
        "peptide_sequence": seq,
        "peptide_length": len(seq),
        "affinity": pd.to_numeric(affinity, errors="coerce"),
        "solubility": pd.to_numeric(solubility, errors="coerce"),
        "stability": pd.to_numeric(stability, errors="coerce"),
        "peptide_pdb_path": str(peptide_pdb),
    }

    ss = get_ss3_composition(peptide_pdb)
    if ss is not None:
        row["ss_H_frac"] = float(ss["H"])
        row["ss_E_frac"] = float(ss["E"])
        row["ss_C_frac"] = float(ss["C"])

    return row


def load_augmented_sample(sample_dir: Path, original_root: Path) -> Optional[Dict]:
    if not sample_dir.is_dir():
        return None

    folder_name = sample_dir.name
    target_id, aug_idx = extract_target_and_aug_idx(folder_name)
    if target_id is None or aug_idx is None:
        return None

    peptide_fa = sample_dir / "peptide.fa"
    peptide_pdb = sample_dir / "peptide.pdb"
    receptor_pdb = sample_dir / "receptor.pdb"
    if not peptide_fa.exists() and not peptide_pdb.exists():
        return None

    seq = get_sequence(peptide_fa, peptide_pdb)
    if not seq:
        return None

    cands_dir = original_root / target_id / "cands"
    hdock_json = cands_dir / "cands_hdock_scores.json"
    sol_json = cands_dir / "cands_solubility_scores.json"
    stab_json = cands_dir / "cands_stability_scores.json"

    affinity = np.nan
    solubility = np.nan
    stability = np.nan
    if hdock_json.exists() and sol_json.exists() and stab_json.exists():
        hdock = read_json(hdock_json)
        sol = read_json(sol_json)
        stab = read_json(stab_json)
        key_abs = str(cands_dir / f"pep_{aug_idx:02d}.pdb")
        key_rel = f"pep_{aug_idx:02d}.pdb"
        affinity = hdock.get(key_abs, hdock.get(key_rel))
        solubility = sol.get(key_abs, sol.get(key_rel))
        stability = stab.get(key_abs, stab.get(key_rel))

    row = {
        "pair_id": folder_name,
        "target_id": target_id,
        "peptide_sequence": seq,
        "peptide_length": len(seq),
        "affinity": pd.to_numeric(affinity, errors="coerce"),
        "solubility": pd.to_numeric(solubility, errors="coerce"),
        "stability": pd.to_numeric(stability, errors="coerce"),
        "peptide_pdb_path": str(peptide_pdb),
        "receptor_pdb_path": str(receptor_pdb),
    }

    ss = get_ss3_composition(peptide_pdb)
    if ss is not None:
        row["ss_H_frac"] = float(ss["H"])
        row["ss_E_frac"] = float(ss["E"])
        row["ss_C_frac"] = float(ss["C"])

    return row


def build_original_dataframe(original_root: Path) -> pd.DataFrame:
    rows = []
    for sample_dir in sorted(original_root.iterdir()):
        row = load_original_sample(sample_dir)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No valid original samples found in {original_root}")
    return pd.DataFrame(rows)


def build_augmented_dataframe(augmented_root: Path, original_root: Path) -> pd.DataFrame:
    rows = []
    for sample_dir in sorted(augmented_root.iterdir()):
        row = load_augmented_sample(sample_dir, original_root)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No valid augmented samples found in {augmented_root}")
    return pd.DataFrame(rows)


# -----------------------------
# Distribution helpers
# -----------------------------
def normalized_value_counts(series: pd.Series, categories: Optional[Sequence[str]] = None) -> pd.Series:
    s = series.dropna().astype(str)
    vc = s.value_counts(normalize=True)
    if categories is not None:
        vc = vc.reindex(categories, fill_value=0.0)
    return vc.sort_index()


def aa_composition(seqs: Iterable[str]) -> pd.Series:
    counts = pd.Series(0.0, index=AA20)
    total = 0
    for seq in seqs:
        if not isinstance(seq, str):
            continue
        for aa in seq:
            if aa in counts.index:
                counts[aa] += 1
                total += 1
    return counts / total if total > 0 else counts


def mean_ss3_composition(df: pd.DataFrame) -> Optional[pd.Series]:
    cols = ["ss_H_frac", "ss_E_frac", "ss_C_frac"]
    if not all(c in df.columns for c in cols):
        return None
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    valid = x.dropna(how="all")
    if len(valid) == 0:
        return None
    mean_vals = valid.mean(axis=0)
    out = pd.Series({
        "H": mean_vals["ss_H_frac"],
        "E": mean_vals["ss_E_frac"],
        "C": mean_vals["ss_C_frac"]
    })
    total = out.sum()
    return out / total if total > 0 else None


def js_divergence(p: pd.Series, q: pd.Series, eps: float = 1e-12) -> float:
    all_idx = sorted(set(p.index).union(q.index))
    p = p.reindex(all_idx, fill_value=0.0).astype(float) + eps
    q = q.reindex(all_idx, fill_value=0.0).astype(float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def wasserstein_1d(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    x = np.sort(x)
    y = np.sort(y)
    n = max(len(x), len(y))
    q = np.linspace(0, 1, n)
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    return float(np.mean(np.abs(xq - yq)))


def ks_statistic(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    vals = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(np.sort(x), vals, side="right") / len(x)
    cdf_y = np.searchsorted(np.sort(y), vals, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def summarize_numeric(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.quantile(0.5)),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def compare_categorical(p: pd.Series, q: pd.Series) -> Dict:
    all_idx = sorted(set(p.index).union(q.index))
    p2 = p.reindex(all_idx, fill_value=0.0)
    q2 = q.reindex(all_idx, fill_value=0.0)
    return {
        "original_summary": p2.to_dict(),
        "augmented_summary": q2.to_dict(),
        "metrics": {
            "js_divergence": js_divergence(p2, q2),
            "l1_distance": float((p2 - q2).abs().sum()),
        },
    }


def compare_numeric(x: pd.Series, y: pd.Series) -> Dict:
    xs = pd.to_numeric(x, errors="coerce")
    ys = pd.to_numeric(y, errors="coerce")
    return {
        "original_summary": summarize_numeric(xs),
        "augmented_summary": summarize_numeric(ys),
        "metrics": {
            "wasserstein_1d": wasserstein_1d(xs, ys),
            "ks_statistic": ks_statistic(xs, ys),
            "mean_shift": float(ys.mean() - xs.mean()),
        },
    }


# -----------------------------
# Plotting
# -----------------------------
def save_barplot(
    series_a: pd.Series,
    series_b: pd.Series,
    title: str,
    ylabel: str,
    outpath_base: str,
    label_a: str = "Original",
    label_b: str = "Augmented"
) -> None:
    idx = sorted(set(series_a.index).union(series_b.index))
    a = series_a.reindex(idx, fill_value=0.0)
    b = series_b.reindex(idx, fill_value=0.0)

    x = np.arange(len(idx))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(idx) * 0.45), 4.8))

    ax.bar(
        x - width / 2, a.values, width=width,
        label=label_a, color=COLOR_ORIG,
        edgecolor="white", linewidth=0.8
    )
    ax.bar(
        x + width / 2, b.values, width=width,
        label=label_b, color=COLOR_AUG,
        edgecolor="white", linewidth=0.8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(idx, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_SPINE)
    ax.spines["bottom"].set_color(COLOR_SPINE)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

    plt.tight_layout()
    save_figure_both(fig, outpath_base)
    plt.close(fig)


def save_histogram(
    series_a: pd.Series,
    series_b: pd.Series,
    title: str,
    xlabel: str,
    outpath_base: str,
    bins: int = 40,
    label_a: str = "Original",
    label_b: str = "Augmented"
) -> None:
    a = pd.to_numeric(series_a, errors="coerce").dropna()
    b = pd.to_numeric(series_b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.8))

    ax.hist(
        a, bins=bins, alpha=0.65, density=True,
        label=label_a, color=COLOR_ORIG,
        edgecolor="white", linewidth=0.6
    )
    ax.hist(
        b, bins=bins, alpha=0.65, density=True,
        label=label_b, color=COLOR_AUG,
        edgecolor="white", linewidth=0.6
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title, pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_SPINE)
    ax.spines["bottom"].set_color(COLOR_SPINE)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

    plt.tight_layout()
    save_figure_both(fig, outpath_base)
    plt.close(fig)


# -----------------------------
# Reporting
# -----------------------------
def md_table_from_dict(d: Dict[str, float], digits: int = 4) -> str:
    lines = ["| Metric | Value |", "|---|---:|"]
    for k, v in d.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.{digits}f} |")
        else:
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def write_report(results: Dict, outdir: str) -> None:
    report_path = Path(outdir) / "report.md"
    lines: List[str] = []
    lines.append("# OT Self-Distillation Augmentation Analysis\n")

    lines.append("## Dataset scale\n")
    lines.append(md_table_from_dict(results["dataset_scale"]))
    lines.append("")

    for key, title in [
        ("peptide_length", "Peptide length distribution"),
        ("amino_acid_composition", "Amino-acid composition"),
        ("secondary_structure_composition", "Secondary-structure composition"),
        ("affinity_distribution", "Affinity label distribution"),
        ("stability_distribution", "Stability label distribution"),
        ("solubility_distribution", "Solubility label distribution"),
    ]:
        if key not in results:
            continue
        item = results[key]
        lines.append(f"## {title}\n")
        lines.append("### Original summary\n")
        lines.append(md_table_from_dict(item["original_summary"]))
        lines.append("")
        lines.append("### Augmented summary\n")
        lines.append(md_table_from_dict(item["augmented_summary"]))
        lines.append("")
        lines.append("### Comparison metrics\n")
        lines.append(md_table_from_dict(item["metrics"]))
        lines.append("")

    if "notes" in results:
        lines.append("## Notes\n")
        for note in results["notes"]:
            lines.append(f"- {note}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main analysis
# -----------------------------
def build_analysis(df_orig: pd.DataFrame, df_aug: pd.DataFrame) -> Dict:
    results: Dict[str, object] = {}
    notes: List[str] = []

    results["dataset_scale"] = {
        "n_original": int(len(df_orig)),
        "n_augmented": int(len(df_aug)),
        "scale_ratio": float(len(df_aug) / max(len(df_orig), 1)),
        "n_targets_original": int(df_orig["target_id"].nunique()),
        "n_targets_augmented": int(df_aug["target_id"].nunique()),
    }

    results["peptide_length"] = compare_numeric(df_orig["peptide_length"], df_aug["peptide_length"])

    aa_orig = aa_composition(df_orig["peptide_sequence"])
    aa_aug = aa_composition(df_aug["peptide_sequence"])
    results["amino_acid_composition"] = compare_categorical(aa_orig, aa_aug)

    ss_orig = mean_ss3_composition(df_orig)
    ss_aug = mean_ss3_composition(df_aug)
    if ss_orig is not None and ss_aug is not None:
        results["secondary_structure_composition"] = compare_categorical(ss_orig, ss_aug)
    else:
        notes.append("Secondary-structure composition was skipped for some or all samples because DSSP/HELIX/SHEET information was unavailable.")

    for label in ["affinity", "stability", "solubility"]:
        if label in df_orig.columns and label in df_aug.columns:
            x = pd.to_numeric(df_orig[label], errors="coerce")
            y = pd.to_numeric(df_aug[label], errors="coerce")
            if x.notna().sum() > 0 and y.notna().sum() > 0:
                results[f"{label}_distribution"] = compare_numeric(x, y)
            else:
                notes.append(f"{label} distribution was skipped because labels were missing for one side.")

    notes.append("Target family distribution was not analyzed because no target-family annotation file was provided.")
    results["notes"] = notes
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze OT self-distillation augmentation from train_data / train_data_augmentation directories."
    )
    parser.add_argument("--original_root", required=True, help="Root directory of original data, e.g. /root/autodl-tmp/train_data")
    parser.add_argument("--augmented_root", required=True, help="Root directory of augmented data, e.g. /root/autodl-tmp/train_data_augmentation")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--save_tables", action="store_true", help="Save parsed original/augmented tables as CSV")
    args = parser.parse_args()

    set_nature_style()
    ensure_dir(args.outdir)

    original_root = Path(args.original_root)
    augmented_root = Path(args.augmented_root)
    outdir = Path(args.outdir)

    df_orig = build_original_dataframe(original_root)
    df_aug = build_augmented_dataframe(augmented_root, original_root)

    results = build_analysis(df_orig, df_aug)

    if args.save_tables:
        df_orig.to_csv(outdir / "original_parsed.csv", index=False)
        df_aug.to_csv(outdir / "augmented_parsed.csv", index=False)

    with open(outdir / "analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    save_histogram(
        df_orig["peptide_length"], df_aug["peptide_length"],
        title="Peptide length distribution",
        xlabel="Peptide length",
        outpath_base=str(outdir / "peptide_length_hist")
    )

    aa_orig = aa_composition(df_orig["peptide_sequence"])
    aa_aug = aa_composition(df_aug["peptide_sequence"])
    save_barplot(
        aa_orig, aa_aug,
        title="Amino-acid composition",
        ylabel="Fraction",
        outpath_base=str(outdir / "aa_composition_bar")
    )

    ss_orig = mean_ss3_composition(df_orig)
    ss_aug = mean_ss3_composition(df_aug)
    if ss_orig is not None and ss_aug is not None:
        save_barplot(
            ss_orig, ss_aug,
            title="Secondary-structure composition",
            ylabel="Fraction",
            outpath_base=str(outdir / "ss3_composition_bar")
        )

    for label in ["affinity", "stability", "solubility"]:
        x = pd.to_numeric(df_orig.get(label, pd.Series(dtype=float)), errors="coerce")
        y = pd.to_numeric(df_aug.get(label, pd.Series(dtype=float)), errors="coerce")
        if x.notna().sum() > 0 and y.notna().sum() > 0:
            save_histogram(
                x, y,
                title=f"{label.capitalize()} distribution",
                xlabel=label,
                outpath_base=str(outdir / f"{label}_hist")
            )

    write_report(results, str(outdir))
    print(f"Done. Results saved to: {outdir}")
    print("Each figure is saved in both PNG (300 dpi) and PDF (vector, Illustrator-editable).")


if __name__ == "__main__":
    main()


'''
python /root/autodl-tmp/Peptide_3D/results/1_OT/1_ot_augmentation_distribution_analysis.py \
  --original_root /root/autodl-tmp/train_data \
  --augmented_root /root/autodl-tmp/train_data_augmentation_strong \
  --outdir /root/autodl-tmp/Peptide_3D/results/1_OT/1_ot_aug_distribution_analysis \
  --save_tables
'''
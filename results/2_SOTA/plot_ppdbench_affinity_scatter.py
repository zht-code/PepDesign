#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mmseqs_similarity import build_train_fasta_from_root, compute_train_similarity
from utils.plotting_2sota import save_figure, setup_publication_style
from utils.structure_metrics import extract_peptide_sequence


DEFAULT_PPD_ROOT = "/root/autodl-tmp/PPDbench"
DEFAULT_BASELINE_DIR = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/baseline_data"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/ppdbench_affinity_figures"
DEFAULT_TRAIN_ROOT = "/root/autodl-tmp/train_data"
DEFAULT_MMSEQS = "/root/autodl-fs/mmseqs-linux-gpu/mmseqs/bin/mmseqs"
DEFAULT_PG_PPD_JSON = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/baseline_data/proteingenerator_PPDbench_hdock_scores.json"
DEFAULT_PG_ZIP = "/root/autodl-tmp/Peptide_3D/results/2_SOTA/baseline_data/proteingenerator.zip"

RF_FILES = [f"RFdiffusion{i}.json" for i in range(1, 6)]
METHOD_ORDER = ["ours", "rfdiffusion", "bindcraft", "proteingenerator"]
METHOD_LABELS = {
    "ours": "Ours",
    "rfdiffusion": "RFdiffusion",
    "bindcraft": "BindCraft",
    "proteingenerator": "ProteinGenerator",
}
OURS_TOPK = 5


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot PPDbench affinity scatter colored by train-set similarity.")
    ap.add_argument("--ppdbench-root", default=DEFAULT_PPD_ROOT)
    ap.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--train-root", default=DEFAULT_TRAIN_ROOT)
    ap.add_argument("--mmseqs", default=DEFAULT_MMSEQS)
    ap.add_argument("--proteingenerator-ppd-json", default=DEFAULT_PG_PPD_JSON)
    ap.add_argument("--proteingenerator-zip", default=DEFAULT_PG_ZIP)
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


def _polymer_chain_sequences_from_lines(lines: Iterable[str]) -> dict[str, str]:
    aa3_to_aa1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "MSE": "M",
    }
    chain_residues: dict[str, dict[tuple[int, str], str]] = defaultdict(dict)
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        resname = line[17:20].strip().upper()
        chain_id = (line[21].strip() or "_") if len(line) > 21 else "_"
        resseq_str = line[22:26].strip() if len(line) > 26 else ""
        icode = (line[26].strip() or "") if len(line) > 26 else ""
        if not resseq_str:
            continue
        try:
            resseq = int(resseq_str)
        except ValueError:
            continue
        key = (resseq, icode)
        if key not in chain_residues[chain_id]:
            chain_residues[chain_id][key] = resname

    out: dict[str, str] = {}
    for chain_id, residue_map in chain_residues.items():
        seq = []
        for key in sorted(residue_map.keys(), key=lambda item: (item[0], item[1])):
            aa = aa3_to_aa1.get(residue_map[key])
            if aa:
                seq.append(aa)
        if seq:
            out[chain_id] = "".join(seq)
    return out


def sequence_from_zip_member(zf: zipfile.ZipFile, member_name: str, reference_length: int | None = None) -> str:
    lines = zf.read(member_name).decode("utf-8", errors="ignore").splitlines()
    chain_map = _polymer_chain_sequences_from_lines(lines)
    if not chain_map:
        return ""
    items = list(chain_map.items())
    if len(items) == 1:
        return items[0][1]
    if reference_length is not None:
        items.sort(key=lambda item: (abs(len(item[1]) - reference_length), len(item[1])))
    else:
        items.sort(key=lambda item: len(item[1]))
    return items[0][1]


def load_native_scores(baseline_dir: Path, output_dir: Path) -> dict[str, float]:
    native_scores: dict[str, float] = {}
    legacy_json = baseline_dir / "Hdock_proteingenerator_vina.json"
    if legacy_json.is_file():
        data = json.loads(legacy_json.read_text(encoding="utf-8"))
        for target_id, entry in data.items():
            properties = entry.get("properties", [])
            if not properties:
                continue
            score = properties[0].get("test Affinity (kcal/mol)")
            if score is None:
                continue
            native_scores[str(target_id)] = float(score)
        if native_scores:
            return native_scores

    # Fallback: reuse native affinities already written by a previous plotting run.
    for csv_name in ["ppdbench_best_affinity_shared_targets.csv", "ppdbench_best_affinity_all_available.csv"]:
        csv_path = output_dir / csv_name
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        if "target_id" not in df.columns or "native_hdock_score" not in df.columns:
            continue
        sub = df[["target_id", "native_hdock_score"]].dropna()
        if sub.empty:
            continue
        for row in sub.itertuples(index=False):
            native_scores[str(row.target_id)] = float(row.native_hdock_score)
        if native_scores:
            return native_scores
    return native_scores


def load_ours(ppd_root: Path) -> list[dict]:
    rows: list[dict] = []
    patt = re.compile(r"pep_(\d{2})\.pdb$")
    for score_json in sorted(ppd_root.glob("*/multi_cands/cands_hdock_scores.json")):
        target_id = score_json.parent.parent.name
        payload = json.loads(score_json.read_text(encoding="utf-8"))

        reference_peptide = ppd_root / target_id / "peptide.pdb"
        for pdb_path, score in payload.items():
            match = patt.search(Path(pdb_path).name)
            if not match:
                continue
            pdb = Path(pdb_path)
            if not pdb.is_file():
                continue
            seq = extract_peptide_sequence(str(pdb), reference_peptide_path=str(reference_peptide))
            rows.append(
                {
                    "method": "ours",
                    "target_id": target_id,
                    "candidate_rank": int(match.group(1)),
                    "hdock_score": float(score),
                    "sequence": seq,
                    "pdb_path": str(pdb),
                }
            )
    return rows


def load_rfdiffusion(baseline_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for rank, file_name in enumerate(RF_FILES, start=1):
        data = json.loads((baseline_dir / file_name).read_text(encoding="utf-8"))
        affinity_key = f"RFdiffusion{rank} Affinity (kcal/mol)"
        for target_id, entry in data.items():
            properties = entry.get("properties", [])
            if not properties:
                continue
            score = properties[0].get(affinity_key)
            seq = str(entry.get("generated_peptide", "")).strip()
            if score is None or not seq:
                continue
            rows.append(
                {
                    "method": "rfdiffusion",
                    "target_id": str(target_id),
                    "candidate_rank": rank,
                    "hdock_score": float(score),
                    "sequence": seq,
                    "pdb_path": "",
                }
            )
    return rows


def load_bindcraft(baseline_dir: Path, ppd_root: Path) -> list[dict]:
    rows: list[dict] = []
    data = json.loads((baseline_dir / "bindcraft_hdock_scores.json").read_text(encoding="utf-8"))
    zip_path = baseline_dir / "result_bindcraft.zip"
    with zipfile.ZipFile(zip_path) as zf:
        for target_id, target_map in data.items():
            reference_peptide = ppd_root / target_id / "peptide.pdb"
            try:
                ref_len = len(extract_peptide_sequence(str(reference_peptide)))
            except Exception:
                ref_len = None
            sorted_items = sorted(
                target_map.items(),
                key=lambda kv: float(kv[1].get("score", float("inf"))),
            )
            for rank, (candidate_path, info) in enumerate(sorted_items[:5], start=1):
                rel = candidate_path.split("/experiments/result_clean_pdb/", 1)[-1]
                member_name = f"result_bindcraft/result_clean_pdb/{rel}"
                seq = ""
                if member_name in zf.namelist():
                    seq = sequence_from_zip_member(zf, member_name, reference_length=ref_len)
                rows.append(
                    {
                        "method": "bindcraft",
                        "target_id": str(target_id),
                        "candidate_rank": rank,
                        "hdock_score": float(info.get("score")),
                        "sequence": seq,
                        "pdb_path": candidate_path,
                    }
                )
    return rows


def _proteingenerator_sequence_from_sources(
    *,
    target_id: str,
    info: dict,
    zip_members: set[str],
    zip_handle: zipfile.ZipFile | None,
) -> tuple[str, str]:
    peptide_path = str(info.get("peptide_pdb", "")).strip()
    if peptide_path and Path(peptide_path).is_file():
        try:
            return extract_peptide_sequence(peptide_path), peptide_path
        except Exception:
            pass

    candidate_name = Path(peptide_path).name if peptide_path else ""
    if candidate_name:
        member_name = f"proteingenerator/{target_id}/{candidate_name}"
        if zip_handle is not None and member_name in zip_members:
            try:
                return sequence_from_zip_member(zip_handle, member_name), member_name
            except Exception:
                pass
    return "", peptide_path


def load_proteingenerator(
    baseline_dir: Path,
    per_candidate_json: Path | None = None,
    proteingenerator_zip: Path | None = None,
) -> list[dict]:
    rows: list[dict] = []
    if per_candidate_json is not None and per_candidate_json.is_file():
        data = json.loads(per_candidate_json.read_text(encoding="utf-8"))
        zip_handle = None
        zip_members: set[str] = set()
        if proteingenerator_zip is not None and proteingenerator_zip.is_file():
            zip_handle = zipfile.ZipFile(proteingenerator_zip)
            zip_members = set(zip_handle.namelist())
        try:
            for target_id, target_map in data.items():
                if not isinstance(target_map, dict):
                    continue
                sorted_items = sorted(
                    (
                        (candidate_path, info)
                        for candidate_path, info in target_map.items()
                        if isinstance(info, dict) and info.get("score") is not None
                    ),
                    key=lambda kv: float(kv[1]["score"]),
                )
                for rank, (candidate_path, info) in enumerate(sorted_items[:5], start=1):
                    seq, resolved_path = _proteingenerator_sequence_from_sources(
                        target_id=str(target_id),
                        info=info,
                        zip_members=zip_members,
                        zip_handle=zip_handle,
                    )
                    rows.append(
                        {
                            "method": "proteingenerator",
                            "target_id": str(target_id),
                            "candidate_rank": rank,
                            "hdock_score": float(info.get("score")),
                            "sequence": seq,
                            "pdb_path": resolved_path or candidate_path,
                        }
                    )
        finally:
            if zip_handle is not None:
                zip_handle.close()
        if rows:
            return rows

    data = json.loads((baseline_dir / "Hdock_proteingenerator_vina.json").read_text(encoding="utf-8"))
    for target_id, entry in data.items():
        properties = entry.get("properties", [])
        if not properties:
            continue
        score = properties[0].get("protein_generator Affinity (kcal/mol)")
        seq = str(entry.get("generated_peptide", "")).strip()
        if score is None or not seq:
            continue
        rows.append(
            {
                "method": "proteingenerator",
                "target_id": str(target_id),
                "candidate_rank": 1,
                "hdock_score": float(score),
                "sequence": seq,
                "pdb_path": "",
            }
        )
    return rows


def compute_similarity(per_candidate_df: pd.DataFrame, train_root: Path, mmseqs: str, output_dir: Path) -> pd.DataFrame:
    train_fasta = build_train_fasta_from_root(train_root, output_dir / "_train_fastas" / "ppdbench_train.fasta")
    query_records = [
        (f"{row.method}|{row.target_id}|{int(row.candidate_rank):02d}", row.sequence)
        for row in per_candidate_df.itertuples(index=False)
    ]
    sim_map = compute_train_similarity(
        dataset_to_queries={"ppdbench": query_records},
        dataset_to_train_fasta={"ppdbench": train_fasta},
        mmseqs=mmseqs,
        output_dir=output_dir,
    )
    id_to_similarity = sim_map.get("ppdbench", {})
    df = per_candidate_df.copy()
    df["query_id"] = df.apply(lambda row: f"{row['method']}|{row['target_id']}|{int(row['candidate_rank']):02d}", axis=1)
    df["train_similarity"] = df["query_id"].map(id_to_similarity)
    return df


def build_best_of_top5(per_candidate_df: pd.DataFrame, native_scores: dict[str, float]) -> pd.DataFrame:
    rows = []
    for (method, target_id), sub in per_candidate_df.groupby(["method", "target_id"], dropna=False):
        sub = sub.copy()
        sub["hdock_score"] = pd.to_numeric(sub["hdock_score"], errors="coerce")
        sub = sub[np.isfinite(sub["hdock_score"])]
        if sub.empty:
            continue
        best = sub.sort_values(["hdock_score", "candidate_rank"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "method": method,
                "target_id": target_id,
                "candidate_rank": int(best["candidate_rank"]),
                "best_hdock_score": float(best["hdock_score"]),
                "train_similarity": float(best["train_similarity"]) if pd.notna(best["train_similarity"]) else math.nan,
                "sequence": best["sequence"],
                "native_hdock_score": native_scores.get(target_id, math.nan),
                "candidate_count": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def build_mean_of_top5(per_candidate_df: pd.DataFrame, native_scores: dict[str, float]) -> pd.DataFrame:
    rows = []
    for (method, target_id), sub in per_candidate_df.groupby(["method", "target_id"], dropna=False):
        sub = sub.copy()
        sub["hdock_score"] = pd.to_numeric(sub["hdock_score"], errors="coerce")
        sub["train_similarity"] = pd.to_numeric(sub["train_similarity"], errors="coerce")
        sub = sub[np.isfinite(sub["hdock_score"])]
        if sub.empty:
            continue
        rows.append(
            {
                "method": method,
                "target_id": target_id,
                "top5_mean_hdock_score": float(sub["hdock_score"].mean()),
                "train_similarity": float(sub["train_similarity"].mean()) if sub["train_similarity"].notna().any() else math.nan,
                "native_hdock_score": native_scores.get(target_id, math.nan),
                "candidate_count": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def select_topk_candidates(per_candidate_df: pd.DataFrame, topk: int) -> pd.DataFrame:
    kept = []
    for (method, target_id), sub in per_candidate_df.groupby(["method", "target_id"], dropna=False):
        sub = sub.copy()
        sub["hdock_score"] = pd.to_numeric(sub["hdock_score"], errors="coerce")
        sub = sub[np.isfinite(sub["hdock_score"])]
        if sub.empty:
            continue
        if method == "ours" and len(sub) > topk:
            sub = sub.sort_values(["hdock_score", "candidate_rank"], ascending=[True, True]).head(topk)
        kept.append(sub)
    if not kept:
        return per_candidate_df.iloc[0:0].copy()
    return pd.concat(kept, ignore_index=True)


def common_target_set(best_df: pd.DataFrame) -> set[str]:
    method_to_targets = {
        method: set(best_df.loc[best_df["method"] == method, "target_id"].astype(str))
        for method in METHOD_ORDER
    }
    target_sets = [targets for targets in method_to_targets.values() if targets]
    if not target_sets:
        return set()
    shared = set.intersection(*target_sets)
    return {target for target in shared if pd.notna(best_df.loc[best_df["target_id"] == target, "native_hdock_score"]).any()}


def add_trendline(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) < 3 or len(np.unique(x)) < 2:
        return
    try:
        coeff = np.polyfit(x, y, deg=1)
    except np.linalg.LinAlgError:
        return
    xp = np.linspace(np.min(x), np.max(x), 100)
    yp = coeff[0] * xp + coeff[1]
    ax.plot(xp, yp, color="black", linewidth=1.0, alpha=0.8)


def panel_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title_prefix: str,
    out_stem: str,
    output_dir: Path,
    dpi: int,
) -> tuple[str, str]:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), sharey=False, constrained_layout=True)
    axes = axes.flatten()
    vmin = 0.0
    vmax = 1.0
    scatter_artist = None

    for ax, method in zip(axes, METHOD_ORDER):
        sub = df[df["method"] == method].copy()
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
        sub["train_similarity"] = pd.to_numeric(sub["train_similarity"], errors="coerce")
        sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col]) & np.isfinite(sub["train_similarity"])]
        if sub.empty:
            ax.set_title(f"{METHOD_LABELS[method]} | no data")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            continue

        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        c = sub["train_similarity"].to_numpy(dtype=float)
        scatter_artist = ax.scatter(
            x,
            y,
            c=c,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=34,
            alpha=0.82,
            edgecolors="black",
            linewidths=0.25,
        )
        add_trendline(ax, x, y)
        if x_col == "native_hdock_score":
            min_xy = min(np.min(x), np.min(y))
            max_xy = max(np.max(x), np.max(y))
            pad = max(5.0, (max_xy - min_xy) * 0.05)
            lo = min_xy - pad
            hi = max_xy + pad
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="#666666", linewidth=0.9, alpha=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_title(f"{title_prefix}{METHOD_LABELS[method]} (n={len(sub)})")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    if scatter_artist is not None:
        cbar = fig.colorbar(scatter_artist, ax=axes.tolist(), shrink=0.92)
        cbar.set_label("Train similarity")
        cbar.set_ticks(np.linspace(vmin, vmax, 6))
    return save_figure(fig, output_dir, out_stem, dpi)


def main() -> None:
    args = parse_args()
    setup_publication_style()
    plt.rcParams["font.family"] = ["DejaVu Serif", "serif"]

    ppd_root = Path(args.ppdbench_root)
    baseline_dir = Path(args.baseline_dir)
    output_dir = ensure_dir(args.output_dir)
    train_root = Path(args.train_root)
    pg_ppd_json = Path(args.proteingenerator_ppd_json)
    pg_zip = Path(args.proteingenerator_zip)

    native_scores = load_native_scores(baseline_dir, output_dir)
    ours_rows = load_ours(ppd_root)
    rf_rows = load_rfdiffusion(baseline_dir)
    bind_rows = load_bindcraft(baseline_dir, ppd_root)
    pg_rows = load_proteingenerator(baseline_dir, pg_ppd_json, pg_zip)

    per_candidate_df = pd.DataFrame(ours_rows + rf_rows + bind_rows + pg_rows)
    per_candidate_df = per_candidate_df.sort_values(["method", "target_id", "candidate_rank"]).reset_index(drop=True)
    per_candidate_df = select_topk_candidates(per_candidate_df, OURS_TOPK)
    per_candidate_df = compute_similarity(per_candidate_df, train_root, args.mmseqs, output_dir)

    best_df = build_best_of_top5(per_candidate_df, native_scores)
    top5_mean_df = build_mean_of_top5(per_candidate_df, native_scores)
    shared_targets = common_target_set(best_df)
    shared_df = best_df[best_df["target_id"].isin(shared_targets)].copy()
    shared_df = shared_df.sort_values(["method", "target_id"]).reset_index(drop=True)
    top5_mean_shared_df = top5_mean_df[top5_mean_df["target_id"].isin(shared_targets)].copy()
    top5_mean_shared_df = top5_mean_shared_df.sort_values(["method", "target_id"]).reset_index(drop=True)

    per_candidate_path = output_dir / "ppdbench_per_candidate_affinity_similarity.csv"
    best_all_path = output_dir / "ppdbench_best_affinity_all_available.csv"
    best_shared_path = output_dir / "ppdbench_best_affinity_shared_targets.csv"
    top5_mean_all_path = output_dir / "ppdbench_top5_mean_affinity_all_available.csv"
    top5_mean_shared_path = output_dir / "ppdbench_top5_mean_affinity_shared_targets.csv"
    summary_path = output_dir / "ppdbench_affinity_plot_summary.json"

    per_candidate_df.to_csv(per_candidate_path, index=False)
    best_df.to_csv(best_all_path, index=False)
    shared_df.to_csv(best_shared_path, index=False)
    top5_mean_df.to_csv(top5_mean_all_path, index=False)
    top5_mean_shared_df.to_csv(top5_mean_shared_path, index=False)

    panel_scatter(
        shared_df,
        x_col="native_hdock_score",
        y_col="best_hdock_score",
        x_label="Native peptide HDOCK score",
        y_label="Best generated HDOCK score from top5",
        title_prefix="PPDbench | ",
        out_stem="ppdbench_native_vs_best_affinity_shared_targets",
        output_dir=output_dir,
        dpi=args.dpi,
    )
    panel_scatter(
        top5_mean_shared_df,
        x_col="native_hdock_score",
        y_col="top5_mean_hdock_score",
        x_label="Native peptide HDOCK score",
        y_label="Mean generated HDOCK score over top5",
        title_prefix="PPDbench | ",
        out_stem="ppdbench_native_vs_top5_mean_affinity_shared_targets",
        output_dir=output_dir,
        dpi=args.dpi,
    )
    panel_scatter(
        shared_df,
        x_col="train_similarity",
        y_col="best_hdock_score",
        x_label="Train similarity",
        y_label="Best generated HDOCK score from top5",
        title_prefix="PPDbench | ",
        out_stem="ppdbench_similarity_vs_best_affinity_shared_targets",
        output_dir=output_dir,
        dpi=args.dpi,
    )
    panel_scatter(
        top5_mean_shared_df,
        x_col="train_similarity",
        y_col="top5_mean_hdock_score",
        x_label="Train similarity",
        y_label="Mean generated HDOCK score over top5",
        title_prefix="PPDbench | ",
        out_stem="ppdbench_similarity_vs_top5_mean_affinity_shared_targets",
        output_dir=output_dir,
        dpi=args.dpi,
    )

    summary = {
        "ppdbench_target_dirs": len([p for p in ppd_root.iterdir() if p.is_dir()]),
        "ours_targets_with_affinity": int(pd.Series(list(native_scores.keys())).nunique()),
        "per_candidate_counts": {
            method: int((per_candidate_df["method"] == method).sum()) for method in METHOD_ORDER
        },
        "best_target_counts_all_available": {
            method: int((best_df["method"] == method).sum()) for method in METHOD_ORDER
        },
        "shared_target_count": int(len(shared_targets)),
        "note": (
            "Ours is read from PPDbench/*/multi_cands/cands_hdock_scores.json and truncated to the best 5 candidates "
            "per target by HDOCK score. Native peptide affinity is taken from Hdock_proteingenerator_vina.json. "
            "ProteinGenerator prefers per-candidate PPDbench scores from proteingenerator_ppdbench_hdock_scores.json "
            "when available, otherwise falls back to the summary affinity in Hdock_proteingenerator_vina.json."
        ),
        "outputs": {
            "per_candidate_csv": str(per_candidate_path),
            "best_all_csv": str(best_all_path),
            "best_shared_csv": str(best_shared_path),
            "top5_mean_all_csv": str(top5_mean_all_path),
            "top5_mean_shared_csv": str(top5_mean_shared_path),
            "native_vs_best_png": str(output_dir / "ppdbench_native_vs_best_affinity_shared_targets.png"),
            "native_vs_best_pdf": str(output_dir / "ppdbench_native_vs_best_affinity_shared_targets.pdf"),
            "native_vs_top5_mean_png": str(output_dir / "ppdbench_native_vs_top5_mean_affinity_shared_targets.png"),
            "native_vs_top5_mean_pdf": str(output_dir / "ppdbench_native_vs_top5_mean_affinity_shared_targets.pdf"),
            "similarity_vs_best_png": str(output_dir / "ppdbench_similarity_vs_best_affinity_shared_targets.png"),
            "similarity_vs_best_pdf": str(output_dir / "ppdbench_similarity_vs_best_affinity_shared_targets.pdf"),
            "similarity_vs_top5_mean_png": str(output_dir / "ppdbench_similarity_vs_top5_mean_affinity_shared_targets.png"),
            "similarity_vs_top5_mean_pdf": str(output_dir / "ppdbench_similarity_vs_top5_mean_affinity_shared_targets.pdf"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved per-candidate table: {per_candidate_path}")
    print(f"Saved target-level table: {best_shared_path}")
    print(f"Shared targets used for plotting: {len(shared_targets)}")
    print(f"Saved figures under: {output_dir}")


if __name__ == "__main__":
    main()

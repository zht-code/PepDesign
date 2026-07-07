from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch

from utils_io import ensure_dir, write_fasta
from utils_sequence import mmseqs_search, sequence_identity

# PDF text as TrueType (type 42), not Type 3 outlines — needed for Adobe Illustrator text editing.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.use14corefonts"] = False

# Sequential colormap for identity heatmaps (perceptually smooth; works well with data-scaled norm).
_HEATMAP_CMAP = "magma"

_PLOT_RC = {
    "figure.facecolor": "#f8f8f7",
    "font.size": 9,
    "axes.labelcolor": "#2d2d2d",
    "axes.edgecolor": "#c4c4c4",
    "axes.linewidth": 0.9,
    "text.color": "#1a1a1a",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "grid.color": "#e6e6e4",
    "grid.linewidth": 0.7,
    "grid.alpha": 1.0,
    # TrueType in PDF/PS so Illustrator / Inkscape can select and edit text (not Type 3 outlines).
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Applied again at PDF write time so figures created outside _PLOT_RC still export editable text.
_PDF_EDITABLE_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.use14corefonts": False,
}


def _save_fig_png_pdf(out_png_path: str | Path, *, dpi: int = 300) -> None:
    """Write PNG (raster) and PDF (vector text as TrueType for Adobe Illustrator editing)."""
    p = Path(out_png_path)
    stem = p.with_suffix("") if p.suffix.lower() == ".png" else p
    plt.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    with plt.rc_context(_PDF_EDITABLE_RC):
        plt.savefig(stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")


def best_similarity_python(query_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_pairs = list(zip(target_df["sample_id"], target_df["receptor_sequence"]))
    for _, q in query_df.iterrows():
        qid = q["sample_id"]
        qseq = q["receptor_sequence"]
        best_id, best_sim = None, -1.0
        for tid, tseq in target_pairs:
            sim = sequence_identity(qseq, tseq)
            if sim > best_sim:
                best_sim = sim
                best_id = tid
        rows.append({"query_id": qid, "best_target_id": best_id, "best_identity": best_sim})
    return pd.DataFrame(rows)


def best_similarity_mmseqs(query_df: pd.DataFrame, target_df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    qf = outdir / "query.fasta"
    tf = outdir / "target.fasta"
    write_fasta(list(zip(query_df["sample_id"], query_df["receptor_sequence"])), qf)
    write_fasta(list(zip(target_df["sample_id"], target_df["receptor_sequence"])), tf)
    tsv = mmseqs_search(str(qf), str(tf), str(outdir / "mmseqs_search"))

    cols = ["query", "target", "pident", "alnlen", "mismatch", "gapopen", "qstart", "qend",
            "tstart", "tend", "evalue", "bits"]
    df = pd.read_csv(tsv, sep="\t", header=None)
    df = df.iloc[:, :len(cols)]
    df.columns = cols
    best = df.sort_values(["query", "bits"], ascending=[True, False]).groupby("query", as_index=False).first()
    best = best.rename(columns={"query": "query_id", "target": "best_target_id", "pident": "best_identity"})
    best["best_identity"] = best["best_identity"] / 100.0
    return best[["query_id", "best_target_id", "best_identity"]]


def plot_similarity(best_df: pd.DataFrame, out_png: str, title: str, *, xlabel: str) -> None:
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
        fig.patch.set_facecolor("#f8f8f7")
        ax.set_facecolor("#ffffff")
        vals = best_df["best_identity"].dropna().values
        ax.hist(
            vals,
            bins=22,
            color="#C1D8E9",
            edgecolor="#E8F1F6",
            linewidth=0.55,
            alpha=0.95,
        )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=12, color="#1a1a1a")
        ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=1.0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    _save_fig_png_pdf(out_png, dpi=300)
    plt.close(fig)


_BEST_SIM_COLS = ("query_id", "best_target_id", "best_identity")


def load_best_similarity_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in _BEST_SIM_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}; expected {_BEST_SIM_COLS}")
    df = df[list(_BEST_SIM_COLS)].copy()
    df["query_id"] = df["query_id"].astype(str)
    return df


# Circos discrete bins: 8 steps interpolated in RGB between user reference stops:
# #92B1D9 → #C1D8E9 → #DBDDEF → #F6C8B6 → #D4D4D4
_CIRCOS_DISCRETE_HEX = (
    "#92B1D9",
    "#ADC7E2",
    "#C5D9EA",
    "#D4DCED",
    "#E3D7DF",
    "#F2CBBE",
    "#E7CDC3",
    "#D4D4D4",
)


def _circos_discrete_norm(
    vals: np.ndarray,
    *,
    scale: str,
    n_bins: int = 8,
) -> tuple[np.ndarray, np.ndarray, ListedColormap, BoundaryNorm, ScalarMappable]:
    """Discrete BoundaryNorm + ListedColormap for Circos-style wheels."""
    vals = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)
    if scale == "fixed":
        vmin, vmax = 0.0, 1.0
    else:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        if hi - lo < 1e-12:
            vmin, vmax = 0.0, 1.0
        else:
            pad = 0.06 * (hi - lo)
            vmin = max(0.0, lo - pad)
            vmax = min(1.0, hi + pad)
    boundaries = np.linspace(vmin, vmax, n_bins + 1)
    colors = list(_CIRCOS_DISCRETE_HEX[:n_bins])
    if len(colors) < n_bins:
        cmap_base = colormaps["Spectral"]
        colors = [cmap_base(i / max(n_bins - 1, 1)) for i in range(n_bins)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return vals, boundaries, cmap, norm, sm


def _short_id(s: str, *, max_len: int = 18) -> str:
    s = str(s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _normalize_identity_colors(vals: np.ndarray, *, scale: str) -> tuple[np.ndarray, Normalize]:
    """Map colors so narrow identity ranges still span the colormap (avoids all-yellow when vmin≈vmax≪1)."""
    vals = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)
    if scale == "fixed":
        return vals, Normalize(vmin=0.0, vmax=1.0)
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return vals, Normalize(vmin=0.0, vmax=1.0)
    if hi - lo < 1e-12:
        return vals, Normalize(vmin=0.0, vmax=1.0)
    span = hi - lo
    pad = 0.06 * span
    vmin = max(0.0, lo - pad)
    vmax = min(1.0, hi + pad)
    if vmax - vmin < 1e-12:
        return vals, Normalize(vmin=0.0, vmax=1.0)
    return vals, Normalize(vmin=vmin, vmax=vmax)


def plot_circos_style_heatmap(
    best_df: pd.DataFrame,
    query_df: pd.DataFrame,
    target_df: pd.DataFrame,
    out_png: str,
    title: str,
    *,
    nn_column_label: str,
    heatmap_scale: str = "data",
) -> None:
    """Circos-style wheel: discrete color bins (blue→red), thin tracks, white gaps, center title + legend."""
    aligned = best_df.set_index("query_id").reindex(query_df["sample_id"].astype(str))
    vals = pd.to_numeric(aligned["best_identity"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        raise ValueError("No samples to plot.")

    vals, boundaries, cmap, norm, sm = _circos_discrete_norm(vals, scale=heatmap_scale, n_bins=8)
    face = sm.to_rgba(vals, alpha=1.0)

    theta_edges = np.linspace(0.0, 2.0 * np.pi, n + 1)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) * 0.5
    width = 2.0 * np.pi / n
    theta_ring = np.linspace(0.0, 2.0 * np.pi, 720)

    with plt.rc_context(_PLOT_RC):
        fig = plt.figure(figsize=(10.0, 10.0), facecolor="#ffffff")
        ax = fig.add_subplot(111, projection="polar", facecolor="#ffffff")
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")

        # Large center, thin outer tracks, white spacers (Circos-like).
        r_hub = 0.40
        r_sp1_top = 0.45
        r_dot = 0.475
        r_sp2_bot = 0.505
        r_sp2_top = 0.53
        r_in = 0.53
        r_out = 0.61
        r_rim = 0.625

        # Hub
        ax.bar(
            0.0,
            r_hub,
            width=2.0 * np.pi,
            bottom=0.0,
            color="#ffffff",
            align="edge",
            edgecolor="#cfcfcd",
            linewidth=0.9,
            zorder=1,
        )
        # Spacer ring (track gap)
        ax.bar(
            0.0,
            r_sp1_top - r_hub,
            width=2.0 * np.pi,
            bottom=r_hub,
            color="#ffffff",
            align="edge",
            edgecolor="none",
            zorder=2,
        )

        # Inner dot track (second thin ring of same encoding)
        dot_s = float(np.clip(4000.0 / max(n, 1), 10.0, 52.0))
        ax.scatter(
            theta_centers,
            np.full(n, r_dot),
            c=vals,
            cmap=cmap,
            norm=norm,
            s=dot_s,
            edgecolors="#ffffff",
            linewidths=0.5,
            zorder=5,
        )

        # Spacer before heatmap
        ax.bar(
            0.0,
            r_sp2_top - r_sp2_bot,
            width=2.0 * np.pi,
            bottom=r_sp2_bot,
            color="#ffffff",
            align="edge",
            edgecolor="none",
            zorder=4,
        )

        # Outer heatmap ring (narrow tiles, white angular gaps)
        for i in range(n):
            ax.bar(
                theta_edges[i],
                r_out - r_in,
                width=width,
                bottom=r_in,
                color=face[i],
                align="edge",
                edgecolor="#ffffff",
                linewidth=0.45,
                zorder=3,
            )

        # Outer reference rim (dashed, like Circos)
        ax.plot(
            theta_ring,
            np.full_like(theta_ring, r_rim),
            color="#8c8c8a",
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            zorder=6,
            clip_on=False,
        )
        ax.plot(
            theta_ring,
            np.full_like(theta_ring, r_in),
            color="#b8b8b6",
            linewidth=0.65,
            linestyle=(0, (2, 2)),
            zorder=2,
            clip_on=False,
        )

        # Outer sample IDs (tangential text; stride if many samples to limit overlap)
        sample_labels = query_df["sample_id"].astype(str).tolist()
        r_label = r_rim + 0.055
        stride = 1
        if n > 56:
            stride = max(1, int(np.ceil(n / 56)))
        fs_outer = float(max(3.2, min(6.8, 2100.0 / max(n // max(stride, 1), 1))))
        for i in range(0, n, stride):
            th = theta_centers[i]
            lab = _short_id(sample_labels[i], max_len=20)
            # Tangential rotation (read along the arc, outside the rim)
            rot_deg = np.degrees(th) - 90.0
            ax.text(
                th,
                r_label,
                lab,
                ha="center",
                va="center",
                rotation=rot_deg,
                rotation_mode="anchor",
                fontsize=fs_outer,
                color="#3d3d3b",
                clip_on=False,
                zorder=10,
            )

        ax.set_ylim(0.0, min(0.94, r_label + 0.18))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["polar"].set_visible(False)

        # Center: title + subtitle + discrete legend (horizontal color tiles)
        ax.text(
            0.5,
            0.58,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="semibold",
            color="#1a1a1a",
        )
        ax.text(
            0.5,
            0.48,
            nn_column_label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#555553",
        )
        legend_handles = []
        n_bins = len(boundaries) - 1
        bin_colors = list(_CIRCOS_DISCRETE_HEX[:n_bins])
        for b in range(n_bins):
            lo, hi = boundaries[b], boundaries[b + 1]
            legend_handles.append(
                Patch(
                    facecolor=bin_colors[b],
                    edgecolor="#ffffff",
                    linewidth=0.4,
                    label=f"{lo:.3f}–{hi:.3f}",
                )
            )
        ax.legend(
            handles=legend_handles,
            loc="center",
            bbox_to_anchor=(0.5, 0.30),
            bbox_transform=ax.transAxes,
            ncol=4,
            frameon=True,
            fancybox=False,
            edgecolor="#c8c8c6",
            fontsize=6.5,
            title="Best sequence identity (bins)",
            title_fontsize=7.5,
            labelspacing=0.35,
            handletextpad=0.45,
            columnspacing=0.6,
        )

        fig.subplots_adjust(top=0.94, bottom=0.06)

    _save_fig_png_pdf(out_png, dpi=300)
    plt.close(fig)


def plot_dot_heatmap(
    best_df: pd.DataFrame,
    query_df: pd.DataFrame,
    target_df: pd.DataFrame,
    out_png: str,
    title: str,
    *,
    nn_column_label: str,
    heatmap_scale: str = "data",
) -> None:
    """One-column dot heatmap: fixed-size circles, color = identity (heatmap-style sequential scale)."""
    aligned = best_df.set_index("query_id").reindex(query_df["sample_id"].astype(str))
    vals = pd.to_numeric(aligned["best_identity"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(vals)
    y = np.arange(n, dtype=float)
    x = np.zeros(n)

    dot_area = 118.0
    fig_w = max(5.4, 3.8 + min(n * 0.015, 1.9))
    fig_h = max(7.2, n * 0.175 + 1.3)

    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
        fig.patch.set_facecolor("#f8f8f7")
        ax.set_facecolor("#ffffff")

        vals, norm = _normalize_identity_colors(vals, scale=heatmap_scale)
        sc = ax.scatter(
            x,
            y,
            s=dot_area,
            c=vals,
            cmap=_HEATMAP_CMAP,
            norm=norm,
            edgecolors="#ffffff",
            linewidths=0.55,
            alpha=1.0,
            zorder=4,
        )
        ax.set_xlim(-0.62, 0.62)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.set_xticks([0.0])
        ax.set_xticklabels([nn_column_label], fontsize=9, rotation=38, ha="right", color="#3d3d3d")
        ax.set_yticks(y)
        ax.set_yticklabels(query_df["sample_id"].astype(str).tolist(), fontsize=6.5, color="#444444")
        ax.set_ylabel("Test sample", fontsize=10, color="#2d2d2d")
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=14, color="#141414")
        ax.grid(True, axis="y", linestyle="-", linewidth=0.65, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["left"].set_edgecolor("#c4c4c4")

        cbar = fig.colorbar(sc, ax=ax, shrink=0.86, pad=0.03, aspect=28)
        cbar.set_label("Best sequence identity", fontsize=9.5, color="#2d2d2d")
        cbar.ax.tick_params(labelsize=8.5, colors="#444444")
        cbar.outline.set_linewidth(0.45)
        cbar.outline.set_edgecolor("#bfbfbf")

    _save_fig_png_pdf(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=False, help="Not used directly if splits already contain receptor_sequence")
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--use-mmseqs", action="store_true")
    ap.add_argument(
        "--nn-target",
        choices=("train", "full"),
        default="train",
        help="train: nearest neighbor among that split's training CSV (leakage check). "
        "full: search all_metadata (legacy; protein/family plots match if test lists match).",
    )
    ap.add_argument(
        "--reuse-csv",
        action="store_true",
        help="Load <outdir>/<split>_best_similarity_nn-<nn-target>.csv and only redraw figures; "
        "skips MMseqs and Python similarity (ignores --use-mmseqs).",
    )
    ap.add_argument(
        "--heatmap-style",
        choices=("circos", "rect"),
        default="circos",
        help="circos: circular rings (Circos-style); rect: vertical linear dot heatmap.",
    )
    ap.add_argument(
        "--heatmap-scale",
        choices=("data", "fixed"),
        default="data",
        help="data: color scale fits min–max of identities (recommended for narrow ranges). "
        "fixed: always 0–1 for cross-figure comparison (may look flat if all values are similar).",
    )
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    splits = Path(args.splits_dir)

    all_meta_path = splits / "all_metadata.csv"
    if not all_meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {all_meta_path}")
    all_df = pd.read_csv(all_meta_path)

    split_specs = [
        ("protein_level_test", splits / "protein_level_test.csv", splits / "protein_level_train.csv"),
        ("family_level_test", splits / "family_level_test.csv", splits / "family_level_train.csv"),
    ]

    for split_name, test_file, train_file in split_specs:
        qdf = pd.read_csv(test_file)
        if args.nn_target == "train":
            if not train_file.exists():
                raise FileNotFoundError(f"Missing training split for {split_name}: {train_file}")
            tdf = pd.read_csv(train_file)
            qids = set(qdf["sample_id"].astype(str))
            tdf = tdf[~tdf["sample_id"].astype(str).isin(qids)].copy()
            if tdf.empty:
                raise RuntimeError(f"No training rows left for {split_name} after excluding test IDs.")
            xlabel = "Best receptor sequence identity to training set (same split)"
            nn_col = "NN in training set"
        else:
            tdf = all_df
            xlabel = "Best receptor sequence identity to full augmented set"
            nn_col = "NN in full augmented set"

        csv_path = outdir / f"{split_name}_best_similarity_nn-{args.nn_target}.csv"
        if args.reuse_csv:
            if not csv_path.is_file():
                raise FileNotFoundError(
                    f"--reuse-csv: expected {csv_path}; run without --reuse-csv first to compute similarity."
                )
            best = load_best_similarity_csv(csv_path)
        elif args.use_mmseqs:
            try:
                best = best_similarity_mmseqs(qdf, tdf, str(outdir / split_name))
            except Exception as e:
                print(f"[WARN] MMseqs search failed for {split_name}, fallback to Python similarity. Error: {e}")
                best = best_similarity_python(qdf, tdf)
            best.to_csv(csv_path, index=False)
        else:
            best = best_similarity_python(qdf, tdf)
            best.to_csv(csv_path, index=False)
        plot_similarity(
            best,
            str(outdir / f"{split_name}_similarity_hist.png"),
            f"{split_name}: best receptor vs {args.nn_target}",
            xlabel=xlabel,
        )
        heatmap_path = str(outdir / f"{split_name}_similarity_heatmap.png")
        ht_title = f"{split_name}: nearest-neighbor ({args.nn_target})"
        hs = args.heatmap_scale
        if args.heatmap_style == "circos":
            plot_circos_style_heatmap(best, qdf, tdf, heatmap_path, ht_title, nn_column_label=nn_col, heatmap_scale=hs)
        else:
            plot_dot_heatmap(best, qdf, tdf, heatmap_path, ht_title, nn_column_label=nn_col, heatmap_scale=hs)


if __name__ == "__main__":
    main()

# Example:
# python /root/autodl-tmp/Peptide_3D/results/2_SOTA/01_visualize_similarity.py \
#   --splits-dir /root/autodl-tmp/Peptide_3D/results/2_SOTA/splits \
#   --outdir /root/autodl-tmp/Peptide_3D/results/2_SOTA/similarity \
#   --use-mmseqs --nn-target train
#
# Regenerate figures only (reuse saved CSV, no MMseqs; default heatmap is Circos-style ring):
# python /root/autodl-tmp/Peptide_3D/results/2_SOTA/01_visualize_similarity.py \
#   --splits-dir /root/autodl-tmp/Peptide_3D/results/2_SOTA/splits \
#   --outdir /root/autodl-tmp/Peptide_3D/results/2_SOTA/similarity \
#   --reuse-csv --nn-target train
#   --heatmap-style circos
# Linear dot heatmap instead: add --heatmap-style rect
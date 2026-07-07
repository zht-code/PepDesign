from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.plotting_2sota import (  # noqa: E402
    DATASET_ORDER,
    METHOD_ORDER,
    ensure_dir,
    plot_bar_from_aggregate,
    plot_boxplot_by_method,
    plot_hdock_scatter,
    setup_publication_style,
    summarize_methods_by_dataset,
)


LOGGER = logging.getLogger("plot_2_sota_metrics")


def resolve_input_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.exists() and path_str.startswith("/autodl-tmp/"):
        alt = Path("/root") / path_str.lstrip("/")
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path_str}")
    return path.resolve()


def resolve_output_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.exists() and path_str.startswith("/autodl-tmp/"):
        path = Path("/root") / path_str.lstrip("/")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    setup_logging()
    setup_publication_style()

    parser = argparse.ArgumentParser(description="Plot unified 2_SOTA metrics.")
    parser.add_argument("--metrics-dir", default="/autodl-tmp/Peptide_3D/results/2_SOTA/metrics_summary")
    parser.add_argument("--output-dir", default="/autodl-tmp/Peptide_3D/results/2_SOTA/figures")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--native-affinity-scatter",
        action="store_true",
        help=(
            "After standard figures, run native vs generated HDOCK scatter plots (PPDbench-style) "
            "for family/protein test; requires results/2_SOTA/baseline_data/*_native_hdock.json."
        ),
    )
    args = parser.parse_args()

    metrics_dir = resolve_input_dir(args.metrics_dir)
    output_dir = ensure_dir(resolve_output_dir(args.output_dir))

    per_candidate_df = pd.read_csv(metrics_dir / "per_candidate_metrics.csv")
    aggregate_df = pd.read_csv(metrics_dir / "aggregate_metrics.csv")
    _top5_df = pd.read_csv(metrics_dir / "top5_summary.csv")

    generated: List[str] = []
    skipped: List[str] = []

    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            ok, paths, note = plot_hdock_scatter(per_candidate_df, dataset, method, output_dir, args.dpi)
            if ok:
                generated.extend(paths)
            else:
                skipped.append(f"hdock_scatter_{dataset}_{method}: {note}")

    for dataset in DATASET_ORDER:
        ok, paths, note = plot_boxplot_by_method(
            per_candidate_df=per_candidate_df,
            dataset=dataset,
            metric="contact_consistency",
            ylabel="Contact consistency",
            output_dir=output_dir,
            dpi=args.dpi,
            stem_prefix="contact_consistency_boxplot",
        )
        if ok:
            generated.extend(paths)
        else:
            skipped.append(f"contact_consistency_boxplot_{dataset}: {note}")

    bar_specs = [
        ("plddt", "Mean pLDDT", "plddt_barplot", False),
        ("ramachandran_compliance", "Ramachandran compliance", "ramachandran_barplot", False),
        ("perplexity", "Perplexity", "perplexity_barplot", True),
        ("repetition_rate", "Repetition rate", "repetition_barplot", False),
        ("novelty", "Novelty", "novelty_barplot", False),
    ]
    for dataset in DATASET_ORDER:
        for metric, ylabel, stem_prefix, allow_skip in bar_specs:
            ok, paths, note = plot_bar_from_aggregate(
                aggregate_df=aggregate_df,
                dataset=dataset,
                metric=metric,
                ylabel=ylabel,
                output_dir=output_dir,
                dpi=args.dpi,
                stem_prefix=stem_prefix,
                skip_if_all_nan=allow_skip,
            )
            if ok:
                generated.extend(paths)
            else:
                skipped.append(f"{stem_prefix}_{dataset}: {note}")

    for dataset in DATASET_ORDER:
        ok, paths, note = plot_boxplot_by_method(
            per_candidate_df=per_candidate_df,
            dataset=dataset,
            metric="clash_score",
            ylabel="Clash score",
            output_dir=output_dir,
            dpi=args.dpi,
            stem_prefix="clash_score_boxplot",
        )
        if ok:
            generated.extend(paths)
        else:
            skipped.append(f"clash_score_boxplot_{dataset}: {note}")

    participation = summarize_methods_by_dataset(
        per_candidate_df,
        metrics=[
            "hdock_score",
            "contact_consistency",
            "plddt",
            "ramachandran_compliance",
            "clash_score",
            "perplexity",
            "repetition_rate",
            "novelty",
        ],
    )

    LOGGER.info("Generated %d figure files.", len(generated))
    for path in generated:
        LOGGER.info("Generated: %s", path)

    if skipped:
        LOGGER.info("Skipped %d figure targets due to insufficient data.", len(skipped))
        for item in skipped:
            LOGGER.info("Skipped: %s", item)

    for dataset in DATASET_ORDER:
        LOGGER.info("Methods with finite data in %s: %s", dataset, participation[dataset])

    if args.native_affinity_scatter:
        native_script = PROJECT_ROOT / "results" / "2_SOTA" / "plot_2sota_native_affinity_scatter.py"
        baseline_dir = metrics_dir.parent / "baseline_data"
        cmd = [
            sys.executable,
            str(native_script),
            "--per-candidate-csv",
            str(metrics_dir / "per_candidate_metrics.csv"),
            "--baseline-dir",
            str(baseline_dir),
            "--output-dir",
            str(output_dir),
            "--dpi",
            str(args.dpi),
        ]
        LOGGER.info("Running native affinity scatter script: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()

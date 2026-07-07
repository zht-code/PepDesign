from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

import libtbx.load_env
import libtbx
from iotbx import pdb
from mmtbx.validation import clashscore, ramalyze


LOGGER = logging.getLogger("run_molprobity_metrics_batch")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _prepare_environment() -> None:
    exe_dir = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{exe_dir}:{os.environ.get('PATH', '')}"

    share_probe_dir = exe_dir.parent / "share" / "cctbx" / "probe" / "exe"
    share_probe_dir.mkdir(parents=True, exist_ok=True)
    probe_src = exe_dir / "probe"
    if probe_src.exists():
        probe_dst = share_probe_dir / "probe"
        if not probe_dst.exists():
            probe_dst.symlink_to(probe_src)

    alias_map = {
        "molprobity.reduce": exe_dir / "reduce",
        "molprobity.probe": exe_dir / "probe",
    }
    for alias, src in alias_map.items():
        dst = exe_dir / alias
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    original_has_module = libtbx.env.has_module
    libtbx.env.has_module = lambda name: True if name in {"probe", "reduce"} else original_has_module(name)


def compute_ramachandran_compliance(pdb_path: str) -> float:
    hierarchy = pdb.input(file_name=pdb_path).construct_hierarchy()
    result = ramalyze.ramalyze(hierarchy, outliers_only=False, quiet=True)
    total = result.get_phi_psi_residues_count()
    if not total:
        return float("nan")
    _, fraction = result.get_favored_count_and_fraction()
    return float(fraction)


def reduce_to_hydrogenated_pdb(pdb_path: str, reduced_output_path: Path) -> str:
    if reduced_output_path.exists() and reduced_output_path.stat().st_size > 0:
        return str(reduced_output_path.resolve())
    reduce_bin = Path(sys.executable).resolve().parent / "molprobity.reduce"
    proc = subprocess.run([str(reduce_bin), pdb_path], capture_output=True, text=True)
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(f"reduce failed for {pdb_path}: rc={proc.returncode}, stderr={proc.stderr[-1000:]}")
    reduced_output_path.parent.mkdir(parents=True, exist_ok=True)
    reduced_output_path.write_text(proc.stdout, encoding="utf-8")
    return str(reduced_output_path.resolve())


def compute_clash_score(pdb_path: str, reduced_output_path: Path) -> float:
    reduced_pdb = reduce_to_hydrogenated_pdb(pdb_path, reduced_output_path)
    hierarchy = pdb.input(file_name=reduced_pdb).construct_hierarchy()
    result = clashscore.clashscore(hierarchy, keep_hydrogens=True, verbose=False)
    return float(result.get_clashscore())


def main() -> None:
    _setup_logging()
    _prepare_environment()

    parser = argparse.ArgumentParser(description="Batch MolProbity Ramachandran and clashscore calculation.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    reduced_cache_dir = output_csv.parent / "reduced_pdbs"
    reduced_cache_dir.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows_in = list(csv.DictReader(handle))
    fieldnames = ["query_id", "peptide_only_pdb", "ramachandran_compliance", "clash_score"]
    rows = []
    completed_query_ids = set()
    if output_csv.exists():
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
        rows.extend(existing_rows)
        completed_query_ids = {str(row["query_id"]) for row in existing_rows}
        LOGGER.info("Resuming MolProbity batch with %d completed rows", len(completed_query_ids))

    def flush_rows() -> None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    for idx, row in enumerate(rows_in, start=1):
        query_id = str(row["query_id"])
        if query_id in completed_query_ids:
            continue
        pdb_path = str(row["pdb_path"])
        rama = float("nan")
        clash = float("nan")
        try:
            rama = compute_ramachandran_compliance(pdb_path)
        except Exception as exc:
            LOGGER.warning("Ramachandran failed for %s: %s", query_id, exc)
        try:
            clash = compute_clash_score(pdb_path, reduced_cache_dir / f"{query_id.replace('|', '_')}__reduced.pdb")
        except Exception as exc:
            LOGGER.warning("Clashscore failed for %s: %s", query_id, exc)
        rows.append(
            {
                "query_id": query_id,
                "peptide_only_pdb": pdb_path,
                "ramachandran_compliance": rama,
                "clash_score": clash,
            }
        )
        if idx % 100 == 0:
            LOGGER.info("Processed %d / %d structures", idx, len(rows_in))
            flush_rows()

    flush_rows()
    LOGGER.info("Saved MolProbity metrics to %s", output_csv)


if __name__ == "__main__":
    main()

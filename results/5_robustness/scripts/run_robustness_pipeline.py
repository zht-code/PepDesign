#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter 5 robustness: final model under target perturbations (PPDbench test set).

All outputs live under results/5_robustness/ (see README.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# project roots
_SCRIPTS = Path(__file__).resolve().parent
_ROB = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from robustness_lib.paths import PROJECT_ROOT, ensure_subdirs  # noqa: E402
from robustness_lib.metrics_eval import load_thresholds, to_higher_better  # noqa: E402
from robustness_lib.aggregate_metrics import robustness_summary_row  # noqa: E402
from robustness_lib.condition_loop import (  # noqa: E402
    condition_fully_recorded,
    merge_part_csvs,
    mp_worker_run,
    run_condition_on_targets,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_PARETO = PROJECT_ROOT / "results" / "3_Pareto_improved"
if str(_PARETO) not in sys.path:
    sys.path.insert(0, str(_PARETO))

from models_DPO import ProteinPeptideModel  # noqa: E402

def _setup_log(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _export_case_candidates(df: pd.DataFrame, dirs: dict[str, Path], logger: logging.Logger) -> None:
    """Targets with favorable clean HDOCK and clear loss under strong structure missing."""
    try:
        sub = df[df["perturb_type"] == "structure_missing"]
        if sub.empty:
            return
        lv_max = float(sub["level_value"].max())
        d0 = sub[np.isclose(sub["level_value"].astype(float), 0.0)]
        d1 = sub[np.isclose(sub["level_value"].astype(float), lv_max)]
        d0 = d0[d0["affinity_hdock"].notna()]
        d1 = d1[d1["affinity_hdock"].notna()]
        merged = d0.merge(d1, on="target_id", suffixes=("_clean", "_pert"))
        if merged.empty:
            return
        merged["hdock_worsening"] = merged["affinity_hdock_pert"].astype(float) - merged["affinity_hdock_clean"].astype(
            float
        )
        merged = merged[merged["affinity_hdock_clean"].astype(float) < -5.0]
        merged = merged.sort_values("hdock_worsening", ascending=False)
        recs = merged.head(2).to_dict(orient="records")
        path = dirs["cases"] / "selected_cases.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2)
        logger.info("case candidates -> %s (n=%d)", path, len(recs))
    except Exception as e:
        logger.warning("case export skipped: %s", e)


def _targets(bench_root: Path, limit: int | None) -> list[Path]:
    dirs = sorted([p for p in bench_root.iterdir() if p.is_dir() and (p / "receptor.pdb").is_file()])
    if limit is not None:
        dirs = dirs[: int(limit)]
    return dirs


def condition_tag(pert: str, level: float, rep: int) -> str:
    lv = str(level).replace(".", "p")
    return f"{pert}_lvl{lv}_r{rep}"


def run(args: argparse.Namespace) -> int:
    dirs = ensure_subdirs()
    logger = _setup_log(dirs["logs"], "robustness")
    defaults_path = Path(args.defaults_config)
    if not defaults_path.is_file():
        defaults_path = _ROB / "configs" / "pipeline_defaults.json"
    defaults = _load_json(defaults_path)
    th_raw = load_thresholds(Path(args.thresholds_config))
    th = th_raw["success_criteria"]

    bench_root = Path(args.bench_root or defaults["bench_root"])
    ckpt = Path(args.ckpt or defaults["final_model_checkpoint"])
    encoder_mode = args.encoder_mode or defaults.get("encoder_mode", "geometry")
    pocket_r = float(args.pocket_radius_A or defaults.get("pocket_radius_A", 10.0))
    min_res = int(args.min_receptor_residues or defaults.get("min_receptor_residues_after_perturb", 8))
    n_rep = int(args.n_repeats if args.n_repeats is not None else defaults.get("n_repeats_default", 3))

    struct_levels = [float(x) for x in (args.structure_levels or defaults["structure_missing_levels_pct"])]
    pocket_levels = [float(x) for x in (args.pocket_levels or defaults["pocket_noise_levels_A"])]
    seq_levels = [float(x) for x in (args.sequence_levels or defaults["sequence_trunc_levels_pct"])]

    pert_map = {
        "structure_missing": struct_levels,
        "pocket_noise": pocket_levels,
        "sequence_trunc": seq_levels,
    }

    if args.perturbation_type != "all":
        pert_map = {args.perturbation_type: pert_map[args.perturbation_type]}

    gpu_start = int(args.gpu)
    if str(args.device).lower() == "cpu":
        use_cuda = False
    elif args.device == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = torch.cuda.is_available()

    num_gpus = int(args.num_gpus)
    if num_gpus == 0:
        num_gpus = torch.cuda.device_count() if use_cuda else 1
    if not use_cuda:
        num_gpus = 1
    num_gpus = max(1, num_gpus)
    if use_cuda:
        avail = torch.cuda.device_count()
        if gpu_start + num_gpus > avail:
            num_gpus = max(1, avail - gpu_start)
            logger.warning("num_gpus 已限制为 %d（gpu_start=%d，可见 GPU=%d）", num_gpus, gpu_start, avail)

    if args.eval_workers is not None:
        eval_w = max(1, int(args.eval_workers))
    elif int(getattr(args, "num_workers", 0) or 0) > 0:
        eval_w = max(1, int(args.num_workers))
    else:
        c = os.cpu_count() or 8
        eval_w = max(1, min(16, (c - 1) // max(1, num_gpus)))

    resume = not args.no_resume

    device: torch.device | None = None
    model: ProteinPeptideModel | None = None
    if num_gpus == 1:
        if args.device == "auto":
            device = torch.device(f"cuda:{gpu_start}" if use_cuda else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda":
            idx = device.index if device.index is not None else gpu_start
            torch.cuda.set_device(idx)

    cache_pep = dirs["cache"] / "peptides"
    tmp_placement = dirs["tmp"] / "placement_pdb"
    hdock_root = Path(args.hdock_work_root)
    foldx_root = Path(args.foldx_work_root)

    logger.info("PROJECT_ROOT=%s", PROJECT_ROOT)
    logger.info("checkpoint=%s exists=%s", ckpt, ckpt.is_file())
    logger.info("bench_root=%s encoder_mode=%s", bench_root, encoder_mode)
    logger.info("perturbations=%s", list(pert_map.keys()))
    logger.info(
        "parallel: num_gpus=%d eval_workers=%d resume=%s gpu_start=%d",
        num_gpus,
        eval_w,
        resume,
        gpu_start,
    )

    if args.only_aggregate:
        return aggregate_only(dirs, logger)

    if args.only_plot:
        return plot_only(dirs, logger)

    if not ckpt.is_file():
        logger.error("Checkpoint missing: %s", ckpt)
        return 1

    if num_gpus == 1:
        assert device is not None
        model = ProteinPeptideModel(device)
        state = torch.load(str(ckpt), map_location=device)
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=False)
        # ESM3 在 models_DPO 里默认 load_local_model(..., device=cpu)，若不整模 .to(device)，会出现
        # sequence_tokens 在 CUDA、esm3 权重在 CPU → index_select 报 cpu/cuda 混用。
        model.to(device)
        model.eval()

    targets = _targets(bench_root, args.max_targets)
    logger.info("targets=%d", len(targets))

    metric_name = args.metric_filter or "all"

    all_rows: list[dict] = []

    for pert, levels in pert_map.items():
        if args.level is not None and float(args.level) not in levels:
            continue
        filt_levels = [float(args.level)] if args.level is not None else levels
        for level in filt_levels:
            for rep in range(n_rep):
                if args.repeat is not None and int(rep) != int(args.repeat):
                    continue
                tag = condition_tag(pert, level, rep)
                out_csv = dirs["tables"] / f"samples_{tag}.csv"
                if args.skip_existing and out_csv.is_file():
                    logger.info("skip existing %s", out_csv)
                    df_prev = pd.read_csv(out_csv)
                    all_rows.extend(df_prev.to_dict(orient="records"))
                    continue
                if resume and condition_fully_recorded(out_csv, targets):
                    logger.info("resume: 条件已完成，跳过 %s", out_csv.name)
                    df_prev = pd.read_csv(out_csv)
                    all_rows.extend(df_prev.to_dict(orient="records"))
                    continue

                rng_seed = int(args.seed) + rep * 1000 + int(level * 17) + hash(pert) % 10000

                if num_gpus == 1:
                    assert model is not None and device is not None
                    rng = np.random.default_rng(rng_seed)
                    rows = run_condition_on_targets(
                        model=model,
                        device=device,
                        targets=targets,
                        out_csv=out_csv,
                        resume=resume,
                        eval_workers=eval_w,
                        logger=logger,
                        pert=pert,
                        level=level,
                        rep=rep,
                        tag=tag,
                        rng=rng,
                        encoder_mode=encoder_mode,
                        pocket_r=pocket_r,
                        min_res=min_res,
                        args=args,
                        cache_pep=cache_pep,
                        tmp_placement=tmp_placement,
                        hdock_root=hdock_root,
                        foldx_root=foldx_root,
                        th=th,
                    )
                    all_rows.extend(rows)
                else:
                    ctx = get_context("spawn")
                    procs: list = []
                    args_ns = vars(args)
                    ckpt_default = str(ckpt)
                    for r in range(num_gpus):
                        shard = [targets[i] for i in range(r, len(targets), num_gpus)]
                        if not shard:
                            continue
                        out_part = dirs["tables"] / f"samples_{tag}_part{r:02d}.csv"
                        payload: dict[str, Any] = {
                            "rank": r,
                            "world_size": num_gpus,
                            "target_paths": [str(p) for p in shard],
                            "out_csv": str(out_part),
                            "gpu_id": gpu_start + r,
                            "use_cuda": use_cuda,
                            "log_path": str(dirs["logs"] / f"robustness_{tag}_gpu{r}.log"),
                            "args_ns": args_ns,
                            "ckpt_default": ckpt_default,
                            "cache_pep": str(cache_pep),
                            "tmp_placement": str(tmp_placement),
                            "hdock_root": str(hdock_root),
                            "foldx_root": str(foldx_root),
                            "th": th,
                            "resume": resume,
                            "eval_workers": eval_w,
                            "pert": pert,
                            "level": float(level),
                            "rep": int(rep),
                            "tag": tag,
                            "rng_seed": rng_seed + r * 7919,
                            "encoder_mode": encoder_mode,
                            "pocket_r": pocket_r,
                            "min_res": min_res,
                        }
                        proc = ctx.Process(target=mp_worker_run, args=(payload,))
                        proc.start()
                        procs.append(proc)
                    exit_codes = []
                    for proc in procs:
                        proc.join()
                        exit_codes.append(proc.exitcode)
                    if any(c != 0 for c in exit_codes):
                        logger.error("多卡子进程异常退出 codes=%s", exit_codes)
                        return 1
                    merge_part_csvs(dirs["tables"], tag, logger)
                    df_merged = pd.read_csv(out_csv)
                    all_rows.extend(df_merged.to_dict(orient="records"))

    # Save combined samples
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv(dirs["tables"] / "robustness_all_samples.csv", index=False)

    if not args.no_aggregate:
        aggregate_tables(dirs, pert_map, th, logger, metric_name)

    if not args.no_plot:
        plot_only(dirs, logger)

    return 0


def aggregate_tables(
    dirs: dict[str, Path],
    pert_map: dict,
    th: dict,
    logger: logging.Logger,
    metric_filter: str,
) -> None:
    samples = sorted(dirs["tables"].glob("samples_*.csv"))
    if not samples:
        logger.warning("no samples_*.csv for aggregation")
        return
    frames = [pd.read_csv(p) for p in samples]
    df = pd.concat(frames, ignore_index=True)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df = df.dropna(subset=["affinity_hdock"], how="all")

    df.to_csv(dirs["tables"] / "robustness_all_samples_merged.csv", index=False)

    # Per condition mean over targets, then over repeats
    metrics = ["affinity_hdock", "stability", "solubility", "success_triple"]
    agg_rows = []
    for (pert, lv), g in df.groupby(["perturb_type", "level_value"]):
        sub = g.copy()
        if "success_triple" in sub.columns:
            def _succ_val(x):
                if x is True or x == 1 or str(x).lower() == "true":
                    return 1.0
                if x is False or x == 0 or str(x).lower() == "false":
                    return 0.0
                return np.nan

            sub["success_triple"] = sub["success_triple"].apply(_succ_val)
        m = sub.groupby("repeat_id")[metrics].mean(numeric_only=True).mean()
        agg_rows.append(
            {
                "perturb_type": pert,
                "level_value": float(lv),
                "n_targets": int(sub["target_id"].nunique()),
                "affinity_mean": m.get("affinity_hdock", np.nan),
                "stability_mean": m.get("stability", np.nan),
                "solubility_mean": m.get("solubility", np.nan),
                "success_rate": m.get("success_triple", np.nan),
            }
        )
    agg_df = pd.DataFrame(agg_rows).sort_values(["perturb_type", "level_value"])
    agg_df.to_csv(dirs["tables"] / "robustness_aggregate_by_condition.csv", index=False)

    # Summary Table_5
    summary = []
    norm_fn = {
        "structure_missing": lambda x: x / 40.0,
        "pocket_noise": lambda x: x / 2.0,
        "sequence_trunc": lambda x: x / 40.0,
    }
    for pert in pert_map:
        sub = agg_df[agg_df["perturb_type"] == pert].sort_values("level_value")
        if sub.empty:
            continue
        levels = sub["level_value"].astype(float).values
        xn = np.array([norm_fn[pert](v) for v in levels])
        for metric_col, mname in [
            ("affinity_mean", "affinity_hdock"),
            ("success_rate", "success_rate"),
            ("stability_mean", "stability"),
            ("solubility_mean", "solubility"),
        ]:
            if metric_filter != "all" and mname != metric_filter:
                continue
            vals = sub[metric_col].astype(float).values
            clean_mask = np.isclose(levels, levels.min()) | np.isclose(levels, 0.0)
            clean_raw = float(vals[clean_mask][0]) if clean_mask.any() else float(vals[0])
            if mname == "affinity_hdock":
                clean_b = float(to_higher_better("affinity_hdock", clean_raw))
                vals_b = np.array(
                    [float(to_higher_better("affinity_hdock", float(v)) or np.nan) for v in vals],
                    dtype=float,
                )
                note = (
                    "clean_mean/max_drop/AUDC use negated HDOCK (higher=better); "
                    "report raw HDOCK in robustness_aggregate_by_condition.csv"
                )
            else:
                clean_b = clean_raw
                vals_b = vals.astype(float)
                note = ""
            row = robustness_summary_row(pert, mname, levels, clean_b, vals_b, xn)
            row["notes"] = note
            summary.append(row)
    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(dirs["tables"] / "Table_5_robustness_summary.csv", index=False)
    sum_df.to_json(dirs["metrics"] / "Table_5_robustness_summary.json", orient="records", indent=2)
    logger.info("aggregate saved (Table_5_robustness_summary.csv)")

    for pert in pert_map:
        subp = agg_df[agg_df["perturb_type"] == pert]
        if not subp.empty:
            subp.to_csv(dirs["tables"] / f"robustness_aggregate_{pert}.csv", index=False)

    _export_case_candidates(df, dirs, logger)


def aggregate_only(dirs: dict[str, Path], logger: logging.Logger) -> int:
    defaults_path = _ROB / "configs" / "pipeline_defaults.json"
    defaults = _load_json(defaults_path)
    th_raw = load_thresholds(_ROB / "configs" / "thresholds.json")
    th = th_raw["success_criteria"]
    pert_map = {
        "structure_missing": [float(x) for x in defaults["structure_missing_levels_pct"]],
        "pocket_noise": [float(x) for x in defaults["pocket_noise_levels_A"]],
        "sequence_trunc": [float(x) for x in defaults["sequence_trunc_levels_pct"]],
    }
    aggregate_tables(dirs, pert_map, th, logger, "all")
    return 0


def plot_only(dirs: dict[str, Path], logger: logging.Logger) -> int:
    import subprocess

    plot_py = _SCRIPTS / "plot_robustness_figure.py"
    r = subprocess.run([sys.executable, str(plot_py)], cwd=str(_SCRIPTS))
    if r.returncode != 0:
        logger.error("plot script failed")
    return r.returncode


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robustness pipeline (final model, PPDbench)")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--bench-root", type=str, default=None)
    p.add_argument("--defaults-config", type=str, default=str(_ROB / "configs" / "pipeline_defaults.json"))
    p.add_argument("--thresholds-config", type=str, default=str(_ROB / "configs" / "thresholds.json"))
    p.add_argument("--encoder-mode", type=str, choices=["geometry", "sequence_only"], default=None)
    p.add_argument("--pocket-radius-A", type=float, default=None)
    p.add_argument("--min-receptor-residues", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-repeats", type=int, default=None)
    p.add_argument("--repeat", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--gpu", type=int, default=0, help="起始 GPU 索引；多卡时为 cuda:{gpu}, cuda:{gpu+1}, …")
    p.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="并行 GPU 进程数；0 表示使用当前可见的全部 GPU。CPU 推理时固定为 1。",
    )
    p.add_argument(
        "--eval-workers",
        type=int,
        default=None,
        help="每进程内对接/物化评测的并发线程数；默认按 CPU 核数与 GPU 数自动分配（单卡约拉满，多卡会均分 CPU）。",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="关闭断点续跑：不跳过已成功靶点（将重写同一路径 CSV）。",
    )
    p.add_argument("--max-targets", type=int, default=None, help="debug: only first N targets")
    p.add_argument("--num-per-target", type=int, default=3)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--max-len", type=int, default=30)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--oversample-factor", type=int, default=3)
    p.add_argument("--perturbation-type", type=str, default="all",
                   choices=["all", "structure_missing", "pocket_noise", "sequence_trunc"])
    p.add_argument("--level", type=float, default=None)
    p.add_argument("--structure-levels", type=float, nargs="*", default=None)
    p.add_argument("--pocket-levels", type=float, nargs="*", default=None)
    p.add_argument("--sequence-levels", type=float, nargs="*", default=None)
    p.add_argument("--metric-filter", type=str, default="all")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-aggregate", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--only-aggregate", action="store_true")
    p.add_argument("--only-plot", action="store_true")
    p.add_argument("--hdock-bin", type=str, default="/root/autodl-fs/HDOCKlite/hdock")
    p.add_argument("--createpl-bin", type=str, default="/root/autodl-fs/HDOCKlite/createpl")
    p.add_argument("--foldx-bin", type=str, default="/root/autodl-tmp/foldx_20270131")
    p.add_argument("--proteinsol-wrapper", type=str,
                   default="/root/autodl-tmp/protein-sol/multiple_prediction_wrapper_export.sh")
    p.add_argument("--hdock-work-root", type=str, default=str(_ROB / "cache" / "hdock_work"))
    p.add_argument("--foldx-work-root", type=str, default=str(_ROB / "cache" / "foldx_work"))
    p.add_argument("--hdock-timeout", type=int, default=900)
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="保留参数：当前按靶点顺序生成；未使用。",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="已弃用：请使用 --eval-workers。",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_argparser().parse_args()))

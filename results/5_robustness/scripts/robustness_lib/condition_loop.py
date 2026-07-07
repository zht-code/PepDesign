"""单条件靶点循环：GPU 串行生成 + 线程池并行评测；多进程 spawn 入口 mp_worker_run。"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .generate_target import generate_for_target, write_peptide_pdbs
from .metrics_eval import evaluate_peptide, success_triple
from .paths import PROJECT_ROOT

_PARETO = PROJECT_ROOT / "results" / "3_Pareto_improved"
if str(_PARETO) not in sys.path:
    sys.path.insert(0, str(_PARETO))

from models_DPO import ProteinPeptideModel  # noqa: E402


def dedupe_rows_by_target(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    if "target_id" not in df.columns:
        return rows
    df = df.drop_duplicates(subset=["target_id"], keep="last")
    return df.sort_values("target_id").to_dict(orient="records")


def write_samples_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dedupe_rows_by_target(rows)).to_csv(path, index=False)


def successful_target_ids(rows: list[dict]) -> set[str]:
    if not rows:
        return set()
    df = pd.DataFrame(rows)
    if "target_id" not in df.columns:
        return set()
    if "error" in df.columns:
        mask = df["error"].isna() | (df["error"].astype(str) == "")
        return set(df.loc[mask, "target_id"].dropna().astype(str))
    return set(df["target_id"].dropna().astype(str))


def load_resume_rows(csv_path: Path) -> tuple[list[dict], set[str]]:
    if not csv_path.is_file():
        return [], set()
    df = pd.read_csv(csv_path)
    rows = df.to_dict(orient="records")
    return rows, successful_target_ids(rows)


def condition_fully_recorded(csv_path: Path, targets: list[Path]) -> bool:
    if not csv_path.is_file():
        return False
    _, done = load_resume_rows(csv_path)
    want = {t.name for t in targets}
    return want <= done


def merge_part_csvs(tables_dir: Path, tag: str, logger: logging.Logger) -> Path:
    pattern = f"samples_{tag}_part*.csv"
    parts = sorted(tables_dir.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"no shard files matching {pattern} under {tables_dir}")
    frames = [pd.read_csv(p) for p in parts]
    merged = pd.concat(frames, ignore_index=True)
    if "target_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["target_id"], keep="last")
    out = tables_dir / f"samples_{tag}.csv"
    merged.sort_values("target_id").to_csv(out, index=False)
    for p in parts:
        try:
            p.unlink()
        except OSError as e:
            logger.warning("could not remove shard %s: %s", p, e)
    logger.info("merged %d shard(s) -> %s rows=%d", len(parts), out, len(merged))
    return out


def eval_build_row(
    *,
    tid: str,
    rep: int,
    pert: str,
    level: float,
    tag: str,
    encoder_mode: str,
    rec_orig: Path,
    pep1: Path,
    seq0: str,
    hdock_bin: str,
    createpl_bin: str,
    foldx_bin: str,
    proteinsol_wrapper: str,
    hdock_root: Path,
    foldx_root: Path,
    hdock_timeout: int,
    th: dict,
) -> dict[str, Any]:
    try:
        ev = evaluate_peptide(
            receptor_pdb=rec_orig,
            peptide_pdb=pep1,
            hdock_bin=hdock_bin,
            createpl_bin=createpl_bin,
            foldx_bin=foldx_bin,
            proteinsol_wrapper=proteinsol_wrapper,
            hdock_work_root=hdock_root / tag / tid,
            foldx_work_root=foldx_root / tag / tid,
            hdock_timeout=int(hdock_timeout),
        )
        succ = success_triple(ev["affinity_hdock"], ev["stability"], ev["solubility"], th)
        return {
            "target_id": tid,
            "repeat_id": rep,
            "perturb_type": pert,
            "level_value": float(level),
            "condition_tag": tag,
            "peptide_path": str(pep1),
            "sequence_top1": seq0,
            "affinity_hdock": ev["affinity_hdock"],
            "stability": ev["stability"],
            "solubility": ev["solubility"],
            "success_triple": succ,
            "encoder_mode": encoder_mode,
        }
    except Exception as e:
        return {
            "target_id": tid,
            "repeat_id": rep,
            "perturb_type": pert,
            "level_value": float(level),
            "condition_tag": tag,
            "error": str(e),
        }


def run_condition_on_targets(
    *,
    model: ProteinPeptideModel,
    device: torch.device,
    targets: list[Path],
    out_csv: Path,
    resume: bool,
    eval_workers: int,
    logger: logging.Logger,
    pert: str,
    level: float,
    rep: int,
    tag: str,
    rng: np.random.Generator,
    encoder_mode: str,
    pocket_r: float,
    min_res: int,
    args: argparse.Namespace,
    cache_pep: Path,
    tmp_placement: Path,
    hdock_root: Path,
    foldx_root: Path,
    th: dict,
) -> list[dict]:
    rows: list[dict] = []
    done_ids: set[str] = set()
    if resume:
        rows, done_ids = load_resume_rows(out_csv)
        if done_ids:
            logger.info("resume %s: %d targets already done", out_csv.name, len(done_ids))

    rows_lock = threading.Lock()
    eval_workers = max(1, int(eval_workers))

    def _append_and_save(row: dict) -> None:
        with rows_lock:
            rows.append(row)
            write_samples_csv(out_csv, rows)

    pending: set = set()
    with ThreadPoolExecutor(max_workers=eval_workers) as ex:
        for tdir in targets:
            tid = tdir.name
            if tid in done_ids:
                continue

            while len(pending) >= eval_workers:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                pending -= done
                for fut in done:
                    _append_and_save(fut.result())

            pk = "clean" if float(level) == 0.0 else pert
            lv = 0.0 if pk == "clean" else float(level)
            try:
                seqs, pdb_pl = generate_for_target(
                    model,
                    str(tdir),
                    encoder_mode=encoder_mode,
                    perturb_kind=pk,
                    level_value=lv,
                    rng=rng,
                    pocket_radius_A=pocket_r,
                    min_residues=min_res,
                    num_keep=int(args.num_per_target),
                    top_k=int(args.top_k),
                    max_len=int(args.max_len),
                    temperature=float(args.temperature),
                    oversample_factor=int(args.oversample_factor),
                    tmp_root=tmp_placement,
                )
                out_dir = cache_pep / tag / tid
                write_peptide_pdbs(seqs, receptor_pdb_for_placement=pdb_pl, out_dir=out_dir)
                pep1 = out_dir / "pep_01.pdb"
                rec_orig = tdir / "receptor.pdb"
                if not pep1.is_file():
                    raise RuntimeError("pep_01 missing")
                seq0 = seqs[0] if seqs else ""
                fut = ex.submit(
                    eval_build_row,
                    tid=tid,
                    rep=rep,
                    pert=pert,
                    level=level,
                    tag=tag,
                    encoder_mode=encoder_mode,
                    rec_orig=rec_orig,
                    pep1=pep1,
                    seq0=seq0,
                    hdock_bin=args.hdock_bin,
                    createpl_bin=args.createpl_bin,
                    foldx_bin=args.foldx_bin,
                    proteinsol_wrapper=args.proteinsol_wrapper,
                    hdock_root=hdock_root,
                    foldx_root=foldx_root,
                    hdock_timeout=int(args.hdock_timeout),
                    th=th,
                )
                pending.add(fut)
            except Exception as e:
                logger.exception("target %s failed: %s", tid, e)
                _append_and_save(
                    {
                        "target_id": tid,
                        "repeat_id": rep,
                        "perturb_type": pert,
                        "level_value": float(level),
                        "condition_tag": tag,
                        "error": str(e),
                    }
                )

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            pending -= done
            for fut in done:
                _append_and_save(fut.result())

    logger.info("wrote %s rows=%d", out_csv, len(dedupe_rows_by_target(rows)))
    return dedupe_rows_by_target(rows)


def mp_worker_run(payload: dict[str, Any]) -> None:
    """子进程入口（multiprocessing spawn）：须位于可导入模块以便 pickle。"""
    rank = int(payload["rank"])
    log_path = Path(payload["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    wlog = logging.getLogger(f"robustness.gpu{rank}")
    wlog.setLevel(logging.INFO)
    wlog.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    wlog.addHandler(fh)

    args = argparse.Namespace(**payload["args_ns"])
    target_paths = [Path(p) for p in payload["target_paths"]]
    out_csv = Path(payload["out_csv"])
    gpu_id = int(payload["gpu_id"])
    use_cuda = bool(payload["use_cuda"])

    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else gpu_id)

    ckpt = Path(args.ckpt or payload["ckpt_default"])
    if not ckpt.is_file():
        wlog.error("Checkpoint missing: %s", ckpt)
        raise SystemExit(1)

    model = ProteinPeptideModel(device)
    state = torch.load(str(ckpt), map_location=device)
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    cache_pep = Path(payload["cache_pep"])
    tmp_placement = Path(payload["tmp_placement"])
    hdock_root = Path(payload["hdock_root"])
    foldx_root = Path(payload["foldx_root"])
    th = payload["th"]

    rng = np.random.default_rng(int(payload["rng_seed"]))
    run_condition_on_targets(
        model=model,
        device=device,
        targets=target_paths,
        out_csv=out_csv,
        resume=bool(payload["resume"]),
        eval_workers=int(payload["eval_workers"]),
        logger=wlog,
        pert=str(payload["pert"]),
        level=float(payload["level"]),
        rep=int(payload["rep"]),
        tag=str(payload["tag"]),
        rng=rng,
        encoder_mode=str(payload["encoder_mode"]),
        pocket_r=float(payload["pocket_r"]),
        min_res=int(payload["min_res"]),
        args=args,
        cache_pep=cache_pep,
        tmp_placement=tmp_placement,
        hdock_root=hdock_root,
        foldx_root=foldx_root,
        th=th,
    )

#!/usr/bin/env python3
"""Export PDBs for targets where Ours beats all four baselines on affinity_hdock (more negative = better)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

BASE = Path("/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline").resolve()
ROB = Path("/root/autodl-tmp/Peptide_3D/results/5_robustness").resolve()
OUT = BASE / "cases"
TABLES_OURS = ROB / "tables"
RAW = BASE / "raw_results"
# Perturbed receptor PDBs (shared across methods for a given condition + target)
PERTURBED_TARGETS = BASE / "cache" / "perturbed_targets"

BASELINES = ["rfdiffusion", "proteingenerator", "bindcraft"]

# (perturbation_type, level_value, label for subfolder)
CONDITIONS: list[tuple[str, float, str]] = [
    ("structure_missing", 20.0, "structure_missing_light_20pct"),
    ("structure_missing", 40.0, "structure_missing_heavy_40pct"),
    ("sequence_trunc", 20.0, "sequence_trunc_light_20pct"),
    ("sequence_trunc", 40.0, "sequence_trunc_heavy_40pct"),
    ("pocket_noise", 1.0, "pocket_noise_light_1p0A"),
    ("pocket_noise", 2.0, "pocket_noise_heavy_2p0A"),
]


def lvl_token(level: float) -> str:
    """Match table filenames: 20.0 -> 20p0, 1.0 -> 1p0, 0.5 -> 0p5."""
    if float(level).is_integer():
        return f"{int(level)}p0"
    return f"{float(level):.1f}".replace(".", "p")


def samples_path_ours(pert: str, level: float) -> Path:
    return TABLES_OURS / f"samples_{pert}_lvl{lvl_token(level)}_r0.csv"


def samples_path_baseline(method: str, pert: str, level: float) -> Path:
    return RAW / method / f"samples_{pert}_lvl{lvl_token(level)}_r0.csv"


def load_ours_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "perturb_type": "perturbation_type",
            "peptide_path": "pdb_path",
        }
    )
    if "method" not in df.columns:
        df["method"] = "ours"
    return df


def affinity_series(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df["affinity_hdock"], errors="coerce")
    return pd.Series(s.values, index=df["target_id"].astype(str).str.lower())


def pdb_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df["pdb_path"].astype(str).values, index=df["target_id"].astype(str).str.lower())


def main() -> None:
    intersect_path = BASE / "tables" / "intersection_targets.csv"
    targets: list[str] | None = None
    if intersect_path.is_file():
        targets = pd.read_csv(intersect_path)["target_id"].astype(str).str.lower().tolist()

    summary_rows: list[dict] = []

    for pert, level, label in CONDITIONS:
        p_ours = samples_path_ours(pert, level)
        if not p_ours.is_file():
            print(f"SKIP {label}: missing {p_ours}")
            continue

        df_o = load_ours_df(p_ours)
        cond_tag = str(df_o["condition_tag"].iloc[0])
        aff_o = affinity_series(df_o)
        pdb_o = pdb_series(df_o)

        baseline_aff: dict[str, pd.Series] = {}
        baseline_pdb: dict[str, pd.Series] = {}
        skip_cond = False
        for m in BASELINES:
            p = samples_path_baseline(m, pert, level)
            if not p.is_file():
                print(f"SKIP {label}: missing baseline {p}")
                skip_cond = True
                break
            d = pd.read_csv(p)
            baseline_aff[m] = affinity_series(d)
            baseline_pdb[m] = pdb_series(d)
        if skip_cond:
            continue

        cand = set(aff_o.dropna().index)
        for m in BASELINES:
            cand &= set(baseline_aff[m].dropna().index)
        if targets is not None:
            cand &= set(targets)

        winners: list[str] = []
        for tid in sorted(cand):
            o = float(aff_o.loc[tid])
            others = [float(baseline_aff[m].loc[tid]) for m in BASELINES]
            if all(o < x for x in others):
                winners.append(tid)

        cond_dir = OUT / label
        cond_dir.mkdir(parents=True, exist_ok=True)

        for tid in winners:
            tdir = cond_dir / tid
            tdir.mkdir(parents=True, exist_ok=True)
            src = pdb_o.loc[tid]
            shutil.copy2(src, tdir / "ours.pdb")
            for m in BASELINES:
                shutil.copy2(baseline_pdb[m].loc[tid], tdir / f"{m}.pdb")
            recv_src = PERTURBED_TARGETS / cond_tag / tid / "receptor_perturbed.pdb"
            recv_ok = recv_src.is_file()
            if recv_ok:
                shutil.copy2(recv_src, tdir / "target_receptor_perturbed.pdb")
            else:
                print(f"WARN missing receptor {recv_src} ({label}/{tid})")
            o_aff = float(aff_o.loc[tid])
            summary_rows.append(
                {
                    "condition": label,
                    "condition_tag": cond_tag,
                    "perturbation_type": pert,
                    "level_value": level,
                    "target_id": tid,
                    "target_receptor_source": str(recv_src) if recv_ok else "",
                    "target_receptor_copied": recv_ok,
                    "affinity_ours": o_aff,
                    **{f"affinity_{m}": float(baseline_aff[m].loc[tid]) for m in BASELINES},
                }
            )

        print(f"{label}: {len(winners)} targets; wrote under {cond_dir}")

    sum_df = pd.DataFrame(summary_rows)
    sum_path = OUT / "winning_cases_summary.csv"
    sum_df.to_csv(sum_path, index=False)
    print(f"Wrote {sum_path} ({len(sum_df)} rows)")


if __name__ == "__main__":
    main()

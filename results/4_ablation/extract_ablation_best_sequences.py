#!/usr/bin/env python3
"""从 PPDbench 与 4_ablation HDOCK 汇总中抽取各方法 HDOCK 最优多肽及受体序列，写入 JSON。"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from Bio.PDB import PDBParser, PPBuilder

BENCH = Path("/root/autodl-tmp/PPDbench")
OUT_DIR = Path("/root/autodl-tmp/Peptide_3D/results/4_ablation")
ABLATION_JSON = {
    "generated_ablation_base": OUT_DIR / "ppdbench_hdock_ablation_base.json",
    "generated_ablation_base_dpo": OUT_DIR / "ppdbench_hdock_ablation_base_dpo.json",
    "generated_ablation_base_ot": OUT_DIR / "ppdbench_hdock_ablation_base_ot.json",
}
OUT_JSON = OUT_DIR / "ppdbench_ablation_best_peptides_sequences.json"


def pdb_sequence(pdb_path: Path) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    ppb = PPBuilder()
    parts: list[str] = []
    for pp in ppb.build_peptides(structure):
        parts.append(str(pp.get_sequence()))
    return "".join(parts)


def best_from_hdock_summary(data: dict) -> dict[str, dict]:
    """按 target_id 分组，取 score 最小（HDOCK 能量越低越有利）的一条。"""
    by_target: dict[str, list[dict]] = defaultdict(list)
    for _k, rec in data.items():
        tid = rec["target_id"]
        by_target[tid].append(rec)
    best: dict[str, dict] = {}
    for tid, rows in by_target.items():
        best[tid] = min(rows, key=lambda r: float(r["score"]))
    return best


def best_multi_cands_scores(tid: str) -> tuple[Path, float] | None:
    jpath = BENCH / tid / "multi_cands" / "cands_hdock_scores.json"
    if not jpath.is_file():
        return None
    scores = json.loads(jpath.read_text())
    if not scores:
        return None
    best_path_str = min(scores, key=lambda p: float(scores[p]))
    return Path(best_path_str), float(scores[best_path_str])


def list_target_ids() -> list[str]:
    ids = []
    for p in sorted(BENCH.iterdir()):
        if p.is_dir() and len(p.name) == 4 and p.name.isalnum():
            ids.append(p.name.lower())
    return ids


def main() -> None:
    target_ids = list_target_ids()
    method_best_hdock: dict[str, dict[str, dict]] = {}
    for method, jpath in ABLATION_JSON.items():
        raw = json.loads(jpath.read_text())
        method_best_hdock[method] = best_from_hdock_summary(raw)

    receptor_seq_cache: dict[str, str] = {}
    out_targets: dict[str, dict] = {}

    for tid in target_ids:
        rec_out: dict = {"target_id": tid, "methods": {}}
        receptor_pdb = BENCH / tid / "receptor.pdb"
        if receptor_pdb.is_file():
            rec_out["receptor_pdb"] = str(receptor_pdb)
            if tid not in receptor_seq_cache:
                try:
                    receptor_seq_cache[tid] = pdb_sequence(receptor_pdb)
                except Exception as e:  # noqa: BLE001
                    receptor_seq_cache[tid] = ""
                    rec_out["receptor_sequence_error"] = str(e)
            rec_out["receptor_sequence"] = receptor_seq_cache[tid]
        else:
            rec_out["receptor_pdb"] = None
            rec_out["receptor_sequence"] = ""

        for method, best_map in method_best_hdock.items():
            row = best_map.get(tid)
            if not row:
                rec_out["methods"][method] = {"error": "missing_in_hdock_summary"}
                continue
            pep_path = Path(row["peptide_pdb"])
            entry = {
                "hdock_score": float(row["score"]),
                "peptide_basename": row["peptide_basename"],
                "peptide_pdb": str(pep_path),
            }
            try:
                entry["peptide_sequence"] = pdb_sequence(pep_path)
            except Exception as e:  # noqa: BLE001
                entry["peptide_sequence"] = ""
                entry["peptide_sequence_error"] = str(e)
            rec_out["methods"][method] = entry

        mc = best_multi_cands_scores(tid)
        if mc is None:
            rec_out["methods"]["multi_cands"] = {"error": "missing_scores_or_empty"}
        else:
            pep_path, sc = mc
            entry = {
                "hdock_score": sc,
                "peptide_basename": pep_path.name,
                "peptide_pdb": str(pep_path),
            }
            try:
                entry["peptide_sequence"] = pdb_sequence(pep_path)
            except Exception as e:  # noqa: BLE001
                entry["peptide_sequence"] = ""
                entry["peptide_sequence_error"] = str(e)
            rec_out["methods"]["multi_cands"] = entry

        out_targets[tid] = rec_out

    payload = {
        "description": (
            "PPDbench 133 靶点：各方法在 HDOCK 打分下最优的一条多肽（score 取最小值，即结合能更有利）"
            "及受体蛋白序列。generated_ablation_* 分数来源为 4_ablation 下 ppdbench_hdock_ablation_*.json；"
            "multi_cands 分数来源为 <target>/multi_cands/cands_hdock_scores.json。"
        ),
        "bench_root": str(BENCH),
        "n_targets": len(target_ids),
        "methods": list(ABLATION_JSON.keys()) + ["multi_cands"],
        "targets": out_targets,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON} ({len(target_ids)} targets)")


if __name__ == "__main__":
    main()

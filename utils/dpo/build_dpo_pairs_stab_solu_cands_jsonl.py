#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import argparse
from tqdm import tqdm
from pathlib import Path
from statistics import mean, pstdev

def tprint(*a):
    print(*a, flush=True)

def read_index(tsv_path: str):
    """
    读取 prompts.tsv:
      receptor_pdb \t peptide_seq \t candidates_dir
    跳过注释和空行
    """
    items = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # 兼容无 header 的情况
        if reader.fieldnames is None or len(reader.fieldnames) < 3:
            f.seek(0)
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split("\t")
                if len(parts) < 3:
                    continue
                rec_pdb, pep_seq, cand_dir = parts[0], parts[1], parts[2]
                items.append({
                    "receptor_pdb": rec_pdb,
                    "peptide_seq": pep_seq,
                    "cand_dir": cand_dir,
                })
        else:
            # 标题要求至少包含这三列
            colmap = {name.lower(): name for name in reader.fieldnames}
            need = ["receptor_pdb", "peptide_seq", "candidates_dir"]
            if not all(any(k in name.lower() for name in reader.fieldnames) for k in need):
                raise ValueError(
                    f"{tsv_path} 需要列: receptor_pdb, peptide_seq, candidates_dir "
                    f"（可以大小写或轻微变形，但含义要对应）"
                )
            for row in reader:
                if not row:
                    continue
                rp = row.get(colmap.get("receptor_pdb", "receptor_pdb"), "").strip()
                pep = row.get(colmap.get("peptide_seq", "peptide_seq"), "").strip()
                cd = row.get(colmap.get("candidates_dir", "candidates_dir"), "").strip()
                if not (rp and pep and cd):
                    continue
                items.append({
                    "receptor_pdb": rp,
                    "peptide_seq": pep,
                    "cand_dir": cd,
                })

    tprint(f"[INFO] loaded {len(items)} prompts from {tsv_path}")
    return items

def load_scores(cache_path: Path):
    """
    读取某个 cands_xxx_scores.json
    返回 {pdb_path: float or None}
    """
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # 过滤非数值
        out = {}
        for k, v in data.items():
            try:
                if v is None:
                    continue
                out[k] = float(v)
            except Exception:
                continue
        return out
    except Exception as e:
        tprint(f"[WARN] load {cache_path} failed: {e}")
        return {}

def zscore(vals):
    """
    标准化，同一 prompt 内用; 长度<2 或方差太小则全0
    """
    if not vals:
        return []
    if len(vals) == 1:
        return [0.0]
    mu = mean(vals)
    sigma = pstdev(vals)
    if sigma < 1e-8:
        return [0.0 for _ in vals]
    return [(v - mu) / max(sigma, 1e-8) for v in vals]

def build_pairs_for_prompt(rec_pdb: str,
                           pep_seq: str,
                           scores: dict,
                           metric_name: str,
                           pairs_per_prompt: int,
                           min_margin: float,
                           higher_is_better: bool):
    """
    基于单个 prompt 的缓存分数构造偏好对。
    scores: {pdb_path: score}
    返回一个 list[dict]，每个 dict 是一条 jsonl。
    """
    rows = []
    for pdb_path, s in scores.items():
        if s is None:
            continue
        if not os.path.exists(pdb_path):
            # 忽略脏路径
            continue
        rows.append({
            "pdb": pdb_path,
            "score_raw": float(s),
        })

    if len(rows) < 2:
        return []

    # 定义“越大越好”的有效分数
    if higher_is_better:
        eff = [r["score_raw"] for r in rows]
    else:
        # 比如 hdock：越负越好 -> eff = -score
        eff = [-r["score_raw"] for r in rows]

    eff_z = zscore(eff)
    for r, rz in zip(rows, eff_z):
        r["R"] = rz

    # 按 R 从大到小排序：越大越优
    rows.sort(key=lambda x: x["R"], reverse=True)

    pairs = []
    M = min(pairs_per_prompt, len(rows) // 2)
    for i in range(M):
        chosen = rows[i]
        rejected = rows[-(i + 1)]
        dR = chosen["R"] - rejected["R"]
        if dR < min_margin:
            continue
        pairs.append({
            "prompt": {
                "receptor_pdb": rec_pdb,
                "peptide_seq": pep_seq,
            },
            "chosen": {
                "type": "pdb",
                "pdb_path": chosen["pdb"],
                "score": {
                    metric_name: chosen["score_raw"],
                    "R": chosen["R"],
                },
            },
            "rejected": {
                "type": "pdb",
                "pdb_path": rejected["pdb"],
                "score": {
                    metric_name: rejected["score_raw"],
                    "R": rejected["R"],
                },
            },
            # pair_weight 用 ΔR，当成偏好强度
            "pair_weight": float(max(dR, 0.0)),
        })

    return pairs

def main():
    ap = argparse.ArgumentParser(
        description="基于 cands_<metric>_scores.json 构建单指标偏好对 JSONL"
    )
    ap.add_argument("--index", type=str, default="/root/autodl-tmp/Peptide_3D/utils/dpo/prompts.tsv",
                    help="prompts.tsv: receptor_pdb, peptide_seq, candidates_dir")
    ap.add_argument("--metric", type=str, default="solubility",
                    help="指标名，比如 affinity / stability / solubility，用于 JSON 字段名")
    ap.add_argument("--cache-name", type=str, default="cands_solubility_scores.json",
                    help="缓存文件名，比如 cands_stability_scores.json / cands_solubility_scores.json")
    ap.add_argument("--out", type=str, default="/root/autodl-tmp/Peptide_3D/utils/dpo/solubility_pairs.jsonl",
                    help="输出 JSONL 路径，比如 stability_pairs.jsonl")
    ap.add_argument("--pairs-per-prompt", type=int, default=3,
                    help="每个 prompt 最多生成多少 (top,bottom) 对")
    '''''
    --min-margin 是“偏好强度阈值”, 值越大 → 只保留分数差更大、更“确定”的偏好对，数量会变少，但更干净。
    值越小 → 偏好对数量变多，但会包含很多“半斤八两”的对，噪声更高。
    '''
    ap.add_argument("--min-margin", type=float, default=0.4,
                    help="要求 z-score 差值 >= 此阈值才接受该 pair")
    ap.add_argument("--higher-is-better", type=int, default=1,
                    help="1 表示分数越大越好；0 表示越小越好（例如 hdock）")

    args = ap.parse_args()

    metric_name = args.metric
    higher_is_better = bool(args.higher_is_better)

    prompts = read_index(args.index)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for it in tqdm(prompts, desc="Build pairs", unit="prompt"):
            rec_pdb  = it["receptor_pdb"]
            pep_seq  = it["peptide_seq"]
            cand_dir = it["cand_dir"]

            cache_path = Path(cand_dir) / args.cache_name
            scores = load_scores(cache_path)
            if not scores:
                # 没有这个指标的缓存就跳过
                continue

            pairs = build_pairs_for_prompt(
                rec_pdb, pep_seq, scores,
                metric_name=metric_name,
                pairs_per_prompt=args.pairs_per_prompt,
                min_margin=args.min_margin,
                higher_is_better=higher_is_better,
            )
            for p in pairs:
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")
            total_pairs += len(pairs)

    tprint(f"[FIN] wrote {total_pairs} pairs to {out_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, statistics as stats
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path

# 进度条（没装 tqdm 也能跑）
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

EPS = 1e-8

def zscore(vals: List[float]) -> List[float]:
    if not vals:
        return []
    mu = sum(vals) / len(vals)
    if len(vals) > 1:
        # population stdev 更稳；也可用 sample stdev
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sd = max(var ** 0.5, EPS)
    else:
        sd = 1.0
    return [(v - mu) / sd for v in vals]

def iqr_clip(values: List[float], k: float) -> Tuple[float, float]:
    """返回 [lo, hi] IQR裁剪边界；values 长度少时退回(-inf, +inf)."""
    if len(values) < 4:
        return float("-inf"), float("+inf")
    s = sorted(values)
    q1 = s[len(s)//4]
    q3 = s[(3*len(s))//4]
    iqr = q3 - q1
    return (q1 - k*iqr, q3 + k*iqr)

def parse_line(ln: str) -> Dict:
    r = json.loads(ln)
    # 最少字段检查
    _ = r["prompt"]["receptor_pdb"]
    _ = r["prompt"]["peptide_seq"]
    _ = r["chosen"]["pdb_path"]
    _ = r["chosen"]["score"]["hdock"]
    _ = r["rejected"]["pdb_path"]
    _ = r["rejected"]["score"]["hdock"]
    return r

def main():
    ap = argparse.ArgumentParser(description="Clean & reweight DPO pairs built from HDOCK.")
    ap.add_argument("--in-jsonl",  default='/root/autodl-tmp/Peptide_3D/utils/dpo/dpo_pairs.jsonl', help="原始 dpo_pairs.jsonl")
    ap.add_argument("--out-jsonl", default='/root/autodl-tmp/Peptide_3D/utils/dpo/dpo_pairs_cleaned.jsonl', help="清洗后的输出")
    # 全局 hdock 可接受范围
    ap.add_argument("--hdock-min", type=float, default=-1000.0, help="保留的最小 HDOCK 分数（默认 -1000）")
    ap.add_argument("--hdock-max", type=float, default=0.0,     help="保留的最大 HDOCK 分数（默认 0）")
    # 组内 IQR 裁剪（对 affinity = -hdock）
    ap.add_argument("--iqr-k", type=float, default=3.0, help="若 >0，按 [Q1-k*IQR, Q3+k*IQR] 裁剪组内 affinity")
    # 最小 R 差阈值
    ap.add_argument("--min-margin", type=float, default=0.25, help="R 差过小的 pair 将被丢弃")
    # pair_weight 限幅
    ap.add_argument("--w-lo", type=float, default=0.5, help="pair_weight 下限（默认 0.5）")
    ap.add_argument("--w-hi", type=float, default=2.0, help="pair_weight 上限（默认 2.0）")
    args = ap.parse_args()

    inp  = Path(args.in_jsonl)
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # 读取 + 基本合法性检查
    raw_rows: List[Dict] = []
    with open(inp, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = parse_line(ln)
                raw_rows.append(r)
            except Exception:
                # 坏行直接跳过
                pass

    print(f"[INFO] loaded {len(raw_rows)} raw pairs from {inp}")

    # 统计 & 过滤：路径存在 + 全局 hdock 范围
    rows_ok: List[Dict] = []
    bad_path = bad_score = 0

    for r in raw_rows:
        rp = r["prompt"]["receptor_pdb"]
        cp = r["chosen"]["pdb_path"]
        rj = r["rejected"]["pdb_path"]
        if not (os.path.exists(rp) and os.path.exists(cp) and os.path.exists(rj)):
            bad_path += 1
            continue
        sc_c = float(r["chosen"]["score"]["hdock"])
        sc_r = float(r["rejected"]["score"]["hdock"])
        if not (args.hdock_min <= sc_c <= args.hdock_max and args.hdock_min <= sc_r <= args.hdock_max):
            bad_score += 1
            continue
        rows_ok.append(r)

    print(f"[INFO] after path & range filters: {len(rows_ok)} kept, {bad_path} bad_path, {bad_score} bad_score")

    # 按 prompt 分组（用 receptor_pdb + peptide_seq）
    def key_of(r):
        p = r["prompt"]
        return (p["receptor_pdb"], p["peptide_seq"])

    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows_ok:
        groups[key_of(r)].append(r)

    # 组内做 z-score；可选 IQR 裁剪（对 affinity）
    cleaned: List[Dict] = []
    dropped_small_margin = 0
    swapped_pairs = 0
    dropped_iqr = 0

    for gkey, plist in tqdm(groups.items(), desc="Groups"):
        # 收集该组所有 hdock（来自所有 chosen+rejected）
        vals = []
        for r in plist:
            vals.append(float(r["chosen"]["score"]["hdock"]))
            vals.append(float(r["rejected"]["score"]["hdock"]))

        # 先转 affinity（越大越好）
        aff_all = [-v for v in vals]

        # 组内 IQR 剪裁：给个 soft 方式 -> 得到允许区间
        if args.iqr_k and args.iqr_k > 0:
            lo, hi = iqr_clip(aff_all, args.iqr_k)
        else:
            lo, hi = float("-inf"), float("+inf")

        # 重新组装该组的“唯一候选 -> affinity”映射（以路径为 key）
        cand_aff: Dict[str, float] = {}
        for r in plist:
            for side in ("chosen", "rejected"):
                path = r[side]["pdb_path"]
                hdock = float(r[side]["score"]["hdock"])
                aff = -hdock
                cand_aff[path] = aff

        # 组内可用于 z-score 的集合（应用 IQR）
        use_aff = [a for a in cand_aff.values() if lo <= a <= hi]
        if len(use_aff) == 0:
            # 这组全被 IQR 剪没了：直接跳过整组
            dropped_iqr += len(plist)
            continue

        # 拿“可用集合”的 z-score 标准
        aff_mu = sum(use_aff)/len(use_aff)
        if len(use_aff) > 1:
            var = sum((a - aff_mu)**2 for a in use_aff)/len(use_aff)
            aff_sd = max(var**0.5, EPS)
        else:
            aff_sd = 1.0

        # 给每个候选算 R（若在 IQR 外，仍然给一个边界内的 R，避免直接丢弃）
        def aff_to_R(a: float) -> float:
            a_clip = min(max(a, lo), hi)
            return (a_clip - aff_mu) / aff_sd

        cand_R: Dict[str, float] = {p: aff_to_R(a) for p, a in cand_aff.items()}

        # 生成清洗后的 pair
        for r in plist:
            ch = r["chosen"]; rj = r["rejected"]
            Rc = cand_R[ch["pdb_path"]]
            Rr = cand_R[rj["pdb_path"]]
            # 若方向反了就交换
            swapped = False
            if Rc < Rr:
                ch, rj = rj, ch
                Rc, Rr = Rr, Rc
                swapped = True

            Rdiff = Rc - Rr
            if Rdiff < args.min_margin:
                dropped_small_margin += 1
                continue

            out = {
                "prompt": r["prompt"],
                "chosen": {
                    "type": ch.get("type", "pdb"),
                    "pdb_path": ch["pdb_path"],
                    "score": {
                        "hdock": float(ch["score"]["hdock"]),
                        "R": float(Rc),
                    }
                },
                "rejected": {
                    "type": rj.get("type", "pdb"),
                    "pdb_path": rj["pdb_path"],
                    "score": {
                        "hdock": float(rj["score"]["hdock"]),
                        "R": float(Rr),
                    }
                },
                "pair_weight": float(max(min(Rdiff, args.w_hi), args.w_lo))
            }
            cleaned.append(out)
            if swapped:
                swapped_pairs += 1

    # 写出
    with open(outp, "w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Summary ===")
    print(f"input pairs          : {len(raw_rows)}")
    print(f"kept after path/range: {len(rows_ok)}")
    print(f"  - bad_path         : {bad_path}")
    print(f"  - bad_score        : {bad_score}")
    print(f"cleaned pairs (out)  : {len(cleaned)}")
    print(f"swapped (fix order)  : {swapped_pairs}")
    print(f"dropped small margin : {dropped_small_margin}")
    print(f"dropped by IQR group : {dropped_iqr}")
    print(f"written to           : {outp}")

if __name__ == "__main__":
    main()

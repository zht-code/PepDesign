#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
并行提取 interface_labels（多肽×受体接触图）到一个总 JSON，带进度条。

示例：
python make_interface_labels_parallel.py \
  --root /root/autodl-tmp/train_data \
  --out  /root/autodl-tmp/Peptide_3D/data/interface_labels.json \
  --cutoff 5.0 \
  --mode all-atom \
  --workers 4
"""

import os
import json
import argparse
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- 进度条 ----
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x  # 无 tqdm 时降级为普通迭代

# 3-letter -> one-letter（此脚本未用到序列，可保留备用）
AA3_TO_AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H",
    "ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
    "TYR":"Y","VAL":"V","SEC":"U","PYL":"O","ASX":"B","GLX":"Z","UNK":"X"
}

# ---------------- 基础功能 ----------------
def _parse_pdb_group_by_residue(pdb_path: str) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """
    解析 PDB，按 (chain_id, resseq) 分组，返回：
      res_ids:  ["A:1", "A:2", ...]
      res_ca:   [ (3,), ... ]  每个残基的 CA 坐标（若无 CA 用该残基所有原子质心）
      res_all:  [ (Ni,3), ... ] 每个残基的所有原子坐标
    """
    residues: Dict[Tuple[str,int], Dict[str, Any]] = {}
    order: List[Tuple[str,int]] = []

    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                atom_name = line[12:16].strip()
                resname   = line[17:20].strip()
                chain_id  = (line[21].strip() or " ")
                resseq    = int(line[22:26])
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except Exception:
                continue

            key = (chain_id, resseq)
            if key not in residues:
                residues[key] = {"atoms": [], "ca": None, "resname": resname}
                order.append(key)

            residues[key]["atoms"].append((x,y,z))
            if atom_name == "CA":
                residues[key]["ca"] = (x,y,z)

    res_ids, res_ca, res_all = [], [], []
    for key in order:
        chain_id, resseq = key
        info = residues[key]
        res_ids.append(f"{chain_id}:{resseq}")
        atoms_np = np.asarray(info["atoms"], dtype=np.float32)
        res_all.append(atoms_np)
        if info["ca"] is not None:
            res_ca.append(np.asarray(info["ca"], dtype=np.float32))
        else:
            center = atoms_np.mean(axis=0) if atoms_np.size > 0 else np.zeros(3, dtype=np.float32)
            res_ca.append(center.astype(np.float32))
    return res_ids, res_ca, res_all


def _pair_min_dist(A: np.ndarray, B: np.ndarray) -> float:
    """两个残基原子集合的最小原子-原子距离（单位 Å）。A: (Na,3), B: (Nb,3)"""
    if A.size == 0 or B.size == 0:
        return math.inf
    diff = A[:, None, :] - B[None, :, :]     # (Na,Nb,3)
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return float(np.sqrt(np.min(d2)))


def _make_interface_labels(
    pep_res_all: List[np.ndarray], pep_res_ca: List[np.ndarray],
    rec_res_all: List[np.ndarray], rec_res_ca: List[np.ndarray],
    cutoff: float = 5.0, mode: str = "all-atom"
) -> np.ndarray:
    """
    生成 L_pep × L_rec 的 0/1 接触矩阵。
    mode:
      - "all-atom": 使用“最小原子-原子距离” < cutoff 判为 1
      - "ca":       使用“CA-CA 距离”       < cutoff 判为 1
    """
    Lp, Lr = len(pep_res_all), len(rec_res_all)
    lab = np.zeros((Lp, Lr), dtype=np.int8)

    if mode == "ca":
        pep_ca = np.stack(pep_res_ca, axis=0)  # (Lp,3)
        rec_ca = np.stack(rec_res_ca, axis=0)  # (Lr,3)
        diff = pep_ca[:, None, :] - rec_ca[None, :, :]   # (Lp,Lr,3)
        d2 = np.einsum("ijk,ijk->ij", diff, diff)        # (Lp,Lr)
        lab = (np.sqrt(d2) < cutoff).astype(np.int8)
        return lab

    # all-atom（逐残基对）
    for i in range(Lp):
        Ai = pep_res_all[i]
        for j in range(Lr):
            Bj = rec_res_all[j]
            if _pair_min_dist(Ai, Bj) < cutoff:
                lab[i, j] = 1
    return lab


def _find_receptor_pdb(dir_path: str, peptide_name: str = "peptide.pdb") -> Optional[str]:
    """在子目录下找受体 pdb（排除 peptide.pdb）。"""
    cands = [p for p in os.listdir(dir_path) if p.lower().endswith(".pdb")]
    cands = [p for p in cands if p != peptide_name]
    if not cands:
        return None
    for n in cands:
        low = n.lower()
        if "receptor" in low or "protein" in low or "rec_" in low or low.startswith("rec"):
            return os.path.join(dir_path, n)
    return os.path.join(dir_path, sorted(cands)[0])


# ---------------- 并行单样本处理 ----------------
def _process_one_sample(root: str, sample_id: str, cutoff: float, mode: str) -> tuple[str, Optional[dict], Optional[str]]:
    """
    处理一个子目录，返回 (sample_id, record or None, skip_reason or None)
    """
    d = os.path.join(root, sample_id)
    pep_pdb = os.path.join(d, "peptide.pdb")
    rec_pdb = _find_receptor_pdb(d, peptide_name="peptide.pdb")

    if not (os.path.isfile(pep_pdb) and rec_pdb and os.path.isfile(rec_pdb)):
        return sample_id, None, "missing receptor/peptide PDB"

    try:
        rec_ids, rec_ca, rec_all = _parse_pdb_group_by_residue(rec_pdb)
        pep_ids, pep_ca, pep_all = _parse_pdb_group_by_residue(pep_pdb)
        if len(rec_ids) == 0 or len(pep_ids) == 0:
            return sample_id, None, "empty residues"
        labels = _make_interface_labels(
            pep_all, pep_ca, rec_all, rec_ca, cutoff=float(cutoff), mode=mode
        ).astype(int).tolist()

        rec = {
            "receptor_pdb": rec_pdb,
            "peptide_pdb": pep_pdb,
            "n_pep": len(pep_ids),
            "n_rec": len(rec_ids),
            "pep_res_ids": pep_ids,
            "rec_res_ids": rec_ids,
            "cutoff_A": float(cutoff),
            "mode": mode,
            "labels": labels
        }
        return sample_id, rec, None
    except Exception as e:
        return sample_id, None, f"exception: {e}"


# ---------------- 主流程 ----------------
def build_interface_json_parallel(root: str, out_path: str, cutoff: float, mode: str, workers: int) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 子目录列表
    subdirs = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    total = len(subdirs)
    if total == 0:
        print(f"[WARN] No subfolders found under: {root}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False)
        return

    results: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}

    # 并行
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_process_one_sample, root, sid, cutoff, mode): sid
            for sid in subdirs
        }
        for fut in tqdm(as_completed(futures), total=total, desc="Processing samples", unit="sample"):
            sid = futures[fut]
            try:
                sample_id, record, reason = fut.result()
                if record is not None:
                    results[sample_id] = record
                else:
                    skipped[sample_id] = reason or "unknown"
            except Exception as e:
                skipped[sid] = f"future-exception: {e}"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"[OK] Wrote interface labels for {len(results)} / {total} samples to: {out_path}")
    if skipped:
        print(f"[WARN] Skipped {len(skipped)} samples. (showing up to 10)")
        for i, (k, v) in enumerate(list(skipped.items())[:10], start=1):
            print(f"  {i:02d}. {k} -> {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=False, default="/root/autodl-tmp/train_data",
                    help="训练集根目录，例如 /root/autodl-tmp/train_data")
    ap.add_argument("--out", required=False, default="/root/autodl-tmp/Peptide_3D/data/interface_labels.json",
                    help="输出 JSON 路径")
    ap.add_argument("--cutoff", type=float, default=5.0, help="接触判定阈值（Å）")
    ap.add_argument("--mode", choices=["all-atom", "ca"], default="all-atom",
                    help="接触距离计算方式：all-atom 最小距离 或 仅 CA-CA 距离")
    ap.add_argument("--workers", type=int, default=80, help="并行进程数（建议与 CPU 核数相当；例：4）")
    args = ap.parse_args()

    build_interface_json_parallel(
        root=args.root,
        out_path=args.out,
        cutoff=args.cutoff,
        mode=args.mode,
        workers=max(1, int(args.workers)),
    )


if __name__ == "__main__":
    main()

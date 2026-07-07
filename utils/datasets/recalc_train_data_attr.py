#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

# =========================
# 路径配置：按需修改
# =========================
TRAIN_ROOT = Path("/root/autodl-tmp/train_data")

OUT_STABILITY_JSON = Path("/root/autodl-tmp/Peptide_3D/data/original_stability_scores.json")
OUT_SOLUBILITY_JSON = Path("/root/autodl-tmp/Peptide_3D/data/original_solubility_scores.json")

FOLDX_BIN = "/root/autodl-tmp/foldx_20270131"  # foldx 可执行文件路径
PROTEINSOL_BIN = "/root/autodl-tmp/protein-sol/multiple_prediction_wrapper_export.sh"

# 建议放到 /tmp 或你有写权限的大盘目录
WORKDIR_ROOT = "/root/autodl-tmp/tmp_attr_eval"

# 是否跳过已经有结果的样本
SKIP_IF_EXISTS = False


# =========================
# 你的函数：原样保留/轻微修正
# =========================
def foldx_stability_score(
    pdb_path: Path,
    foldx_bin: str = "foldx",
    workdir_root: str | None = None
) -> float | None:
    try:
        if workdir_root is not None:
            os.makedirs(workdir_root, exist_ok=True)
            workdir = tempfile.mkdtemp(prefix="foldx_", dir=workdir_root)
        else:
            workdir = tempfile.mkdtemp(prefix="foldx_")

        workdir = Path(workdir)

        local_pdb = workdir / "peptide.pdb"
        local_pdb.write_text(Path(pdb_path).read_text())

        cmd = [foldx_bin, "--command=Stability", "--pdb=peptide.pdb"]
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )

        fxout = workdir / "peptide_0_ST.fxout"
        if not fxout.exists():
            return None

        with fxout.open() as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        # 更稳一点：遍历每一行找可解析结果
        for line in lines:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                dg = float(parts[1])
                return -dg  # 越大越稳定
            except Exception:
                continue

        return None

    except Exception:
        return None

    return None

def solubility_score_from_seq(
    seq: str,
    proteinsol_bin: str = "multiple_prediction_wrapper_export.sh",
    workdir_root: str | None = None
) -> float | None:
    """
    使用 Protein-Sol 从序列预测溶解性 (scaled-sol)。
    如果 Protein-Sol 失败，将启用基于氨基酸组成的简易 fallback。
    """
    seq = (seq or "").strip().upper()
    if not seq:
        return None

    def fallback_solubility(sequence: str) -> float:
        # 简单启发式：极性/带电高 -> 溶解性高，疏水高 -> 溶解性低
        hydrophobic = sum(sequence.count(x) for x in "AILMFWYV")
        polar = sum(sequence.count(x) for x in "NQST")
        negative = sum(sequence.count(x) for x in "DE")
        positive = sum(sequence.count(x) for x in "KRH")
        charge = negative - positive
        score = 0.5 + 0.02 * polar - 0.02 * hydrophobic + 0.01 * charge
        return max(0.0, min(1.0, score))

    try:
        ps_bin_path = Path(proteinsol_bin).resolve()
        ps_dir = ps_bin_path.parent

        td_kwargs = {}
        if workdir_root is not None:
            os.makedirs(workdir_root, exist_ok=True)
            td_kwargs["dir"] = workdir_root

        with tempfile.TemporaryDirectory(prefix="proteinsol_", **td_kwargs) as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            fasta_path = tmpdir / "input.fasta"
            with fasta_path.open("w") as f:
                f.write(">pep\n")
                f.write(seq + "\n")

            cmd = [str(ps_bin_path), str(fasta_path)]
            proc = subprocess.run(
                cmd,
                cwd=str(ps_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
            )

            pred_path = ps_dir / "seq_prediction.txt"
            if not pred_path.exists():
                return fallback_solubility(seq)

            with pred_path.open("r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            try:
                pred_path.unlink()
            except Exception:
                pass

            header_cols = None
            for ln in lines:
                if ln.startswith("HEADERS PREDICTIONS"):
                    parts = [p.strip() for p in ln.split(",")]
                    try:
                        id_idx = parts.index("ID")
                        header_cols = parts[id_idx:]
                    except ValueError:
                        if len(parts) > 2:
                            header_cols = parts[2:]
                        else:
                            header_cols = None
                    break

            if header_cols is None:
                return fallback_solubility(seq)

            for ln in lines:
                if ln.startswith("SEQUENCE PREDICTIONS"):
                    parts = [p.strip() for p in ln.split(",")]
                    start_idx = None
                    for i, p in enumerate(parts):
                        if p.startswith(">"):
                            start_idx = i
                            break
                    if start_idx is None:
                        continue

                    values = parts[start_idx:start_idx + len(header_cols)]
                    if len(values) < len(header_cols):
                        continue

                    colmap = {h: v for h, v in zip(header_cols, values)}

                    if "scaled-sol" in colmap:
                        try:
                            return float(colmap["scaled-sol"])
                        except ValueError:
                            break

            return fallback_solubility(seq)

    except Exception:
        return fallback_solubility(seq)


# =========================
# 序列读取
# =========================
AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}
AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def load_fasta_sequence(path: Path) -> str:
    if not path.exists():
        return ""
    seqs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seqs.append(line)
    seq = "".join(seqs).strip().upper()
    seq = "".join([aa for aa in seq if aa.isalpha()])
    return seq


def load_pdb_sequence(path: Path) -> str:
    if not path.exists():
        return ""
    seen = set()
    seq = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            aa = AA3_TO_1.get(resname, "X")
            if aa in AA20:
                seq.append(aa)
    return "".join(seq)


def get_peptide_sequence(sample_dir: Path) -> str:
    fa_path = sample_dir / "peptide.fa"
    pdb_path = sample_dir / "peptide.pdb"

    seq = load_fasta_sequence(fa_path)
    if seq:
        return seq

    return load_pdb_sequence(pdb_path)


# =========================
# JSON 读写
# =========================
def load_existing_json(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================
# 主流程
# =========================
def main():
    if not TRAIN_ROOT.exists():
        raise FileNotFoundError(f"训练集目录不存在: {TRAIN_ROOT}")

    stability_scores = load_existing_json(OUT_STABILITY_JSON)
    solubility_scores = load_existing_json(OUT_SOLUBILITY_JSON)

    sample_dirs = [p for p in sorted(TRAIN_ROOT.iterdir()) if p.is_dir()]

    total = 0
    ok_stab = 0
    ok_sol = 0

    for sample_dir in sample_dirs:
        total += 1
        peptide_pdb = sample_dir / "peptide.pdb"
        receptor_pdb = sample_dir / "receptor.pdb"

        if not peptide_pdb.exists():
            print(f"[WARN] 缺少 peptide.pdb，跳过: {sample_dir}")
            continue

        peptide_key = str(peptide_pdb.resolve())

        # ---------- stability ----------
        if SKIP_IF_EXISTS and peptide_key in stability_scores:
            stab_score = stability_scores[peptide_key]
        else:
            stab_score = foldx_stability_score(
                peptide_pdb,
                foldx_bin=FOLDX_BIN,
                workdir_root=WORKDIR_ROOT,
            )
            print(stab_score)
            stability_scores[peptide_key] = stab_score

        if stab_score is not None:
            ok_stab += 1

        # ---------- solubility ----------
        if SKIP_IF_EXISTS and peptide_key in solubility_scores:
            sol_score = solubility_scores[peptide_key]
        else:
            seq = get_peptide_sequence(sample_dir)
            sol_score = solubility_score_from_seq(
                seq,
                proteinsol_bin=PROTEINSOL_BIN,
                workdir_root=WORKDIR_ROOT,
            )
            solubility_scores[peptide_key] = sol_score

        if sol_score is not None:
            ok_sol += 1

        # 每处理一个样本就落盘，防止中途崩掉白跑
        save_json(stability_scores, OUT_STABILITY_JSON)
        save_json(solubility_scores, OUT_SOLUBILITY_JSON)

        print(
            f"[{total}/{len(sample_dirs)}] {sample_dir.name} | "
            f"stability={stab_score} | solubility={sol_score}"
        )

    print("\n========== 完成 ==========")
    print(f"总样本数: {len(sample_dirs)}")
    print(f"成功得到 stability 的样本数: {ok_stab}")
    print(f"成功得到 solubility 的样本数: {ok_sol}")
    print(f"稳定性结果已保存到: {OUT_STABILITY_JSON}")
    print(f"溶解性结果已保存到: {OUT_SOLUBILITY_JSON}")


if __name__ == "__main__":
    main()



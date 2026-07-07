#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, subprocess, tempfile
from pathlib import Path
from tqdm import tqdm

from Bio import PDB
from Bio.SeqUtils import seq1 as _seq1

_AA3_CUSTOM = {
    "MSE": "M","SEC": "U","PYL": "O",
    "HID": "H","HIE": "H","HIP": "H",
    "CYX": "C","ASX": "B","GLX": "Z","UNK": "X",
}

def resname_to_one(resname: str) -> str:
    try:
        return _seq1(resname.strip(), custom_map=_AA3_CUSTOM, undef_code="X")
    except Exception:
        return "X"

def extract_peptide_seq(pdb_path: Path) -> str:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pep", str(pdb_path))
    residues = [res for res in structure.get_residues()
                if PDB.is_aa(res, standard=False)]
    seq = "".join(resname_to_one(res.get_resname()) for res in residues)
    return seq

# ========= FoldX（原样保留，可视需求微调） =========

def foldx_stability_score(pdb_path: Path,
                          foldx_bin: str = "foldx",
                          workdir_root: str | None = None) -> float | None:
    try:
        # 自定义 workdir 根目录（如果提供）
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
            # print(proc.stdout)  # 可选调试
            return None

        with fxout.open() as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if len(lines) >= 0:
            parts = lines[0].split("\t")
            try:
                dg = float(parts[1])
                return -dg  # 越大越稳定
            except Exception:
                return None
    except Exception:
        return None

    return None


# ========= Protein-Sol 溶解性预测 =========

def solubility_score_from_seq(
    seq: str,
    proteinsol_bin: str = "multiple_prediction_wrapper_export.sh",
    workdir_root: str | None = None
) -> float | None:
    """
    使用 Protein-Sol 从序列预测溶解性 (scaled-sol).
    流程:
      - 在临时目录写 input.fasta
      - 在 Protein-Sol 安装目录下运行 wrapper
      - 在该目录读取 seq_prediction.txt, 解析 HEADERS PREDICTIONS / SEQUENCE PREDICTIONS
    """
    seq = (seq or "").strip().upper()
    if not seq:
        return None

    try:
        # Protein-Sol 安装目录
        ps_bin_path = Path(proteinsol_bin).resolve()
        ps_dir = ps_bin_path.parent

        # 临时目录
        td_kwargs = {}
        if workdir_root is not None:
            os.makedirs(workdir_root, exist_ok=True)
            td_kwargs["dir"] = workdir_root

        with tempfile.TemporaryDirectory(prefix="proteinsol_", **td_kwargs) as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            # 写 FASTA（绝对路径，传给 wrapper）
            fasta_path = tmpdir / "input.fasta"
            with fasta_path.open("w") as f:
                f.write(">pep\n")
                f.write(seq + "\n")

            # 在 Protein-Sol 目录下执行脚本
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
                # 调试时可以打开下面两行看看:
                # print(proc.stdout)
                # print((ps_dir / "run.log").read_text())
                return None

            with pred_path.open("r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            # 用完可选清理，防止堆积（如要调试可暂时注释）
            try:
                pred_path.unlink()
            except Exception:
                pass

            # 解析:
            # 先找到 HEADERS PREDICTIONS 行，拿到列名
            header_cols = None
            for ln in lines:
                if ln.startswith("HEADERS PREDICTIONS"):
                    # e.g. HEADERS PREDICTIONS LINE,ID,percent-sol,scaled-sol,population-sol,pI
                    parts = [p.strip() for p in ln.split(",")]
                    # parts[0] = 'HEADERS PREDICTIONS LINE'
                    # 实际有用的是从 ID 开始:
                    # 兼容点粗暴写法：从第二个逗号之后的字段当 header
                    # 找到第一个真正的 "ID"
                    try:
                        id_idx = parts.index("ID")
                        header_cols = parts[id_idx:]  # ['ID', 'percent-sol', 'scaled-sol', ...]
                    except ValueError:
                        # 如果找不到，就从第三个元素开始兜底
                        if len(parts) > 2:
                            header_cols = parts[2:]
                        else:
                            header_cols = None
                    break

            if header_cols is None:
                return None

            # 再找到 SEQUENCE PREDICTIONS 行，对应数值
            for ln in lines:
                if ln.startswith("SEQUENCE PREDICTIONS"):
                    # e.g. SEQUENCE PREDICTIONS,>pep,96.187,0.842,0.446,1.000
                    parts = [p.strip() for p in ln.split(",")]
                    # 同理，找到和 header 对齐的起始位置
                    # 一般是从 ID (这里是 >pep) 开始
                    # 先找到第一个以 '>' 开头的序列ID位置
                    start_idx = None
                    for i, p in enumerate(parts):
                        if p.startswith(">"):
                            start_idx = i
                            break
                    if start_idx is None:
                        continue

                    values = parts[start_idx:start_idx + len(header_cols)]
                    if len(values) < len(header_cols):
                        # 不够长就跳过这行
                        continue

                    colmap = {h: v for h, v in zip(header_cols, values)}

                    # 优先使用 scaled-sol
                    if "scaled-sol" in colmap:
                        try:
                            return float(colmap["scaled-sol"])
                        except ValueError:
                            pass

                    # # 退一步，用 percent-sol（/100 也行，看你定义）
                    # if "percent-sol" in colmap:
                    #     try:
                    #         return float(colmap["percent-sol"])
                    #     except ValueError:
                    #         pass

                    # # 再不行，兜底从这一行抓一个能转 float 的数
                    # for tok in reversed(values):
                    #     try:
                    #         return float(tok)
                    #     except ValueError:
                    #         continue

            return None

    except Exception:
        return None




# ==================================================

def load_prompts(tsv_path: Path):
    rows = []
    with tsv_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            rows.append({
                "receptor_pdb": parts[0],
                "peptide_seq": parts[1],
                "candidates_dir": parts[2],
            })
    return rows

def discover_candidates(cand_dir: Path):
    pdbs = []
    for ext in ("*.pdb", "*.ent", "*.pdbqt"):
        pdbs.extend(cand_dir.glob(ext))
    return sorted(set(pdbs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=str,
                    default="/root/autodl-tmp/Peptide_3D/utils/dpo/prompts.tsv")
    ap.add_argument("--foldx", type=str,
                    default="/root/autodl-tmp/foldx_20251231")
    ap.add_argument("--proteinsol", type=str,
                    default="/root/autodl-tmp/protein-sol/multiple_prediction_wrapper_export.sh")
    ap.add_argument("--workdir", type=str, default="/root/autodl-tmp/tmp_runs",
                    help="自定义临时计算目录根路径，用于 FoldX 和 Protein-Sol 运行")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts))
    print(f"[INFO] loaded {len(prompts)} prompts")

    for p in tqdm(prompts, desc="build stability/solubility caches"):
        cand_dir = Path(p["candidates_dir"])
        if not cand_dir.exists():
            continue

        stab_path = cand_dir / "cands_stability_scores.json"
        solu_path = cand_dir / "cands_solubility_scores.json"

        stab_scores = {} if args.overwrite or not stab_path.exists() else json.load(open(stab_path))
        solu_scores = {} if args.overwrite or not solu_path.exists() else json.load(open(solu_path))

        for pdb_path in discover_candidates(cand_dir):
            sp = str(pdb_path)

            # 稳定性
            if sp not in stab_scores:
                s = foldx_stability_score(
                    pdb_path,
                    foldx_bin=args.foldx,
                    workdir_root=args.workdir,
                )
                stab_scores[sp] = s

            # 溶解性（Protein-Sol）    序列大于21aa时才计算溶解性
            if sp not in solu_scores:
                seq = extract_peptide_seq(pdb_path)
                s2 = solubility_score_from_seq(
                    seq,
                    proteinsol_bin=args.proteinsol,
                    workdir_root=args.workdir,
                )
                solu_scores[sp] = s2

        with stab_path.open("w") as f:
            json.dump(stab_scores, f, indent=2)
        with solu_path.open("w") as f:
            json.dump(solu_scores, f, indent=2)

    print("[FIN] caches built (stability + solubility).")


if __name__ == "__main__":
    main()

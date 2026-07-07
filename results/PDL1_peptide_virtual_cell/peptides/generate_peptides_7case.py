#!/usr/bin/env python3
"""
为多肽对接/分析生成模型采样的多肽全原子 PDB。

未指定 --pdb 时：仍使用内置案例列表，在 --case-dir 下查找对应受体 PDB（与旧版 7_case 一致）。
指定 --pdb 时：按你的蛋白受体 PDB 路径生成，输出在 --output-root/<PDB 主文件名>/cands/。

流程与 utils/reference/train_data_generate_top10.py 的 worker 一致：序列生成 → interface 重排 →
α-螺旋初始全原子构象 → 口袋附近刚体摆放 → OpenMM 约束最小化。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

CASE_DIR = Path(__file__).resolve().parent


def _find_peptide_3d_root(start: Path) -> Path:
    """从当前脚本目录向上查找 Peptide_3D 根（含 utils/reference/train_data_generate_top10.py）。"""
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        ref = p / "utils" / "reference" / "train_data_generate_top10.py"
        if ref.is_file():
            return p
    raise FileNotFoundError(
        f"未找到 train_data_generate_top10.py（已从 {start} 向上搜索）。"
        "请确认 Peptide_3D 工程完整，或将本脚本放在 Peptide_3D 子目录下。"
    )


PEPTIDE_3D_ROOT = _find_peptide_3d_root(CASE_DIR)
REF_DIR = PEPTIDE_3D_ROOT / "utils" / "reference"

for p in (str(PEPTIDE_3D_ROOT), str(REF_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

_gen_path = REF_DIR / "train_data_generate_top10.py"
_spec = importlib.util.spec_from_file_location("train_data_generate_top10", _gen_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"无法加载 {_gen_path}")
gen_ref = importlib.util.module_from_spec(_spec)
sys.modules["train_data_generate_top10"] = gen_ref
_spec.loader.exec_module(gen_ref)

# 未传 --pdb 时，在 --case-dir 下按下列文件名查找（主文件名不含路径）
DEFAULT_CASE_RECEPTOR_PDBS = (
    "3V2A-vegf.pdb",
    "6LML-GPCR.pdb",
    "7OUN-PD-L1.pdb",
)


def build_prot_list_legacy(case_dir: Path) -> list[tuple[str, str]]:
    """内置三案例：返回 (sample_dir, receptor_pdb)；sample_dir 用于存放该靶点的 cands/。"""
    pairs: list[tuple[str, str]] = []
    for fname in DEFAULT_CASE_RECEPTOR_PDBS:
        pdb_path = case_dir / fname
        if not pdb_path.is_file():
            raise FileNotFoundError(f"未找到受体 PDB：{pdb_path}")
        stem = pdb_path.stem
        sample_dir = case_dir / stem
        pairs.append((str(sample_dir), str(pdb_path.resolve())))
    return pairs


def build_prot_list_custom(
    pdb_paths: list[Path],
    output_root: Path,
) -> list[tuple[str, str]]:
    """自定义受体 PDB：每个文件输出到 output_root/<stem>/cands/。"""
    pairs: list[tuple[str, str]] = []
    for p in pdb_paths:
        p = p.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"未找到受体 PDB：{p}")
        if p.suffix.lower() != ".pdb":
            raise ValueError(f"应为 .pdb 文件：{p}")
        stem = p.stem
        sample_dir = output_root / stem
        pairs.append((str(sample_dir.resolve()), str(p)))
    return pairs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="按受体 PDB 生成多肽（复用 train_data_generate_top10.worker）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "示例：\n"
            "  %(prog)s --pdb ../data/my_target.pdb\n"
            "  %(prog)s --pdb a.pdb b.pdb --output-root ./runs --num-per-protein 10\n"
            "  %(prog)s   # 无 --pdb：使用 --case-dir 下三套默认文件名\n"
        ),
    )
    ap.add_argument(
        "--pdb",
        nargs="+",
        default=None,
        metavar="PATH",
        help="一个或多个受体蛋白 PDB 路径；省略则使用内置三案例（见 --case-dir）。",
    )
    ap.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help=f"内置案例 PDB 所在目录；默认与本脚本同级（当前为 {CASE_DIR}）。仅在未指定 --pdb 时使用。",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="自定义 --pdb 时，各靶点输出根目录 <root>/<PDB主名>/cands/；默认与本脚本同级。",
    )
    ap.add_argument(
        "--ckpt-path",
        default="/root/autodl-tmp/Peptide_3D/logs_Ranger_no_DPO/best_model_epoch_72_loss_2.0048.pth",
        help="ProteinPeptideModel 权重路径。",
    )
    ap.add_argument("--num-per-protein", type=int, default=100, help="每个靶点保留的多肽条数。")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--oversample-factor", type=int, default=3)
    ap.add_argument("--num-gpus", type=int, default=1, help="使用的 GPU 数量（多进程）。")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.pdb:
        out_root = (args.output_root or CASE_DIR).expanduser().resolve()
        pdb_paths = [Path(x) for x in args.pdb]
        prot_list = build_prot_list_custom(pdb_paths, out_root)
    else:
        case_dir = (args.case_dir or CASE_DIR).expanduser().resolve()
        prot_list = build_prot_list_legacy(case_dir)
    print(f"将处理 {len(prot_list)} 个靶点，输出各自子目录下的 cands/：")
    for sample_dir, pdb_path in prot_list:
        print(f"  {pdb_path} -> {sample_dir}/cands/")

    avail = torch.cuda.device_count()
    if avail == 0:
        print("未检测到 CUDA，使用 CPU 单进程。")
        world_size = 1
        shards = [prot_list]
    else:
        world_size = min(args.num_gpus, avail)
        indices = np.array_split(np.arange(len(prot_list)), world_size)
        shards = [[prot_list[i] for i in idx.tolist()] for idx in indices]

    cfg = dict(
        ckpt_path=args.ckpt_path,
        num_per_protein=args.num_per_protein,
        top_k=args.top_k,
        max_len=args.max_len,
        temperature=args.temperature,
        num_gpus=world_size,
        oversample_factor=args.oversample_factor,
    )

    if world_size == 1:
        gen_ref.worker(0, shards[0], cfg)
        return

    ctx = get_context("spawn")
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=gen_ref.worker, args=(rank, shards[rank], cfg), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()


'''

# 指定单个自定义受体 PDB（输出 peptides/MyTarget/cands/）
python /root/autodl-tmp/Peptide_3D/results/PDL1_peptide_virtual_cell/peptides/generate_peptides_7case.py --pdb /root/autodl-tmp/Peptide_3D/results/PDL1_peptide_virtual_cell/data/7OUN-PD-L1_target_A.pdb --output-root /root/autodl-tmp/Peptide_3D/results/PDL1_peptide_virtual_cell/peptides

# 多个靶点 + 自定义输出根目录（输出 ./runs/TargetA/cands/、runs/TargetB/cands/）
python generate_peptides_7case.py --pdb /path/a.pdb /path/b.pdb --output-root /root/autodl-tmp/Peptide_3D/results/PDL1_peptide_virtual_cell/peptides

# 与以前相同：在当前 --case-dir 下找三套默认文件名
python generate_peptides_7case.py --num-per-protein 5 --num-gpus 1

'''
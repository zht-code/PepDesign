#!/usr/bin/env python3
"""将 GeneFormer 与 scFoundation（细胞）权重快照到 models/perturb_virtual_cell/。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "models" / "perturb_virtual_cell"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-geneformer", action="store_true")
    ap.add_argument("--skip-scfoundation", action="store_true")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装: pip install huggingface_hub", file=sys.stderr)
        raise SystemExit(1)

    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_geneformer:
        dst = out / "geneformer"
        print("下载 ctheodoris/Geneformer ->", dst)
        snapshot_download(
            repo_id="ctheodoris/Geneformer",
            local_dir=str(dst),
            local_dir_use_symlinks=False,
        )

    if not args.skip_scfoundation:
        dst = out / "scfoundation_cell"
        print("下载 perturblab/scfoundation-cell ->", dst)
        snapshot_download(
            repo_id="perturblab/scfoundation-cell",
            local_dir=str(dst),
            local_dir_use_symlinks=False,
        )

    print("完成。")


if __name__ == "__main__":
    main()

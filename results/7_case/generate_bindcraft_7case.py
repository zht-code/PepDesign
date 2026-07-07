#!/usr/bin/env python3
"""
使用 BindCraft 对 7_case 三个靶点各运行 binder/多肽设计。

优先使用项目内副本：
  <Peptide_3D>/results/5_robustness/baseline/repos/BindCraft
若不存在则克隆到：
  <本目录>/external/BindCraft

为每个靶点生成临时 settings JSON（starting_pdb 为仅含靶链、重编号为 A1..N 的 PDB），
并调用官方入口 bindcraft.py。

依赖：需按 BindCraft 文档配置 ColabDesign、JAX、PyRosetta、AlphaFold 参数目录等；
settings_advanced 中的 af_params_dir / dalphaball_path 等需在你环境中填写或通过 UI 同步。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CASE_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("3V2A-vegf", "3V2A-vegf.pdb", "R"),
    ("6LML-GPCR", "6LML-GPCR.pdb", "R"),
    ("7OUN-PD-L1", "7OUN-PD-L1.pdb", "A"),
)

BINDCRAFT_GIT = "https://github.com/martinpacesa/BindCraft.git"
TARGET_OVERRIDES: dict[str, dict[str, object]] = {
    "6LML-GPCR": {
        "binder_len_min": 15,
        "binder_len_max": 27,
        "advanced_name": "bindcraft_advanced_6LML-GPCR_fast.json",
        # 默认 peptide_filters 对 MPNN 后 AF2 门槛偏高，6LML 常 0 条写入 mpnn_design_stats 却永不退出
        "filters_name": "peptide_filters_6LML_relaxed.json",
    }
}


def peptide_3d_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_repo() -> Path:
    bundled = peptide_3d_root() / "results" / "5_robustness" / "baseline" / "repos" / "BindCraft"
    if bundled.is_dir() and (bundled / "bindcraft.py").is_file():
        return bundled
    return Path(__file__).resolve().parent / "external" / "BindCraft"


def ensure_repo(repo: Path) -> Path:
    if (repo / "bindcraft.py").is_file():
        return repo
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", BINDCRAFT_GIT, str(repo)],
        check=True,
    )
    return repo


def write_chain_only_renumbered(pdb_path: Path, chain_id: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping: dict[tuple[str, str], int] = {}
    n = 0
    lines_out: list[str] = []
    serial = 0
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 27:
                continue
            ch = line[21]
            if ch.strip() != chain_id.strip():
                continue
            resseq = line[22:26]
            icode = line[26] if len(line) > 26 else " "
            key = (resseq, icode)
            if key not in mapping:
                n += 1
                mapping[key] = n
            new_res = mapping[key]
            serial += 1
            new_line = f"{line[:6]}{serial:5d}{line[11:21]}A{new_res:4d}{line[26:].rstrip()}\n"
            lines_out.append(new_line)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines_out)
        handle.write("END\n")
    return n


def pick_hotspot_residue_str(n_res: int) -> str:
    """BindCraft / ColabDesign 常用逗号分隔的残基编号（靶链上的序号）。"""
    if n_res <= 0:
        return "1"
    r = max(1, min(n_res, n_res // 2))
    return str(r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="7_case 三靶点 BindCraft 批跑。")
    p.add_argument("--repo-root", type=str, default="", help="BindCraft 仓库根目录。")
    p.add_argument(
        "--filters",
        type=str,
        default="",
        help="filters JSON，默认使用仓库内 settings_filters/peptide_filters.json",
    )
    p.add_argument(
        "--advanced",
        type=str,
        default="",
        help="advanced JSON，默认 settings_advanced/peptide_3stage_multimer.json",
    )
    p.add_argument("--binder-len-min", type=int, default=15)
    p.add_argument("--binder-len-max", type=int, default=35)
    p.add_argument("--num-final-designs", type=int, default=1, help="每个靶点接受的最终设计数。")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = Path(__file__).resolve().parent
    repo = Path(args.repo_root).resolve() if args.repo_root else default_repo()
    if not (repo / "bindcraft.py").is_file():
        print(f"未找到 BindCraft，正在克隆到: {repo}")
        if args.dry_run:
            print(f"[dry-run] git clone {BINDCRAFT_GIT} {repo}")
        else:
            ensure_repo(repo)

    default_filters = repo / "settings_filters" / "peptide_filters.json"
    advanced = Path(args.advanced) if args.advanced else repo / "settings_advanced" / "peptide_3stage_multimer.json"
    if not advanced.is_file():
        raise FileNotFoundError(f"缺少 advanced 配置: {advanced}")

    prep_dir = case_dir / "bindcraft_inputs"
    settings_dir = case_dir / "bindcraft_settings_7case"
    settings_dir.mkdir(parents=True, exist_ok=True)

    for stem, pdb_name, chain in CASE_TARGETS:
        raw_pdb = case_dir / pdb_name
        if not raw_pdb.is_file():
            raise FileNotFoundError(raw_pdb)
        clean_pdb = prep_dir / f"{stem}_target_A.pdb"
        n_res = write_chain_only_renumbered(raw_pdb, chain, clean_pdb)
        design_path = (case_dir / "bindcraft_runs" / stem).resolve()
        design_path.mkdir(parents=True, exist_ok=True)
        target_override = TARGET_OVERRIDES.get(stem, {})
        binder_len_min = int(target_override.get("binder_len_min", args.binder_len_min))
        binder_len_max = int(target_override.get("binder_len_max", args.binder_len_max))
        target_advanced = advanced
        advanced_name = target_override.get("advanced_name")
        if advanced_name:
            candidate_advanced = case_dir / str(advanced_name)
            if candidate_advanced.is_file():
                target_advanced = candidate_advanced
        target_filters = Path(args.filters) if args.filters else default_filters
        filters_name = target_override.get("filters_name")
        if filters_name:
            candidate_filters = case_dir / str(filters_name)
            if candidate_filters.is_file():
                target_filters = candidate_filters
        if not target_filters.is_file():
            raise FileNotFoundError(f"缺少 filters 配置: {target_filters}")

        settings = {
            "design_path": str(design_path) + "/",
            "binder_name": stem,
            "starting_pdb": str(clean_pdb.resolve()),
            "chains": "A",
            "target_hotspot_residues": pick_hotspot_residue_str(n_res),
            "lengths": [binder_len_min, binder_len_max],
            "number_of_final_designs": args.num_final_designs,
        }
        settings_path = settings_dir / f"{stem}_settings.json"
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            str(repo / "bindcraft.py"),
            "--settings",
            str(settings_path),
            "--filters",
            str(target_filters),
            "--advanced",
            str(target_advanced),
        ]
        print("\n>>>", stem)
        print(" ", " \\\n  ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(repo), check=False)


if __name__ == "__main__":
    main()


'''

python "/root/autodl-tmp/Peptide_3D/results/7_case/generate_bindcraft_7case.py" \
  --advanced /root/autodl-tmp/Peptide_3D/results/7_case/bindcraft_advanced_6LML-GPCR_fast.json

'''
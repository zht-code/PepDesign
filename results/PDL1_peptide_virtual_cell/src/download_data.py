"""
GEO / 公开数据下载占位与校验。

真实 GEO 矩阵格式不一，建议用户手动下载后放入 data/raw/。
此处提供路径检查与元数据提示。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def print_geo_instructions(config: dict[str, Any], project_root: Path) -> None:
    geo = config.get("geo", {})
    raw = project_root / "data" / "raw"
    log.info("原始数据目录: %s", raw)
    for acc, meta in geo.items():
        note = meta.get("note", "")
        url = meta.get("url_hint", "")
        log.info("[%s] %s 参考: %s → 放入 %s/", acc, note, url, raw)


def check_raw_directory(project_root: Path) -> list[Path]:
    raw = project_root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    files = list(raw.rglob("*"))
    files = [p for p in files if p.is_file()]
    if not files:
        log.warning("data/raw/ 为空 — 将使用内置 demo 单细胞数据跑通流程。")
    return files


def try_download_gse_placeholder(gse_id: str, dest_dir: Path) -> bool:
    """
    预留：可通过 entrez-direct / GEOparse 等实现自动下载。
    当前返回 False，提示手动下载。
    """
    log.warning(
        "自动下载 %s 未实现（避免未安装 GEO 依赖）。请将文件放入 %s",
        gse_id,
        dest_dir,
    )
    return False

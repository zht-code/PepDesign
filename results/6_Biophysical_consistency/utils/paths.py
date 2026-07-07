from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """本工程根目录（含 config.yaml、scripts/ 的目录）。"""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")
    # 允许用环境变量覆盖外部项目根（只读数据）
    env_root = os.environ.get("PEPTIDE_3D_ROOT")
    if env_root:
        data["project_root"] = env_root
    return data


@dataclass(frozen=True)
class ProjectPaths:
    """将 config 中的相对路径解析为绝对路径。"""

    root: Path
    data_inventory: Path
    intermediate: Path
    tables: Path
    figures: Path
    case_studies: Path
    logs: Path

    @staticmethod
    def from_config(cfg: dict[str, Any], root: Path | None = None) -> "ProjectPaths":
        base = root or project_root()
        rel = (cfg.get("paths") or {}) if isinstance(cfg.get("paths"), dict) else {}
        return ProjectPaths(
            root=base,
            data_inventory=base / rel.get("data_inventory", "data_inventory"),
            intermediate=base / rel.get("intermediate", "intermediate"),
            tables=base / rel.get("tables", "tables"),
            figures=base / rel.get("figures", "figures"),
            case_studies=base / rel.get("case_studies", "case_studies"),
            logs=base / rel.get("logs", "logs"),
        )

    def ensure_dirs(self) -> None:
        for p in (
            self.data_inventory,
            self.intermediate,
            self.tables,
            self.figures,
            self.case_studies,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)

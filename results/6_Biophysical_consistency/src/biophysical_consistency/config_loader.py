from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .paths import resolve_project_root


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out


def load_merged_config(config_dir: Path) -> dict[str, Any]:
    default_path = config_dir / "default_config.yaml"
    local_path = config_dir / "local_config.yaml"
    cfg = load_yaml(default_path)
    if local_path.exists():
        cfg = merge_config(cfg, load_yaml(local_path))
    cfg["project_root"] = str(resolve_project_root(cfg.get("project_root", "")))
    return cfg

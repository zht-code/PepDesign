"""确保 scGPT 预训练权重位于项目目录（优先 HuggingFace，国内可用 hf-mirror）。"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REQUIRED_FILES = ("best_model.pt", "vocab.json", "args.json")


def _model_dir(project_root: Path, config: dict[str, Any]) -> Path:
    rel = str(config.get("scgpt_model_dir", "models/scgpt_whole_human")).strip()
    return project_root / rel


def checkpoint_ready(model_dir: Path) -> bool:
    return all((model_dir / f).exists() for f in REQUIRED_FILES)


def ensure_scgpt_checkpoint(project_root: Path, config: dict[str, Any]) -> Path | None:
    """
    若缺少权重则尝试下载到 config['scgpt_model_dir']。
    返回可用模型目录，失败返回 None（调用方应回退 simple_signature）。
    """
    model_dir = _model_dir(project_root, config)
    model_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_ready(model_dir):
        return model_dir

    repo = str(config.get("scgpt_hf_repo", "perturblab/scgpt-human")).strip()
    mirrors = list(config.get("scgpt_hf_mirrors", []) or [])
    if not mirrors:
        mirrors = [
            "https://hf-mirror.com",
            "https://huggingface.co",
        ]

    for name in REQUIRED_FILES:
        dest = model_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        ok = False
        for base in mirrors:
            url = f"{base.rstrip('/')}/{repo}/resolve/main/{name}"
            try:
                log.info("下载 scGPT 权重: %s -> %s", url, dest)
                urllib.request.urlretrieve(url, str(dest))  # noqa: S310
                if dest.exists() and dest.stat().st_size > 0:
                    ok = True
                    break
            except OSError as exc:
                log.warning("下载失败 (%s): %s", url, exc)
        if not ok:
            log.error("无法下载 %s，请检查网络或手动放置到 %s", name, model_dir)
            return None

    return model_dir if checkpoint_ready(model_dir) else None

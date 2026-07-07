from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return set(str(x) for x in data)
    if isinstance(data, dict) and "completed" in data:
        return set(str(x) for x in data["completed"])
    return set()


def save_set(path: Path, items: set[str], meta: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"completed": sorted(items)}
    if meta:
        payload["meta"] = meta
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_done(checkpoint_path: Path, sample_uid: str) -> bool:
    return sample_uid in load_set(checkpoint_path)


def mark_done(checkpoint_path: Path, sample_uid: str, all_done: set[str]) -> None:
    all_done.add(sample_uid)
    save_set(checkpoint_path, all_done)

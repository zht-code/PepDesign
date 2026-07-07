from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PACKAGE_ROOT


def resolve_project_root(config_root: str | Path) -> Path:
    """Resolve Peptide_3D root from config or PEPTIDE_3D_ROOT."""
    env = os.environ.get("PEPTIDE_3D_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    p = Path(config_root).expanduser().resolve()
    return p

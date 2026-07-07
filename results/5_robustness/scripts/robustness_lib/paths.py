from __future__ import annotations

import sys
from pathlib import Path

# run_robustness_pipeline.py lives in results/5_robustness/scripts/
_SCRIPTS = Path(__file__).resolve().parents[1]
ROBUSTNESS_ROOT = _SCRIPTS.parent
# ROBUSTNESS_ROOT = .../Peptide_3D/results/5_robustness → parents[1] == Peptide_3D（parents[2] 会错到 autodl-tmp）
PROJECT_ROOT = ROBUSTNESS_ROOT.parents[1]

# 必须先于 `import model...`：否则在 scripts 目录下直接跑 pipeline 会报 No module named 'model'
_root_s = str(PROJECT_ROOT)
if _root_s not in sys.path:
    sys.path.insert(0, _root_s)


def ensure_subdirs() -> dict[str, Path]:
    """Create standard output directories under 5_robustness."""
    sub = {
        "scripts": ROBUSTNESS_ROOT / "scripts",
        "configs": ROBUSTNESS_ROOT / "configs",
        "logs": ROBUSTNESS_ROOT / "logs",
        "cache": ROBUSTNESS_ROOT / "cache",
        "tables": ROBUSTNESS_ROOT / "tables",
        "figures": ROBUSTNESS_ROOT / "figures",
        "cases": ROBUSTNESS_ROOT / "cases",
        "metrics": ROBUSTNESS_ROOT / "metrics",
        "tmp": ROBUSTNESS_ROOT / "tmp",
    }
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    return sub

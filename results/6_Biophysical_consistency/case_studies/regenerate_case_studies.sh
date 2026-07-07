#!/usr/bin/env bash
# 从已有 Table_S11 / S8 / S1 等结果重新筛选案例并生成图与 PyMOL 素材
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python3 "${ROOT}/scripts/07_select_case_studies.py" --config "${ROOT}/config.yaml" "$@"

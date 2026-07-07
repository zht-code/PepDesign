#!/usr/bin/env bash
# Peptide biophysical consistency — 一键流水线（01→06）
# 依赖：Python3、requirements_biophysical.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CFG="${ROOT}/config.yaml"
LOG="${ROOT}/logs"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOG}" "${ROOT}/data_inventory" "${ROOT}/intermediate" \
  "${ROOT}/tables" "${ROOT}/figures" "${ROOT}/case_studies"

echo "==> [01] Build master table (Table S1)"
python3 "${ROOT}/scripts/01_build_master_table.py" \
  --config "${CFG}" \
  --log-dir "${LOG}"

echo "==> [02] Free peptide structure metrics (Table S2)"
python3 "${ROOT}/scripts/02_analyze_free_peptides.py" \
  --config "${CFG}" \
  --log-dir "${LOG}"

echo "==> [03] Complex interface analysis (Table S3, S8, …)"
python3 "${ROOT}/scripts/03_analyze_complex_interfaces.py" \
  --config "${CFG}" \
  --log-dir "${LOG}"

echo "==> [04] Solubility & aggregation hotspots (Table S5, S6, S7, …)"
python3 "${ROOT}/scripts/04_analyze_solubility_and_hotspots.py" \
  --config "${CFG}" \
  --log-dir "${LOG}"

echo "==> [05] Summary scores (Table S4, S11, S12, summary_report.md)"
python3 "${ROOT}/scripts/05_build_summary_scores.py" \
  --config "${CFG}" \
  --log-dir "${LOG}"

echo "==> [06] Figures (Fig7a–f, supplementary, figure_manifest.md)"
python3 "${ROOT}/scripts/06_make_figures.py" \
  --config "${CFG}" \
  --log-dir "${LOG}"

echo "==> [07] Case studies (selected_cases.json, case_*/)"
python3 "${ROOT}/scripts/07_select_case_studies.py" \
  --config "${CFG}"

echo "[run_all.sh] Done. ROOT=${ROOT}"

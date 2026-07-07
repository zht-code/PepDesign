#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline"
PY="${BASE_DIR}/scripts/run_baseline_robustness.py"
PLOT="${BASE_DIR}/scripts/plot_robustness_comparison.py"

METHODS="${1:-all}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-20260415}"

python "${PY}" --methods "${METHODS}" --build-index-only
python "${PY}" --methods "${METHODS}" --rfdiffusion-only-postprocess --device "${DEVICE}"

for perturb in structure_missing pocket_noise sequence_trunc; do
  case "${perturb}" in
    structure_missing|sequence_trunc)
      levels=(0 10 20 30 40)
      ;;
    pocket_noise)
      levels=(0 0.5 1.0 1.5 2.0)
      ;;
  esac

  for level in "${levels[@]}"; do
    python "${PY}" \
      --methods "${METHODS}" \
      --perturbation-type "${perturb}" \
      --perturbation-strength "${level}" \
      --skip-existing \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}" \
      --seed "${SEED}"
  done
done

python "${PY}" --methods "${METHODS}" --aggregate-only
python "${PLOT}"

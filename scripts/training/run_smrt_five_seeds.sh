#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# TC-TopoRT SMRT five-seed experiment
#
# Paper seeds:
#   1, 5, 79, 123, 256
#
# Usage:
#   bash scripts/training/run_smrt_five_seeds.sh
#
# Preview all five commands without training:
#   DRY_RUN=1 bash scripts/training/run_smrt_five_seeds.sh
#
# Generated files are written under ARTIFACT_ROOT.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SINGLE_RUNNER="${SCRIPT_DIR}/run_smrt_single_seed.sh"

if [[ ! -f "${SINGLE_RUNNER}" ]]; then
    echo "[ERROR] Single-seed runner not found: ${SINGLE_RUNNER}" >&2
    exit 1
fi

SEEDS=(1 5 79 123 256)

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "TC-TopoRT SMRT five-seed experiment"
echo "Seeds         : ${SEEDS[*]}"
echo "Artifact root : ${ARTIFACT_ROOT}"
echo "Resume        : ${RESUME}"
echo "Dry run       : ${DRY_RUN}"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo
    echo "############################################################"
    echo "Starting seed ${seed}"
    echo "############################################################"

    SEED="${seed}" \
    ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    RESUME="${RESUME}" \
    DRY_RUN="${DRY_RUN}" \
    bash "${SINGLE_RUNNER}"

    echo "Finished seed ${seed}"
done

echo
echo "============================================================"
echo "All five TC-TopoRT SMRT runs completed."
echo "Seeds: ${SEEDS[*]}"
echo "============================================================"

#!/usr/bin/env bash
set -euo pipefail

# Validate the checked-in original and strict tautomer-standardized
# SMRT views before model training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
VALIDATOR="${SCRIPT_DIR}/validate_paired_views.py"

ORIGIN_TRAIN="${REPO_ROOT}/gwn/data/SMRT_train.csv"
ORIGIN_TEST="${REPO_ROOT}/gwn/data/SMRT_test.csv"

TAUT_TRAIN="${REPO_ROOT}/gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv"
TAUT_TEST="${REPO_ROOT}/gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv"

required_files=(
    "${VALIDATOR}"
    "${ORIGIN_TRAIN}"
    "${ORIGIN_TEST}"
    "${TAUT_TRAIN}"
    "${TAUT_TEST}"
)

for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Required file not found: ${path}" >&2
        exit 1
    fi
done

echo "============================================================"
echo "TC-TopoRT SMRT paired-view validation"
echo "Original train : ${ORIGIN_TRAIN}"
echo "Original test  : ${ORIGIN_TEST}"
echo "Tautomer train : ${TAUT_TRAIN}"
echo "Tautomer test  : ${TAUT_TEST}"
echo "============================================================"

"${PYTHON}" "${VALIDATOR}" \
    --origin_train "${ORIGIN_TRAIN}" \
    --origin_test "${ORIGIN_TEST}" \
    --taut_train "${TAUT_TRAIN}" \
    --taut_test "${TAUT_TEST}"

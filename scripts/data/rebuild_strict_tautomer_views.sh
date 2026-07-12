#!/usr/bin/env bash
set -euo pipefail

# Rebuild strict tautomer-standardized SMRT views from the checked-in
# original train/test split, then validate row and RT alignment.
#
# Generated files are written under ARTIFACT_ROOT and should not
# be committed to Git.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts}"

OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/data/strict_tautomer_generated}"

BUILDER="${SCRIPT_DIR}/build_strict_tautomer_views.py"
VALIDATOR="${SCRIPT_DIR}/validate_paired_views.py"

ORIGIN_TRAIN="${REPO_ROOT}/gwn/data/SMRT_train.csv"
ORIGIN_TEST="${REPO_ROOT}/gwn/data/SMRT_test.csv"

GENERATED_TRAIN="${OUTPUT_DIR}/SMRT_train_tautomer_strict.csv"
GENERATED_TEST="${OUTPUT_DIR}/SMRT_test_tautomer_strict.csv"

required_files=(
    "${BUILDER}"
    "${VALIDATOR}"
    "${ORIGIN_TRAIN}"
    "${ORIGIN_TEST}"
)

for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Required file not found: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "TC-TopoRT strict tautomer-view reconstruction"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"

cd "${REPO_ROOT}"

"${PYTHON}" "${BUILDER}" \
    --train_csv "${ORIGIN_TRAIN}" \
    --test_csv "${ORIGIN_TEST}" \
    --out_dir "${OUTPUT_DIR}"

echo
echo "============================================================"
echo "Validate generated paired views"
echo "============================================================"

"${PYTHON}" "${VALIDATOR}" \
    --origin_train "${ORIGIN_TRAIN}" \
    --origin_test "${ORIGIN_TEST}" \
    --taut_train "${GENERATED_TRAIN}" \
    --taut_test "${GENERATED_TEST}"

echo
echo "Strict tautomer views rebuilt and validated."
echo "Generated files: ${OUTPUT_DIR}"

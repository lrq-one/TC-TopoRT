#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# TC-TopoRT structural ablation runner
#
# Usage:
#   bash scripts/ablation/run_structural_ablation.sh no2cell
#   bash scripts/ablation/run_structural_ablation.sh cwn0
#
# Optional environment variables:
#   PYTHON=python
#   SEED=5
#   K_FOLDS=5
#   EPOCHS=150
#   PATIENCE=30
#   BATCH_SIZE=64
#   EVAL_BATCH_SIZE=64
#   NUM_WORKERS=4
#   LR=1e-4
#   WEIGHT_DECAY=1e-2
#   HUBER_BETA=1.0
#   RESUME=0
#   ARTIFACT_ROOT=/path/to/local/artifacts
#
# Generated checkpoints, caches, logs, and results are written
# under ARTIFACT_ROOT and should not be committed to Git.
# ============================================================

MODE="${1:-}"

case "${MODE}" in
    no2cell)
        MAX_RING_SIZE=2
        CWN_LAYERS=6
        CACHE_TAG="no2cell"
        DESCRIPTION="without explicit ring 2-cells"
        ;;
    cwn0)
        MAX_RING_SIZE=6
        CWN_LAYERS=0
        CACHE_TAG="ring6"
        DESCRIPTION="without CWN message passing"
        ;;
    *)
        cat <<'USAGE'
Usage:
  bash scripts/ablation/run_structural_ablation.sh no2cell
  bash scripts/ablation/run_structural_ablation.sh cwn0

Modes:
  no2cell   Set max_ring_size=2 and cwn_layers=6.
  cwn0      Set max_ring_size=6 and cwn_layers=0.
USAGE
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-5}"
K_FOLDS="${K_FOLDS:-5}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
HUBER_BETA="${HUBER_BETA:-1.0}"
RESUME="${RESUME:-0}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts}"

CACHE_ROOT="${CACHE_ROOT:-${ARTIFACT_ROOT}/cache/structural_ablation/${CACHE_TAG}}"
OUT_DIR="${OUT_DIR:-${ARTIFACT_ROOT}/results/structural_ablation/${MODE}_seed${SEED}}"
LOG_DIR="${LOG_DIR:-${ARTIFACT_ROOT}/logs/structural_ablation}"
LOG_FILE="${LOG_DIR}/${MODE}_seed${SEED}.log"

ORIGIN_TRAIN_CSV="${REPO_ROOT}/gwn/data/SMRT_train.csv"
ORIGIN_TEST_CSV="${REPO_ROOT}/gwn/data/SMRT_test.csv"

TAUT_TRAIN_CSV="${REPO_ROOT}/gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv"
TAUT_TEST_CSV="${REPO_ROOT}/gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv"

TRAIN_ENTRY="${REPO_ROOT}/gwn/train_oof_dualview_stack.py"

required_files=(
    "${TRAIN_ENTRY}"
    "${ORIGIN_TRAIN_CSV}"
    "${ORIGIN_TEST_CSV}"
    "${TAUT_TRAIN_CSV}"
    "${TAUT_TEST_CSV}"
)

for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Required file not found: ${path}" >&2
        exit 1
    fi
done

mkdir -p \
    "${CACHE_ROOT}" \
    "${OUT_DIR}" \
    "${LOG_DIR}"

export PYTHONPATH="${REPO_ROOT}/gwn${PYTHONPATH:+:${PYTHONPATH}}"

echo "============================================================"
echo "TC-TopoRT structural ablation"
echo "Mode             : ${MODE}"
echo "Description      : ${DESCRIPTION}"
echo "Seed             : ${SEED}"
echo "Folds            : ${K_FOLDS}"
echo "Maximum ring size: ${MAX_RING_SIZE}"
echo "CWN layers       : ${CWN_LAYERS}"
echo "Resume           : ${RESUME}"
echo "Cache root       : ${CACHE_ROOT}"
echo "Output directory : ${OUT_DIR}"
echo "Log file         : ${LOG_FILE}"
echo "============================================================"

"${PYTHON}" -u "${TRAIN_ENTRY}" \
    --k "${K_FOLDS}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --huber_beta "${HUBER_BETA}" \
    --max_ring_size "${MAX_RING_SIZE}" \
    --cwn_layers "${CWN_LAYERS}" \
    --resume "${RESUME}" \
    --origin_train_csv "${ORIGIN_TRAIN_CSV}" \
    --origin_test_csv "${ORIGIN_TEST_CSV}" \
    --taut_train_csv "${TAUT_TRAIN_CSV}" \
    --taut_test_csv "${TAUT_TEST_CSV}" \
    --origin_train_root "${CACHE_ROOT}/origin_train" \
    --origin_test_root "${CACHE_ROOT}/origin_test" \
    --taut_train_root "${CACHE_ROOT}/taut_train" \
    --taut_test_root "${CACHE_ROOT}/taut_test" \
    --out_dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo
echo "============================================================"
echo "Structural ablation completed"
echo "Mode      : ${MODE}"
echo "Results   : ${OUT_DIR}"
echo "Log       : ${LOG_FILE}"
echo "============================================================"

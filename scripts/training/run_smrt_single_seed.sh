#!/usr/bin/env bash
set -euo pipefail

# TC-TopoRT SMRT single-seed training.
#
# Usage:
#   bash scripts/training/run_smrt_single_seed.sh [SEED]
#
# Examples:
#   bash scripts/training/run_smrt_single_seed.sh 5
#   SEED=5 bash scripts/training/run_smrt_single_seed.sh
#   DRY_RUN=1 bash scripts/training/run_smrt_single_seed.sh 5

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'USAGE'
Usage:
  bash scripts/training/run_smrt_single_seed.sh [SEED]

The positional seed overrides the SEED environment variable.
Generated caches, checkpoints, predictions, metrics, and logs are written
under ARTIFACT_ROOT (default: ./artifacts).
USAGE
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
SEED="${1:-${SEED:-1}}"
K_FOLDS="${K_FOLDS:-5}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
HUBER_BETA="${HUBER_BETA:-1.0}"
MAX_RING_SIZE="${MAX_RING_SIZE:-6}"
CWN_LAYERS="${CWN_LAYERS:-6}"
CWN_HIDDEN="${CWN_HIDDEN:-256}"
STACK_TEMPERATURE="${STACK_TEMPERATURE:-5.0}"
HUBER_ALPHA="${HUBER_ALPHA:-1e-4}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts}"
CACHE_ROOT="${CACHE_ROOT:-${ARTIFACT_ROOT}/cache/smrt_ring6}"
OUT_DIR="${OUT_DIR:-${ARTIFACT_ROOT}/results/smrt/seed${SEED}}"
LOG_DIR="${LOG_DIR:-${ARTIFACT_ROOT}/logs/smrt}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/seed${SEED}.log}"

ENTRY="${REPO_ROOT}/gwn/train_oof_dualview_stack.py"
ORIGIN_TRAIN_CSV="${REPO_ROOT}/gwn/data/SMRT_train.csv"
ORIGIN_TEST_CSV="${REPO_ROOT}/gwn/data/SMRT_test.csv"
TAUT_TRAIN_CSV="${REPO_ROOT}/gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv"
TAUT_TEST_CSV="${REPO_ROOT}/gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv"

for path in \
    "${ENTRY}" \
    "${ORIGIN_TRAIN_CSV}" \
    "${ORIGIN_TEST_CSV}" \
    "${TAUT_TRAIN_CSV}" \
    "${TAUT_TEST_CSV}"
do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Required file not found: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${CACHE_ROOT}" "${OUT_DIR}" "${LOG_DIR}"
export PYTHONPATH="${REPO_ROOT}/gwn${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
    "${PYTHON}" -u "${ENTRY}"
    --k "${K_FOLDS}"
    --seed "${SEED}"
    --epochs "${EPOCHS}"
    --patience "${PATIENCE}"
    --batch_size "${BATCH_SIZE}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --huber_beta "${HUBER_BETA}"
    --max_ring_size "${MAX_RING_SIZE}"
    --cwn_layers "${CWN_LAYERS}"
    --cwn_hidden "${CWN_HIDDEN}"
    --stack_temperature "${STACK_TEMPERATURE}"
    --huber_alpha "${HUBER_ALPHA}"
    --resume "${RESUME}"
    --origin_train_csv "${ORIGIN_TRAIN_CSV}"
    --origin_test_csv "${ORIGIN_TEST_CSV}"
    --taut_train_csv "${TAUT_TRAIN_CSV}"
    --taut_test_csv "${TAUT_TEST_CSV}"
    --origin_train_root "${CACHE_ROOT}/origin_train"
    --origin_test_root "${CACHE_ROOT}/origin_test"
    --taut_train_root "${CACHE_ROOT}/taut_train"
    --taut_test_root "${CACHE_ROOT}/taut_test"
    --out_dir "${OUT_DIR}"
)

printf 'TC-TopoRT seed=%s, folds=%s, output=%s\n' "${SEED}" "${K_FOLDS}" "${OUT_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY RUN] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

#!/usr/bin/env bash
set -euo pipefail

# Reproduce the conventional atom-bond GNN comparison reported
# in the TC-TopoRT Supporting Information.
#
# The default configuration corresponds to the reported seed-5 result.
#
# Usage:
#   bash scripts/ablation/run_atom_bond_gnn.sh
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
#   ARTIFACT_ROOT=/path/to/local/artifacts

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

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts}"
OUT_DIR="${OUT_DIR:-${ARTIFACT_ROOT}/results/atom_bond_gnn/seed${SEED}}"
CACHE_DIR="${CACHE_DIR:-${ARTIFACT_ROOT}/cache/atom_bond_gnn}"

ENTRY="${REPO_ROOT}/scripts/ablation/train_atom_bond_gnn.py"

if [[ ! -f "${ENTRY}" ]]; then
    echo "[ERROR] Training entry not found: ${ENTRY}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}" "${CACHE_DIR}"

echo "============================================================"
echo "TC-TopoRT conventional atom-bond GNN baseline"
echo "Seed       : ${SEED}"
echo "Folds      : ${K_FOLDS}"
echo "Epochs     : ${EPOCHS}"
echo "Output     : ${OUT_DIR}"
echo "Cache      : ${CACHE_DIR}"
echo "============================================================"

cd "${REPO_ROOT}"

"${PYTHON}" -u "${ENTRY}" \
    --seed "${SEED}" \
    --k "${K_FOLDS}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --huber_beta 1.0 \
    --hidden 256 \
    --layers 6 \
    --dropout 0.0 \
    --huber_alpha 1e-4 \
    --scheduler_t0 20 \
    --strata_bins 10 \
    --grad_clip 5.0 \
    --out_dir "${OUT_DIR}" \
    --cache_dir "${CACHE_DIR}"

echo
echo "Completed."
echo "Results: ${OUT_DIR}"

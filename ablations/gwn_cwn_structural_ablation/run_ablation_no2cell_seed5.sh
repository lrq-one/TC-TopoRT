#!/usr/bin/env bash
set -euo pipefail

# Reproduce structural ablation:
# w/o explicit ring 2-cells
#
# This uses the same dual-view OOF training entry as the full model,
# but sets max_ring_size=2 so no explicit molecular ring 2-cells are lifted.
# CWN message passing is kept with cwn_layers=6.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH=.

OUT_DIR="results_Ablation_No2Cell_DualView_Stack_seed5"
LOG_DIR="logs_ablation_cwn"
LOG_FILE="${LOG_DIR}/no2cell_dualview_stack_seed5.log"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Ablation A: w/o explicit ring 2-cells"
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "============================================================"

echo
echo "=== Start No2Cell training/resume ==="
echo "This keeps cwn_layers=6 but sets max_ring_size=2."

PYTHONPATH=. python -u train_oof_dualview_stack.py \
  --k 5 \
  --seed 5 \
  --epochs 150 \
  --patience 30 \
  --batch_size 64 \
  --eval_batch_size 64 \
  --num_workers 4 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --huber_beta 1.0 \
  --max_ring_size 2 \
  --cwn_layers 6 \
  --resume 1 \
  --origin_train_csv data/SMRT_train.csv \
  --origin_test_csv data/SMRT_test.csv \
  --taut_train_csv data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv \
  --taut_test_csv data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv \
  --origin_train_root smrt_ablation_no2cell_origin_train \
  --origin_test_root smrt_ablation_no2cell_origin_test \
  --taut_train_root smrt_ablation_no2cell_taut_train \
  --taut_test_root smrt_ablation_no2cell_taut_test \
  --out_dir "${OUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"

echo
echo "============================================================"
echo "No2Cell ablation finished."
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "============================================================"

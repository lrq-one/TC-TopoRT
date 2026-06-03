#!/usr/bin/env bash
set -e

SEEDS=(79 123 256 5)

for SEED in "${SEEDS[@]}"; do
  OUT_DIR="results_OOF_DualView_Stack_seed${SEED}"
  LOG_FILE="oof_dualview_seed${SEED}.log"

  echo "============================================================"
  echo "Running seed=${SEED}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
  echo "============================================================"

  PYTHONPATH=. python train_oof_dualview_stack.py \
    --k 5 \
    --seed ${SEED} \
    --epochs 150 \
    --patience 30 \
    --batch_size 64 \
    --eval_batch_size 64 \
    --num_workers 4 \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --huber_beta 1.0 \
    --origin_train_csv data/SMRT_train.csv \
    --origin_test_csv data/SMRT_test.csv \
    --taut_train_csv data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv \
    --taut_test_csv data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv \
    --origin_train_root smrt_cwn_oof_origin_train \
    --origin_test_root smrt_cwn_oof_origin_test \
    --taut_train_root smrt_cwn_oof_taut_train \
    --taut_test_root smrt_cwn_oof_taut_test \
    --out_dir "${OUT_DIR}" 2>&1 | tee "${LOG_FILE}"

  echo "Finished seed=${SEED}"
done

echo "All seeds finished."

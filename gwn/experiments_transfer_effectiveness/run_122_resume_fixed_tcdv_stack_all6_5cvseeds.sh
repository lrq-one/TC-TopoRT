#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/lrq_q/ABCoRT-main/gwn
export PYTHONPATH=.

mkdir -p experiments_transfer_effectiveness/logs

CVSEEDS=(1 12 123 1234 12345)

for CVSEED in "${CVSEEDS[@]}"
do
  BASE_DIR="experiments_transfer_effectiveness/results_122_base_tl_all6_seed5_src0to4_cvseed_${CVSEED}"
  STACK_DIR="experiments_transfer_effectiveness/results_122_fixed_tcdv_stack_all6_seed5_src0to4_cvseed_${CVSEED}"
  BASE_CSV="${BASE_DIR}/external_tl_predictions.csv"
  STACK_SUMMARY="${STACK_DIR}/tcdv_fixed_noleak_stack_summary.csv"

  echo "============================================================"
  echo "cv_seed=${CVSEED}"
  echo "============================================================"

  if python experiments_transfer_effectiveness/126_validate_122_base_predictions.py --pred_csv "${BASE_CSV}"
  then
    echo "[SKIP 119] base predictions are complete: ${BASE_CSV}"
  else
    echo "[RERUN 119] base predictions missing or incomplete. Remove and rerun: ${BASE_DIR}"
    rm -rf "${BASE_DIR}"

    python -u experiments_transfer_effectiveness/119_external_tcdv_scratch_vs_tl.py \
      --out_dir "${BASE_DIR}" \
      --datasets Eawag_XBridgeC18_364 FEM_lipids_72 FEM_long_412 IPB_Halle_82 LIFE_new_184 LIFE_old_194 \
      --run_keys seed5 \
      --source_folds 0 1 2 3 4 \
      --init_mode tl \
      --freeze_mode rt_head_full \
      --reset_out_lin 1 \
      --cv_folds 10 \
      --cv_seed "${CVSEED}" \
      --epochs 150 \
      --early_stop_train 999 \
      --batch_size 8 \
      --eval_batch_size 64 \
      --lr 1e-4 \
      2>&1 | tee "experiments_transfer_effectiveness/logs/results_122_base_tl_all6_seed5_src0to4_cvseed_${CVSEED}.log"
  fi

  if [ -s "${STACK_SUMMARY}" ]
  then
    echo "[SKIP 122] stack summary exists: ${STACK_SUMMARY}"
  else
    echo "[RUN 122] no-leak stack for cv_seed=${CVSEED}"
    rm -rf "${STACK_DIR}"

    python -u experiments_transfer_effectiveness/122_external_tcdv_fixed_oof_stack.py \
      --pred_csv "${BASE_CSV}" \
      --out_dir "${STACK_DIR}" \
      --source_folds 0 1 2 3 4 \
      --stacker huber \
      2>&1 | tee "experiments_transfer_effectiveness/logs/results_122_fixed_tcdv_stack_all6_seed5_src0to4_cvseed_${CVSEED}.log"
  fi
done

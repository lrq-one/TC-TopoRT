#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/lrq_q/ABCoRT-main/gwn
export PYTHONPATH=.

rm -rf experiments_transfer_effectiveness/abcort_like_fixed_tl_hard3_seed5_fold0

for CVSEED in 1 12 123 1234 12345
do
  echo "=============================="
  echo "ABCoRT-like fixed TL hard3 cv_seed=${CVSEED}"
  echo "=============================="

  python -u experiments_transfer_effectiveness/119_external_tcdv_scratch_vs_tl.py \
    --out_dir experiments_transfer_effectiveness/abcort_like_fixed_tl_hard3_seed5_fold0/cvseed_${CVSEED} \
    --datasets IPB_Halle_82 Eawag_XBridgeC18_364 FEM_long_412 \
    --run_keys seed5 \
    --source_folds 0 \
    --init_mode tl \
    --freeze_mode rt_head_full \
    --reset_out_lin 0 \
    --cv_folds 10 \
    --cv_seed ${CVSEED} \
    --epochs 150 \
    --early_stop_train 999 \
    --batch_size 8 \
    --eval_batch_size 64 \
    --lr 1e-4 \
    2>&1 | tee experiments_transfer_effectiveness/logs/abcort_like_fixed_tl_hard3_seed5_fold0_cvseed_${CVSEED}.log
done

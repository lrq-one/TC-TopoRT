#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

mkdir -p experiments_transfer_effectiveness/logs

python -u experiments_transfer_effectiveness/external_scratch_all10.py \
  --out_dir experiments_transfer_effectiveness/results_external_scratch_all10_seed5_cvseed1 \
  --run_key seed5 \
  --source_fold 0 \
  --cv_seed 1 \
  --cv_folds 10 \
  --epochs 150 \
  --early_stop_train 999 \
  --batch_size 8 \
  --eval_batch_size 64 \
  --lr 1e-4 \
  2>&1 | tee experiments_transfer_effectiveness/logs/scratch_all10_seed5_cvseed1.log

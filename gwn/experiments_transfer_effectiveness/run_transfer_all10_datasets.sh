#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

mkdir -p experiments_transfer_effectiveness/logs

python -u experiments_transfer_effectiveness/136_external_transfer_all10.py \
  --out_root experiments_transfer_effectiveness/fixed_raw_autoselect_all10_cvseed1 \
  --cv_seeds 1 \
  --run_keys seed5 \
  --source_folds 0 1 2 3 4 \
  --epochs 150 \
  --batch_size 8 \
  --eval_batch_size 64 \
  --lr 1e-4 \
  --skip_existing_base 1 \
  --skip_existing_stack 1 \
  2>&1 | tee experiments_transfer_effectiveness/logs/transfer_all10_cvseed1.log

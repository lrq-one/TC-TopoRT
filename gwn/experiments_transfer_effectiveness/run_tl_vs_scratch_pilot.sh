#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p experiments_transfer_effectiveness/logs

DATASETS=(FEM_lipids_72 IPB_Halle_82 LIFE_old_194)

echo "=== Controlled TL pilot ==="
PYTHONPATH=. python -u experiments_transfer_effectiveness/119_external_tcdv_scratch_vs_tl.py \
  --out_dir experiments_transfer_effectiveness/results_pilot_tl_seed1_src0 \
  --datasets "${DATASETS[@]}" \
  --run_keys seed1 \
  --source_folds 0 \
  --init_mode tl \
  --freeze_mode rt_head_full \
  --reset_out_lin 1 \
  --cv_folds 10 \
  --epochs 150 \
  --early_stop_train 30 \
  --batch_size 8 \
  --eval_batch_size 64 \
  --lr 1e-4 \
  2>&1 | tee experiments_transfer_effectiveness/logs/pilot_tl_seed1_src0.log

echo "=== Controlled scratch pilot ==="
PYTHONPATH=. python -u experiments_transfer_effectiveness/119_external_tcdv_scratch_vs_tl.py \
  --out_dir experiments_transfer_effectiveness/results_pilot_scratch_seed1 \
  --datasets "${DATASETS[@]}" \
  --run_keys seed1 \
  --source_folds 0 \
  --init_mode scratch \
  --freeze_mode all \
  --reset_out_lin 0 \
  --cv_folds 10 \
  --epochs 150 \
  --early_stop_train 30 \
  --batch_size 8 \
  --eval_batch_size 64 \
  --lr 1e-4 \
  2>&1 | tee experiments_transfer_effectiveness/logs/pilot_scratch_seed1.log

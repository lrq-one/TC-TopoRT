#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p experiments_transfer_effectiveness/logs

mapfile -t DATASETS < <(python - <<'PY'
import pandas as pd
df = pd.read_csv("paper_analysis_stage4_external/external_predret10_stage4_meta.csv")
vc = df["dataset_name"].value_counts()
# 和 ABCoRT Figure 4 对齐：只跑样本数 >= 30 的外部数据集
for name in sorted(vc[vc >= 30].index):
    print(name)
PY
)

echo "=== Datasets for Figure4-style TL effectiveness ==="
printf '%s\n' "${DATASETS[@]}"

echo "=== TL pretrained branch ==="
PYTHONPATH=. python -u experiments_transfer_effectiveness/external_train_tcdv_transfer_or_scratch.py \
  --out_dir experiments_transfer_effectiveness/results_figure4_tl_seed1_src0 \
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
  2>&1 | tee experiments_transfer_effectiveness/logs/figure4_tl_seed1_src0.log

echo "=== Scratch random-init branch ==="
PYTHONPATH=. python -u experiments_transfer_effectiveness/external_train_tcdv_transfer_or_scratch.py \
  --out_dir experiments_transfer_effectiveness/results_figure4_scratch_seed1 \
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
  2>&1 | tee experiments_transfer_effectiveness/logs/figure4_scratch_seed1.log

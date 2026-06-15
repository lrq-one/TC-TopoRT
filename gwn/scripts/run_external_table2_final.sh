#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-all}"

cd "$(dirname "$0")/.."
mkdir -p final_external_results

run_eawag_stack () {
  PYTHONPATH=. python -u diagnostics/108_external_noleak_learned_stacking.py \
    --out_dir final_external_results/Eawag_final_top5 \
    --datasets Eawag_XBridgeC18_364 \
    --top_n 5 \
    --cv_seed 1
}

run_femlong_stack () {
  PYTHONPATH=. python -u diagnostics/108_external_noleak_learned_stacking.py \
    --out_dir final_external_results/FEMlong_final_top18 \
    --datasets FEM_long_412 \
    --top_n 18 \
    --cv_seed 1
}

collect_table () {
  PYTHONPATH=. python diagnostics/109_external_table2_result_audit.py \
    | tee final_external_results/table2_best_current.log

  PYTHONPATH=. python diagnostics/110_external_table2_from_manifest.py \
    | tee final_external_results/table2_final_from_manifest.log
}

case "$DATASET" in
  eawag|Eawag_XBridgeC18_364)
    run_eawag_stack
    ;;
  femlong|FEM_long_412)
    run_femlong_stack
    ;;
  collect)
    collect_table
    ;;
  all)
    run_eawag_stack
    run_femlong_stack
    collect_table
    ;;
  *)
    echo "[ERROR] unknown DATASET=$DATASET"
    exit 1
    ;;
esac

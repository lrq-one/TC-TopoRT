# TCDV-TopoRT experiment-to-code map

This file maps manuscript experiments to code entry points, run scripts, result folders, and final paper tables.

## 1. Main SMRT 5-seed TCDV-TopoRT benchmark

Purpose:

Train the final dual-view topology-aware RT predictor on SMRT.

Code entry:

- gwn/train_oof_dualview_stack.py

Model chain:

- gwn/train_oof_dualview_stack.py
- gwn/net/topocellrt_cwn_replace.py
- gwn/net/cwn_hypergraph_adapter.py
- gwn/net/cwn_abcort_transformer.py
- gwn/net/cwn.py

Run scripts:

- Single seed example: gwn/train_oof_dualview_stack.py
- Additional seeds: gwn/run_oof_multiseed_5runs.sh

Single seed command:

cd gwn

PYTHONPATH=. python train_oof_dualview_stack.py \
  --k 5 \
  --seed 1 \
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
  --out_dir results_OOF_DualView_Stack_v1

Formal result directories:

- gwn/results_OOF_DualView_Stack_v1
- gwn/results_OOF_DualView_Stack_seed5
- gwn/results_OOF_DualView_Stack_seed79
- gwn/results_OOF_DualView_Stack_seed123
- gwn/results_OOF_DualView_Stack_seed256

Final tables:

- paper_results_TCDV_TopoRT/tables/01_smrt_single_seed_results.csv
- paper_results_TCDV_TopoRT/tables/02_smrt_5seed_prediction_ensemble_summary.csv

## 2. Dual-view and fusion ablation

Purpose:

Compare original-view only, tautomer-view only, mean fusion, and OOF Huber stacking.

Main source predictions:

- gwn/results_OOF_DualView_Stack_*/oof_base_predictions.csv
- gwn/results_OOF_DualView_Stack_*/test_base_predictions.csv
- gwn/results_OOF_DualView_Stack_*/oof_predictions.csv
- gwn/results_OOF_DualView_Stack_*/test_predictions.csv

Analysis code:

- gwn/diagnostics/113_smrt_dualview_ablation_from_predictions.py
- gwn/diagnostics/118_build_final_paper_tables.py

Final table:

- paper_results_TCDV_TopoRT/tables/03_dualview_and_fusion_ablation.csv

## 3. Structural ablation

Purpose:

Measure the contributions of explicit ring 2-cells and CWN message passing.

Shared training code:

- ablations/gwn_cwn_structural_ablation/train_oof_dualview_stack.py

Full TCDV-TopoRT seed5:

- Result: gwn/results_OOF_DualView_Stack_seed5/final_metrics.json
- max_ring_size = 6
- cwn_layers = 6

w/o explicit ring 2-cells:

- Result: ablations/gwn_cwn_structural_ablation/results_Ablation_No2Cell_DualView_Stack_seed5/final_metrics.json
- max_ring_size = 2
- cwn_layers = 6
- origin_train_root = smrt_ablation_no2cell_origin_train
- origin_test_root = smrt_ablation_no2cell_origin_test
- taut_train_root = smrt_ablation_no2cell_taut_train
- taut_test_root = smrt_ablation_no2cell_taut_test

Note:

The No2Cell result exists and its config is saved. The reproduction helper script has now been added:

- ablations/gwn_cwn_structural_ablation/run_ablation_no2cell_seed5.sh

w/o CWN message passing:

- Run script: ablations/gwn_cwn_structural_ablation/run_ablation_cwn0_seed5.sh
- Result: ablations/gwn_cwn_structural_ablation/results_Ablation_CWN0_DualView_Stack_seed5/final_metrics.json
- max_ring_size = 6
- cwn_layers = 0

Collector:

- ablations/gwn_cwn_structural_ablation/collect_cwn_structural_ablation_seed5.py

Final table:

- paper_results_TCDV_TopoRT/tables/04_structural_ablation_seed5.csv

## 4. External transfer Table 2

Purpose:

Evaluate fixed raw AutoSelect transfer on six external datasets.

Code directory:

- gwn/experiments_transfer_effectiveness

Important scripts:

- gwn/experiments_transfer_effectiveness/external_train_tcdv_transfer_or_scratch.py
- gwn/experiments_transfer_effectiveness/external_stack_fixed_oof.py
- gwn/experiments_transfer_effectiveness/external_stack_fixed_oof_from_wide.py
- gwn/experiments_transfer_effectiveness/external_stack_fixed_raw_autoselect.py
- gwn/experiments_transfer_effectiveness/validate_external_base_predictions.py
- gwn/experiments_transfer_effectiveness/save_external_transfer_result_package.py
- gwn/experiments_transfer_effectiveness/external_run_fixed_raw_autoselect.py

Formal result package:

- gwn/experiments_transfer_effectiveness/paper_saved_external_transfer_fixed_raw_autoselect_cvseed1
- paper_results_TCDV_TopoRT/external_transfer

Final table:

- paper_results_TCDV_TopoRT/tables/05_external_transfer_fixed_raw_autoselect.csv

## 5. Transfer learning versus scratch

Purpose:

Compare transfer-pretrained TCDV-TopoRT against scratch/random initialization on external datasets.

Code directory:

- gwn/experiments_transfer_effectiveness

Important scripts:

- gwn/experiments_transfer_effectiveness/133_external_scratch_only.py
- gwn/experiments_transfer_effectiveness/134_external_tcdv_transfer_only.py
- gwn/experiments_transfer_effectiveness/135_make_tl_vs_scratch_figure.py
- gwn/experiments_transfer_effectiveness/run_figure4_tl_vs_scratch_full.sh

Final outputs:

- paper_results_TCDV_TopoRT/tables/06_tl_vs_scratch_summary.csv
- paper_results_TCDV_TopoRT/tables/06b_tl_vs_scratch_overall_summary.csv
- paper_results_TCDV_TopoRT/tl_vs_scratch
- manuscript_figures_final/fig_tl_vs_scratch_bar.pdf
- manuscript_figures_final/fig_tl_vs_scratch_bar.png

## 6. Candidate filtering and guarded soft reranking

Purpose:

Evaluate RT-aware reranking/filtering on MetaboBase and RIKEN_PlaSMA candidate sets.

Code and analysis locations:

- gwn/experiments_candidate_filtering
- gwn/diagnostics/63_candidate_filtering_smrt_formula_pool.py  # archived in ../TCDV-TopoRT_cleanup_backups/round14_gwn_obsolete_diagnostics_code_20260627
- gwn/diagnostics/86_eval_tl_rank_guard_and_soft_rerank.py  # archived in ../TCDV-TopoRT_cleanup_backups/round14_gwn_obsolete_diagnostics_code_20260627
- gwn/diagnostics/97_eval_evaluable45_rank_guard_soft.py
- gwn/diagnostics/99_make_final_experiment_A_table.py
- gwn/diagnostics/108_make_final_riken_experiment_tables.py

Formal results:

- paper_results_TCDV_TopoRT/candidate_filtering
- paper_results_TCDV_TopoRT/tables/07_candidate_filtering_reranking_summary.csv
- manuscript_figures_final/*candidate*

Preserved checkpoints:

- gwn/experiments_candidate_filtering/**/best_model.pth

## 7. Final table generation

Code:

- gwn/diagnostics/118_build_final_paper_tables.py

Final result root:

- paper_results_TCDV_TopoRT

## 8. Current cleanup rule

Do not remove:

- gwn/results_OOF_DualView_Stack_*
- gwn/smrt_cwn_oof_*
- ablations/gwn_cwn_structural_ablation
- gwn/experiments_candidate_filtering/**/best_model.pth
- paper_results_TCDV_TopoRT
- manuscript_figures_final
- gwn/final_smrt_results
- gwn/final_external_results

Future renaming should be done only after this map is complete.

## External all10 transfer vs scratch entry points

The cleaned external experiment wrappers are:

- `gwn/experiments_transfer_effectiveness/external_transfer_all10.py`
- `gwn/experiments_transfer_effectiveness/external_scratch_all10.py`
- `gwn/experiments_transfer_effectiveness/run_transfer_all10_datasets.sh`
- `gwn/experiments_transfer_effectiveness/run_scratch_all10_datasets.sh`

Dataset list:

- `gwn/configs/external_all10_datasets.csv`

The all10 list combines the previous six Table-2 external datasets with the four additional external datasets used for transfer-vs-scratch comparison.

## External all10 transfer-vs-scratch result figure

Existing all10 transfer-vs-scratch results are summarized and plotted by:

- `gwn/experiments_transfer_effectiveness/make_external_all10_transfer_vs_scratch_figure.py`

This script does not train models. It formats the already obtained all10 scratch and transfer-learning MAE values into paper-ready CSV/Markdown/TXT summaries and a bar figure.

Output directory:

- `gwn/experiments_transfer_effectiveness/all10_transfer_vs_scratch_final/`

## Semantic external experiment script names

The external transfer/scratch scripts were renamed from numbered development names to semantic task-oriented names.

Main external experiment scripts:

- `gwn/experiments_transfer_effectiveness/external_train_tcdv_transfer_or_scratch.py`
- `gwn/experiments_transfer_effectiveness/external_run_fixed_raw_autoselect.py`
- `gwn/experiments_transfer_effectiveness/external_stack_fixed_raw_autoselect.py`
- `gwn/experiments_transfer_effectiveness/external_transfer_all10.py`
- `gwn/experiments_transfer_effectiveness/external_scratch_all10.py`
- `gwn/experiments_transfer_effectiveness/make_external_all10_transfer_vs_scratch_figure.py`

The older numbered names were development-stage names and should not be used as the paper-facing entry points.

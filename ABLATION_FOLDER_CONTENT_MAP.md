# Ablation folder content map

Target folder:

- ablations/gwn_cwn_structural_ablation

This folder is a standalone copied subproject for the CWN structural ablation experiments of TCDV-TopoRT.

## 1. Core purpose

This folder preserves the code, data, checkpoints, predictions, logs, and paper-ready summaries needed for the seed-5 structural ablation experiments:

- Full / normal CWN setting
- w/o explicit ring 2-cells, also called No2Cell
- w/o CWN message passing, also called CWN0

The main formal outputs are used for the structural ablation table in the paper.

## 2. Must keep

### Core data

- data/SMRT_train.csv
- data/SMRT_test.csv
- data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv
- data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv
- data_taut_strict_origin_order/*_reorder_audit.csv

Purpose:

- Original SMRT view and strict tautomer-canonical view for dual-view training.
- Required by train_oof_dualview_stack.py and the ablation run scripts.

### Core model and dataset code

- mp/
- net/
- train_oof_dualview_stack.py

Purpose:

- mp/ contains SMRT dataset construction, Complex/ComplexBatch objects, ring/cell-complex processing, and data utilities.
- net/ contains CWN and TCDV-TopoRT model modules.
- train_oof_dualview_stack.py is the dual-view OOF training entry.

### Structural ablation scripts

- run_ablation_cwn0_seed5.sh
- run_ablation_no2cell_seed5.sh
- collect_cwn_structural_ablation_seed5.py

Purpose:

- run_ablation_cwn0_seed5.sh reproduces the w/o CWN ablation.
- run_ablation_no2cell_seed5.sh reproduces the w/o explicit ring 2-cells ablation.
- collect_cwn_structural_ablation_seed5.py collects Full / No2Cell / CWN0 results into final structural ablation summaries.

### Formal structural ablation results

- results_Ablation_CWN0_DualView_Stack_seed5/
- results_Ablation_No2Cell_DualView_Stack_seed5/
- paper_saved_ablation_cwn0_seed5/
- paper_saved_ablation_no2cell_seed5/
- paper_saved_cwn_structural_ablation_seed5/

Purpose:

- These contain formal metrics, predictions, logs, summaries, folds, and checkpoints for the structural ablation table.
- The result folders contain full training artifacts and checkpoints.
- The paper_saved folders contain compact paper-facing outputs.

### Dataset cache roots

- smrt_ablation_no2cell_origin_train/
- smrt_ablation_no2cell_origin_test/
- smrt_ablation_no2cell_taut_train/
- smrt_ablation_no2cell_taut_test/
- smrt_cwn_oof_origin_train/
- smrt_cwn_oof_origin_test/
- smrt_cwn_oof_taut_train/
- smrt_cwn_oof_taut_test/

Purpose:

- smrt_ablation_no2cell_* stores the r2 processed cache for No2Cell.
- smrt_cwn_oof_* is the r6 cache root used by CWN-style runs. In this subproject it may be rebuilt automatically by the run script if the processed files are absent.
- These are local cache folders and are ignored by Git.

## 3. Diagnostics kept in this subproject

After cleanup, diagnostics/ keeps only 14 SMRT / OOF / dual-view / figure / paper-analysis scripts:

- 40_check_dualview_pair_data.py
- 41_reorder_existing_taut_to_origin_order.py
- 60_oof_ablation_subgroup_shuffle_error.py
- 61_safe_fusion_scan.py
- 62_integrity_stats_rtbin_stacker.py
- 74a_inspect_tcdv_model_loading.py
- 91_make_panelA_rdkit_views.py
- 92_make_panelA_compact_white.py
- 111_smrt_multiseed_result_audit.py
- 112_smrt_main_from_final_metrics.py
- 113_smrt_dualview_ablation_from_predictions.py
- 114_smrt_taut_changed_subgroup.py
- 115_smrt_shuffle_taut_pairing_ablation.py
- 116_smrt_tail_hard_molecule_analysis.py

Purpose:

- These scripts are kept because they relate to SMRT, dual-view pairing, OOF/fusion analysis, tail analysis, or paper figure/table generation.

## 4. Moved out during cleanup

The following non-structural or duplicate peripheral files were moved to backup, not deleted:

- run_ablation_cwn0_seed5.sh.bak_conda_error
- configs/external_table2_final_manifest.csv
- scripts/run_external_table2_final.sh
- external_data/
- external_splits/
- compare_all_external_tl_vs_ABCORT.csv
- compare_stage4AD_new_results_vs_ABCoRT.csv
- existing_prediction_capacity_check.csv
- diagnostics/external_candidate_filtering_audit/
- 86 duplicate diagnostics scripts that also exist identically under gwn/diagnostics/

Backup locations:

- ../TCDV-TopoRT_cleanup_backups/round6_ablation_nonstructural_peripheral_20260627
- ../TCDV-TopoRT_cleanup_backups/round7_ablation_diagnostics_duplicates_20260627

## 5. Do not delete

Do not delete the following during future cleanup:

- results_Ablation_CWN0_DualView_Stack_seed5/
- results_Ablation_No2Cell_DualView_Stack_seed5/
- paper_saved_ablation_cwn0_seed5/
- paper_saved_ablation_no2cell_seed5/
- paper_saved_cwn_structural_ablation_seed5/
- run_ablation_cwn0_seed5.sh
- run_ablation_no2cell_seed5.sh
- collect_cwn_structural_ablation_seed5.py
- mp/
- net/
- train_oof_dualview_stack.py
- data/
- data_taut_strict_origin_order/

These are the structural ablation reproduction assets.
===== append live checks =====

```text
Remaining diagnostics files:
ablations/gwn_cwn_structural_ablation/diagnostics/111_smrt_multiseed_result_audit.py
ablations/gwn_cwn_structural_ablation/diagnostics/112_smrt_main_from_final_metrics.py
ablations/gwn_cwn_structural_ablation/diagnostics/113_smrt_dualview_ablation_from_predictions.py
ablations/gwn_cwn_structural_ablation/diagnostics/114_smrt_taut_changed_subgroup.py
ablations/gwn_cwn_structural_ablation/diagnostics/115_smrt_shuffle_taut_pairing_ablation.py
ablations/gwn_cwn_structural_ablation/diagnostics/116_smrt_tail_hard_molecule_analysis.py
ablations/gwn_cwn_structural_ablation/diagnostics/40_check_dualview_pair_data.py
ablations/gwn_cwn_structural_ablation/diagnostics/41_reorder_existing_taut_to_origin_order.py
ablations/gwn_cwn_structural_ablation/diagnostics/60_oof_ablation_subgroup_shuffle_error.py
ablations/gwn_cwn_structural_ablation/diagnostics/61_safe_fusion_scan.py
ablations/gwn_cwn_structural_ablation/diagnostics/62_integrity_stats_rtbin_stacker.py
ablations/gwn_cwn_structural_ablation/diagnostics/74a_inspect_tcdv_model_loading.py
ablations/gwn_cwn_structural_ablation/diagnostics/91_make_panelA_rdkit_views.py
ablations/gwn_cwn_structural_ablation/diagnostics/92_make_panelA_compact_white.py

Remaining diagnostics count:
14

No2Cell checkpoints:
10
CWN0 checkpoints:
10

Ablation folder size:
3.3G	ablations/gwn_cwn_structural_ablation
1.1G	ablations/gwn_cwn_structural_ablation/results_Ablation_No2Cell_DualView_Stack_seed5/folds
1.1G	ablations/gwn_cwn_structural_ablation/results_Ablation_No2Cell_DualView_Stack_seed5
876M	ablations/gwn_cwn_structural_ablation/smrt_ablation_no2cell_taut_train/processed_r2_Full46D_Embedded_E
876M	ablations/gwn_cwn_structural_ablation/smrt_ablation_no2cell_taut_train
876M	ablations/gwn_cwn_structural_ablation/smrt_ablation_no2cell_origin_train/processed_r2_Full46D_Embedded_E
876M	ablations/gwn_cwn_structural_ablation/smrt_ablation_no2cell_origin_train
254M	ablations/gwn_cwn_structural_ablation/results_Ablation_CWN0_DualView_Stack_seed5
216M	ablations/gwn_cwn_structural_ablation/results_Ablation_CWN0_DualView_Stack_seed5/folds
97M	ablations/gwn_cwn_structural_ablation/smrt_ablation_no2cell_taut_test/processed_r2_Full46D_Embedded_E
97M	ablations/gwn_cwn_structural_ablation/smrt_ablation_no2cell_taut_test
97M	ablations/gwn_cwn_structural_ablation/smrt_ablation_no2cell_origin_test/processed_r2_Full46D_Embedded_E
97M	ablations/gwn_cwn_structural_ablation/smrt_ablation_no2cell_origin_test
36M	ablations/gwn_cwn_structural_ablation/paper_saved_ablation_no2cell_seed5
36M	ablations/gwn_cwn_structural_ablation/paper_saved_ablation_cwn0_seed5
26M	ablations/gwn_cwn_structural_ablation/data_taut_strict_origin_order
4.6M	ablations/gwn_cwn_structural_ablation/data
452K	ablations/gwn_cwn_structural_ablation/figure_assets
264K	ablations/gwn_cwn_structural_ablation/figure_assets/panelA_compact_white
196K	ablations/gwn_cwn_structural_ablation/diagnostics
184K	ablations/gwn_cwn_structural_ablation/figure_assets/panelA_rdkit_views
160K	ablations/gwn_cwn_structural_ablation/mp
40K	ablations/gwn_cwn_structural_ablation/net
16K	ablations/gwn_cwn_structural_ablation/paper_saved_cwn_structural_ablation_seed5
16K	ablations/gwn_cwn_structural_ablation/paper_final_results
12K	ablations/gwn_cwn_structural_ablation/smrt_cwn_oof_taut_train
12K	ablations/gwn_cwn_structural_ablation/smrt_cwn_oof_taut_test
12K	ablations/gwn_cwn_structural_ablation/smrt_cwn_oof_origin_train
12K	ablations/gwn_cwn_structural_ablation/smrt_cwn_oof_origin_test
4.0K	ablations/gwn_cwn_structural_ablation/smrt_cwn_oof_taut_train/raw
4.0K	ablations/gwn_cwn_structural_ablation/smrt_cwn_oof_taut_train/processed_r6_Full46D_Embedded_E
4.0K	ablations/gwn_cwn_structural_ablation/smrt_cwn_oof_taut_test/raw
4.0K	ablations/gwn_cwn_structural_ablation/smrt_cwn_oof_taut_test/processed_r6_Full46D_Embedded_E
4.0K	ablations/gwn_cwn_structural_ablation/smrt_cwn_oof_origin_train/raw
4.0K	ablations/gwn_cwn_structural_ablation/smrt_cwn_oof_origin_train/processed_r6_Full46D_Embedded_E
4.0K	ablations/gwn_cwn_structural_ablation/smrt_cwn_oof_origin_test/raw
4.0K	ablations/gwn_cwn_structural_ablation/smrt_cwn_oof_origin_test/processed_r6_Full46D_Embedded_E
4.0K	ablations/gwn_cwn_structural_ablation/smrt_ablation_no2cell_taut_train/raw
4.0K	ablations/gwn_cwn_structural_ablation/smrt_ablation_no2cell_taut_test/raw
4.0K	ablations/gwn_cwn_structural_ablation/smrt_ablation_no2cell_origin_train/raw
4.0K	ablations/gwn_cwn_structural_ablation/smrt_ablation_no2cell_origin_test/raw
4.0K	ablations/gwn_cwn_structural_ablation/scripts
4.0K	ablations/gwn_cwn_structural_ablation/experiments_cwn_structure_ablation
4.0K	ablations/gwn_cwn_structural_ablation/configs
```

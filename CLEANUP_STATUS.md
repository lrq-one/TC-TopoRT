# TCDV-TopoRT cleanup status

Date: 2026-06-27

## Current status

The repository has been cleaned to the current stable state.

Final cleanup check showed:

- git status --short: clean
- git diff --stat: clean
- git ls-files -d: clean

No core source code, model code, data loader, diagnostics, or formal result directory was unintentionally modified.

## Preserved formal experiment assets

### Main SMRT 5-seed OOF runs

Preserved result directories:

- gwn/results_OOF_DualView_Stack_v1
- gwn/results_OOF_DualView_Stack_seed5
- gwn/results_OOF_DualView_Stack_seed79
- gwn/results_OOF_DualView_Stack_seed123
- gwn/results_OOF_DualView_Stack_seed256

Expected checkpoints:

- 5 seeds x 5 folds x 2 views = 50 best_model.pth

Observed after cleanup:

- OOF checkpoints: 50

### Structural ablation

Preserved directories:

- ablations/gwn_cwn_structural_ablation/results_Ablation_No2Cell_DualView_Stack_seed5
- ablations/gwn_cwn_structural_ablation/results_Ablation_CWN0_DualView_Stack_seed5
- ablations/gwn_cwn_structural_ablation/paper_saved_ablation_no2cell_seed5
- ablations/gwn_cwn_structural_ablation/paper_saved_ablation_cwn0_seed5
- ablations/gwn_cwn_structural_ablation/paper_saved_cwn_structural_ablation_seed5

Observed after cleanup:

- No2Cell checkpoints: 10
- CWN0 checkpoints: 10

### Candidate filtering

Preserved directory:

- gwn/experiments_candidate_filtering

Observed after cleanup:

- candidate filtering checkpoints: 8

### Final paper results

Preserved directories:

- paper_results_TCDV_TopoRT
- manuscript
- manuscript_figures_final
- manuscript_figures_jcim
- gwn/final_smrt_results
- gwn/final_external_results
- gwn/final_paper_tables

Important final tables checked:

- paper_results_TCDV_TopoRT/00_MASTER_RESULTS_SUMMARY.md
- paper_results_TCDV_TopoRT/tables/01_smrt_single_seed_results.csv
- paper_results_TCDV_TopoRT/tables/02_smrt_5seed_prediction_ensemble_summary.csv
- paper_results_TCDV_TopoRT/tables/03_dualview_and_fusion_ablation.csv
- paper_results_TCDV_TopoRT/tables/04_structural_ablation_seed5.csv
- paper_results_TCDV_TopoRT/tables/05_external_transfer_fixed_raw_autoselect.csv
- paper_results_TCDV_TopoRT/tables/06_tl_vs_scratch_summary.csv
- paper_results_TCDV_TopoRT/tables/06b_tl_vs_scratch_overall_summary.csv
- paper_results_TCDV_TopoRT/tables/07_candidate_filtering_reranking_summary.csv
- manuscript_figures_final/fig_tl_vs_scratch_bar.pdf
- manuscript_figures_final/fig_tl_vs_scratch_bar.png

All were present after cleanup.

## Moved out during cleanup

Non-final, failed, duplicated, old, or intermediate outputs were moved to:

- ../TCDV-TopoRT_cleanup_backups/

Do not delete backup folders until the manuscript and code reorganization are fully finished.

## Cleanup boundary for future work

Do not move or delete:

- *.pth under formal result directories
- gwn/results_OOF_DualView_Stack_*
- gwn/smrt_cwn_oof_*
- ablations/gwn_cwn_structural_ablation/results_Ablation_*
- gwn/experiments_candidate_filtering/**/best_model.pth
- paper_results_TCDV_TopoRT
- manuscript_figures_final
- gwn/final_smrt_results
- gwn/final_external_results

## Next phase

Next phase should focus on:

1. building an experiment-to-code map;
2. adding missing reproduction helper scripts;
3. renaming numeric diagnostic scripts into meaningful names;
4. reorganizing code without changing model logic.

## Root-level helper script cleanup

Archived obsolete root-level helper scripts and cleanup records.

Backup:

- ../TCDV-TopoRT_cleanup_backups/round18_root_obsolete_scripts_20260627

Archived categories:

- old ABCORT-style figure attempts
- old clean/overlay/radar figure variants
- old candidate filtering audit helper
- temporary cleanup CSV/TXT records

Retained root-level scripts:

- build_tautomer_strict_csv.py
- make_jcim_style_figures.py
- make_ablation_delta_figure_final.py
- make_final_ablation_radar_like_abcort.py
- make_final_candidate_filtering_summary_correct.py
- make_final_formula_candidate_bar_like_abcort.py
- make_formula_level_guarded_soft_final_plot.py
- make_formula_level_bar_brostyle.py
- make_formula_level_bar_pretty.py

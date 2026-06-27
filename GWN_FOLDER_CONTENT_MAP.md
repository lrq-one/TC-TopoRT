# GWN folder content map

Target folder:

- gwn/

This folder is the main working subproject for TCDV-TopoRT. It contains the core model code, SMRT dual-view OOF training assets, external transfer assets, candidate filtering assets, and final paper-facing result summaries.

## 1. Core purpose

The gwn/ folder is kept as the main reproducibility workspace for the final TCDV-TopoRT experiments:

- SMRT dual-view topology-aware retention time prediction
- 5-seed OOF prediction-level stacking
- dual-view / tautomer / tail / pairing ablations
- external transfer evaluation
- candidate filtering / reranking experiments
- final table and figure generation inputs

## 2. Must keep: core code

Keep:

- gwn/mp/
- gwn/net/
- gwn/train_oof_dualview_stack.py
- gwn/run_oof_multiseed_5runs.sh

Purpose:

- mp/ contains SMRT dataset construction, Complex/ComplexBatch objects, ring/cell-complex utilities, and PyG/cell-complex data processing.
- net/ contains TCDV-TopoRT, CWN, adapter, and model definition code.
- train_oof_dualview_stack.py is the main dual-view OOF training entry.
- run_oof_multiseed_5runs.sh is the main 5-seed SMRT OOF run script.

## 3. Must keep: SMRT data and cache

Keep:

- gwn/data/
- gwn/data_taut_strict_origin_order/
- gwn/smrt_cwn_oof_origin_train/
- gwn/smrt_cwn_oof_origin_test/
- gwn/smrt_cwn_oof_taut_train/
- gwn/smrt_cwn_oof_taut_test/

Purpose:

- data/ stores SMRT_train.csv and SMRT_test.csv.
- data_taut_strict_origin_order/ stores strict tautomer-canonical SMRT files and reorder audit files.
- smrt_cwn_oof_* folders are processed cache roots for the main r6 CWN dual-view OOF runs.

These cache folders are large but kept because they avoid rebuilding SMRT graph/cell-complex datasets.

## 4. Must keep: main SMRT OOF results

Keep:

- gwn/results_OOF_DualView_Stack_v1/
- gwn/results_OOF_DualView_Stack_seed5/
- gwn/results_OOF_DualView_Stack_seed79/
- gwn/results_OOF_DualView_Stack_seed123/
- gwn/results_OOF_DualView_Stack_seed256/

Purpose:

- These are the formal 5-seed SMRT dual-view OOF results.
- Each seed has 5 folds and 2 views.
- Expected total main checkpoints: 50 best_model.pth files.

Do not delete these directories.

## 5. Must keep: final paper result folders

Keep:

- gwn/final_smrt_results/
- gwn/final_external_results/
- gwn/final_paper_tables/
- gwn/paper_final_results/
- paper_results_TCDV_TopoRT/
- manuscript_figures_final/
- manuscript_figures_jcim/

Purpose:

- final_smrt_results/ contains SMRT main result, dual-view ablation, tautomer subgroup, shuffle pairing, tail, hard molecule, and no-leak audit outputs.
- final_external_results/ contains final external Table 2 / manifest-selected external transfer outputs.
- final_paper_tables/ contains generated paper tables from final_smrt_results and final_external_results.
- paper_results_TCDV_TopoRT/ is the formal compact paper result package.
- manuscript_figures_final/ and manuscript_figures_jcim/ contain figure outputs for the manuscript.

## 6. Must keep: external transfer final package

Keep:

- gwn/external_data/
- gwn/external_splits/
- gwn/paper_analysis_stage4_external/
- gwn/experiments_transfer_effectiveness/
- gwn/experiments_transfer_effectiveness/paper_saved_external_transfer_fixed_raw_autoselect_cvseed1/

Purpose:

- external_data/ and external_splits/ store raw external datasets and split definitions.
- paper_analysis_stage4_external/ stores external metadata, temporary origin/taut files, and processed cache for external transfer.
- experiments_transfer_effectiveness/ contains scripts and the final saved external transfer package.
- paper_saved_external_transfer_fixed_raw_autoselect_cvseed1/ is the compact final package for the external transfer result chain.

## 7. Must keep: candidate filtering final chains

Keep:

- gwn/experiments_candidate_filtering/msfinder_exports/
- gwn/experiments_candidate_filtering/msfinder_queries/
- gwn/experiments_candidate_filtering/parsed_candidates/
- gwn/experiments_candidate_filtering/riken_parsed/
- gwn/experiments_candidate_filtering/riken_tl_exact85/
- gwn/experiments_candidate_filtering/riken_tl_exact85_training/
- gwn/experiments_candidate_filtering/riken_exact85_predictions_tl_seed42/
- gwn/experiments_candidate_filtering/riken_exact85_rank_guard_soft_eval_seed42/
- gwn/experiments_candidate_filtering/metabobase_tl_evaluable45/
- gwn/experiments_candidate_filtering/metabobase_tl_evaluable45_training/
- gwn/experiments_candidate_filtering/metabobase_evaluable45_predictions_tl_seed42/
- gwn/experiments_candidate_filtering/metabobase_evaluable45_rank_guard_soft_eval_seed42/

Purpose:

- These are the retained final candidate filtering / reranking chains.
- RIKEN exact85 and MetaboBase evaluable45 are kept as the final candidate-filtering evidence chain.
- Expected candidate filtering checkpoints after cleanup: 4 best_model.pth files.

## 8. Diagnostics

Keep:

- gwn/diagnostics/

Purpose:

- diagnostics/ contains scripts that generate or audit final SMRT results, external transfer results, and candidate filtering outputs.
- Some older scripts remain for provenance, but large output folders have been moved out.

## 9. Moved out during cleanup

The following categories were moved to backup, not deleted:

### Round 9: old external stage4 intermediate outputs

Backup:

- ../TCDV-TopoRT_cleanup_backups/round9_gwn_stage4_intermediate_20260627

Moved examples:

- root-level temporary compare/result CSV files
- old paper_analysis_stage4B/I/J/N/P/Q/V intermediate external transfer result folders
- posthoc calibration intermediate folders

### Round 10: old candidate filtering routes

Backup:

- ../TCDV-TopoRT_cleanup_backups/round10_gwn_candidate_filtering_intermediate_20260627

Moved examples:

- MetaboBase S10 route
- MetaboBase sample45 route
- MetaboBase exact39 route
- MetaboBase calibration route
- old full45/sample recovery intermediates where untracked

### Round 11: old paper_analysis outputs

Backup:

- ../TCDV-TopoRT_cleanup_backups/round11_gwn_old_paper_analysis_20260627

Moved examples:

- gwn/paper_analysis/
- gwn/paper_analysis_62/
- gwn/paper_analysis_63/
- gwn/paper_analysis_63_final/
- gwn/paper_analysis_external/
- gwn/paper_analysis_safe_fusion/

### Round 12: obsolete tracked leftover artifacts

Backup:

- ../TCDV-TopoRT_cleanup_backups/round12_tracked_leftovers_removed_20260627

Removed from Git and backed up:

- gwn/paper_analysis_stage4C_embtr_lifeold_seed1_fold0/external_embedding_features_LIFE_old_194_seed1_fold0.npz
- gwn/experiments_candidate_filtering/metabobase_full45_recovery/*.tsv
- gwn/experiments_candidate_filtering/metabobase_full45_recovery_wide/*.tsv

These were obsolete intermediate artifacts, not final paper assets.

## 10. Do not delete during future cleanup

Do not delete:

- gwn/mp/
- gwn/net/
- gwn/train_oof_dualview_stack.py
- gwn/run_oof_multiseed_5runs.sh
- gwn/data/
- gwn/data_taut_strict_origin_order/
- gwn/smrt_cwn_oof_origin_train/
- gwn/smrt_cwn_oof_origin_test/
- gwn/smrt_cwn_oof_taut_train/
- gwn/smrt_cwn_oof_taut_test/
- gwn/results_OOF_DualView_Stack_v1/
- gwn/results_OOF_DualView_Stack_seed5/
- gwn/results_OOF_DualView_Stack_seed79/
- gwn/results_OOF_DualView_Stack_seed123/
- gwn/results_OOF_DualView_Stack_seed256/
- gwn/final_smrt_results/
- gwn/final_external_results/
- gwn/final_paper_tables/
- gwn/paper_final_results/
- gwn/paper_analysis_stage4_external/
- gwn/experiments_transfer_effectiveness/
- gwn/experiments_candidate_filtering/
- paper_results_TCDV_TopoRT/
- manuscript_figures_final/
- manuscript_figures_jcim/

These are the retained main reproducibility and paper-result assets.
===== append live checks =====

```text
Git commit at doc creation:
d145993 Remove obsolete tracked intermediate analysis artifacts

Main OOF checkpoint count:
50

Candidate filtering checkpoint count:
4

Remaining paper_analysis dirs:
gwn/paper_analysis_stage4_external

Remaining candidate filtering dirs:
gwn/experiments_candidate_filtering
gwn/experiments_candidate_filtering/abcort_reference
gwn/experiments_candidate_filtering/metabobase_evaluable45_predictions_tl_seed42
gwn/experiments_candidate_filtering/metabobase_evaluable45_predictions_tl_seed42/datasets
gwn/experiments_candidate_filtering/metabobase_evaluable45_rank_guard_soft_eval_seed42
gwn/experiments_candidate_filtering/metabobase_tl_evaluable45
gwn/experiments_candidate_filtering/metabobase_tl_evaluable45/seed42
gwn/experiments_candidate_filtering/metabobase_tl_evaluable45_training
gwn/experiments_candidate_filtering/metabobase_tl_evaluable45_training/datasets
gwn/experiments_candidate_filtering/metabobase_tl_evaluable45_training/seed42
gwn/experiments_candidate_filtering/msfinder_exports
gwn/experiments_candidate_filtering/msfinder_exports/meta
gwn/experiments_candidate_filtering/msfinder_exports/riken
gwn/experiments_candidate_filtering/msfinder_queries
gwn/experiments_candidate_filtering/parsed_candidates
gwn/experiments_candidate_filtering/riken_audit
gwn/experiments_candidate_filtering/riken_exact85_predictions_tl_seed42
gwn/experiments_candidate_filtering/riken_exact85_predictions_tl_seed42/datasets
gwn/experiments_candidate_filtering/riken_exact85_rank_guard_soft_eval_seed42
gwn/experiments_candidate_filtering/riken_parsed
gwn/experiments_candidate_filtering/riken_tl_exact85
gwn/experiments_candidate_filtering/riken_tl_exact85_training
gwn/experiments_candidate_filtering/riken_tl_exact85_training/datasets
gwn/experiments_candidate_filtering/riken_tl_exact85_training/seed42

gwn size depth 1:
8.3G	gwn
1.1G	gwn/smrt_cwn_oof_taut_train
1.1G	gwn/smrt_cwn_oof_origin_train
1.1G	gwn/results_OOF_DualView_Stack_v1
1.1G	gwn/results_OOF_DualView_Stack_seed79
1.1G	gwn/results_OOF_DualView_Stack_seed5
1.1G	gwn/results_OOF_DualView_Stack_seed256
1.1G	gwn/results_OOF_DualView_Stack_seed123
492M	gwn/experiments_candidate_filtering
123M	gwn/smrt_cwn_oof_taut_test
123M	gwn/smrt_cwn_oof_origin_test
50M	gwn/paper_analysis_stage4_external
26M	gwn/data_taut_strict_origin_order
20M	gwn/final_smrt_results
4.6M	gwn/data
3.5M	gwn/experiments_transfer_effectiveness
2.2M	gwn/diagnostics
2.0M	gwn/final_external_results
984K	gwn/external_data
452K	gwn/figure_assets
160K	gwn/mp
64K	gwn/paper_final_results
48K	gwn/final_paper_tables
40K	gwn/net
20K	gwn/external_splits
8.0K	gwn/scripts
8.0K	gwn/configs
```

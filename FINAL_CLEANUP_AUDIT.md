# Final cleanup audit

This document records the final repository cleanup audit for TCDV-TopoRT.

## Current commit

```text
24968e0 Archive obsolete root-level helper scripts
```

## Git status at audit generation

```text
?? FINAL_CLEANUP_AUDIT.md
```

## Main retained assets

The following groups are retained:

- root documentation and cleanup maps
- gwn/ core code
- gwn/ SMRT data and processed cache
- gwn/ 5-seed OOF results
- gwn/ final SMRT, external, and paper table results
- gwn/ external transfer final package
- gwn/ candidate filtering final chains
- manuscript figure folders
- ablations/gwn_cwn_structural_ablation final structural ablation assets

## Checkpoint counts

```text
main_oof_checkpoints=50
candidate_filtering_checkpoints=4
ablation_no2cell_checkpoints=10
ablation_cwn0_checkpoints=10
```

Expected:

- main_oof_checkpoints = 50
- candidate_filtering_checkpoints = 4
- ablation_no2cell_checkpoints should remain available if the ablation result folder is retained
- ablation_cwn0_checkpoints should remain available if the ablation result folder is retained

## Diagnostics scripts

```text
diagnostics_py_count=28
```

The gwn diagnostics folder was reduced to the retained final audit/reproduction scripts.

## Remaining root-level scripts

```text
./build_tautomer_strict_csv.py
./make_ablation_delta_figure_final.py
./make_final_ablation_radar_like_abcort.py
./make_final_candidate_filtering_summary_correct.py
./make_final_formula_candidate_bar_like_abcort.py
./make_formula_level_bar_brostyle.py
./make_formula_level_bar_pretty.py
./make_formula_level_guarded_soft_final_plot.py
./make_jcim_style_figures.py
```

## Remaining gwn diagnostics scripts

```text
gwn/diagnostics/102_parse_riken_msfinder_candidates.py
gwn/diagnostics/103_build_riken_exact85_from_tableS11.py
gwn/diagnostics/104_train_riken_tcdv_tl_exact85.py
gwn/diagnostics/105_predict_riken_exact85_candidates_tl.py
gwn/diagnostics/108_make_final_riken_experiment_tables.py
gwn/diagnostics/109_external_table2_result_audit.py
gwn/diagnostics/110_external_table2_from_manifest.py
gwn/diagnostics/111_smrt_multiseed_result_audit.py
gwn/diagnostics/112_smrt_main_from_final_metrics.py
gwn/diagnostics/113_smrt_dualview_ablation_from_predictions.py
gwn/diagnostics/114_smrt_taut_changed_subgroup.py
gwn/diagnostics/115_smrt_shuffle_taut_pairing_ablation.py
gwn/diagnostics/116_smrt_tail_hard_molecule_analysis.py
gwn/diagnostics/117_pairing_noleak_audit.py
gwn/diagnostics/118_build_final_paper_tables.py
gwn/diagnostics/124_audit_external_candidate_filtering_feasibility.py
gwn/diagnostics/40_check_dualview_pair_data.py
gwn/diagnostics/41_reorder_existing_taut_to_origin_order.py
gwn/diagnostics/50_make_oof_paper_figures.py
gwn/diagnostics/70_convert_raw_libraries_to_msfinder_msp.py
gwn/diagnostics/71_parse_msfinder_structure_results.py
gwn/diagnostics/91_make_panelA_rdkit_views.py
gwn/diagnostics/92_make_panelA_compact_white.py
gwn/diagnostics/94_build_metabobase_evaluable45_split.py
gwn/diagnostics/95_train_metabobase_tcdv_tl_evaluable45.py
gwn/diagnostics/96_predict_metabobase_evaluable45_candidates_tl.py
gwn/diagnostics/97_eval_evaluable45_rank_guard_soft.py
gwn/diagnostics/99_make_final_experiment_A_table.py
```

## Repository size overview

```text
16G	.
8.3G	./gwn
3.5G	./.git
3.3G	./ablations
141M	./manuscript_figures_final
19M	./paper_results_TCDV_TopoRT
1.2M	./manuscript
820K	./manuscript_figures_jcim
144K	./__pycache__
```

## gwn size overview

```text
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
3.6M	gwn/experiments_transfer_effectiveness
2.0M	gwn/final_external_results
1.4M	gwn/diagnostics
984K	gwn/external_data
452K	gwn/figure_assets
280K	gwn/mp
76K	gwn/net
64K	gwn/paper_final_results
48K	gwn/final_paper_tables
24K	gwn/__pycache__
20K	gwn/external_splits
8.0K	gwn/configs
```

## Cleanup backups

Major cleanup backups are under:

- ../TCDV-TopoRT_cleanup_backups/

Important cleanup rounds include:

- round6_ablation_nonstructural_peripheral_20260627
- round7_ablation_diagnostics_duplicates_20260627
- round8_ablation_draft_docs_20260627
- round9_gwn_stage4_intermediate_20260627
- round10_gwn_candidate_filtering_intermediate_20260627
- round11_gwn_old_paper_analysis_20260627
- round12_tracked_leftovers_removed_20260627
- round14_gwn_obsolete_diagnostics_code_20260627
- round16_gwn_small_obsolete_scripts_20260627
- round18_root_obsolete_scripts_20260627
- final_repo_audit_20260627

## Conclusion

The repository is now cleaned and organized around the final TCDV-TopoRT reproducibility assets. Large retained folders are mostly formal OOF checkpoints, processed SMRT cache, candidate filtering final chains, final paper results, and manuscript figures.

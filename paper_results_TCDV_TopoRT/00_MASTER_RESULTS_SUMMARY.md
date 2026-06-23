# TCDV-TopoRT Paper Results Package

Generated at: 2026-06-23 14:03:40

Project root: `/home/lwh/projects/lrq_q/TCDV-TopoRT`

## 1. SMRT main benchmark

- Five independent single-seed runs: **MAE = 25.055 ± 0.039 s; MedAE mean = 11.317 s; RMSE mean = 55.671 s; R2 mean = 0.8983**
- Five-seed prediction ensemble: **MAE = 24.920 s; MedAE = 11.164 s; RMSE = 55.540 s; R2 = 0.8988; >100 s = 318; >200 s = 133**

Important distinction:

- `25.055 ± 0.039 s` is the mean ± std of five independent single-seed runs.
- `24.920 s` is the MAE after molecule-wise averaging of the five seed predictions.

Key files:

- `tables/01_smrt_single_seed_results.csv`
- `tables/01b_smrt_single_seed_mean_std.csv`
- `tables/02_smrt_5seed_prediction_ensemble_summary.csv`
- `figures/smrt_error_analysis/`

## 2. Dual-view and fusion ablation

- Strict tautomer view is slightly stronger than original view.
- Dual-view prediction fusion provides the main gain.
- OOF Huber stacking gives small additional stabilization.

Key file: `tables/03_dualview_and_fusion_ablation.csv`

## 3. Structural ablation

- Removing explicit ring 2-cells mildly degrades performance.
- Disabling CWN message passing is an extreme structural ablation and causes large degradation.

Key file: `tables/04_structural_ablation_seed5.csv`

## 4. External transferability

- Fixed raw-only no-leak AutoSelect protocol.
- Better than reported ABCoRT-TL on 4/6 external datasets.
- Competitive on the remaining 2/6 datasets.

Key file: `tables/05_external_transfer_fixed_raw_autoselect.csv`

## 5. TL vs scratch

- Transfer learning improves MAE on 6/10 external datasets in the Figure-4-style comparison.
- Mean MAE improvement is about 4.879 s.

Key files:

- `tables/06_tl_vs_scratch_summary.csv`
- `tables/06b_tl_vs_scratch_overall_summary.csv`

## 6. Candidate filtering / reranking

- MetaboBase candidate-evaluable 45: ours improves candidate reduction and Top-k accuracy.
- RIKEN_PlaSMA exact85: ours improves candidate reduction and Top-k accuracy.

Key file: `tables/07_candidate_filtering_reranking_summary.csv`

## 7. Paper status

No additional mandatory training experiment is required.

Remaining work:

1. Use these tables and figures in manuscript writing.
2. Keep raw predictions and raw metrics for reproducibility.
3. Do not upload checkpoints, processed caches, or large training directories unless required.

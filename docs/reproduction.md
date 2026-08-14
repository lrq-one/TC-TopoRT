# Reproduction guide

Run commands from the repository root. Generated files are written under
`artifacts/`, which is intentionally ignored by Git. The final column
distinguishes artifacts listed in the author Figshare archive manifest from
analyses that require separately supplied inputs. The author archive is available
at [DOI 10.6084/m9.figshare.33252810](https://doi.org/10.6084/m9.figshare.33252810).

| Paper result | Command | Required inputs | Expected output | Archive/reference status |
|---|---|---|---|---|
| SMRT five-seed benchmark | `bash scripts/training/run_smrt_five_seeds.sh` then `python scripts/analysis/summarize_smrt_results.py --prediction 1=... --prediction 5=... --prediction 79=... --prediction 123=... --prediction 256=... --verify-paper-results` | Four paired SMRT CSVs | Per-seed metrics; `25.055090 ± 0.039094 s` MAE | Five checkpoints and test/OOF prediction files |
| O+O, T+T, and O+T controls | `python scripts/analysis/build_dualview_ablation.py --input <dualview_control_metrics.csv> --verify-paper-results` | Per-seed control metric table | Six-control mean/SD table | Control metric inputs are not separately listed in the Figshare manifest |
| No2Cell | `bash scripts/ablation/run_structural_ablation.sh no2cell` | Paired SMRT CSVs | Five-seed `25.121 ± 0.091 s` MAE after running all seeds | No2Cell models and metrics are not separately listed in the Figshare manifest |
| Parameter-matched GINE | `bash scripts/ablation/run_atom_bond_gnn.sh` | Paired SMRT CSVs | Five-seed `25.701 ± 0.069 s` MAE | Matched-GINE predictions and metrics are not separately listed in the Figshare manifest |
| Structural result collection | `python scripts/analysis/collect_structural_ablation.py --input <structural_metrics.csv> --verify-paper-results` | Per-seed metric/parameter table | Full, No2Cell, and matched-GINE summary | Requires the corresponding structural metric table; not separately listed in the Figshare manifest |
| Transfer versus scratch | `python scripts/transfer/train_transfer_all10.py` and `python scripts/transfer/train_scratch_all10.py` | Prepared ten-dataset tables; transfer also requires SMRT checkpoints | Table 8 under `artifacts/results/external_transfer/` | Final transfer configuration metadata are archived; prediction/result tables are not separately listed in the manifest |
| Candidate filtering | `python scripts/filtering/run_candidate_filtering.py --dataset <dataset> --input <candidate.csv>` | Candidate-level predictions and `configs/candidate_filtering.yaml` | Candidate, query, and dataset summaries | Frozen paper candidate lists, development records, thresholds, and final outputs |
| Pairing/no-leakage audit | `python scripts/analysis/audit_pairing_and_noleakage.py [--oof-prediction <csv>]` | Paired SMRT tables; optional OOF predictions | Alignment, split-overlap, and OOF-coverage JSON | SMRT OOF predictions are archived; paired source CSVs are not redistributed |
| Tautomer subgroup | `python scripts/analysis/analyze_tautomer_subgroups.py --prediction 1=<csv> ...` | Five test prediction tables with view predictions and change flag | Changed/unchanged detail and summary | Final SMRT predictions |
| Ring subgroup | `python scripts/analysis/analyze_ring_subgroups.py --full <csv> --no2cell <csv>` | Aligned Full and No2Cell test predictions with SMILES | Overlapping ring-context subgroup table | Requires aligned Full/No2Cell predictions; the No2Cell input is not separately listed in the Figshare manifest |
| Shuffled pairing | `python scripts/analysis/shuffled_pairing_control.py --run 1,<oof.csv>,<test.csv> ...` | Five base OOF/test prediction pairs | Paired versus shuffled MAE table | Base predictions |
| Early RT coverage | `python scripts/analysis/audit_smrt_early_rt_coverage.py --input <full_SMRT.csv> --verify-paper-scope` | Complete public 80,038-record source table | `2,058/80,038` (`2.57%`) operational scope JSON | Source table comes from the original SMRT DOI; audit output is not separately listed in the author archive manifest |
| Tautomer collision audit | `python scripts/analysis/audit_tautomer_standardization_collisions.py --verify-paper-scope` | Four paired retained-SMRT tables | Representation and collision tables | Requires the paired retained-SMRT tables; audit outputs are not separately listed in the Figshare manifest |
| Tautomer change types | `python scripts/analysis/analyze_tautomer_change_types.py` | Four paired retained-SMRT tables | Rule-based change-category statistics and representative examples | Requires the paired retained-SMRT tables; outputs are not separately listed in the Figshare manifest |
| RIKEN matched scope and target-domain RT | `python scripts/analysis/analyze_riken_same_formula_candidates.py --input <matched.csv> --full-target-metrics <full_metrics.csv> --no2cell-target-metrics <no2cell_metrics.csv> --verify-paper-scope --verify-target-metrics` | Matched Full/No2Cell candidate table and two independent target-metric tables | Scope JSON and validated Full/No2Cell target-domain metrics | Not separately claimed in the Figshare archive; the public script reproduces the analysis when the matched inputs are supplied |
| Full versus No2Cell pairwise | `python scripts/analysis/analyze_full_vs_no2cell_pairwise.py --input <matched.csv> --verify-paper-results` | Matched Full/No2Cell candidate table | Query-clustered macro accuracies and bootstrap intervals | Not separately claimed in the Figshare archive; requires the matched Full/No2Cell candidate table |

## SMRT inputs and training

Use `scripts/data/build_strict_tautomer_views.py` and
`scripts/data/validate_paired_views.py` to build and validate the standardized
partner while preserving the dataset-provided row identity and RT label. The
retained train/test sizes are 70,182 and 7,798 after applying `RT > 300 s`.
Strict standardization changes 37,724 train and 4,242 test representations.

Training performs five-fold OOF prediction independently for the two views. The
Huber stacker is fitted only to OOF predictions. Full five-seed training is
expensive and was run on an RTX 3090 24 GB; the test suite does not train it.

## Final benchmark and controls

The locked five-run result is:

| Metric | Mean ± sample SD |
|---|---:|
| MAE | 25.055090 ± 0.039094 s |
| MRE | 3.161936 ± 0.004679% |
| MedAE | 11.316787 ± 0.097631 s |
| RMSE | 55.671332 ± 0.100621 s |
| R² | 0.898308 ± 0.000368 |

Full TC-TopoRT has 26,943,049 trainable parameters; matched GINE has
26,928,385 (0.0544% difference). No2Cell is a precise removal of explicit ring
2-cells, not removal of all ring information.

## Final candidate-filtering computation

`configs/candidate_filtering.yaml` records development `N=30`, all four predictor
MAEs, and thresholds. Within each candidate dataset, the same fixed 30
development query identifiers and experimental RT labels are used across the
four predictors. These development queries are disjoint from the final 45
MetaboBase and 85 RIKEN-PlaSMA evaluation queries; no final-evaluation query or
RT label contributes to threshold calibration.

The development queries were excluded from the reduced training pools used to
fit the calibration models: the MetaboBase parent/reduced pools contain 181/151
compounds, and the RIKEN-PlaSMA parent/reduced pools contain 341/311 compounds.
Development MAE is evaluated on the 30 queries held out with respect to
calibration-model fitting. Each threshold is then set as
`T_m = 3 × MAE_dev,m` and frozen before final evaluation.

Final candidate predictions were generated by the corresponding target-domain
predictors trained on the larger parent pools, which include the 30 development
compounds. Thus, the development set was not held out from the final predictor.
The final 45/85 evaluation queries remain molecule-disjoint from the final
target-predictor training pools.

The script retains missing predictions and otherwise retains
`abs_rt_delta <= T_m`. It never changes original MS-FINDER ordering. Candidate
reduction uses candidate records as denominator; retention and Top-k use queries
as denominator.

Paper-reported/reference TC-TopoRT results:

| Dataset | Queries | Initial rows | Reduction | True retained | Retained % | FN | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MetaboBase | 45 | 3,023 | 41.05% | 42/45 | 93.33% | 3 | 21 | 34 | 39 |
| RIKEN-PlaSMA | 85 | 5,044 | 30.35% | 84/85 | 98.82% | 1 | 45 | 63 | 72 |

These analyses include only evaluable queries for which the true structure was
already present in the initial MS-FINDER candidate list. No final-test parameter
search or order-changing reranking is part of the final protocol.

The table records manuscript reference values for the final protocol.
Dataset-free unit and smoke tests validate the hard-filter semantics without
hard-coding target result counts.

## Frozen versus regenerated artifacts

GitHub regenerates model training and analyses when the required inputs and
compute are available. Figshare supplies the immutable checkpoints,
predictions, candidate-filtering tables, calibration records, configuration
snapshot, and checksums listed in its manifest. Matched Full-versus-No2Cell,
pairwise, and Norharman-specific analysis inputs are not claimed as archive
contents. The original SMRT source itself remains at DOI
10.6084/m9.figshare.8038913 and is not republished as author-generated data.

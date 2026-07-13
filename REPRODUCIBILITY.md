# TC-TopoRT reproducibility map

This document maps the reported computational experiments to their public inputs, commands, and generated outputs. Commands are run from the repository root. Generated files are written under `artifacts/` and are excluded from Git.

## 1. Environment and fast validation

Install the environment:

```bash
conda env create -f environment.yml
conda activate tc-toport
```

Run the minimal data-construction and model-forward test:

```bash
python scripts/tests/smoke_test.py
```

This test builds ring-aware cell complexes for two SMRT molecules and performs one TC-TopoRT forward pass. It does not train a model or reproduce paper accuracy.

## 2. SMRT paired views

| Item | Value |
|---|---|
| Input | `gwn/data/SMRT_train.csv`, `gwn/data/SMRT_test.csv` |
| Command | `bash scripts/data/rebuild_strict_tautomer_views.sh` |
| Validation | `bash scripts/data/validate_smrt_paired_views.sh` |
| Generated output | `artifacts/data/strict_tautomer_generated/` |
| Expected retained rows | train 70,182; test 7,798 |
| Expected genuine changes | train 37,724; test 4,242 |

The checked paired views used in the paper are retained under `gwn/data_taut_strict_origin_order/`.

## 3. SMRT main prediction experiment

**Paper scope:** TC-TopoRT-S five-run summary, TC-TopoRT-E five-seed ensemble, per-seed supplementary table, and prediction-based SMRT figures.

| Stage | Command | Output |
|---|---|---|
| One seed | `bash scripts/training/run_smrt_single_seed.sh 5` | `artifacts/results/smrt/seed5/` |
| Five paper seeds | `bash scripts/training/run_smrt_five_seeds.sh` | `artifacts/results/smrt/seed{1,5,79,123,256}/` |
| Summary and ensemble | `python scripts/analysis/summarize_smrt_results.py` | `artifacts/results/paper_tables/smrt/` |
| SMRT figures | `python scripts/figures/make_smrt_figures.py --result_dir artifacts/results/smrt/seed5` | `artifacts/figures/` |

Headline checks:

- seed-wise mean MAE: approximately 25.055 s;
- seed-wise MAE SD: approximately 0.039 s;
- five-seed ensemble MAE: approximately 24.920 s;
- every reported run records `selected_stacker = huber_stack`.

The training code evaluates several OOF-only fusion controls. The formal five reported runs selected the Huber stacker from training-set OOF predictions, and that selected stacker generated the independent-test predictions.

## 4. Dual-view and fusion ablation

**Paper scope:** original view, strict tautomer view, same-seed paired mean fusion, and OOF Huber stack.

```bash
python scripts/analysis/build_dualview_ablation.py
```

Inputs:

```text
artifacts/results/smrt/seed*/test_base_predictions.csv
artifacts/results/smrt/seed*/test_predictions.csv
artifacts/results/smrt/seed*/final_metrics.json
```

Outputs:

```text
artifacts/results/paper_tables/ablation/dualview_fusion_ablation_by_seed.csv
artifacts/results/paper_tables/ablation/dualview_fusion_ablation_summary.csv
```

## 5. Structural ablations and atom-bond GNN control

Train the two structural variants:

```bash
bash scripts/ablation/run_structural_ablation.sh no2cell
bash scripts/ablation/run_structural_ablation.sh cwn0
```

Train the conventional atom-bond GNN under the same paired-view OOF protocol:

```bash
bash scripts/ablation/run_atom_bond_gnn.sh
```

Collect the paper-facing comparison:

```bash
python scripts/analysis/collect_structural_ablation.py
```

Expected headline MAEs under the seed-5 comparison:

| Variant | MAE (s) |
|---|---:|
| Full TC-TopoRT | about 25.012 |
| Without explicit ring 2-cells | about 25.102 |
| Conventional atom-bond GNN | about 28.252 |
| Without CWN message passing | about 39.645 |

Generated table:

```text
artifacts/results/paper_tables/ablation/structural_ablation_seed5.csv
```

## 6. External transfer learning and scratch comparison

The external dataset manifest is:

```text
configs/external_datasets.csv
```

Prepare the three processed inputs from a standardized combined PredRet CSV:

```bash
python scripts/data/prepare_external_predret.py \
  --input_csv /path/to/combined_predret.csv \
  --out_dir artifacts/data/external
```

Train from scratch:

```bash
python scripts/transfer/train_scratch_all10.py \
  --stage4_meta_csv artifacts/data/external/external_predret10_stage4_meta.csv \
  --origin_csv artifacts/data/external/temp_external_predret10_origin.csv \
  --taut_csv artifacts/data/external/temp_external_predret10_taut.csv
```

Run SMRT-pretrained transfer learning after the five SMRT runs are available:

```bash
python scripts/transfer/train_transfer_all10.py \
  --stage4_meta_csv artifacts/data/external/external_predret10_stage4_meta.csv \
  --origin_csv artifacts/data/external/temp_external_predret10_origin.csv \
  --taut_csv artifacts/data/external/temp_external_predret10_taut.csv \
  --smrt_runs_root artifacts/results/smrt
```

The transfer script creates the combined transfer-versus-scratch table when the scratch summary is present. The expected overall headline is lower transfer MAE on 8 of 10 datasets, with mean and median improvements of approximately 9.164 s and 3.677 s, respectively.

The six-dataset literature comparison in the manuscript uses Eawag-XBridgeC18, FEM-lipids, FEM-long, IPB-Halle, LIFE-new, and LIFE-old. Literature baseline values are cited results and are not retrained by this repository.

## 7. Candidate filtering and reranking

Primary processed inputs:

```text
data/candidate_filtering/metabobase_candidate_predictions.csv
data/candidate_filtering/riken_candidate_predictions.csv
```

Run the fixed paper operating points:

```bash
python scripts/filtering/run_candidate_filtering.py
```

Generated outputs include candidate-level reranking, query-level metrics, selected operating-point summaries, and the main candidate-filtering comparison table under:

```text
artifacts/results/candidate_filtering/
```

Expected TC-TopoRT checks:

| Dataset | Initial | Retained | Reduction | True retained | Top-1 | Top-5 | Top-10 | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MetaboBase | 3,023 | 933 | 69.14% | 93.33% | 55.56% | 82.22% | 88.89% | 3 |
| RIKEN-PlaSMA | 5,044 | 2,712 | 46.23% | 97.65% | 54.12% | 77.65% | 89.41% | 2 |

The script validates these values before reporting a successful reproduction.

## 8. Candidate-filtering sensitivity

Precomputed full-grid source tables:

```text
data/candidate_filtering/metabobase_rank_guard_soft_grid.csv
data/candidate_filtering/riken_rank_guard_soft_grid.csv
```

Run:

```bash
python scripts/filtering/run_filtering_sensitivity.py
```

This produces:

- parameter-grid summaries;
- selected main and higher-reduction operating points;
- five-metric non-dominated rows;
- sensitivity trade-off figures.

The retained grids contain 9,600 MetaboBase and 9,900 RIKEN-PlaSMA four-parameter combinations, in addition to reference rows.

## 9. Figure generation

```bash
python scripts/figures/make_candidate_filtering_figure.py
python scripts/figures/make_transfer_figure.py
python scripts/figures/make_ablation_figure.py
python scripts/figures/make_smrt_figures.py
```

Data-driven figures should be regenerated from experiment summaries or the retained compact source tables. The architecture overview, CWN schematic, and graphical abstract are explanatory vector diagrams rather than outputs of numerical experiments and therefore are not required to be regenerated by Python.

## 10. Implementation notes

- Ring 2-cells are constructed from a NetworkX minimum cycle basis of the molecular graph. Cycles with sizes from 3 through `max_ring_size` are retained; the paper configuration uses `max_ring_size = 6`.
- Original and strict tautomer views remain aligned to the same molecule, RT label, and split.
- Fusion parameters and the Huber stacker are fitted using training-set OOF predictions only.
- Independent-test labels are not used for model selection, stacker fitting, calibration, or candidate-filtering parameter optimization.
- Checkpoints, weights, caches, logs, predictions, generated tables, and generated figures are local outputs and are not committed.

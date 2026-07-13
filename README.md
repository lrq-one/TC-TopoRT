# TC-TopoRT

TC-TopoRT is a topology-aware framework for small-molecule retention-time prediction and retention-time-guided candidate prioritization.

This public repository contains the minimal code, configurations, processed inputs, and result-aggregation utilities needed to reproduce the reported computational workflows. Historical experiments, checkpoints, caches, logs, generated figures, manuscript copies, and backup files are intentionally excluded.

## Method overview

TC-TopoRT combines:

- paired dataset-provided and strict tautomer-canonical molecular views;
- ring-aware cell complexes with atom, bond, and ring cells;
- CWN-based topology-aware message passing;
- leakage-free out-of-fold prediction-level fusion;
- external transfer-learning evaluation;
- guarded RT filtering and soft reranking of MS-FINDER candidates.

Ring 2-cells are constructed from a NetworkX minimum cycle basis. Cycles of sizes 3 through `max_ring_size` are retained; the paper configuration uses `max_ring_size = 6`.

## Reported results

### SMRT

- TC-TopoRT-S: **25.055 ± 0.039 s MAE** across five runs.
- TC-TopoRT-E: **24.920 s MAE** for the five-model ensemble.
- Conventional atom-bond GNN comparison: **28.252 s MAE**.

The training workflow evaluates several OOF-only fusion controls. The formal reported runs selected `huber_stack`, fitted only on training-set OOF predictions, and used it to generate independent-test predictions.

### External transfer learning

Across ten external chromatographic datasets, transfer learning reduced MAE on **8/10 datasets**, with a mean improvement of **9.164 s** and a median improvement of **3.677 s**.

### Candidate filtering

| Dataset | Reduction | True retained | Top-1 | Top-5 | Top-10 | FN |
|---|---:|---:|---:|---:|---:|---:|
| MetaboBase | 69.14% | 93.33% | 55.56% | 82.22% | 88.89% | 3 |
| RIKEN-PlaSMA | 46.23% | 97.65% | 54.12% | 77.65% | 89.41% | 2 |

## Repository structure

```text
.
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── configs/
│   ├── smrt.yaml
│   ├── external_transfer.yaml
│   ├── candidate_filtering.yaml
│   └── external_datasets.csv
├── data/
│   ├── ablation/
│   ├── candidate_filtering/
│   └── external/
├── gwn/
│   ├── train_oof_dualview_stack.py
│   ├── mp/                         # cell-complex data structures and CWN layers
│   ├── net/                        # TC-TopoRT model definition
│   ├── data/                       # dataset-provided SMRT split
│   └── data_taut_strict_origin_order/
└── scripts/
    ├── training/
    ├── data/
    ├── ablation/
    ├── transfer/
    ├── filtering/
    ├── analysis/
    ├── figures/
    └── tests/
```

Generated caches, checkpoints, predictions, metrics, logs, tables, and figures are written under `artifacts/`, which is excluded from Git.

## Installation

Using Conda:

```bash
conda env create -f environment.yml
conda activate tc-toport
```

Using pip:

```bash
python -m pip install -r requirements.txt
```

PyTorch Geometric extension wheels may need to be installed for the local PyTorch/CUDA combination.

## Fast validation

Run the lightweight syntax and data checks:

```bash
bash scripts/tests/run_static_checks.sh
```

Run the minimal data-construction and model-forward test in a complete PyTorch/PyG environment:

```bash
python scripts/tests/smoke_test.py
```

The smoke test builds ring-aware cell complexes for two SMRT molecules and performs one TC-TopoRT forward pass. It does not train a model or reproduce paper accuracy.

# Data sources

## 1. METLIN SMRT

**Primary publication**

X. Domingo-Almenara et al., *The METLIN small molecule dataset for machine learning-based retention time prediction*, Nature Communications 10, 5811 (2019).

- Article: https://www.nature.com/articles/s41467-019-13680-7
- Original Figshare dataset and code: https://doi.org/10.6084/m9.figshare.8038913

**Files included in this repository**

```text
gwn/data/SMRT_train.csv
gwn/data/SMRT_test.csv
gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv
gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv
```

**Processing used in TC-TopoRT**

- retain compounds with RT greater than 300 s;
- require an RDKit-valid molecular structure;
- preserve the dataset-provided train/test split and RT labels;
- build a conservative strict tautomer-canonical paired view;
- keep the original SMILES when canonical tautomerization does not produce a genuine tautomeric change;
- reject a transformed view if molecular formula or heavy-atom count changes.

After filtering, the retained split contains 70,182 training molecules and 7,798 test molecules. Strict tautomer canonicalization changes 37,724 training representations and 4,242 test representations while preserving molecular formulae.

## 2. PredRet external chromatographic datasets

PredRet is used for the **external RT transfer-learning experiments**. It is **not** the candidate-filtering dataset.

**Primary publication**

J. Stanstrup, S. Neumann, and U. Vrhovsek, *PredRet: Prediction of Retention Time by Direct Mapping between Multiple Chromatographic Systems*, Analytical Chemistry 87, 9421–9428 (2015).

- Publication DOI: https://doi.org/10.1021/acs.analchem.5b02287

The ten external systems used in the transfer-versus-scratch analysis are listed in:

```text
configs/external_datasets.csv
```

The six systems used for direct comparison with literature transfer-learning results are Eawag-XBridgeC18, FEM-lipids, FEM-long, IPB-Halle, LIFE-new, and LIFE-old. The broader ten-system transfer-versus-scratch analysis additionally includes FEM-short, UniToyama-Atlantis, MTBLS87, and Cao-HILIC.

**Processed external inputs included in this repository**

```text
data/external/external_predret10_stage4_meta.csv
data/external/temp_external_predret10_origin.csv
data/external/temp_external_predret10_taut.csv
```

These three aligned tables contain **1,787 records from 10 external chromatographic systems**. The metadata table stores the dataset identifier, experimental RT, dataset-provided molecular view, strict tautomer-canonical view, and audit fields. The two graph-construction tables provide the original and strict-tautomer views in the same row order used by the external training workflows.

These files are processed inputs for reproducing the reported external experiments; they are not a redistribution of the complete PredRet database. The optional utility below can reconstruct the same input format from a standardized combined PredRet export:

```bash
python scripts/data/prepare_external_predret.py \
  --input_csv /path/to/combined_predret.csv \
  --out_dir artifacts/data/external
```

The graph-construction CSVs contain a dummy RT above 300 s because the shared `SMRTComplexDataset` loader applies the SMRT retained-compound filter. Actual external RT values remain in the metadata table and replace the dummy targets during external training.

## 3. MetaboBase candidate filtering

**Primary publication**

Z. Lei et al., *Construction of an Ultrahigh Pressure Liquid Chromatography-Tandem Mass Spectral Library of Plant Natural Products and Comparative Spectral Analyses*, Analytical Chemistry 87, 7373–7381 (2015).

- Publication DOI: https://doi.org/10.1021/acs.analchem.5b01559

**Processed candidate-level input included here**

```text
data/candidate_filtering/metabobase_candidate_predictions.csv
```

This table contains the 45 candidate-evaluable queries and 3,023 candidate records used in the guarded filtering and soft-reranking analysis. It includes original MS-FINDER ranks, candidate structure identifiers, TC-TopoRT candidate RT predictions, RT discrepancies, and true-candidate labels required to recompute reduction, Top-k, true-retention, and false-negative metrics.

It is a processed analysis table, not a redistribution of the complete MetaboBase spectral library.

## 4. RIKEN-PlaSMA / MassBank-related candidate filtering

**Primary publication**

H. Tsugawa et al., *A cheminformatics approach to characterize metabolomes in stable-isotope-labeled organisms*, Nature Methods 16, 295–298 (2019).

- Article: https://www.nature.com/articles/s41592-019-0358-2
- Public data DOI: https://doi.org/10.21228/M8XM40
- RIKEN PRIMe: http://prime.psc.riken.jp/
- MassBank: https://massbank.eu/MassBank/

**Processed candidate-level input included here**

```text
data/candidate_filtering/riken_candidate_predictions.csv
```

This table contains the 85 exact-ground-truth queries and 5,044 candidate records used in the reported analysis. It is sufficient to recompute the TC-TopoRT filtering and reranking metrics, but it is not a complete redistribution of PlaSMA, MassBank, or raw spectral files.

## 5. Candidate-filtering sensitivity grids

```text
data/candidate_filtering/metabobase_rank_guard_soft_grid.csv
data/candidate_filtering/riken_rank_guard_soft_grid.csv
```

These compact, precomputed parameter-grid tables are retained for immediate verification and plotting of the four-parameter sensitivity audit. The candidate-level CSVs remain the primary processed records for the filtering results.

# Reproducibility map

Commands below are run from the repository root. Generated outputs are written under `artifacts/`.

## 1. Rebuild and validate the paired SMRT views

```bash
bash scripts/data/rebuild_strict_tautomer_views.sh
bash scripts/data/validate_smrt_paired_views.sh
```

Expected checks:

| Item | Expected value |
|---|---:|
| Training rows | 70,182 |
| Test rows | 7,798 |
| Changed training representations | 37,724 |
| Changed test representations | 4,242 |

## 2. Train the SMRT model

Train one run:

```bash
bash scripts/training/run_smrt_single_seed.sh 5
```

Inspect the command without starting training:

```bash
DRY_RUN=1 bash scripts/training/run_smrt_single_seed.sh 5
```

Train all five reported runs and build the summary and ensemble:

```bash
bash scripts/training/run_smrt_five_seeds.sh
python scripts/analysis/summarize_smrt_results.py
```

Headline checks:

- mean single-run MAE: approximately 25.055 s;
- single-run MAE SD: approximately 0.039 s;
- five-model ensemble MAE: approximately 24.920 s;
- every formal run records `selected_stacker = huber_stack`.

## 3. Dual-view and fusion ablation

```bash
python scripts/analysis/build_dualview_ablation.py
```

This regenerates the original-view, strict-tautomer-view, paired-mean, and OOF-Huber comparison from the formal SMRT prediction files.

## 4. Structural ablations and atom-bond GNN control

```bash
bash scripts/ablation/run_structural_ablation.sh no2cell
bash scripts/ablation/run_structural_ablation.sh cwn0
bash scripts/ablation/run_atom_bond_gnn.sh
python scripts/analysis/collect_structural_ablation.py
```

Expected headline MAEs:

| Variant | MAE (s) |
|---|---:|
| Full TC-TopoRT | about 25.012 |
| Without explicit ring 2-cells | about 25.102 |
| Conventional atom-bond GNN | about 28.252 |
| Without CWN message passing | about 39.645 |

## 5. External transfer learning and scratch comparison

The processed external inputs are already included under `data/external/`.

Validate their alignment before training:

```bash
python scripts/data/validate_external_predret_inputs.py \
  --meta_csv data/external/external_predret10_stage4_meta.csv \
  --origin_csv data/external/temp_external_predret10_origin.csv \
  --taut_csv data/external/temp_external_predret10_taut.csv
```

Train from scratch:

```bash
python scripts/transfer/train_scratch_all10.py \
  --stage4_meta_csv data/external/external_predret10_stage4_meta.csv \
  --origin_csv data/external/temp_external_predret10_origin.csv \
  --taut_csv data/external/temp_external_predret10_taut.csv
```

Run SMRT-pretrained transfer learning after the SMRT source-fold checkpoints are available:

```bash
python scripts/transfer/train_transfer_all10.py \
  --stage4_meta_csv data/external/external_predret10_stage4_meta.csv \
  --origin_csv data/external/temp_external_predret10_origin.csv \
  --taut_csv data/external/temp_external_predret10_taut.csv \
  --smrt_runs_root artifacts/results/smrt
```

Expected overall headline: lower transfer MAE on 8 of 10 datasets, with mean and median improvements of approximately 9.164 s and 3.677 s.

Literature baseline values in the six-dataset comparison are cited results and are not retrained by this repository.

## 6. Candidate filtering and reranking

Primary processed inputs:

```text
data/candidate_filtering/metabobase_candidate_predictions.csv
data/candidate_filtering/riken_candidate_predictions.csv
```

Run the fixed paper operating points:

```bash
python scripts/filtering/run_candidate_filtering.py
```

Expected checks:

| Dataset | Initial | Retained | Reduction | True retained | Top-1 | Top-5 | Top-10 | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MetaboBase | 3,023 | 933 | 69.14% | 93.33% | 55.56% | 82.22% | 88.89% | 3 |
| RIKEN-PlaSMA | 5,044 | 2,712 | 46.23% | 97.65% | 54.12% | 77.65% | 89.41% | 2 |

## 7. Candidate-filtering sensitivity

```bash
python scripts/filtering/run_filtering_sensitivity.py
```

The retained grids contain 9,600 MetaboBase and 9,900 RIKEN-PlaSMA four-parameter combinations, in addition to reference rows. The script produces parameter summaries, non-dominated operating points, and sensitivity figures.

## 8. Figure generation

```bash
python scripts/figures/make_candidate_filtering_figure.py
python scripts/figures/make_transfer_figure.py
python scripts/figures/make_ablation_figure.py
python scripts/figures/make_smrt_figures.py
```

Data-driven figures are regenerated from experiment summaries or retained compact source tables. The architecture overview, CWN schematic, and graphical abstract are explanatory vector diagrams rather than numerical experiment outputs.

## Leakage control

- Original and strict tautomer views remain aligned to the same molecule, RT label, and split.
- Fusion parameters and the Huber stacker are fitted using training-set OOF predictions only.
- Independent-test labels are not used for model selection, stacker fitting, calibration, or candidate-filtering parameter optimization.
- Candidate-filtering comparisons use consistent candidate lists, query sets, experimental RT values, original MS-FINDER ranks, guarded-retention rules, soft-reranking definitions, and evaluation metrics.

## Outputs not committed

The following are generated locally and intentionally excluded from Git:

- model checkpoints and weights;
- graph and cell-complex caches;
- fold-level predictions and logs;
- generated paper tables and figures;
- external transfer intermediate predictions;
- historical experiment directories.

They can be regenerated from the code, configurations, and processed inputs described above.

## License

This repository is released under the [MIT License](LICENSE). Third-party libraries and datasets remain subject to their respective licenses and terms of use.

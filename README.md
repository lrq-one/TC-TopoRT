# TC-TopoRT

TC-TopoRT is a topology-aware framework for small-molecule retention-time prediction and retention-time-guided candidate prioritization.

This public repository intentionally contains only the reproducibility entry points, the minimal model/data code required by those entry points, the SMRT paired views, and the compact candidate-filtering source tables. Historical experiments, diagnostics, checkpoints, logs, generated figures, manuscript copies, and backup files are not included in the current file tree.

## Method overview

TC-TopoRT combines:

- paired dataset-provided and strict tautomer-canonical molecular views;
- ring-aware cell complexes with atom, bond, and ring cells;
- CWN-based topology-aware message passing;
- leakage-free out-of-fold prediction-level fusion;
- external transfer-learning evaluation;
- guarded RT filtering and soft reranking of MS-FINDER candidates.

## Reported results

### SMRT

- TC-TopoRT-S: **25.055 ± 0.039 s MAE** across five seeds.
- TC-TopoRT-E: **24.920 s MAE** for the five-seed ensemble.
- Seeds: `1, 5, 79, 123, 256`.
- Conventional atom-bond GNN comparison: **28.252 s MAE**.

### External transfer

Across ten external datasets, transfer learning reduced MAE on **8/10 datasets**, with a mean improvement of **9.164 s** and a median improvement of **3.677 s**.

### Candidate filtering

| Dataset | Reduction | True retained | Top-1 | Top-5 | Top-10 | FN |
|---|---:|---:|---:|---:|---:|---:|
| MetaboBase | 69.14% | 93.33% | 55.56% | 82.22% | 88.89% | 3 |
| RIKEN-PlaSMA | 46.23% | 97.65% | 54.12% | 77.65% | 89.41% | 2 |

## Repository structure

```text
.
├── README.md
├── environment.yml
├── requirements.txt
├── data/
│   ├── ablation/
│   └── candidate_filtering/
├── gwn/
│   ├── train_oof_dualview_stack.py
│   ├── mp/                         # cell-complex data structures and CWN layers
│   ├── net/                        # TC-TopoRT model definition
│   ├── data/                       # dataset-provided SMRT split
│   ├── data_taut_strict_origin_order/
│   └── paper_analysis_stage4_external/README.md
└── scripts/
    ├── training/
    ├── data/
    ├── ablation/
    ├── transfer/
    ├── filtering/
    └── figures/
```

All generated caches, checkpoints, predictions, metrics, logs, tables, and figures are written under `artifacts/`, which is excluded from Git.

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

## SMRT data

The repository includes the dataset-provided SMRT split and the paired strict tautomer-canonical view:

```text
gwn/data/SMRT_train.csv
gwn/data/SMRT_test.csv
gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv
gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv
```

After the `rt > 300 s` and RDKit-validity filters, the retained split contains 70,182 training molecules and 7,798 test molecules. Strict tautomer canonicalization changes 37,724 training representations and 4,242 test representations while preserving molecular formulae.

## Reproduction

Run commands from the repository root.

### 1. Rebuild and validate the paired views

```bash
bash scripts/data/rebuild_strict_tautomer_views.sh
bash scripts/data/validate_smrt_paired_views.sh
```

### 2. Train one SMRT seed

```bash
bash scripts/training/run_smrt_single_seed.sh 5
```

The seed can also be supplied through the `SEED` environment variable. To inspect the generated command without starting training:

```bash
DRY_RUN=1 bash scripts/training/run_smrt_single_seed.sh 5
```

### 3. Train all five paper seeds

```bash
bash scripts/training/run_smrt_five_seeds.sh
```

### 4. Structural and atom-bond ablations

```bash
bash scripts/ablation/run_structural_ablation.sh no2cell
bash scripts/ablation/run_structural_ablation.sh cwn0
bash scripts/ablation/run_atom_bond_gnn.sh
```

### 5. Candidate filtering

```bash
python scripts/filtering/run_candidate_filtering.py
python scripts/filtering/run_filtering_sensitivity.py
```

The compact candidate-level inputs are included under `data/candidate_filtering/`.

### 6. External transfer and scratch comparison

The transfer scripts are included, but the three processed external PredRet input tables are not redistributed in this repository. Supply them through the command-line arguments documented in `gwn/paper_analysis_stage4_external/README.md`.

```bash
python scripts/transfer/train_scratch_all10.py --help
python scripts/transfer/train_transfer_all10.py --help
```

Transfer learning also expects the SMRT source-fold checkpoints produced by the SMRT workflow under `artifacts/results/smrt/`.

### 7. Generate figures

```bash
python scripts/figures/make_candidate_filtering_figure.py
python scripts/figures/make_ablation_figure.py
python scripts/figures/make_transfer_figure.py
python scripts/figures/make_smrt_figures.py
```

## Leakage control

The original and tautomer-canonical views share identical labels and split assignments. Prediction-level fusion is fitted only from training-set out-of-fold predictions. Independent test labels are not used for model selection, stacker fitting, calibration, or filtering-parameter optimization.

Candidate-filtering comparisons use consistent candidate lists, query sets, experimental RT values, original MS-FINDER ranks, guarded-retention rules, soft-reranking definitions, and evaluation metrics.

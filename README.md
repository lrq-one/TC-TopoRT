# TC-TopoRT

TC-TopoRT is a topology-aware framework for small-molecule retention-time prediction and RT-guided candidate prioritization.

It combines:

- paired original and strict tautomer-canonical molecular views;
- ring-aware cell complexes with atom, bond, and ring cells;
- topology-aware CWN message passing;
- leakage-free out-of-fold prediction-level fusion;
- external transfer learning;
- RT-guided candidate filtering and soft reranking.

## Reported results

### SMRT prediction

- TC-TopoRT-S: **25.055 ± 0.039 s MAE**
- TC-TopoRT-E, five-seed ensemble: **24.920 s MAE**
- Seeds: `1, 5, 79, 123, 256`

The conventional atom-bond GNN baseline obtained **28.252 s MAE**, which is 3.240 s higher than TC-TopoRT seed 5.

### External transfer learning

Across ten external datasets:

- transfer learning achieved lower MAE on **8/10 datasets**;
- mean MAE improvement: **9.164 s**;
- median MAE improvement: **3.677 s**.

### Candidate filtering

| Dataset | Reduction | True retained | Top-1 | Top-5 | Top-10 | FN |
|---|---:|---:|---:|---:|---:|---:|
| MetaboBase | 69.14% | 93.33% | 55.56% | 82.22% | 88.89% | 3 |
| RIKEN-PlaSMA | 46.23% | 97.65% | 54.12% | 77.65% | 89.41% | 2 |

## Repository structure

~~~text
scripts/
├── training/    SMRT training entries
├── data/        paired-view construction and validation
├── ablation/    structural and atom-bond GNN ablations
├── transfer/    external transfer and scratch workflows
├── filtering/   candidate filtering and sensitivity analysis
└── figures/     public figure-generation entries

data/
├── candidate_filtering/
└── ablation/

artifacts/
├── cache/
├── results/
└── figures/
~~~

Generated outputs under `artifacts/` are excluded from Git.

## Installation

Using Conda:

~~~bash
conda env create -f environment.yml
conda activate tc-toport
~~~

Using pip:

~~~bash
python -m pip install -r requirements.txt
~~~

PyTorch and PyTorch Geometric may need installation commands compatible with the local CUDA version.

## Reproduction

Run all commands from the repository root.

### 1. Build and validate paired SMRT views

~~~bash
bash scripts/data/rebuild_strict_tautomer_views.sh
bash scripts/data/validate_smrt_paired_views.sh
~~~

Expected validation:

~~~text
Train: 70,182
Test: 7,798
Train changed: 37,724
Test changed: 4,242
Formula preserved: all
Invalid SMILES: 0
~~~

### 2. Train TC-TopoRT

Single seed:

~~~bash
bash scripts/training/run_smrt_single_seed.sh 5
~~~

Five seeds:

~~~bash
bash scripts/training/run_smrt_five_seeds.sh
~~~

Outputs are written under `artifacts/results/smrt/`.

### 3. Structural ablations

~~~bash
bash scripts/ablation/run_structural_ablation.sh no2cell
bash scripts/ablation/run_structural_ablation.sh cwn0
~~~

### 4. Atom-bond GNN baseline

~~~bash
bash scripts/ablation/run_atom_bond_gnn.sh
~~~

### 5. External transfer and scratch experiments

~~~bash
python scripts/transfer/train_scratch_all10.py
python scripts/transfer/train_transfer_all10.py
~~~

The combined comparison table is written to:

~~~text
artifacts/results/external_transfer/Table_8_transfer_learning_effectiveness.csv
~~~

### 6. Candidate filtering

~~~bash
python scripts/filtering/run_candidate_filtering.py
python scripts/filtering/run_filtering_sensitivity.py
~~~

### 7. Generate figures

~~~bash
python scripts/figures/make_candidate_filtering_figure.py
python scripts/figures/make_transfer_figure.py
python scripts/figures/make_ablation_figure.py
python scripts/figures/make_smrt_figures.py
~~~

The SMRT figure entry expects:

~~~text
artifacts/results/smrt/seed5/test_predictions.csv
~~~

A different result directory can be supplied with:

~~~bash
python scripts/figures/make_smrt_figures.py \
  --result_dir artifacts/results/smrt/<result-directory>
~~~

## Leakage control

The paired molecular views share identical labels and splits. Prediction-level fusion is fitted from out-of-fold training predictions. Independent test labels are not used for model selection, stacker fitting, calibration, or filtering-parameter optimization.

The candidate-filtering comparisons use consistent candidate lists, query sets, experimental RT values, original MS-FINDER ranks, filtering rules, reranking definitions, and evaluation metrics.

## Data policy

The public candidate-filtering CSV files and the compact ablation source table are included under `data/`.

Large caches, checkpoints, weights, logs, generated predictions, tables, and figures remain under `artifacts/` and are not committed.

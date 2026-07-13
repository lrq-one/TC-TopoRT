# TC-TopoRT

TC-TopoRT is a topology-aware framework for small-molecule retention-time prediction and retention-time-guided candidate prioritization.

This public repository contains the reproducibility entry points, the minimal model/data code required by those entry points, the SMRT paired views, compact candidate-filtering records, paper-facing configurations, and result-aggregation scripts. Historical experiments, checkpoints, logs, generated figures, manuscript copies, and backup files are intentionally excluded.

Detailed documentation:

- [Data sources and redistribution scope](DATA_SOURCES.md)
- [Paper-to-code reproducibility map](REPRODUCIBILITY.md)
- [Paper-facing configurations](configs/)

## Method overview

TC-TopoRT combines:

- paired dataset-provided and strict tautomer-canonical molecular views;
- ring-aware cell complexes with atom, bond, and ring cells;
- CWN-based topology-aware message passing;
- leakage-free out-of-fold prediction-level fusion;
- external transfer-learning evaluation;
- guarded RT filtering and soft reranking of MS-FINDER candidates.

Ring 2-cells are constructed from a NetworkX minimum cycle basis of each molecular graph. Cycles of sizes 3 through `max_ring_size` are retained; the paper configuration uses `max_ring_size = 6`.

## Reported results

### SMRT

- TC-TopoRT-S: **25.055 ± 0.039 s MAE** across five seeds.
- TC-TopoRT-E: **24.920 s MAE** for the five-seed ensemble.
- Seeds: `1, 5, 79, 123, 256`.
- Conventional atom-bond GNN comparison: **28.252 s MAE**.

The training workflow evaluates multiple OOF-only fusion controls. The formal five reported runs selected `huber_stack`, fitted only on training-set OOF predictions, and used it to generate independent-test predictions.

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
├── DATA_SOURCES.md
├── REPRODUCIBILITY.md
├── environment.yml
├── requirements.txt
├── configs/
│   ├── smrt.yaml
│   ├── external_transfer.yaml
│   ├── candidate_filtering.yaml
│   └── external_datasets.csv
├── data/
│   ├── ablation/
│   └── candidate_filtering/
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

## Fast smoke test

```bash
python scripts/tests/smoke_test.py
```

The test builds ring-aware cell complexes for two SMRT molecules and performs one model forward pass. It does not train a model or reproduce paper accuracy.

## SMRT data

The repository includes the dataset-provided SMRT split and paired strict tautomer-canonical view:

```text
gwn/data/SMRT_train.csv
gwn/data/SMRT_test.csv
gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv
gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv
```

After the `rt > 300 s` and RDKit-validity filters, the retained split contains 70,182 training molecules and 7,798 test molecules. Strict tautomer canonicalization changes 37,724 training representations and 4,242 test representations while preserving molecular formulae.

## Main reproduction commands

Run commands from the repository root. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete input-command-output map.

### Rebuild and validate paired views

```bash
bash scripts/data/rebuild_strict_tautomer_views.sh
bash scripts/data/validate_smrt_paired_views.sh
```

### Train one SMRT seed

```bash
bash scripts/training/run_smrt_single_seed.sh 5
```

Inspect the generated command without starting training:

```bash
DRY_RUN=1 bash scripts/training/run_smrt_single_seed.sh 5
```

### Train all five paper seeds and summarize

```bash
bash scripts/training/run_smrt_five_seeds.sh
python scripts/analysis/summarize_smrt_results.py
python scripts/analysis/build_dualview_ablation.py
```

### Structural and atom-bond ablations

```bash
bash scripts/ablation/run_structural_ablation.sh no2cell
bash scripts/ablation/run_structural_ablation.sh cwn0
bash scripts/ablation/run_atom_bond_gnn.sh
python scripts/analysis/collect_structural_ablation.py
```

### Candidate filtering and sensitivity

```bash
python scripts/filtering/run_candidate_filtering.py
python scripts/filtering/run_filtering_sensitivity.py
```

The compact candidate-level inputs are included under `data/candidate_filtering/`.

### External transfer and scratch comparison

The repository does not redistribute a complete PredRet database export. Convert a standardized combined PredRet table into the three inputs used by the public training scripts:

```bash
python scripts/data/prepare_external_predret.py \
  --input_csv /path/to/combined_predret.csv \
  --out_dir artifacts/data/external
```

Then run:

```bash
python scripts/transfer/train_scratch_all10.py \
  --stage4_meta_csv artifacts/data/external/external_predret10_stage4_meta.csv \
  --origin_csv artifacts/data/external/temp_external_predret10_origin.csv \
  --taut_csv artifacts/data/external/temp_external_predret10_taut.csv

python scripts/transfer/train_transfer_all10.py \
  --stage4_meta_csv artifacts/data/external/external_predret10_stage4_meta.csv \
  --origin_csv artifacts/data/external/temp_external_predret10_origin.csv \
  --taut_csv artifacts/data/external/temp_external_predret10_taut.csv \
  --smrt_runs_root artifacts/results/smrt
```

Transfer learning expects the SMRT source-fold checkpoints produced by the five-seed workflow.

### Generate figures

```bash
python scripts/figures/make_candidate_filtering_figure.py
python scripts/figures/make_ablation_figure.py
python scripts/figures/make_transfer_figure.py
python scripts/figures/make_smrt_figures.py
```

## Leakage control

The original and tautomer-canonical views share identical labels and split assignments. Prediction-level fusion is fitted only from training-set out-of-fold predictions. Independent-test labels are not used for model selection, stacker fitting, calibration, or filtering-parameter optimization.

Candidate-filtering comparisons use consistent candidate lists, query sets, experimental RT values, original MS-FINDER ranks, guarded-retention rules, soft-reranking definitions, and evaluation metrics.

## Repository scope

The repository intentionally does not commit model weights, checkpoints, caches, fold logs, generated predictions, final PDF/PNG figures, or manuscript files. These are generated outputs rather than required public inputs. Dataset provenance and third-party redistribution boundaries are documented in [DATA_SOURCES.md](DATA_SOURCES.md).

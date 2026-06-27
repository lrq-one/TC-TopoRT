# TCDV-TopoRT

TCDV-TopoRT is a topology-aware dual-view graph neural network pipeline for small-molecule retention time (RT) prediction on the SMRT dataset.

The current repository is organized around the final **TCDV-TopoRT** route, not the earlier legacy TopoCellRT experiments. The main idea is to represent the same SMRT compound with two paired molecular views:

1. the official/original SMILES graph;
2. a strict tautomer-canonical graph generated without changing the RT label or the train/test split.

Each view is encoded by the same CWN-based topology-aware model. The out-of-fold (OOF) predictions from the two views are then combined by a prediction-level stacker selected only on OOF validation predictions.

## Key idea

TCDV-TopoRT uses three design choices.

- **Dual-view molecular construction**: each molecule keeps the original SMRT RT label, while an additional strict tautomer-canonical view is generated for the same compound.
- **Topology-aware molecular representation**: atom, bond, and ring-cell information is encoded by a CWN-style cell-complex backbone and summarized as atom/bond/ring tokens.
- **OOF prediction-level stacking**: 5-fold OOF predictions from the original and tautomer views are used to train a robust Huber stacker, avoiding test-set tuning.

## Repository structure

```text
.
├── build_tautomer_strict_csv.py          # strict tautomer CSV generator
├── README.md
└── gwn/
    ├── train_oof_dualview_stack.py       # main TCDV-TopoRT training/evaluation entry
    ├── run_oof_multiseed_5runs.sh        # multi-seed OOF runs
    ├── data/
    │   ├── SMRT_train.csv
    │   └── SMRT_test.csv
    ├── data_taut_strict_origin_order/
    │   ├── SMRT_train_tautomer_strict.csv
    │   ├── SMRT_test_tautomer_strict.csv
    │   ├── SMRT_train_tautomer_strict_reorder_audit.csv
    │   └── SMRT_test_tautomer_strict_reorder_audit.csv
    ├── diagnostics/
    │   ├── 40_check_dualview_pair_data.py
    │   ├── 41_reorder_existing_taut_to_origin_order.py
    │   └── 50_make_oof_paper_figures.py
    ├── mp/                               # cell-complex data structures and CWN layers
    └── net/
        ├── cwn.py
        ├── cwn_abcort_transformer.py
        ├── cwn_hypergraph_adapter.py
        └── topocellrt_cwn_replace.py
```

The old exploratory scripts, old models, logs, checkpoints, and intermediate results were intentionally removed from the tracked repository. Local archives are ignored by `.gitignore`.

## Environment

The code was developed with a Python/PyTorch Geometric/RDKit environment. The main dependencies are:

```text
Python 3.10
PyTorch
PyTorch Geometric
RDKit
NumPy
Pandas
scikit-learn
tqdm
torch-scatter
torch-sparse
torch-cluster
```

Example environment activation used in experiments:

```bash
conda activate lrq_q
```

## Data

The final repository includes the official SMRT split and the paired strict tautomer-canonical view:

```text
gwn/data/SMRT_train.csv
gwn/data/SMRT_test.csv
gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv
gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv
```

The training pipeline filters molecules using the same rule as the dataset code: `rt > 300.0`, followed by RDKit-valid molecule filtering.

The current paired data check gives:

```text
Train valid rows: 70182
Test valid rows : 7798
Train strict tautomer changed: 37724 / 70182
Test strict tautomer changed : 4242 / 7798
```

## Regenerate strict tautomer data

The strict tautomer CSV generator now uses the repository-internal `gwn/data/` files by default and writes generated outputs to a non-final output directory to avoid overwriting the curated paired data.

```bash
python build_tautomer_strict_csv.py \
  --train_csv gwn/data/SMRT_train.csv \
  --test_csv gwn/data/SMRT_test.csv \
  --out_dir gwn/data_taut_strict_generated
```

The generated files should be checked before use.

## Check original/tautomer pairing

Run the pairing diagnostic before training or after regenerating tautomer data:

```bash
cd gwn

PYTHONPATH=. python diagnostics/40_check_dualview_pair_data.py \
  --origin_train data/SMRT_train.csv \
  --origin_test data/SMRT_test.csv \
  --taut_train data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv \
  --taut_test data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv
```

A successful check should report zero RT mismatch and pass both TRAIN and TEST pair checks.

## Train TCDV-TopoRT

Run a single 5-fold OOF dual-view stack experiment:

```bash
cd gwn

PYTHONPATH=. python train_oof_dualview_stack.py \
  --k 5 \
  --seed 1 \
  --epochs 150 \
  --patience 30 \
  --batch_size 64 \
  --eval_batch_size 64 \
  --num_workers 4 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --huber_beta 1.0 \
  --origin_train_csv data/SMRT_train.csv \
  --origin_test_csv data/SMRT_test.csv \
  --taut_train_csv data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv \
  --taut_test_csv data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv \
  --origin_train_root smrt_cwn_oof_origin_train \
  --origin_test_root smrt_cwn_oof_origin_test \
  --taut_train_root smrt_cwn_oof_taut_train \
  --taut_test_root smrt_cwn_oof_taut_test \
  --out_dir results_OOF_DualView_Stack_v1
```

To run the additional seeds used in the final multi-seed summary:

```bash
cd gwn
bash run_oof_multiseed_5runs.sh
```

The helper script runs seeds `79`, `123`, `256`, and `5`. Seed `1` is usually run separately as `results_OOF_DualView_Stack_v1`.

## Resume cached OOF predictions

If fold-level cached prediction files already exist, the training script can rebuild the OOF/test prediction tables and stacker results without retraining the GNNs:

```bash
cd gwn

PYTHONPATH=. python train_oof_dualview_stack.py \
  --resume 1 \
  --k 5 \
  --seed 1 \
  --epochs 150 \
  --patience 30 \
  --batch_size 64 \
  --eval_batch_size 64 \
  --num_workers 4 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --huber_beta 1.0 \
  --origin_train_csv data/SMRT_train.csv \
  --origin_test_csv data/SMRT_test.csv \
  --taut_train_csv data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv \
  --taut_test_csv data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv \
  --origin_train_root smrt_cwn_oof_origin_train \
  --origin_test_root smrt_cwn_oof_origin_test \
  --taut_train_root smrt_cwn_oof_taut_train \
  --taut_test_root smrt_cwn_oof_taut_test \
  --out_dir results_OOF_DualView_Stack_v1
```

For a complete 5-fold dual-view run, the resume sanity check should print 10 cached-load messages:

```text
5 folds × 2 views = 10 [RESUME] messages
```

## Model files

The final model chain is:

```text
train_oof_dualview_stack.py
  -> net/topocellrt_cwn_replace.py
  -> net/cwn_hypergraph_adapter.py
  -> net/cwn_abcort_transformer.py
  -> net/cwn.py
```

`TopoCellRTCWNReplace` uses a CWN adapter to obtain three topology-aware tokens per molecule: atom-token, bond-token, and ring-token. These tokens are transformed, pooled, gated, combined with 24-dimensional molecular global context, and finally passed to the RT regression head.

## Final local 5-seed result

The final local 5-seed test summary was:

```text
MAE   : 25.0551 ± 0.0391 s
MedAE : 11.3168 ± 0.0976 s
RMSE  : 55.6713 ± 0.1006 s
R2    : 0.8983 ± 0.0004
P95   : 86.4581 ± 0.5962 s
P99   : 282.7597 ± 4.0937 s
>100s : 319.6 ± 5.8 molecules
>200s : 134.2 ± 2.2 molecules
Bias  : 1.7807 ± 0.1121 s
```

The selected final prediction-level stacker was `huber_stack`.

## Output files

Each run writes files such as:

```text
results_OOF_DualView_Stack_*/config.json
results_OOF_DualView_Stack_*/final_metrics.json
results_OOF_DualView_Stack_*/oof_predictions.csv
results_OOF_DualView_Stack_*/test_predictions.csv
results_OOF_DualView_Stack_*/oof_base_predictions.csv
results_OOF_DualView_Stack_*/test_base_predictions.csv
```

Generated result folders, logs, local archives, and cached graph datasets are ignored by `.gitignore`.

## Notes

- The repository keeps the final TCDV-TopoRT route only.
- The strict tautomer view must remain paired with the original SMRT order and label.
- Do not tune the prediction-level stacker on the independent test set.
- Use the OOF validation predictions to select stacking/fusion parameters.

## External all10 transfer-vs-scratch reproduction

The external transfer-vs-scratch comparison is organized into two clear lines: transfer learning and from-scratch training.

### Dataset list

The all10 external dataset list is stored in:

    gwn/configs/external_all10_datasets.csv

It combines the previous six Table-2 external datasets with four additional external datasets:

    FEM_short_73
    UniToyama_Atlantis_143
    FEM_long_412
    Eawag_XBridgeC18_364
    LIFE_old_194
    MTBLS87_147
    LIFE_new_184
    Cao_HILIC_116
    IPB_Halle_82
    FEM_lipids_72

### Transfer-learning line

Paper-facing wrapper:

    gwn/experiments_transfer_effectiveness/external_transfer_all10.py

Shell entry point:

    cd gwn
    bash experiments_transfer_effectiveness/run_transfer_all10_datasets.sh

This line uses the TCDV-TopoRT transfer-learning protocol with fixed raw AutoSelect aggregation.

### From-scratch line

Paper-facing wrapper:

    gwn/experiments_transfer_effectiveness/external_scratch_all10.py

Shell entry point:

    cd gwn
    bash experiments_transfer_effectiveness/run_scratch_all10_datasets.sh

This line uses random initialization / scratch training on the same all10 external datasets.

### Existing all10 result summary and figure

The already obtained all10 transfer-vs-scratch MAE values are summarized and plotted by:

    cd gwn
    python experiments_transfer_effectiveness/make_external_all10_transfer_vs_scratch_figure.py

This script does not train models. It formats existing all10 scratch and transfer-learning MAE values into CSV, Markdown, TXT summaries, and a bar figure.

Output directory:

    gwn/experiments_transfer_effectiveness/all10_transfer_vs_scratch_final/

### Relationship to core model code

The external all10 wrappers do not change the core TCDV-TopoRT model implementation.

Core model code remains under:

    gwn/mp/
    gwn/net/
    gwn/train_oof_dualview_stack.py


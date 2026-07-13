# TC-TopoRT

TC-TopoRT is a topology-aware framework for small-molecule retention-time prediction and RT-guided candidate prioritization.

This repository provides the model code, experiment configurations, processed inputs, and scripts used for the SMRT benchmark, ablation studies, external transfer learning, and candidate-filtering experiments.

## Environment

Create the recommended Conda environment:

```bash
conda env create -f environment.yml
conda activate tc-toport
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

PyTorch Geometric extension packages may need to be installed separately for the local PyTorch and CUDA versions.

## Repository layout

```text
configs/                    Experiment configurations
data/external/              Processed PredRet inputs
data/candidate_filtering/   Candidate-level filtering inputs
gwn/data/                   SMRT train/test data
gwn/data_taut_strict_origin_order/
                             Strict tautomer paired views
gwn/mp/                     Cell-complex construction and CWN layers
gwn/net/                    TC-TopoRT model
scripts/training/           SMRT training workflows
scripts/ablation/           Ablation and atom-bond GNN controls
scripts/transfer/           External transfer and scratch workflows
scripts/filtering/          Candidate filtering and sensitivity analysis
scripts/analysis/           Result aggregation
scripts/tests/              Lightweight validation
```

Generated checkpoints, caches, predictions, metrics, tables, and figures are written under `artifacts/`.

## Data

### SMRT

The filtered official SMRT split and aligned strict tautomer views are included in:

```text
gwn/data/SMRT_train.csv
gwn/data/SMRT_test.csv
gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv
gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv
```

Source: Domingo-Almenara et al., *Nature Communications* (2019), [Figshare dataset](https://doi.org/10.6084/m9.figshare.8038913).

### External PredRet datasets

The processed inputs used for the ten-dataset transfer-versus-scratch experiment are included in:

```text
data/external/external_predret10_stage4_meta.csv
data/external/temp_external_predret10_origin.csv
data/external/temp_external_predret10_taut.csv
```

The three files contain 1,787 aligned records from ten chromatographic systems. Dataset names and settings are listed in `configs/external_datasets.csv`.

Source: Stanstrup et al., *Analytical Chemistry* (2015), [official publication record](https://doi.org/10.1021/acs.analchem.5b02287).

### Candidate filtering

Processed candidate-level inputs are included in:

```text
data/candidate_filtering/metabobase_candidate_predictions.csv
data/candidate_filtering/riken_candidate_predictions.csv
```

The corresponding parameter-sensitivity grids are also provided in `data/candidate_filtering/`.

Sources:

- MetaboBase: Lei et al., *Analytical Chemistry* (2015), [official publication record](https://doi.org/10.1021/acs.analchem.5b01559)
- RIKEN-PlaSMA: [RIKEN PRIMe](http://prime.psc.riken.jp/), [Metabolomics Workbench data DOI](https://doi.org/10.21228/M8XM40), and [publication](https://www.nature.com/articles/s41592-019-0358-2)
- MassBank: [official database](https://massbank.eu/MassBank/)

## Validation

Run syntax and data checks:

```bash
bash scripts/tests/run_static_checks.sh
```

Run a minimal model-forward test:

```bash
python scripts/tests/smoke_test.py
```

## SMRT training

Run one seed:

```bash
bash scripts/training/run_smrt_single_seed.sh 5
```

Run the five reported seeds and aggregate the results:

```bash
bash scripts/training/run_smrt_five_seeds.sh
python scripts/analysis/summarize_smrt_results.py
```

## Ablation studies

Dual-view and fusion analysis:

```bash
python scripts/analysis/build_dualview_ablation.py
```

Structural ablations and atom-bond GNN control:

```bash
bash scripts/ablation/run_structural_ablation.sh no2cell
bash scripts/ablation/run_structural_ablation.sh cwn0
bash scripts/ablation/run_atom_bond_gnn.sh
python scripts/analysis/collect_structural_ablation.py
```

## External transfer learning

Validate the aligned PredRet inputs:

```bash
python scripts/data/validate_external_predret_inputs.py \
  --meta_csv data/external/external_predret10_stage4_meta.csv \
  --origin_csv data/external/temp_external_predret10_origin.csv \
  --taut_csv data/external/temp_external_predret10_taut.csv
```

Train the scratch control:

```bash
python scripts/transfer/train_scratch_all10.py \
  --stage4_meta_csv data/external/external_predret10_stage4_meta.csv \
  --origin_csv data/external/temp_external_predret10_origin.csv \
  --taut_csv data/external/temp_external_predret10_taut.csv
```

Run transfer learning after the SMRT source-fold checkpoints have been generated:

```bash
python scripts/transfer/train_transfer_all10.py \
  --stage4_meta_csv data/external/external_predret10_stage4_meta.csv \
  --origin_csv data/external/temp_external_predret10_origin.csv \
  --taut_csv data/external/temp_external_predret10_taut.csv \
  --smrt_runs_root artifacts/results/smrt
```

## Candidate filtering

Recompute the fixed filtering and reranking results:

```bash
python scripts/filtering/run_candidate_filtering.py
```

Run the parameter-sensitivity analysis:

```bash
python scripts/filtering/run_filtering_sensitivity.py
```

## Notes

Published baseline values reported in the manuscript were taken from the corresponding original studies and were not retrained in this repository.

Model checkpoints and generated experiment outputs are not committed because they can be regenerated from the provided code, configurations, and inputs.

## License

This repository is released under the [MIT License](LICENSE). Third-party datasets and libraries remain subject to their original licenses and terms.

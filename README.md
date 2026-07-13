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
data/external/              Processed external RT inputs
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

## Data and acquisition

The processed CSV files included in this repository are the exact inputs used in the reported experiments. The links below identify where the original public data can be obtained. When no stable public bulk-download page was identified, the source publication is provided and the processed records used in this study are included here.

### 1. SMRT benchmark

Included files:

```text
gwn/data/SMRT_train.csv
gwn/data/SMRT_test.csv
gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv
gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv
```

Original public source:

- Domingo-Almenara et al., *Nature Communications* (2019)
- [Official Figshare dataset](https://doi.org/10.6084/m9.figshare.8038913)

The repository contains the filtered official train/test split and the aligned strict tautomer views used by TC-TopoRT.

### 2. External RT transfer datasets

Included processed files:

```text
data/external/external_predret10_stage4_meta.csv
data/external/temp_external_predret10_origin.csv
data/external/temp_external_predret10_taut.csv
```

These files contain 1,787 aligned records from the ten chromatographic systems used in the transfer-versus-scratch experiment. Dataset names, sample counts, and settings are also listed in `configs/external_datasets.csv`.

| Dataset | Records | Use in the paper | Original source / access |
|---|---:|---|---|
| Eawag-XBridgeC18 | 364 | Literature comparison and transfer-vs-scratch | [PredRet publication and supporting information](https://doi.org/10.1021/acs.analchem.5b02287) |
| FEM-lipids | 72 | Literature comparison and transfer-vs-scratch | [PredRet publication and supporting information](https://doi.org/10.1021/acs.analchem.5b02287) |
| FEM-long | 412 | Literature comparison and transfer-vs-scratch | [PredRet publication and supporting information](https://doi.org/10.1021/acs.analchem.5b02287) |
| IPB-Halle | 82 | Literature comparison and transfer-vs-scratch | [PredRet publication and supporting information](https://doi.org/10.1021/acs.analchem.5b02287) |
| LIFE-new | 184 | Literature comparison and transfer-vs-scratch | [PredRet publication and supporting information](https://doi.org/10.1021/acs.analchem.5b02287) |
| LIFE-old | 194 | Literature comparison and transfer-vs-scratch | [PredRet publication and supporting information](https://doi.org/10.1021/acs.analchem.5b02287) |
| FEM-short | 73 | Transfer-vs-scratch | [PredRet publication and supporting information](https://doi.org/10.1021/acs.analchem.5b02287) |
| UniToyama-Atlantis | 143 | Transfer-vs-scratch | [PredRet publication and supporting information](https://doi.org/10.1021/acs.analchem.5b02287) |
| MTBLS87 | 147 | Transfer-vs-scratch | [MetaboLights accession MTBLS87](https://www.ebi.ac.uk/metabolights/MTBLS87); also compiled in PredRet |
| Cao-HILIC | 116 | Transfer-vs-scratch | [PredRet publication and supporting information](https://doi.org/10.1021/acs.analchem.5b02287) |

For systems without a separate stable public download page, the aligned processed inputs used in this study are already provided under `data/external/`.

### 3. Candidate-filtering datasets

Included processed files:

```text
data/candidate_filtering/metabobase_candidate_predictions.csv
data/candidate_filtering/riken_candidate_predictions.csv
data/candidate_filtering/metabobase_rank_guard_soft_grid.csv
data/candidate_filtering/riken_rank_guard_soft_grid.csv
```

| Dataset | Queries | Use in the paper | Original source / access |
|---|---:|---|---|
| MetaboBase | 45 | Candidate filtering, reranking, Top-k, true-retention, and false-negative analysis | Lei et al., *Analytical Chemistry* (2015), [publication DOI](https://doi.org/10.1021/acs.analchem.5b01559). A stable public bulk-download page for the complete MetaboBase library was not identified; the exact processed candidate records used here are included in this repository. |
| RIKEN-PlaSMA | 85 | Candidate filtering, reranking, Top-k, true-retention, and false-negative analysis | [RIKEN PRIMe](http://prime.psc.riken.jp/), [Metabolomics Workbench data DOI](https://doi.org/10.21228/M8XM40), [source publication](https://www.nature.com/articles/s41592-019-0358-2), and [MassBank](https://massbank.eu/MassBank/) |

Candidate lists and original ranks were generated with MS-FINDER. The official software page is available at [MS-FINDER](https://systemsomicslab.github.io/compms/msfinder/main.html). The two candidate-level CSV files above contain the exact query/candidate records, original ranks, predicted RT values, and truth labels used to recompute the reported filtering results.

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

Validate the aligned external inputs:

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

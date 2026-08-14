# TC-TopoRT

TC-TopoRT is a topology-aware cell-complex neural network for LC-MS retention
time prediction and RT-assisted metabolite candidate filtering. The model uses
paired dataset-provided and conservatively tautomer-standardized molecular
views, explicit ring 2-cells, cellular message passing, and leakage-free
out-of-fold (OOF) fusion.

The paired views are representation controls. They do not model physical
solution-phase tautomer populations, pH-specific microspecies, or ESI tautomer
equilibria.

## Repository contents

- `gwn/mp/` and `gwn/net/`: cell-complex construction and TC-TopoRT model code.
- `gwn/train_oof_dualview_stack.py`: paired-view five-fold OOF training and Huber fusion.
- `configs/`: final SMRT, external-transfer, and candidate-filtering configurations.
- `scripts/data/`: paired-view and external-data preparation.
- `scripts/training/`: one-seed and five-seed SMRT entry points.
- `scripts/ablation/`: No2Cell and parameter-matched atom-bond GINE controls.
- `scripts/transfer/`: ten-dataset transfer-versus-scratch experiment.
- `scripts/filtering/`: final development-calibrated hard candidate filter.
- `scripts/analysis/` and `scripts/figures/`: final-paper and SI analyses.
- `scripts/tests/`: dataset-free static, smoke, and filtering-rule checks.
- `docs/`: detailed reproduction and supplementary-analysis notes.

## Data and archived outputs

GitHub contains code, configurations, environment definitions, and reproduction
instructions. It intentionally excludes checkpoints, graph caches, large source
tables, prediction dumps, candidate-level outputs, and generated figures.

The original METLIN SMRT benchmark source is available at
[DOI 10.6084/m9.figshare.8038913](https://doi.org/10.6084/m9.figshare.8038913).
This is the original source benchmark, not the authors' TC-TopoRT archive.
External datasets remain subject to their original sources and terms.

The separate author Figshare archive contains trained models, final predictions,
candidate lists, candidate-level filtering outputs, development-calibration
records, frozen thresholds, final configuration snapshots, and a checksum
manifest. Its DOI will be added after the deposit is finalized; no DOI is
invented here. See [data/README.md](data/README.md) for local placement.

## Environment

The paper workflow used Python 3.10, PyTorch 2.3.0, CUDA 12.1, PyTorch
Geometric, RDKit, and an NVIDIA RTX 3090 with 24 GB memory.

```bash
conda env create -f environment.yml
conda activate tc-toport
```

A pip installation can use `pip install -r requirements.txt` in a clean Python
3.10 environment. Install the `torch-scatter` and `torch-sparse` binary wheels
that match the selected PyTorch/CUDA build; PyG extension wheels are not
interchangeable across arbitrary PyTorch/CUDA versions. CPU-only validation is
possible, but full training is computationally expensive.

## Quick validation

These checks do not require the full SMRT dataset:

```bash
bash scripts/tests/run_static_checks.sh
python scripts/tests/smoke_test.py
python scripts/tests/test_candidate_filtering.py
```

## SMRT benchmark

Place the four paired CSVs as described in `data/README.md`. The benchmark uses
70,182 training compounds, 7,798 independent test compounds, `RT > 300 s`, and
paper seeds `1 5 79 123 256`.

```bash
SEED=1 bash scripts/training/run_smrt_single_seed.sh
bash scripts/training/run_smrt_five_seeds.sh

python scripts/analysis/summarize_smrt_results.py \
  --prediction 1=artifacts/results/smrt/seed1/test_predictions.csv \
  --prediction 5=artifacts/results/smrt/seed5/test_predictions.csv \
  --prediction 79=artifacts/results/smrt/seed79/test_predictions.csv \
  --prediction 123=artifacts/results/smrt/seed123/test_predictions.csv \
  --prediction 256=artifacts/results/smrt/seed256/test_predictions.csv \
  --verify-paper-results
```

The final five-run MAE is `25.055090 ± 0.039094 s`; averaging the five final
test predictions gives approximately `24.920 s` MAE.

## Dual-view and structural controls

Final controls comprise Original-only, tautomer-standardized-only, O+O, T+T,
same-seed O+T arithmetic averaging, the final O+T OOF Huber fusion, shuffled
pairing, No2Cell, and the parameter-matched atom-bond GINE. No2Cell removes
explicit higher-order ring 2-cells while retaining conventional atom, bond, and
global ring descriptors.

```bash
bash scripts/ablation/run_structural_ablation.sh no2cell
bash scripts/ablation/run_atom_bond_gnn.sh
```

Independent-model fusion is the major contributor; tautomer standardization
provides a smaller representation-robustness contribution. Supplied per-seed
control tables can be checked with `build_dualview_ablation.py` and
`collect_structural_ablation.py`; see [docs/reproduction.md](docs/reproduction.md).

## External transfer

The final experiment compares SMRT-pretrained transfer with random initialization
on ten datasets:

```bash
python scripts/transfer/train_transfer_all10.py --dry_run 1
python scripts/transfer/train_scratch_all10.py --dry_run 1
```

Remove `--dry_run 1` for training after preparing inputs and source checkpoints.
Transfer is better on 8/10 datasets; mean and median MAE improvements are 9.164 s
and 3.677 s. Scratch is better on Cao-HILIC and IPB-Halle; MTBLS87 retains a
small positive transfer benefit.

## Candidate filtering

For predictor `m`, the threshold is calibrated on a fixed development set held
out from the calibration-model fit:

```text
T_m = 3 × MAE_dev,m
```

Within each candidate dataset, all four predictors use the same fixed 30
development queries and shared experimental RT labels. Final evaluation queries
are excluded from calibration. A candidate is retained when its prediction is
missing or `|experimental RT - predicted RT| <= T_m`; retained candidates keep
their original MS-FINDER order. There is no rank guard, soft reranking, `g`,
`tau`, `alpha`, hybrid score, or final-test threshold tuning.

TC-TopoRT uses `T = 174.868 s` for MetaboBase and `T = 80.977 s` for
RIKEN-PlaSMA. The final evaluation denominators are 45 and 85 queries,
respectively, restricted to evaluable queries whose true candidate is present in
the initial MS-FINDER list.

```bash
python scripts/filtering/run_candidate_filtering.py \
  --dataset metabobase \
  --input data/local/candidate_filtering/metabobase_candidate_predictions.csv

python scripts/filtering/run_candidate_filtering.py \
  --dataset riken_plasma \
  --input data/local/candidate_filtering/riken_plasma_candidate_predictions.csv
```

If the frozen inputs are absent, the script exits with an explicit Figshare/local
placement message and never fabricates an output.

The manuscript reports the following RIKEN-PlaSMA reference result for the
final protocol: 84/85 true candidates retained (98.82%, FN = 1), Top-1 = 45/85,
Top-5 = 63/85, and Top-10 = 72/85. These values are the manuscript-reported
reference results for the final evaluation protocol. Exact numerical
reproduction requires the corresponding candidate-level inputs and calibration
records used for that evaluation. The frozen candidate-filtering inputs and
calibration records are provided in the author Figshare archive.

## Supplementary analyses

See [docs/supplementary_analyses.md](docs/supplementary_analyses.md) for the
early-RT scope audit, tautomer-standardization collision audit, overlapping ring
subgroups, matched Full-versus-No2Cell RIKEN analysis, same-formula pairwise
discrimination, and the illustrative Norharman case.

## Reproducibility

[docs/reproduction.md](docs/reproduction.md) maps each paper result to its
command, required inputs, generated output, and archive availability. Code
regenerates analyses when the required inputs are supplied; the separate archive
provides the frozen research artifacts listed in its manifest so that GitHub
does not duplicate large data or trained models.

## Citation and license

Citation metadata are provided in [CITATION.cff](CITATION.cff). The manuscript
DOI is omitted until assigned. Repository code is MIT licensed. Third-party data
retain their original licenses and are not relicensed by this repository.

# Reviewer 2 reproducibility additions

This document records the new analyses and figure-generation steps prepared for the Reviewer 2 revision. The scripts do not alter the locked retained-compound SMRT benchmark. They add separate, explicitly labelled sensitivity analyses.

## 1. Formula-matched isomer discrimination

The analysis compares full TC-TopoRT with the no-ring-2-cell control and, when supplied, the atom-bond GNN control. It reports pairwise RT-ordering accuracy and RT-difference error for constitutional isomers, stereoisomers, operationally defined positional-like pairs, and ring-topology challenges.

```bash
python scripts/analysis/analyze_isomer_discrimination.py \
  --full_predictions artifacts/results/smrt/seed5/test_predictions.csv \
  --no2cell_predictions artifacts/results/structural_ablation/no2cell_seed5/test_predictions.csv \
  --atom_bond_predictions artifacts/results/atom_bond_gnn/seed5/test_predictions.csv \
  --out_dir artifacts/results/paper_tables/isomer_discrimination
```

The positional-like label is deliberately operational rather than ontological: same molecular formula, same Murcko scaffold, different constitutional graph, matching ring signature, and Morgan similarity at or above the configured threshold. The manuscript must report the exact definition and should not call these pairs manually curated positional isomers.

Expected outputs:

```text
isomer_pair_records.csv
isomer_pair_summary.csv
representative_isomer_pairs.csv
representative_isomer_pairs.png
```

## 2. Full-range and early-retention sensitivity

Download the official `SMRT_dataset.sdf` from the original Figshare release. Then construct an extended full-range split while preserving every retained-compound train/test assignment:

```bash
python scripts/data/prepare_smrt_full_range.py \
  --sdf /path/to/SMRT_dataset.sdf \
  --retained_train_csv gwn/data/SMRT_train.csv \
  --retained_test_csv gwn/data/SMRT_test.csv \
  --out_dir artifacts/data/smrt_full_range
```

The script assigns the previously excluded `RT <= 300 s` compounds using a deterministic stratified 90/10 split. This extended split is a sensitivity analysis and is not described as the official SMRT split.

Generate strict tautomer views for the extended data:

```bash
python scripts/data/build_strict_tautomer_views.py \
  --train_csv artifacts/data/smrt_full_range/SMRT_full_train.csv \
  --test_csv artifacts/data/smrt_full_range/SMRT_full_test.csv \
  --out_dir artifacts/data/smrt_full_range/strict_tautomer
```

Run a full-range model:

```bash
bash scripts/training/run_smrt_full_range_single_seed.sh 5
```

Evaluate all, early-retention, retained-region, and RT-bin performance:

```bash
python scripts/analysis/evaluate_early_retention.py \
  --predictions artifacts/results/smrt_full_range/seed5/test_predictions.csv \
  --out_dir artifacts/results/paper_tables/early_retention
```

For a final paper claim, repeat the sensitivity experiment across the same five seeds as the retained benchmark and report mean, SD, and the exact early-test count.

## 3. Concrete 0/1/2-cell illustration

Generate the explanatory molecular lifting figure:

```bash
python scripts/figures/draw_cell_complex_example.py \
  --name Caffeine \
  --out artifacts/figures/cell_complex_example.png
```

The example produces a structure-labelled schematic with atom 0-cells, bond 1-cells, and ring 2-cells. It is an explanatory figure, not an additional performance experiment.

## 4. Chromatographic metadata

`configs/external_chromatography_metadata.csv` records the SMRT method and the ten external target systems. `NR` means that the field was not reported in the cited article or was unavailable in RepoRT. Do not impute missing pH, temperature, or flow-rate values.

## 5. Interpretation guardrails

- RT is system-specific; external deployment requires target-system adaptation and validation.
- The dual-view branch is a representation-consistency device, not a model of solution-phase tautomer populations or exchange kinetics.
- Candidate filtering is conditional on the true structure being present in the upstream MS-FINDER list.
- The no-ring-2-cell experiment is the direct topology-specific ablation. The atom-bond GNN is an architecture-level reference because its feature dimensions differ from TC-TopoRT.

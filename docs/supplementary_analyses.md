# Supplementary controls and subgroup analyses

The scripts below restore the compact, paper-facing analyses used for the supplementary controls. They read formal prediction outputs under `artifacts/results/` and write regenerated tables under `artifacts/results/paper_tables/`.

Run from the repository root after the corresponding SMRT and structural-ablation experiments have completed.

## Shuffled tautomer-pairing control

```bash
python scripts/analysis/shuffled_pairing_control.py
```

This applies 50 deterministic permutations per seed, destroys molecule-wise original/tautomer pairing, refits the Huber stacker on shuffled training OOF predictions only, and evaluates the independent test predictions.

## Pairing and no-leakage audit

```bash
python scripts/analysis/audit_pairing_and_noleakage.py
```

The audit checks paired-view row and RT alignment, train/test structural overlap, OOF fold coverage, finite predictions, OOF-only stacker fitting, and test-prediction averaging without test-label use.

## Tautomer-changed and unchanged subgroups

```bash
python scripts/analysis/analyze_tautomer_subgroups.py
```

This reports original-view, strict-tautomer-view, and final fused MAE for changed, unchanged, and all test molecules across the five formal seeds.

## Ring-context subgroups

```bash
python scripts/analysis/analyze_ring_subgroups.py
```

This compares the full seed-5 model with the no-ring-2-cell ablation for acyclic, ring-containing, aromatic-ring, heterocycle-containing, and multi-ring molecules. The groups are not mutually exclusive.

## Strict-tautomer change types

```bash
python scripts/analysis/analyze_tautomer_change_types.py
```

This rebuilds retained and changed counts, verifies molecular-formula preservation, and generates a deterministic rule-based summary of representation-level tautomer canonicalization patterns. The categories are not claims about dominant solution-phase tautomers.

## Required generated inputs

The default commands expect:

```text
artifacts/results/smrt/seed{1,5,79,123,256}/
artifacts/results/structural_ablation/no2cell_seed5/test_predictions.csv
```

Generated outputs remain under `artifacts/` and should not be committed.

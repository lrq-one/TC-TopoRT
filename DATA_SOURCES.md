# Data sources and redistribution scope

This repository contains the minimum processed inputs needed for the public TC-TopoRT workflows. It does not redistribute third-party raw mass-spectral libraries, MS-FINDER software, model checkpoints, graph caches, or complete external-database mirrors.

## 1. METLIN SMRT

**Primary publication**

X. Domingo-Almenara et al., *The METLIN small molecule dataset for machine learning-based retention time prediction*, Nature Communications 10, 5811 (2019).

- Article: https://www.nature.com/articles/s41467-019-13680-7
- Original Figshare dataset and code: https://doi.org/10.6084/m9.figshare.8038913

**Files included here**

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
- keep the original SMILES whenever canonical tautomerization does not produce a genuine tautomeric change;
- reject a transformed view if molecular formula or heavy-atom count changes.

The paired-view construction and audit are reproducible with:

```bash
bash scripts/data/rebuild_strict_tautomer_views.sh
bash scripts/data/validate_smrt_paired_views.sh
```

## 2. PredRet external chromatographic datasets

**Primary publication**

J. Stanstrup, S. Neumann, and U. Vrhovsek, *PredRet: Prediction of Retention Time by Direct Mapping between Multiple Chromatographic Systems*, Analytical Chemistry 87, 9421–9428 (2015).

- Publication DOI: https://doi.org/10.1021/acs.analchem.5b02287

The ten external systems used in the transfer-versus-scratch analysis are listed in:

```text
configs/external_datasets.csv
```

The six systems used for direct comparison with literature transfer-learning results are Eawag-XBridgeC18, FEM-lipids, FEM-long, IPB-Halle, LIFE-new, and LIFE-old. The broader ten-system analysis additionally includes FEM-short, UniToyama-Atlantis, MTBLS87, and Cao-HILIC.

**Redistribution status**

The repository does not redistribute a complete PredRet database export. Users should obtain the source records from the public PredRet resources and the corresponding publications cited in the manuscript. A standardized combined CSV can then be converted into the exact three tables consumed by the public transfer scripts:

```bash
python scripts/data/prepare_external_predret.py \
  --input_csv /path/to/combined_predret.csv \
  --out_dir artifacts/data/external
```

The standardized input must contain a dataset identifier, SMILES, and experimental RT. Accepted column aliases are documented by `python scripts/data/prepare_external_predret.py --help`.

Generated files:

```text
artifacts/data/external/external_predret10_stage4_meta.csv
artifacts/data/external/temp_external_predret10_origin.csv
artifacts/data/external/temp_external_predret10_taut.csv
```

The graph-construction CSVs contain a dummy RT above 300 s because the shared `SMRTComplexDataset` loader applies the SMRT retained-compound filter. Actual external RT values are stored in the metadata table and replace the dummy target during external training.

## 3. MetaboBase candidate filtering

**Primary publication**

Z. Lei et al., *Construction of an Ultrahigh Pressure Liquid Chromatography-Tandem Mass Spectral Library of Plant Natural Products and Comparative Spectral Analyses*, Analytical Chemistry 87, 7373–7381 (2015).

- Publication DOI: https://doi.org/10.1021/acs.analchem.5b01559

**Processed candidate-level input included here**

```text
data/candidate_filtering/metabobase_candidate_predictions.csv
```

This table contains the 45 candidate-evaluable queries and 3,023 candidate records used in the reported guarded filtering and soft-reranking analysis. It includes the original MS-FINDER rank, candidate structure identifiers, TC-TopoRT candidate RT predictions, RT discrepancies, and true-candidate labels required to recompute reduction, Top-k, true-retention, and false-negative metrics.

It is a processed analysis table, not a redistribution of the complete MetaboBase spectral library.

## 4. RIKEN-PlaSMA / MassBank-related candidate filtering

**Primary publication**

H. Tsugawa et al., *A cheminformatics approach to characterize metabolomes in stable-isotope-labeled organisms*, Nature Methods 16, 295–298 (2019).

- Article: https://www.nature.com/articles/s41592-019-0358-2
- Public data DOI reported by the article: https://doi.org/10.21228/M8XM40
- RIKEN PRIMe resource: http://prime.psc.riken.jp/
- MassBank: https://massbank.eu/MassBank/

**Processed candidate-level input included here**

```text
data/candidate_filtering/riken_candidate_predictions.csv
```

This table contains the 85 exact-ground-truth queries and 5,044 candidate records used in the reported analysis. It is sufficient to recompute the TC-TopoRT filtering and reranking metrics, but it is not a complete redistribution of PlaSMA, MassBank, or raw spectral files.

## 5. Sensitivity-grid source tables

```text
data/candidate_filtering/metabobase_rank_guard_soft_grid.csv
data/candidate_filtering/riken_rank_guard_soft_grid.csv
```

These are compact, precomputed parameter-grid result tables retained to permit immediate verification and plotting of the four-parameter sensitivity audit. The candidate-level inputs remain the primary processed records for the reported filtering results.

## 6. Outputs not redistributed

The following are generated locally under `artifacts/` and are intentionally not committed:

- checkpoints and model weights;
- graph/cell-complex caches;
- fold-level predictions and logs;
- generated paper tables and figures;
- external transfer intermediate predictions;
- historical experiment directories.

They can be regenerated from the code, configurations, and data sources described above.

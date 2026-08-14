# Data placement

This GitHub repository intentionally does not redistribute complete source
datasets, large candidate tables, checkpoints, predictions, or generated
outputs.

The original METLIN SMRT source benchmark is available from
[Figshare DOI 10.6084/m9.figshare.8038913](https://doi.org/10.6084/m9.figshare.8038913).
That DOI identifies the original benchmark source; it is not the authors'
TC-TopoRT archive. External transfer datasets must likewise be obtained from
their original public resources and used under their original terms.

The separate TC-TopoRT author Figshare archive is available at
[DOI 10.6084/m9.figshare.33252810](https://doi.org/10.6084/m9.figshare.33252810).
It provides trained models, frozen predictions, candidate lists, candidate-level
filtering outputs, development-calibration records, thresholds, final
configuration snapshots, and checksums.

Matched Full-versus-No2Cell, pairwise, and Norharman-specific analysis inputs
are not claimed as contents of the author archive. The corresponding public
analysis scripts operate when those study inputs are supplied separately.

## SMRT partition provenance

The original SMRT deposit provides the complete 80,038-record files
`SMRT_dataset.csv` and `SMRT_dataset.sdf`; it does not provide named train and
test CSVs. TC-TopoRT uses the train/test partition used in this study,
represented locally as `SMRT_train_set.txt` and `SMRT_test_set.txt`, converts
only the column names and CSV serialization, applies `RT > 300 s`, and requires
RDKit-valid structures.
The resulting retained inputs contain 70,182 training records and 7,798 test
records. This study does not generate a new train/test split.

The complete split files are third-party data and are not tracked here. The
current public repository also does not contain an independent split-ID manifest
or a DOI-to-partition generation procedure, so reconstructing the exact
partition from `SMRT_dataset.csv` alone is not claimed. Users reproducing the
benchmark must supply the same study partition at the paths below.

After downloading or preparing the inputs, use this local layout:

```text
gwn/data/
├── SMRT_train.csv
└── SMRT_test.csv

gwn/data_taut_strict_origin_order/
├── SMRT_train_tautomer_strict.csv
└── SMRT_test_tautomer_strict.csv

data/local/candidate_filtering/
├── metabobase_candidate_predictions.csv
└── riken_plasma_candidate_predictions.csv

gwn/paper_analysis_stage4_external/
├── external_predret10_stage4_meta.csv
├── temp_external_predret10_origin.csv
└── temp_external_predret10_taut.csv
```

Create the external files from a combined source table with
`scripts/data/prepare_external_predret.py`; this local directory is ignored by
Git. The SMRT training wrappers retain their established `gwn/data` convention.
The candidate-filtering script accepts `--input` when a frozen archive is placed
elsewhere. Generated caches, checkpoints, predictions, and summaries are written
under `artifacts/` and are ignored by Git.

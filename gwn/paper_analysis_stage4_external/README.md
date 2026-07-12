# External transfer inputs

The public transfer and scratch scripts expect three processed PredRet tables:

```text
external_predret10_stage4_meta.csv
temp_external_predret10_origin.csv
temp_external_predret10_taut.csv
```

These external processed inputs are not redistributed in this repository. Place them in this directory using the names above, or provide their locations explicitly:

```bash
python scripts/transfer/train_scratch_all10.py \
  --stage4_meta_csv /path/to/external_predret10_stage4_meta.csv \
  --origin_csv /path/to/temp_external_predret10_origin.csv \
  --taut_csv /path/to/temp_external_predret10_taut.csv
```

Use the same three arguments with `train_transfer_all10.py`. Transfer learning additionally requires the SMRT source-fold checkpoints generated under `artifacts/results/smrt/`, unless another `--smrt_runs_root` is supplied.

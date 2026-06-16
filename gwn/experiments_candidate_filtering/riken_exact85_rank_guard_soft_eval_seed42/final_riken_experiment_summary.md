# Final RIKEN exact85 candidate filtering results

## Experiment A: Ours ablation

| Method                             | N  | Candidates before | Candidates after | Reduction (%) | True retention (%) | Top1 (%) | Delta Top1 vs MS-FINDER | Top5 (%) | Delta Top5 vs MS-FINDER | Top10 (%) | Delta Top10 vs MS-FINDER | Threshold | Guard k | Tau   | Alpha |
| ---------------------------------- | -- | ----------------- | ---------------- | ------------- | ------------------ | -------- | ----------------------- | -------- | ----------------------- | --------- | ------------------------ | --------- | ------- | ----- | ----- |
| MS-FINDER original                 | 85 | 5044              | 5044             | 0.0           | 100.0              | 47.06    | 0.0                     | 70.59    | 0.0                     | 82.35     | 0.0                      |           |         |       |       |
| Hard RT filter                     | 85 | 5044              | 3829             | 24.09         | 100.0              | 52.94    | 5.88                    | 76.47    | 5.88                    | 85.88     | 3.53                     | 100.0     |         |       |       |
| RT soft rerank                     | 85 | 5044              | 5044             | 0.0           | 100.0              | 55.29    | 8.24                    | 75.29    | 4.71                    | 83.53     | 1.18                     |           |         | 25.66 | 1.5   |
| RT-aware guarded soft rerank       | 85 | 5044              | 2712             | 46.23         | 97.65              | 54.12    | 7.06                    | 77.65    | 7.06                    | 89.41     | 7.06                     | 50.0      | 2.0     | 25.66 | 2.0   |
| High-reduction guarded soft rerank | 85 | 5044              | 2316             | 54.08         | 95.29              | 54.12    | 7.06                    | 77.65    | 7.06                    | 87.06     | 4.71                     | 40.0      | 2.0     | 25.66 | 2.0   |


## Experiment B: ABCoRT-TL reported vs ours

| Method                            | N  | Reduction (%) | Delta Reduction vs ABCoRT-TL | Top1 (%) | Delta Top1 vs ABCoRT-TL | Top5 (%) | Delta Top5 vs ABCoRT-TL | Top10 (%) | Delta Top10 vs ABCoRT-TL | True retention (%) | Threshold | Guard k | Tau   | Alpha |
| --------------------------------- | -- | ------------- | ---------------------------- | -------- | ----------------------- | -------- | ----------------------- | --------- | ------------------------ | ------------------ | --------- | ------- | ----- | ----- |
| ABCoRT-TL reported                | 85 | 28.46         | 0.0                          | 52.94    | 0.0                     | 76.47    | 0.0                     | 83.53     | 0.0                      |                    | 76.98     |         |       |       |
| MS-FINDER original                | 85 | 0.0           | -28.46                       | 47.06    | -5.88                   | 70.59    | -5.88                   | 82.35     | -1.18                    | 100.0              |           |         |       |       |
| Hard RT filter                    | 85 | 24.09         | -4.37                        | 52.94    | 0.0                     | 76.47    | 0.0                     | 85.88     | 2.35                     | 100.0              | 100.0     |         |       |       |
| Ours guarded soft, balanced       | 85 | 46.23         | 17.77                        | 54.12    | 1.18                    | 77.65    | 1.18                    | 89.41     | 5.88                     | 97.65              | 50.0      | 2.0     | 25.66 | 2.0   |
| Ours guarded soft, high-reduction | 85 | 54.08         | 25.62                        | 54.12    | 1.18                    | 77.65    | 1.18                    | 87.06     | 3.53                     | 95.29              | 40.0      | 2.0     | 25.66 | 2.0   |


## RIKEN TL RT metrics

| view         | split              | best_epoch | best_val_mae       | MAE                | RMSE               | MedAE             | bias                |
| ------------ | ------------------ | ---------- | ------------------ | ------------------ | ------------------ | ----------------- | ------------------- |
| origin       | train341           | 37         | 26.543915692497706 | 8.444282867342146  | 20.23493446896869  | 3.73333740234375  | -0.8233487606048584 |
| origin       | riken_exact85_test | 37         | 26.543915692497706 | 25.15687938017004  | 34.965818662529884 | 16.26190185546875 | 2.886482954025269   |
| taut         | train341           | 13         | 26.451297535615808 | 13.910937983269566 | 22.666188118554928 | 9.21295166015625  | -7.589876651763916  |
| taut         | riken_exact85_test | 13         | 26.451297535615808 | 26.444976806640625 | 35.61089108202495  | 19.4342041015625  | -4.083080768585205  |
| dualview_avg | train341           | -1         |                    | 10.55725231897796  | 20.21497300021401  | 6.5540771484375   | -4.20661295334265   |
| dualview_avg | riken_exact85_test | -1         |                    | 23.40818625057445  | 30.80247360567641  | 18.26058959960937 | -0.5982989142922794 |


Notes:

- RIKEN exact85 is reconstructed from Table S11.

- Candidate coverage is 85/85 true candidates.

- ABCoRT-TL reported baseline: reduction 28.46%, Top1 52.94%, Top5 76.47%, Top10 83.53%.

- Main selected ours balanced method: rank_guard_filter_soft_th50.0_g2_tau25.66_alpha2.0.

- High-reduction method: rank_guard_filter_soft_th40.0_g2_tau25.66_alpha2.0.

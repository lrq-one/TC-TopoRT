# Final experiment status summary
## Completed experiments
- SMRT main benchmark: completed, 5 seeds.
- External Table 2 transfer: completed, 6 datasets fixed by manifest.
- Dual-view ablation: completed, 5 seeds.
- Tautomer-changed subgroup analysis: completed, 5 seeds.
- Shuffle tautomer pairing ablation: completed, 50 shuffles per seed.
- Tail and hard-molecule analysis: completed.
- Pairing / no-leakage audit: completed.

## Key SMRT result
- Final stack MAE = 25.055090 ± 0.039094; RMSE = 55.671332 ± 0.100621; R2 = 0.898308 ± 0.000368.

## Key ablation result
- Origin only MAE = 25.278126 ± 0.054086.
- Tautomer only MAE = 25.217209 ± 0.070277.
- Mean fusion MAE = 25.059187 ± 0.037986.
- Huber stack MAE = 25.055090 ± 0.039094.

## Shuffle pairing result
- Paired MAE = 25.055090 ± 0.039094.
- Shuffled tautomer MAE = 25.274680 ± 0.044849.
- Shuffle minus paired delta = 0.219590 ± 0.053742.

## Pairing / leakage audit
- Audit status counts: {'PASS': 137}.
- FAIL checks: 0; WARN checks: 0.
- Origin/tautomer train and test row order and RT labels passed all checks.
- Exact SMILES and InChIKey train/test overlaps are zero.

## Optional remaining experiments
- Transfer-vs-scratch on six external datasets, if time allows.
- Candidate filtering application, if the manuscript wants to fully mirror ABCoRT's downstream application.

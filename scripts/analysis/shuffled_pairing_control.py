#!/usr/bin/env python3
"""Evaluate the final OOF Huber fusion after breaking paired-view identity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_run(value: str) -> tuple[int, Path, Path]:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use SEED,OOF_CSV,TEST_CSV")
    return int(parts[0]), Path(parts[1]), Path(parts[2])


def features(original: np.ndarray, standardized: np.ndarray, changed: np.ndarray) -> np.ndarray:
    difference = np.abs(original - standardized)
    mean = 0.5 * (original + standardized)
    return np.column_stack(
        [
            original, standardized, difference, mean,
            np.minimum(original, standardized), np.maximum(original, standardized),
            changed, difference * changed,
            original * changed / 1000.0, standardized * changed / 1000.0,
        ]
    )


def fit_predict(actual: np.ndarray, train_features: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(), HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
    )
    model.fit(train_features, actual)
    return model.predict(test_features)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="SEED,OOF_CSV,TEST_CSV")
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=20260614)
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/shuffled_pairing_control.csv"))
    args = parser.parse_args()
    rows = []
    for run_index, (seed, oof_path, test_path) in enumerate(args.run):
        oof = pd.read_csv(oof_path)
        test = pd.read_csv(test_path)
        required_oof = {"Actual_RT", "Origin_OOF_Pred", "Taut_OOF_Pred", "Taut_Changed"}
        required_test = {"Actual_RT", "Origin_Test_Pred", "Taut_Test_Pred", "Taut_Changed"}
        if required_oof - set(oof.columns) or required_test - set(test.columns):
            raise ValueError(f"Prediction schema mismatch for seed {seed}.")
        y_oof = oof["Actual_RT"].to_numpy(float)
        y_test = test["Actual_RT"].to_numpy(float)
        original_oof = oof["Origin_OOF_Pred"].to_numpy(float)
        standardized_oof = oof["Taut_OOF_Pred"].to_numpy(float)
        changed_oof = oof["Taut_Changed"].to_numpy(float)
        original_test = test["Origin_Test_Pred"].to_numpy(float)
        standardized_test = test["Taut_Test_Pred"].to_numpy(float)
        changed_test = test["Taut_Changed"].to_numpy(float)
        paired = fit_predict(
            y_oof,
            features(original_oof, standardized_oof, changed_oof),
            features(original_test, standardized_test, changed_test),
        )
        rows.append({"seed": seed, "condition": "paired", "permutation": -1, "MAE_seconds": float(np.abs(y_test - paired).mean())})
        for permutation in range(args.permutations):
            rng = np.random.default_rng(args.base_seed + run_index * 1000 + permutation)
            shuffled = fit_predict(
                y_oof,
                features(original_oof, standardized_oof[rng.permutation(len(oof))], changed_oof),
                features(original_test, standardized_test[rng.permutation(len(test))], changed_test),
            )
            rows.append({"seed": seed, "condition": "shuffled_standardized_partner", "permutation": permutation, "MAE_seconds": float(np.abs(y_test - shuffled).mean())})
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.groupby("condition")["MAE_seconds"].agg(["count", "mean", "std"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

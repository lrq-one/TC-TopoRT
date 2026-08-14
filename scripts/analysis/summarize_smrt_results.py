#!/usr/bin/env python3
"""Summarize the five final SMRT test-prediction files and their average."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_SEEDS = [1, 5, 79, 123, 256]
EXPECTED = {
    "MAE": (25.055090, 0.039094),
    "MRE": (3.161936, 0.004679),
    "MedAE": (11.316787, 0.097631),
    "RMSE": (55.671332, 0.100621),
    "R2": (0.898308, 0.000368),
}


def parse_prediction(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use SEED=PATH, for example 1=artifacts/.../test_predictions.csv") from error
    return seed, Path(path_text)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    absolute = np.abs(actual - predicted)
    relative = absolute / (np.abs(actual) + 1e-8) * 100.0
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "MAE": float(absolute.mean()),
        "MRE": float(relative.mean()),
        "MedAE": float(np.median(absolute)),
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "R2": float(1.0 - residual / total),
    }


def pick(frame: pd.DataFrame, names: tuple[str, ...], label: str) -> str:
    return next((name for name in names if name in frame.columns), None) or (_ for _ in ()).throw(
        ValueError(f"Cannot find {label} column in {list(frame.columns)}")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", type=parse_prediction, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis/smrt_summary"))
    parser.add_argument("--verify-paper-results", action="store_true")
    args = parser.parse_args()
    mapping = dict(args.prediction)
    if sorted(mapping) != PAPER_SEEDS:
        raise ValueError(f"Exactly the five paper seeds are required: {PAPER_SEEDS}; found {sorted(mapping)}")

    rows = []
    actual_reference = None
    predictions = []
    for seed in PAPER_SEEDS:
        frame = pd.read_csv(mapping[seed])
        actual_col = pick(frame, ("Actual_RT", "experimental_rt", "rt"), "actual RT")
        prediction_col = pick(frame, ("Final_Pred", "final_prediction", "predicted_rt"), "prediction")
        actual = pd.to_numeric(frame[actual_col], errors="raise").to_numpy(float)
        prediction = pd.to_numeric(frame[prediction_col], errors="raise").to_numpy(float)
        if len(actual) != 7798:
            raise ValueError(f"Seed {seed} has {len(actual)} rows; expected 7,798.")
        if actual_reference is None:
            actual_reference = actual
        elif not np.allclose(actual_reference, actual, atol=1e-8, rtol=0.0):
            raise ValueError(f"Seed {seed} target rows are not aligned with the other seeds.")
        predictions.append(prediction)
        rows.append({"seed": seed, **metrics(actual, prediction)})

    detail = pd.DataFrame(rows)
    summary = {
        metric: {
            "mean": float(detail[metric].mean()),
            "sample_sd": float(detail[metric].std(ddof=1)),
        }
        for metric in EXPECTED
    }
    assert actual_reference is not None
    ensemble = metrics(actual_reference, np.mean(np.vstack(predictions), axis=0))
    if args.verify_paper_results:
        errors = []
        for metric, (mean, sd) in EXPECTED.items():
            if not np.isclose(summary[metric]["mean"], mean, atol=5e-6):
                errors.append(f"{metric} mean={summary[metric]['mean']}")
            if not np.isclose(summary[metric]["sample_sd"], sd, atol=5e-6):
                errors.append(f"{metric} sd={summary[metric]['sample_sd']}")
        if errors:
            raise AssertionError("SMRT summary differs from locked values: " + "; ".join(errors))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "five_seed_metrics.csv", index=False)
    result = {"five_seed_summary": summary, "five_model_average_prediction": ensemble}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


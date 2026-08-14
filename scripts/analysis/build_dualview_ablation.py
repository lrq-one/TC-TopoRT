#!/usr/bin/env python3
"""Summarize a supplied per-seed dual-view control metric table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_MAE = {
    "Original only": (25.278, 0.054),
    "Tautomer-standardized only": (25.217, 0.070),
    "O+O": (25.110, 0.041),
    "T+T": (25.056, 0.037),
    "same-seed O+T arithmetic mean": (25.059, 0.038),
    "Final O+T OOF Huber": (25.055, 0.039),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with control, seed, and mae_seconds.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/dualview_controls.csv"))
    parser.add_argument("--verify-paper-results", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    required = {"control", "seed", "mae_seconds"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    frame["mae_seconds"] = pd.to_numeric(frame["mae_seconds"], errors="raise")
    summary = frame.groupby("control", as_index=False).agg(
        runs=("seed", "nunique"),
        mae_mean_seconds=("mae_seconds", "mean"),
        mae_sample_sd_seconds=("mae_seconds", "std"),
    )
    if args.verify_paper_results:
        indexed = summary.set_index("control")
        errors = []
        for control, (mean, sd) in EXPECTED_MAE.items():
            if control not in indexed.index:
                errors.append(f"missing {control}")
                continue
            row = indexed.loc[control]
            if not np.isclose(row["mae_mean_seconds"], mean, atol=0.0005):
                errors.append(f"{control} mean={row['mae_mean_seconds']}")
            if not np.isclose(row["mae_sample_sd_seconds"], sd, atol=0.0005):
                errors.append(f"{control} sd={row['mae_sample_sd_seconds']}")
        if errors:
            raise AssertionError("Dual-view controls differ from locked values: " + "; ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

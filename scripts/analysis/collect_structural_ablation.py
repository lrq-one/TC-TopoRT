#!/usr/bin/env python3
"""Summarize supplied five-seed Full, No2Cell, and matched-GINE metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {
    "Full TC-TopoRT": (25.055, 0.039, 26943049),
    "w/o explicit ring 2-cells": (25.121, 0.091, None),
    "parameter-matched atom-bond GINE": (25.701, 0.069, 26928385),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with variant, seed, mae_seconds, parameters.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/structural_ablation.csv"))
    parser.add_argument("--verify-paper-results", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    required = {"variant", "seed", "mae_seconds", "parameters"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    summary = frame.groupby("variant", as_index=False).agg(
        runs=("seed", "nunique"),
        mae_mean_seconds=("mae_seconds", "mean"),
        mae_sample_sd_seconds=("mae_seconds", "std"),
        parameters=("parameters", "max"),
    )
    if args.verify_paper_results:
        indexed = summary.set_index("variant")
        errors = []
        for variant, (mean, sd, parameters) in EXPECTED.items():
            if variant not in indexed.index:
                errors.append(f"missing {variant}")
                continue
            row = indexed.loc[variant]
            if not np.isclose(row["mae_mean_seconds"], mean, atol=0.0005):
                errors.append(f"{variant} mean={row['mae_mean_seconds']}")
            if not np.isclose(row["mae_sample_sd_seconds"], sd, atol=0.0005):
                errors.append(f"{variant} sd={row['mae_sample_sd_seconds']}")
            if parameters is not None and int(row["parameters"]) != parameters:
                errors.append(f"{variant} parameters={int(row['parameters'])}")
        if errors:
            raise AssertionError("Structural controls differ from locked values: " + "; ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

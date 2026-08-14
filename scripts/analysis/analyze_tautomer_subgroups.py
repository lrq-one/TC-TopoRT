#!/usr/bin/env python3
"""Summarize changed/unchanged tautomer-representation prediction subgroups."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_input(value: str) -> tuple[int, Path]:
    seed, path = value.split("=", 1)
    return int(seed), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", type=parse_input, required=True, help="SEED=CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis/tautomer_subgroups"))
    args = parser.parse_args()
    rows = []
    for seed, path in args.prediction:
        frame = pd.read_csv(path)
        required = {"Actual_RT", "Origin_Test_Pred", "Taut_Test_Pred", "Final_Pred", "Taut_Changed"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        changed = pd.to_numeric(frame["Taut_Changed"], errors="raise").astype(bool)
        for group_name, mask in [("changed", changed), ("unchanged", ~changed), ("all", np.ones(len(frame), bool))]:
            actual = pd.to_numeric(frame.loc[mask, "Actual_RT"], errors="raise").to_numpy(float)
            for method, column in [("Original", "Origin_Test_Pred"), ("Tautomer-standardized", "Taut_Test_Pred"), ("Fusion", "Final_Pred")]:
                predicted = pd.to_numeric(frame.loc[mask, column], errors="raise").to_numpy(float)
                rows.append({"seed": seed, "group": group_name, "method": method, "N": len(actual), "MAE_seconds": float(np.abs(actual - predicted).mean())})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["group", "method"], as_index=False).agg(
        runs=("seed", "nunique"), N=("N", "max"), mae_mean_seconds=("MAE_seconds", "mean"), mae_sample_sd_seconds=("MAE_seconds", "std")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "detail.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


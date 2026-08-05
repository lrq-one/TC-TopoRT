#!/usr/bin/env python3
"""Evaluate full-range TC-TopoRT predictions in the early-retention region."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

RT_CANDIDATES = ("Actual_RT", "actual_rt", "rt", "RT")
PRED_CANDIDATES = ("Final_Pred", "final_pred", "prediction", "pred_rt", "Predicted_RT")


def pick_column(df: pd.DataFrame, candidates: tuple[str, ...], path: Path) -> str:
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(f"{path}: none of {candidates} found; columns={list(df.columns)}")


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    err = p - y
    abs_err = np.abs(err)
    safe = np.maximum(np.abs(y), 1e-12)
    denom = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(len(y)),
        "mae_s": float(np.mean(abs_err)),
        "medae_s": float(np.median(abs_err)),
        "rmse_s": float(np.sqrt(np.mean(err**2))),
        "mre_pct": float(100.0 * np.mean(abs_err / safe)),
        "medre_pct": float(100.0 * np.median(abs_err / safe)),
        "r2": float(1.0 - np.sum(err**2) / denom) if denom > 0 else math.nan,
        "bias_s": float(np.mean(err)),
        "p90_abs_error_s": float(np.quantile(abs_err, 0.90)),
        "p95_abs_error_s": float(np.quantile(abs_err, 0.95)),
        "p99_abs_error_s": float(np.quantile(abs_err, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="artifacts/results/smrt_full_range/seed5/test_predictions.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/results/paper_tables/early_retention",
    )
    parser.add_argument("--threshold", type=float, default=300.0)
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.predictions)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    rt_col = pick_column(df, RT_CANDIDATES, path)
    pred_col = pick_column(df, PRED_CANDIDATES, path)
    work = pd.DataFrame(
        {
            "actual_rt": pd.to_numeric(df[rt_col], errors="raise").astype(float),
            "pred_rt": pd.to_numeric(df[pred_col], errors="raise").astype(float),
        }
    )
    if not np.isfinite(work.to_numpy(float)).all():
        raise RuntimeError("Non-finite labels or predictions found")

    masks = [
        ("All full-range test molecules", pd.Series(True, index=work.index)),
        (f"Early RT <= {args.threshold:g} s", work["actual_rt"] <= args.threshold),
        (f"Retained RT > {args.threshold:g} s", work["actual_rt"] > args.threshold),
    ]
    summary_rows = []
    for group, mask in masks:
        sub = work.loc[mask]
        if sub.empty:
            continue
        summary_rows.append({"group": group, **metrics(sub["actual_rt"], sub["pred_rt"])})
    summary = pd.DataFrame(summary_rows)

    bins = pd.qcut(work["actual_rt"], q=args.n_bins, labels=False, duplicates="drop")
    bin_rows = []
    for bin_id in sorted(pd.Series(bins).dropna().unique()):
        sub = work.loc[bins == bin_id]
        bin_rows.append(
            {
                "rt_quantile_bin": int(bin_id) + 1,
                "rt_min_s": float(sub["actual_rt"].min()),
                "rt_max_s": float(sub["actual_rt"].max()),
                **metrics(sub["actual_rt"], sub["pred_rt"]),
            }
        )
    by_bin = pd.DataFrame(bin_rows)

    detail = work.copy()
    detail["absolute_error_s"] = np.abs(detail["pred_rt"] - detail["actual_rt"])
    detail["signed_error_s"] = detail["pred_rt"] - detail["actual_rt"]
    detail["early_region"] = detail["actual_rt"] <= args.threshold

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "early_retention_summary.csv", index=False)
    by_bin.to_csv(out_dir / "full_range_rt_bin_summary.csv", index=False)
    detail.to_csv(out_dir / "full_range_test_errors.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()

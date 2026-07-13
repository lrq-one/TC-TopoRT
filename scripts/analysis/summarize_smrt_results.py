#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = [1, 5, 79, 123, 256]


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    err = np.abs(y - p)
    rel = err / (np.abs(y) + 1e-8) * 100.0
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(len(y)),
        "mae": float(err.mean()),
        "mre_pct": float(rel.mean()),
        "medae": float(np.median(err)),
        "medre_pct": float(np.median(rel)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "p99": float(np.percentile(err, 99)),
        "gt100": int((err > 100).sum()),
        "gt200": int((err > 200).sum()),
        "bias": float(np.mean(p - y)),
    }


def seed_dir(root: Path, seed: int) -> Path:
    return root / f"seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the five TC-TopoRT SMRT runs and build the five-seed ensemble."
    )
    parser.add_argument("--results_root", default="artifacts/results/smrt")
    parser.add_argument("--out_dir", default="artifacts/results/paper_tables/smrt")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--require_huber",
        type=int,
        choices=[0, 1],
        default=1,
        help="Require every reported run to record selected_stacker=huber_stack.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    predictions: list[np.ndarray] = []
    reference_y: np.ndarray | None = None
    reference_index: np.ndarray | None = None

    for seed in args.seeds:
        run_dir = seed_dir(results_root, seed)
        metrics_path = run_dir / "final_metrics.json"
        prediction_path = run_dir / "test_predictions.csv"

        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        selected = str(payload.get("selected_stacker", ""))
        if args.require_huber and selected != "huber_stack":
            raise RuntimeError(
                f"seed {seed}: expected selected_stacker=huber_stack, found {selected!r}"
            )

        frame = pd.read_csv(prediction_path)
        required = {"Actual_RT", "Final_Pred"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{prediction_path}: missing columns {sorted(missing)}")

        y = frame["Actual_RT"].to_numpy(dtype=float)
        p = frame["Final_Pred"].to_numpy(dtype=float)
        source_index = (
            frame["Source_Index"].to_numpy()
            if "Source_Index" in frame.columns
            else np.arange(len(frame))
        )

        if reference_y is None:
            reference_y = y
            reference_index = source_index
        else:
            if len(y) != len(reference_y) or not np.allclose(y, reference_y, atol=1e-6, rtol=0.0):
                raise RuntimeError(f"seed {seed}: test labels do not match the first run")
            if not np.array_equal(source_index, reference_index):
                raise RuntimeError(f"seed {seed}: test row order does not match the first run")

        rows.append({"seed": seed, "selected_stacker": selected, **metrics(y, p)})
        predictions.append(p)

    if reference_y is None or reference_index is None:
        raise RuntimeError("No SMRT runs were found.")

    per_seed = pd.DataFrame(rows)
    numeric_cols = [c for c in per_seed.columns if c not in {"seed", "selected_stacker"}]
    summary_rows = []
    for statistic, func in [("mean", np.mean), ("sd", lambda x: np.std(x, ddof=1))]:
        row: dict[str, float | str] = {"statistic": statistic}
        for column in numeric_cols:
            row[column] = float(func(per_seed[column].to_numpy(dtype=float)))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    ensemble_pred = np.mean(np.vstack(predictions), axis=0)
    ensemble_metrics = pd.DataFrame([
        {"method": "TC-TopoRT-E five-seed ensemble", **metrics(reference_y, ensemble_pred)}
    ])
    ensemble_table = pd.DataFrame({
        "Source_Index": reference_index,
        "Actual_RT": reference_y,
        "Ensemble_Pred": ensemble_pred,
        "Absolute_Error": np.abs(reference_y - ensemble_pred),
    })

    per_seed.to_csv(out_dir / "smrt_per_seed_metrics.csv", index=False)
    summary.to_csv(out_dir / "smrt_single_seed_mean_sd.csv", index=False)
    ensemble_metrics.to_csv(out_dir / "smrt_five_seed_ensemble_metrics.csv", index=False)
    ensemble_table.to_csv(out_dir / "smrt_five_seed_ensemble_predictions.csv", index=False)

    print("\nPer-seed metrics")
    print(per_seed.to_string(index=False))
    print("\nMean and SD")
    print(summary.to_string(index=False))
    print("\nFive-seed ensemble")
    print(ensemble_metrics.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

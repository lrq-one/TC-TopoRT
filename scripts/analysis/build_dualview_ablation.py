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
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(len(y)),
        "mae": float(err.mean()),
        "medae": float(np.median(err)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the original/tautomer/mean/OOF-Huber ablation table from SMRT run outputs."
    )
    parser.add_argument("--results_root", default="artifacts/results/smrt")
    parser.add_argument("--out_dir", default="artifacts/results/paper_tables/ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--require_huber", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []

    for seed in args.seeds:
        run_dir = results_root / f"seed{seed}"
        base_path = run_dir / "test_base_predictions.csv"
        final_path = run_dir / "test_predictions.csv"
        metrics_path = run_dir / "final_metrics.json"

        for path in [base_path, final_path, metrics_path]:
            if not path.is_file():
                raise FileNotFoundError(path)

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        selected = str(payload.get("selected_stacker", ""))
        if args.require_huber and selected != "huber_stack":
            raise RuntimeError(
                f"seed {seed}: expected selected_stacker=huber_stack, found {selected!r}"
            )

        base = pd.read_csv(base_path)
        final = pd.read_csv(final_path)
        required_base = {"Actual_RT", "Origin_Test_Pred", "Taut_Test_Pred"}
        required_final = {"Actual_RT", "Final_Pred"}
        if missing := required_base.difference(base.columns):
            raise ValueError(f"{base_path}: missing columns {sorted(missing)}")
        if missing := required_final.difference(final.columns):
            raise ValueError(f"{final_path}: missing columns {sorted(missing)}")

        y = base["Actual_RT"].to_numpy(dtype=float)
        y_final = final["Actual_RT"].to_numpy(dtype=float)
        if len(y) != len(y_final) or not np.allclose(y, y_final, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"seed {seed}: base/final test labels do not match")

        origin = base["Origin_Test_Pred"].to_numpy(dtype=float)
        tautomer = base["Taut_Test_Pred"].to_numpy(dtype=float)
        mean_pred = 0.5 * (origin + tautomer)
        huber = final["Final_Pred"].to_numpy(dtype=float)

        for method, prediction in [
            ("Original view only", origin),
            ("Tautomer view only", tautomer),
            ("Same-seed paired mean fusion", mean_pred),
            ("OOF Huber stack", huber),
        ]:
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "selected_stacker": selected,
                    **metrics(y, prediction),
                }
            )

    by_seed = pd.DataFrame(rows)
    summary_rows = []
    for method, sub in by_seed.groupby("method", sort=False):
        row: dict[str, float | int | str] = {"method": method, "runs": int(len(sub))}
        for column in ["mae", "medae", "rmse", "r2"]:
            values = sub[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    reference = float(
        summary.loc[summary["method"].eq("OOF Huber stack"), "mae_mean"].iloc[0]
    )
    summary["delta_mae_vs_huber"] = summary["mae_mean"] - reference

    by_seed.to_csv(out_dir / "dualview_fusion_ablation_by_seed.csv", index=False)
    summary.to_csv(out_dir / "dualview_fusion_ablation_summary.csv", index=False)

    print("\nBy seed")
    print(by_seed.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

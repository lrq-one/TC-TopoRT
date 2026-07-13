#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEEDS = [1, 5, 79, 123, 256]


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(y - pred)))


def require_columns(df: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the SMRT tautomer-changed/unchanged subgroup table from the "
            "five formal single-seed prediction outputs."
        )
    )
    parser.add_argument("--results_root", default="artifacts/results/smrt")
    parser.add_argument(
        "--out_dir",
        default="artifacts/results/paper_tables/subgroups/tautomer_change",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
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
        require_columns(
            base,
            {
                "Actual_RT",
                "Origin_Test_Pred",
                "Taut_Test_Pred",
                "Taut_Changed",
            },
            base_path,
        )
        require_columns(final, {"Actual_RT", "Final_Pred"}, final_path)

        y = base["Actual_RT"].to_numpy(dtype=float)
        y_final = final["Actual_RT"].to_numpy(dtype=float)
        if len(y) != len(y_final) or not np.allclose(
            y, y_final, atol=1e-6, rtol=0.0
        ):
            raise RuntimeError(f"seed {seed}: base/final test labels do not match")

        changed = base["Taut_Changed"].to_numpy(dtype=float) >= 0.5
        methods = {
            "Original view": base["Origin_Test_Pred"].to_numpy(dtype=float),
            "Tautomer view": base["Taut_Test_Pred"].to_numpy(dtype=float),
            "Fusion": final["Final_Pred"].to_numpy(dtype=float),
        }
        groups = {
            "Tautomer-changed molecules": changed,
            "Unchanged molecules": ~changed,
            "All molecules": np.ones(len(base), dtype=bool),
        }

        for group, mask in groups.items():
            if not np.any(mask):
                raise RuntimeError(f"seed {seed}: empty subgroup {group!r}")
            for method, pred in methods.items():
                rows.append(
                    {
                        "seed": seed,
                        "group": group,
                        "method": method,
                        "n": int(np.sum(mask)),
                        "mae": mae(y[mask], pred[mask]),
                        "selected_stacker": selected,
                    }
                )

    by_seed = pd.DataFrame(rows)
    summary_rows: list[dict[str, float | int | str]] = []
    group_order = [
        "Tautomer-changed molecules",
        "Unchanged molecules",
        "All molecules",
    ]
    method_order = ["Original view", "Tautomer view", "Fusion"]

    for group in group_order:
        for method in method_order:
            sub = by_seed[
                by_seed["group"].eq(group) & by_seed["method"].eq(method)
            ]
            if len(sub) != len(args.seeds):
                raise RuntimeError(
                    f"{group}/{method}: expected {len(args.seeds)} runs, found {len(sub)}"
                )
            n_values = sub["n"].to_numpy(dtype=int)
            if not np.all(n_values == n_values[0]):
                raise RuntimeError(f"{group}: subgroup size differs across seeds")
            values = sub["mae"].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "group": group,
                    "method": method,
                    "n": int(n_values[0]),
                    "runs": int(len(values)),
                    "mae_mean": float(np.mean(values)),
                    "mae_sd": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    ),
                }
            )

    summary_long = pd.DataFrame(summary_rows)
    summary_wide = (
        summary_long.pivot(
            index=["group", "n"],
            columns="method",
            values=["mae_mean", "mae_sd"],
        )
        .reset_index()
    )
    summary_wide.columns = [
        "_".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary_wide.columns
    ]
    summary_wide = summary_wide.rename(
        columns={
            "mae_mean_Original view": "original_mae_mean",
            "mae_sd_Original view": "original_mae_sd",
            "mae_mean_Tautomer view": "tautomer_mae_mean",
            "mae_sd_Tautomer view": "tautomer_mae_sd",
            "mae_mean_Fusion": "fusion_mae_mean",
            "mae_sd_Fusion": "fusion_mae_sd",
        }
    )
    summary_wide["_order"] = summary_wide["group"].map(
        {name: i for i, name in enumerate(group_order)}
    )
    summary_wide = summary_wide.sort_values("_order").drop(columns="_order")

    by_seed.to_csv(out_dir / "tautomer_subgroup_by_seed.csv", index=False)
    summary_long.to_csv(out_dir / "tautomer_subgroup_summary_long.csv", index=False)
    summary_wide.to_csv(out_dir / "tautomer_subgroup_table.csv", index=False)

    print("\nSummary table")
    print(summary_wide.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

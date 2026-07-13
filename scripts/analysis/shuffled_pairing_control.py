#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_SEEDS = [1, 5, 79, 123, 256]


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(y - pred)))


def build_stack_features(
    origin_pred: np.ndarray,
    taut_pred: np.ndarray,
    changed: np.ndarray,
) -> np.ndarray:
    origin_pred = np.asarray(origin_pred, dtype=float)
    taut_pred = np.asarray(taut_pred, dtype=float)
    changed = np.asarray(changed, dtype=float)

    diff = np.abs(origin_pred - taut_pred)
    mean_pred = 0.5 * (origin_pred + taut_pred)
    min_pred = np.minimum(origin_pred, taut_pred)
    max_pred = np.maximum(origin_pred, taut_pred)

    return np.vstack(
        [
            origin_pred,
            taut_pred,
            diff,
            mean_pred,
            min_pred,
            max_pred,
            changed,
            diff * changed,
            origin_pred * changed / 1000.0,
            taut_pred * changed / 1000.0,
        ]
    ).T


def fit_huber(
    y_oof: np.ndarray,
    x_oof: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=alpha, max_iter=1000),
    )
    model.fit(x_oof, y_oof)
    return model.predict(x_test)


def require_columns(df: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the shuffled tautomer-pairing control. Tautomer-view "
            "predictions are permuted across molecules in both OOF and test tables, "
            "then the Huber stacker is refitted only on shuffled OOF predictions."
        )
    )
    parser.add_argument("--results_root", default="artifacts/results/smrt")
    parser.add_argument(
        "--out_dir",
        default="artifacts/results/paper_tables/ablation/shuffled_pairing",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--n_permutations", type=int, default=50)
    parser.add_argument("--base_seed", type=int, default=20260614)
    parser.add_argument("--huber_alpha", type=float, default=1e-4)
    parser.add_argument(
        "--require_huber",
        type=int,
        choices=[0, 1],
        default=1,
        help="Require the formal run to record selected_stacker=huber_stack.",
    )
    args = parser.parse_args()

    if args.n_permutations < 1:
        raise ValueError("--n_permutations must be at least 1")

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    permutation_rows: list[dict[str, float | int | str]] = []
    seed_rows: list[dict[str, float | int]] = []

    for seed_index, seed in enumerate(args.seeds):
        run_dir = results_root / f"seed{seed}"
        oof_path = run_dir / "oof_base_predictions.csv"
        test_base_path = run_dir / "test_base_predictions.csv"
        test_final_path = run_dir / "test_predictions.csv"
        metrics_path = run_dir / "final_metrics.json"

        for path in [oof_path, test_base_path, test_final_path, metrics_path]:
            if not path.is_file():
                raise FileNotFoundError(path)

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        selected = str(payload.get("selected_stacker", ""))
        if args.require_huber and selected != "huber_stack":
            raise RuntimeError(
                f"seed {seed}: expected selected_stacker=huber_stack, found {selected!r}"
            )

        oof_df = pd.read_csv(oof_path)
        test_df = pd.read_csv(test_base_path)
        final_df = pd.read_csv(test_final_path)

        require_columns(
            oof_df,
            {"Actual_RT", "Origin_OOF_Pred", "Taut_OOF_Pred", "Taut_Changed"},
            oof_path,
        )
        require_columns(
            test_df,
            {"Actual_RT", "Origin_Test_Pred", "Taut_Test_Pred", "Taut_Changed"},
            test_base_path,
        )
        require_columns(final_df, {"Actual_RT", "Final_Pred"}, test_final_path)

        y_oof = oof_df["Actual_RT"].to_numpy(dtype=float)
        y_test = test_df["Actual_RT"].to_numpy(dtype=float)
        y_final = final_df["Actual_RT"].to_numpy(dtype=float)
        if len(y_test) != len(y_final) or not np.allclose(
            y_test, y_final, atol=1e-6, rtol=0.0
        ):
            raise RuntimeError(f"seed {seed}: base/final test labels do not match")

        origin_oof = oof_df["Origin_OOF_Pred"].to_numpy(dtype=float)
        taut_oof = oof_df["Taut_OOF_Pred"].to_numpy(dtype=float)
        changed_oof = oof_df["Taut_Changed"].to_numpy(dtype=float)

        origin_test = test_df["Origin_Test_Pred"].to_numpy(dtype=float)
        taut_test = test_df["Taut_Test_Pred"].to_numpy(dtype=float)
        changed_test = test_df["Taut_Changed"].to_numpy(dtype=float)

        paired_pred = final_df["Final_Pred"].to_numpy(dtype=float)
        paired_mae = mae(y_test, paired_pred)

        shuffled_maes: list[float] = []
        for permutation_id in range(args.n_permutations):
            permutation_seed = (
                args.base_seed + seed_index * 1000 + permutation_id
            )
            rng = np.random.default_rng(permutation_seed)
            shuffled_oof = taut_oof[rng.permutation(len(taut_oof))]
            shuffled_test = taut_test[rng.permutation(len(taut_test))]

            x_oof = build_stack_features(origin_oof, shuffled_oof, changed_oof)
            x_test = build_stack_features(origin_test, shuffled_test, changed_test)
            shuffled_pred = fit_huber(
                y_oof=y_oof,
                x_oof=x_oof,
                x_test=x_test,
                alpha=args.huber_alpha,
            )
            shuffled_mae = mae(y_test, shuffled_pred)
            shuffled_maes.append(shuffled_mae)
            permutation_rows.append(
                {
                    "seed": seed,
                    "permutation_id": permutation_id,
                    "permutation_seed": permutation_seed,
                    "paired_mae": paired_mae,
                    "shuffled_mae": shuffled_mae,
                    "delta_shuffle_minus_paired": shuffled_mae - paired_mae,
                }
            )

        shuffled_values = np.asarray(shuffled_maes, dtype=float)
        seed_rows.append(
            {
                "seed": seed,
                "paired_mae": paired_mae,
                "shuffled_mae_mean": float(np.mean(shuffled_values)),
                "shuffled_mae_sd_within_seed": (
                    float(np.std(shuffled_values, ddof=1))
                    if len(shuffled_values) > 1
                    else 0.0
                ),
                "delta_shuffle_minus_paired": (
                    float(np.mean(shuffled_values)) - paired_mae
                ),
                "n_permutations": args.n_permutations,
            }
        )

    permutation_table = pd.DataFrame(permutation_rows)
    by_seed = pd.DataFrame(seed_rows)

    paired_values = by_seed["paired_mae"].to_numpy(dtype=float)
    shuffled_values = by_seed["shuffled_mae_mean"].to_numpy(dtype=float)
    delta_values = by_seed["delta_shuffle_minus_paired"].to_numpy(dtype=float)

    summary = pd.DataFrame(
        [
            {
                "condition": "Correctly paired dual-view predictions",
                "mae_mean": float(np.mean(paired_values)),
                "mae_sd_across_seeds": (
                    float(np.std(paired_values, ddof=1))
                    if len(paired_values) > 1
                    else 0.0
                ),
                "runs": len(by_seed),
            },
            {
                "condition": "Shuffled tautomer predictions",
                "mae_mean": float(np.mean(shuffled_values)),
                "mae_sd_across_seeds": (
                    float(np.std(shuffled_values, ddof=1))
                    if len(shuffled_values) > 1
                    else 0.0
                ),
                "runs": len(by_seed),
            },
            {
                "condition": "Shuffle minus paired",
                "mae_mean": float(np.mean(delta_values)),
                "mae_sd_across_seeds": (
                    float(np.std(delta_values, ddof=1))
                    if len(delta_values) > 1
                    else 0.0
                ),
                "runs": len(by_seed),
            },
        ]
    )

    permutation_table.to_csv(
        out_dir / "shuffled_pairing_permutation_results.csv", index=False
    )
    by_seed.to_csv(out_dir / "shuffled_pairing_by_seed.csv", index=False)
    summary.to_csv(out_dir / "shuffled_pairing_summary.csv", index=False)

    print("\nBy seed")
    print(by_seed.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

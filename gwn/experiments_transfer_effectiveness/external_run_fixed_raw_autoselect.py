#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run fixed external transfer protocol for TCDV-TopoRT.

One-command entry:
  1. Run external transfer fine-tuning prediction generation with 119.
  2. Run fixed no-leak raw AutoSelect aggregation with 122c.
  3. Collect a Table-2-style comparison against ABCoRT-TL.

Default is cv_seed=1 for the paper-safe fixed protocol pilot.
For 5 CV seeds:
  --cv_seeds 1 12 123 1234 12345
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ABCoRT_TL = {
    "Eawag_XBridgeC18_364": 45.30,
    "FEM_lipids_72": 85.46,
    "FEM_long_412": 87.16,
    "IPB_Halle_82": 13.81,
    "LIFE_new_184": 15.62,
    "LIFE_old_194": 9.97,
}


DATASETS = [
    "Eawag_XBridgeC18_364",
    "FEM_lipids_72",
    "FEM_long_412",
    "IPB_Halle_82",
    "LIFE_new_184",
    "LIFE_old_194",
]


def run_cmd(cmd, dry_run=False):
    print("\n" + "=" * 100)
    print("RUN:")
    print(" ".join(map(str, cmd)))
    print("=" * 100)

    if dry_run:
        return

    subprocess.run(list(map(str, cmd)), check=True)


def read_summary(stack_dir: Path, cv_seed: int):
    p = stack_dir / "tcdv_fixed_noleak_autocal_summary.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    df = df[df["method"] == "tcdv_fixed_noleak_autocal"].copy()
    df["cv_seed"] = int(cv_seed)
    return df


def collect_results(out_root: Path, cv_seeds):
    rows = []

    for cv_seed in cv_seeds:
        stack_dir = out_root / f"results_122c_fixed_raw_autoselect_seed5_src0to4_cvseed_{cv_seed}"
        rows.append(read_summary(stack_dir, cv_seed))

    all_df = pd.concat(rows, ignore_index=True)

    metric_cols = ["mae", "medae", "rmse", "r2", "pearson", "spearman", "bias"]
    summary_rows = []

    for ds, sub in all_df.groupby("dataset_name"):
        row = {
            "dataset_name": ds,
            "method": "TCDV-TopoRT fixed raw AutoSelect",
            "num_cv_seeds": int(sub["cv_seed"].nunique()),
            "abcort_tl_mae": ABCoRT_TL.get(ds),
        }

        for c in metric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0

        if row["abcort_tl_mae"] is not None:
            row["delta_vs_abcort_mae"] = row["mae_mean"] - row["abcort_tl_mae"]
            row["better_than_abcort"] = bool(row["mae_mean"] < row["abcort_tl_mae"])
        else:
            row["delta_vs_abcort_mae"] = None
            row["better_than_abcort"] = None

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("dataset_name").reset_index(drop=True)

    out_root.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_root / "external_table2_fixed_raw_autoselect_per_cvseed.csv", index=False)
    summary.to_csv(out_root / "external_table2_fixed_raw_autoselect_summary.csv", index=False)

    print("\n=== Final fixed raw AutoSelect summary ===")
    show_cols = [
        "dataset_name",
        "mae_mean",
        "mae_std",
        "abcort_tl_mae",
        "delta_vs_abcort_mae",
        "better_than_abcort",
    ]
    print(summary[show_cols].to_string(index=False))

    print("\n[SAVE]", out_root / "external_table2_fixed_raw_autoselect_per_cvseed.csv")
    print("[SAVE]", out_root / "external_table2_fixed_raw_autoselect_summary.csv")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_root", default="experiments_transfer_effectiveness/fixed_raw_autoselect_table2")
    ap.add_argument("--cv_seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--run_keys", nargs="+", default=["seed5"])
    ap.add_argument("--source_folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)

    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--skip_existing_base", type=int, default=1)
    ap.add_argument("--skip_existing_stack", type=int, default=1)
    ap.add_argument("--dry_run", type=int, default=0)

    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for cv_seed in args.cv_seeds:
        base_dir = out_root / f"results_119_base_tl_seed5_src0to4_cvseed_{cv_seed}"
        stack_dir = out_root / f"results_122c_fixed_raw_autoselect_seed5_src0to4_cvseed_{cv_seed}"
        base_csv = base_dir / "external_tl_predictions.csv"
        stack_summary = stack_dir / "tcdv_fixed_noleak_autocal_summary.csv"

        if args.skip_existing_base and base_csv.exists():
            print(f"[SKIP 119] existing base predictions: {base_csv}")
        else:
            cmd119 = [
                sys.executable,
                "-u",
                "experiments_transfer_effectiveness/external_train_tcdv_transfer_or_scratch.py",
                "--out_dir", base_dir,
                "--datasets", *args.datasets,
                "--run_keys", *args.run_keys,
                "--source_folds", *map(str, args.source_folds),
                "--init_mode", "tl",
                "--freeze_mode", "rt_head_full",
                "--reset_out_lin", "1",
                "--cv_folds", "10",
                "--cv_seed", str(cv_seed),
                "--epochs", str(args.epochs),
                "--early_stop_train", "999",
                "--batch_size", str(args.batch_size),
                "--eval_batch_size", str(args.eval_batch_size),
                "--lr", str(args.lr),
            ]
            run_cmd(cmd119, dry_run=bool(args.dry_run))

        if args.skip_existing_stack and stack_summary.exists():
            print(f"[SKIP 122c] existing stack summary: {stack_summary}")
        else:
            cmd122c = [
                sys.executable,
                "-u",
                "experiments_transfer_effectiveness/external_stack_fixed_raw_autoselect.py",
                "--pred_csv", base_csv,
                "--out_dir", stack_dir,
                "--cv_seed", str(cv_seed),
                "--cv_folds", "10",
                "--source_folds", *map(str, args.source_folds),
                "--calib_modes", "raw",
                "--selection_metric", "mae",
            ]
            run_cmd(cmd122c, dry_run=bool(args.dry_run))

    if not args.dry_run:
        collect_results(out_root, args.cv_seeds)


if __name__ == "__main__":
    main()

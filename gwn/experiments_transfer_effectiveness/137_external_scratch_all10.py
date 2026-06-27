#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run TCDV-TopoRT from-scratch training on the all10 external datasets.

This is the paper-facing scratch-training wrapper.

It calls:
  experiments_transfer_effectiveness/119_external_tcdv_scratch_vs_tl.py

Protocol:
  - init_mode = scratch
  - freeze_mode = all
  - reset_out_lin = 0
  - source_fold = 0 only as a placeholder, to avoid repeated equivalent scratch runs
  - cv_folds = 10
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ALL10_DATASETS = [
    "FEM_short_73",
    "UniToyama_Atlantis_143",
    "FEM_long_412",
    "Eawag_XBridgeC18_364",
    "LIFE_old_194",
    "MTBLS87_147",
    "LIFE_new_184",
    "Cao_HILIC_116",
    "IPB_Halle_82",
    "FEM_lipids_72",
]


def run_cmd(cmd, dry_run: bool = False):
    print("\n" + "=" * 100)
    print("RUN:")
    print(" ".join(map(str, cmd)))
    print("=" * 100)

    if dry_run:
        return

    subprocess.run(list(map(str, cmd)), check=True)


def copy_as_scratch_names(out_dir: Path):
    mapping = {
        "external_tl_predictions.csv": "external_scratch_predictions.csv",
        "external_tl_fold_metrics.csv": "external_scratch_fold_metrics.csv",
        "external_tl_metrics_by_run.csv": "external_scratch_metrics_by_run.csv",
        "external_tl_summary.csv": "external_scratch_summary.csv",
    }

    for old, new in mapping.items():
        src = out_dir / old
        dst = out_dir / new
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[COPY] {src} -> {dst}")
        else:
            print(f"[MISS] {src}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out_dir",
        default="experiments_transfer_effectiveness/results_external_scratch_all10_seed5_cvseed1",
    )
    ap.add_argument("--datasets", nargs="+", default=ALL10_DATASETS)

    ap.add_argument("--run_key", default="seed5")
    ap.add_argument("--source_fold", type=int, default=0)

    ap.add_argument("--cv_seed", type=int, default=1)
    ap.add_argument("--cv_folds", type=int, default=10)

    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--early_stop_train", type=int, default=999)
    ap.add_argument("--dry_run", type=int, default=0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "experiments_transfer_effectiveness/119_external_tcdv_scratch_vs_tl.py",
        "--out_dir", out_dir,
        "--datasets", *args.datasets,
        "--run_keys", args.run_key,
        "--source_folds", str(args.source_fold),
        "--init_mode", "scratch",
        "--freeze_mode", "all",
        "--reset_out_lin", "0",
        "--cv_folds", str(args.cv_folds),
        "--cv_seed", str(args.cv_seed),
        "--epochs", str(args.epochs),
        "--early_stop_train", str(args.early_stop_train),
        "--batch_size", str(args.batch_size),
        "--eval_batch_size", str(args.eval_batch_size),
        "--lr", str(args.lr),
    ]

    run_cmd(cmd, dry_run=bool(args.dry_run))

    if not args.dry_run:
        copy_as_scratch_names(out_dir)

    print("\n✅ scratch all10 wrapper done:", out_dir)


if __name__ == "__main__":
    main()

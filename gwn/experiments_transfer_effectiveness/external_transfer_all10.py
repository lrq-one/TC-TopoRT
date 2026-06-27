#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run TCDV-TopoRT external transfer learning on the all10 external datasets.

This is the paper-facing transfer-learning wrapper.

It calls:
  experiments_transfer_effectiveness/external_run_fixed_raw_autoselect.py

Protocol:
  - init_mode = tl
  - freeze_mode = rt_head_full
  - reset_out_lin = 1
  - source_folds = 0 1 2 3 4
  - cv_folds = 10
  - fixed raw AutoSelect aggregation through 122c
"""

from __future__ import annotations

import argparse
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out_root",
        default="experiments_transfer_effectiveness/fixed_raw_autoselect_all10_cvseed1",
    )
    ap.add_argument("--datasets", nargs="+", default=ALL10_DATASETS)

    ap.add_argument("--cv_seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--run_keys", nargs="+", default=["seed5"])
    ap.add_argument("--source_folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])

    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--skip_existing_base", type=int, default=1)
    ap.add_argument("--skip_existing_stack", type=int, default=1)
    ap.add_argument("--dry_run", type=int, default=0)

    args = ap.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "experiments_transfer_effectiveness/external_run_fixed_raw_autoselect.py",
        "--out_root", args.out_root,
        "--cv_seeds", *map(str, args.cv_seeds),
        "--run_keys", *args.run_keys,
        "--source_folds", *map(str, args.source_folds),
        "--datasets", *args.datasets,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--eval_batch_size", str(args.eval_batch_size),
        "--lr", str(args.lr),
        "--skip_existing_base", str(args.skip_existing_base),
        "--skip_existing_stack", str(args.skip_existing_stack),
        "--dry_run", str(args.dry_run),
    ]

    run_cmd(cmd, dry_run=bool(args.dry_run))

    print("\n✅ transfer all10 wrapper done:", args.out_root)


if __name__ == "__main__":
    main()

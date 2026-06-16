#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import sys

EXPECTED_N = {
    "Eawag_XBridgeC18_364": 364,
    "FEM_lipids_72": 72,
    "FEM_long_412": 412,
    "IPB_Halle_82": 82,
    "LIFE_new_184": 184,
    "LIFE_old_194": 194,
}

EXPECTED_SOURCE_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_METHODS = ["origin_tl", "taut_tl"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    args = ap.parse_args()

    p = Path(args.pred_csv)
    if not p.exists():
        print("[INVALID] missing:", p)
        sys.exit(1)

    try:
        df = pd.read_csv(p)
    except Exception as e:
        print("[INVALID] cannot read:", p, repr(e))
        sys.exit(1)

    required_cols = ["dataset_name", "method", "source_fold", "cv_fold", "y_true", "y_pred"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("[INVALID] missing columns:", missing)
        print("columns:", list(df.columns))
        sys.exit(1)

    df = df[df["method"].isin(EXPECTED_METHODS)].copy()

    problems = []

    for ds, n in EXPECTED_N.items():
        for sf in EXPECTED_SOURCE_FOLDS:
            for method in EXPECTED_METHODS:
                sub = df[
                    (df["dataset_name"] == ds)
                    & (df["source_fold"].astype(int) == sf)
                    & (df["method"] == method)
                ]
                if len(sub) != n:
                    problems.append(
                        f"{ds} source_fold={sf} method={method}: expected {n}, got {len(sub)}"
                    )
                if len(sub) > 0:
                    if sub["y_pred"].isna().any() or sub["y_true"].isna().any():
                        problems.append(
                            f"{ds} source_fold={sf} method={method}: NaN in y_true/y_pred"
                        )
                    folds = sorted(sub["cv_fold"].astype(int).unique().tolist())
                    if folds != list(range(10)):
                        problems.append(
                            f"{ds} source_fold={sf} method={method}: cv_folds={folds}"
                        )

    if problems:
        print("[INVALID]", p)
        for x in problems[:50]:
            print(" -", x)
        if len(problems) > 50:
            print(" ... more problems:", len(problems) - 50)
        sys.exit(1)

    print("[VALID]", p)
    print("rows:", len(df))
    sys.exit(0)

if __name__ == "__main__":
    main()

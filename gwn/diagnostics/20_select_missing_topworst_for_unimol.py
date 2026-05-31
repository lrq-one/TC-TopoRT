import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def norm_pred(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    smi = cols.get("smiles", "SMILES")
    y = cols.get("actual_rt", None) or cols.get("y", None) or cols.get("y_true", None)
    p = cols.get("predicted_rt", None) or cols.get("y_pred", None) or cols.get("pred", None)

    if y is None or p is None:
        raise ValueError(f"bad columns in {path}: {df.columns.tolist()}")

    out = df[[smi, y, p]].copy()
    out.columns = ["SMILES", "Actual_RT", "Predicted_RT"]
    out["SMILES"] = out["SMILES"].astype(str)
    out["Actual_RT"] = out["Actual_RT"].astype(float)
    out["Predicted_RT"] = out["Predicted_RT"].astype(float)
    out["Abs_Error"] = (out["Actual_RT"] - out["Predicted_RT"]).abs()
    return out


def take_missing(df, split, top_k, err_thr, existing):
    df = df.sort_values("Abs_Error", ascending=False).copy()

    a = df.head(top_k)
    b = df[df["Abs_Error"] >= err_thr]
    out = pd.concat([a, b], axis=0).drop_duplicates("SMILES")

    out = out[~out["SMILES"].astype(str).isin(existing)].copy()
    out["split"] = split
    out = out.sort_values("Abs_Error", ascending=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing_npz", required=True)
    ap.add_argument("--train_pred", required=True)
    ap.add_argument("--val_pred", required=True)
    ap.add_argument("--test_pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_top", type=int, default=3000)
    ap.add_argument("--val_top", type=int, default=1000)
    ap.add_argument("--test_top", type=int, default=1000)
    ap.add_argument("--err_thr", type=float, default=80.0)
    args = ap.parse_args()

    z = np.load(args.existing_npz, allow_pickle=False)
    existing = set(z["smiles"].astype(str).tolist())

    train = norm_pred(args.train_pred)
    val = norm_pred(args.val_pred)
    test = norm_pred(args.test_pred)

    tr = take_missing(train, "train", args.train_top, args.err_thr, existing)
    va = take_missing(val, "val", args.val_top, args.err_thr, existing)
    te = take_missing(test, "test", args.test_top, args.err_thr, existing)

    all_df = pd.concat([tr, va, te], axis=0)
    all_df = all_df.drop_duplicates("SMILES")
    all_df = all_df.sort_values(["split", "Abs_Error"], ascending=[True, False])

    all_df.to_csv(args.out, index=False)

    print("existing 3D:", len(existing))
    print("saved:", args.out)
    print("new missing total:", len(all_df))

    for name, sub in [("train", tr), ("val", va), ("test", te)]:
        if len(sub) == 0:
            print(name, "N=0")
        else:
            print(
                name,
                "N=", len(sub),
                "min_err=", round(sub["Abs_Error"].min(), 3),
                "median_err=", round(sub["Abs_Error"].median(), 3),
                "max_err=", round(sub["Abs_Error"].max(), 3),
            )

    print("\nTop 30 new test missing:")
    print(te.head(30)[["SMILES", "Actual_RT", "Predicted_RT", "Abs_Error"]].to_string(index=False))


if __name__ == "__main__":
    main()

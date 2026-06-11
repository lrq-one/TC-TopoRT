import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def norm_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for x in candidates:
        if x.lower() in lower:
            return lower[x.lower()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", default="paper_analysis_stage4_external/external_predret10_stage4_meta.csv")
    ap.add_argument("--origin_csv", default="paper_analysis_stage4_external/temp_external_predret10_origin.csv")
    ap.add_argument("--taut_csv", default="paper_analysis_stage4_external/temp_external_predret10_taut.csv")
    ap.add_argument("--dataset", default="LIFE_old_194")
    ap.add_argument("--head", type=int, default=10)
    args = ap.parse_args()

    meta = norm_cols(pd.read_csv(args.meta_csv)).sort_values("stage4_index").reset_index(drop=True)
    ori = norm_cols(pd.read_csv(args.origin_csv)).reset_index(drop=True)
    tau = norm_cols(pd.read_csv(args.taut_csv)).reset_index(drop=True)

    print("=== columns ===")
    print("meta:", list(meta.columns))
    print("origin:", list(ori.columns))
    print("taut:", list(tau.columns))

    print("\n=== lengths ===")
    print("meta:", len(meta))
    print("origin_csv:", len(ori))
    print("taut_csv:", len(tau))

    if len(meta) != len(ori) or len(meta) != len(tau):
        print("[WARN] length mismatch!")

    # stage4_index column check
    for name, df in [("meta", meta), ("origin", ori), ("taut", tau)]:
        if "stage4_index" in df.columns:
            bad = int((df["stage4_index"].values != np.arange(len(df))).sum())
            print(f"{name}: stage4_index sequential mismatch count = {bad}")
        else:
            print(f"{name}: no stage4_index column")

    sub = meta[meta["dataset_name"] == args.dataset].copy()
    if len(sub) == 0:
        raise ValueError(f"dataset not found: {args.dataset}")

    idx = sub["stage4_index"].values.astype(int)
    print("\n=== dataset subset ===")
    print("dataset:", args.dataset)
    print("n:", len(sub))
    print("stage4_index min/max:", int(idx.min()), int(idx.max()))
    print("unique stage4_index:", len(np.unique(idx)))

    # compare metadata rows by stage4_index
    ori_sub = ori.iloc[idx].copy()
    tau_sub = tau.iloc[idx].copy()

    print("\n=== sampled paired rows ===")
    show_cols = []
    for c in ["stage4_index", "dataset_name", "rt", "smiles", "smile", "orig_smile", "canonical_smiles", "inchikey"]:
        if c in meta.columns or c in ori.columns or c in tau.columns:
            show_cols.append(c)

    for k in range(min(args.head, len(idx))):
        i = idx[k]
        print(f"\n--- k={k}, stage4_index={i} ---")
        for name, df in [("meta", meta), ("origin", ori), ("taut", tau)]:
            row = df.iloc[i]
            small = {}
            for c in ["stage4_index", "dataset_name", "rt", "smiles", "smile", "orig_smile", "canonical_smiles", "inchikey"]:
                if c in df.columns:
                    small[c] = row[c]
            print(name, small)

    # compare common columns
    print("\n=== common-column mismatch in subset ===")
    common_checks = ["stage4_index", "dataset_name", "rt", "inchikey", "canonical_smiles"]
    for c in common_checks:
        if c in meta.columns and c in ori.columns:
            m = meta.iloc[idx][c].astype(str).values
            o = ori_sub[c].astype(str).values
            print(f"meta vs origin {c}: mismatch = {int((m != o).sum())} / {len(idx)}")
        if c in meta.columns and c in tau.columns:
            m = meta.iloc[idx][c].astype(str).values
            t = tau_sub[c].astype(str).values
            print(f"meta vs taut   {c}: mismatch = {int((m != t).sum())} / {len(idx)}")

    # RT numerical check
    if "rt" in meta.columns and "rt" in ori.columns:
        dm = np.abs(meta.iloc[idx]["rt"].astype(float).values - ori_sub["rt"].astype(float).values)
        print("meta-origin rt max abs diff:", float(np.nanmax(dm)))
    if "rt" in meta.columns and "rt" in tau.columns:
        dt = np.abs(meta.iloc[idx]["rt"].astype(float).values - tau_sub["rt"].astype(float).values)
        print("meta-taut rt max abs diff:", float(np.nanmax(dt)))

    print("\n✅ audit done")


if __name__ == "__main__":
    main()

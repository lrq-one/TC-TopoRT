#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

CAND = Path("experiments_candidate_filtering/metabobase_s10_predictions_tl_exact39/metabobase_s10_candidate_predictions_tl_seed42.csv")
META = Path("experiments_candidate_filtering/metabobase_tl_exact39/metabobase_test_exact39_metadata.csv")
TEST_PRED = Path("experiments_candidate_filtering/metabobase_tl_exact39_training/seed42/dualview_avg/test_predictions_avg.csv")

OUT = Path("experiments_candidate_filtering/metabobase_s10_tl_true_outlier_audit")
OUT.mkdir(parents=True, exist_ok=True)


def first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


cand = pd.read_csv(CAND)
meta = pd.read_csv(META).reset_index(drop=True)
tp = pd.read_csv(TEST_PRED).reset_index(drop=True)

cand["is_true"] = cand["is_true"].astype(bool)

# official exact39 prediction table, row order should match test metadata
official = pd.DataFrame()
official["official_name"] = meta["name"]
official["official_inchikey"] = meta["inchikey"]
official["official_smiles"] = meta["smiles"]
official["official_rt_sec"] = meta["rt_sec"].astype(float)
official["official_tl_pred_rt"] = tp["avg_pred"].astype(float)
official["official_tl_abs_err"] = tp["abs_err"].astype(float)

true_rows = cand[cand["is_true"]].copy()
true_rows = true_rows.sort_values(["s10_row", "candidate_rank"]).copy()

# keep first true candidate per query
true_first = true_rows.groupby("s10_row", as_index=False).first()

true_ik_col = first_existing(true_first, ["true_inchikey", "inchikey", "query_inchikey"])
cand_ik_col = first_existing(true_first, ["candidate_inchikey", "InChIKey", "inchikey"])

if true_ik_col is None:
    raise RuntimeError("Cannot find true_inchikey-like column in candidate file")

audit = true_first.merge(
    official,
    left_on=true_ik_col,
    right_on="official_inchikey",
    how="left",
)

audit["candidate_true_pred_rt"] = audit["candidate_pred_rt"].astype(float)
audit["candidate_true_abs_delta"] = audit["abs_rt_delta"].astype(float)
audit["official_minus_candidate_pred_gap"] = audit["official_tl_pred_rt"] - audit["candidate_true_pred_rt"]
audit["candidate_delta_minus_official_err"] = audit["candidate_true_abs_delta"] - audit["official_tl_abs_err"]

name_cols = [
    "s10_row",
    "true_name",
    "candidate_rank",
    "candidate_name",
    "rt_sec",
    "candidate_true_pred_rt",
    "candidate_true_abs_delta",
    "official_name",
    "official_rt_sec",
    "official_tl_pred_rt",
    "official_tl_abs_err",
    "candidate_delta_minus_official_err",
    "official_minus_candidate_pred_gap",
]
name_cols = [c for c in name_cols if c in audit.columns]

# add possible inchikey columns
for c in [true_ik_col, cand_ik_col, "official_inchikey"]:
    if c and c in audit.columns and c not in name_cols:
        name_cols.append(c)

audit_sorted = audit.sort_values("candidate_true_abs_delta", ascending=False).copy()
audit_sorted.to_csv(OUT / "true_candidate_outliers_sorted.csv", index=False)

print("=" * 100)
print("candidate file:", CAND)
print("metadata test rows:", len(meta))
print("test pred rows:", len(tp))
print("true candidate rows:", len(true_rows))
print("first true candidates by query:", len(true_first))
print("merged official matches:", int(audit["official_name"].notna().sum()))
print("=" * 100)

print("\nTrue candidate error summary:")
print(audit["candidate_true_abs_delta"].describe().to_string())

print("\nOfficial exact39 test prediction error summary:")
print(official["official_tl_abs_err"].describe().to_string())

print("\nTop true-candidate outliers:")
print(audit_sorted[name_cols].head(25).to_string(index=False))

print("\nSaved:")
print(OUT / "true_candidate_outliers_sorted.csv")
print("=" * 100)

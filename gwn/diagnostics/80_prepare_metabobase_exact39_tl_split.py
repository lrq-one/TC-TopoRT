#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

REF = Path("experiments_candidate_filtering/abcort_reference/metabobase_abcort_s10_test_inchikeys.csv")
META = Path("experiments_candidate_filtering/msfinder_queries/msfinder_query_metadata.csv")

OUT = Path("experiments_candidate_filtering/metabobase_tl_exact39")
OUT.mkdir(parents=True, exist_ok=True)

def pick_inchikey_col(df):
    for c in df.columns:
        if "inchikey" in c.lower():
            return c
    return df.columns[0]

ref = pd.read_csv(REF)
meta = pd.read_csv(META)

ref_col = pick_inchikey_col(ref)
ref["ref_inchikey"] = ref[ref_col].astype(str).str.strip()

meta = meta[meta["dataset"].eq("MetaboBase")].copy()
meta["inchikey"] = meta["inchikey"].astype(str).str.strip()

ref_keys = set(ref["ref_inchikey"].tolist())

meta["is_s10_exact39"] = meta["inchikey"].isin(ref_keys)

test = meta[meta["is_s10_exact39"]].copy()
train = meta[~meta["is_s10_exact39"]].copy()

missing = ref[~ref["ref_inchikey"].isin(set(meta["inchikey"].tolist()))].copy()

# 给 SMRTComplexDataset 用，只需要 smiles,rt
train_model = train[["smiles", "rt_sec"]].rename(columns={"rt_sec": "rt"}).copy()
test_model = test[["smiles", "rt_sec"]].rename(columns={"rt_sec": "rt"}).copy()

train_model.to_csv(OUT / "metabobase_train_exact39_for_model.csv", index=False)
test_model.to_csv(OUT / "metabobase_test_exact39_for_model.csv", index=False)

train.to_csv(OUT / "metabobase_train_exact39_metadata.csv", index=False)
test.to_csv(OUT / "metabobase_test_exact39_metadata.csv", index=False)
missing.to_csv(OUT / "metabobase_s10_missing6_from_metadata.csv", index=False)

print("=" * 100)
print("reference S10 rows:", len(ref))
print("MetaboBase metadata rows:", len(meta))
print("exact S10 matched test rows:", len(test))
print("train rows:", len(train))
print("missing S10 rows:", len(missing))
print("-" * 100)
print("train model csv:", OUT / "metabobase_train_exact39_for_model.csv")
print("test model csv:", OUT / "metabobase_test_exact39_for_model.csv")
print("missing csv:", OUT / "metabobase_s10_missing6_from_metadata.csv")
print("=" * 100)

print("\nMissing 6:")
print(missing[["ref_inchikey"]].to_string(index=False))

print("\nTest exact39 preview:")
cols = [c for c in ["source_row", "name", "inchikey", "smiles", "rt_sec"] if c in test.columns]
print(test[cols].head(50).to_string(index=False))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

REF = Path("experiments_candidate_filtering/abcort_reference/metabobase_abcort_s10_test_inchikeys.csv")
META = Path("experiments_candidate_filtering/msfinder_queries/msfinder_query_metadata.csv")
MATCHED = Path("experiments_candidate_filtering/metabobase_s10_subset/metabobase_s10_test_queries_matched_metadata.csv")
CAND = Path("experiments_candidate_filtering/metabobase_s10_subset/metabobase_s10_candidates_valid.csv")

OUT = Path("experiments_candidate_filtering/metabobase_s10_split_audit")
OUT.mkdir(parents=True, exist_ok=True)


def inchikey14(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    return s.split("-")[0] if s else ""


ref = pd.read_csv(REF)
meta = pd.read_csv(META)
matched = pd.read_csv(MATCHED)
cand = pd.read_csv(CAND)

meta = meta[meta["dataset"].eq("MetaboBase")].copy()

# 自动找 reference inchikey 列
ref_cols = ref.columns.tolist()
ik_col = None
for c in ref_cols:
    if "inchikey" in c.lower():
        ik_col = c
        break
if ik_col is None:
    ik_col = ref_cols[0]

ref["ref_inchikey"] = ref[ik_col].astype(str).str.strip()
ref["ref_inchikey14"] = ref["ref_inchikey"].map(inchikey14)

meta["meta_inchikey"] = meta["inchikey"].astype(str).str.strip()
meta["meta_inchikey14"] = meta["meta_inchikey"].map(inchikey14)

matched["matched_inchikey"] = matched["inchikey"].astype(str).str.strip()
matched["matched_inchikey14"] = matched["matched_inchikey"].map(inchikey14)

cand_true = cand[cand["is_true"].astype(bool)].copy()
cand_true["cand_true_inchikey14"] = cand_true["true_inchikey"].map(inchikey14)

ref_to_meta_full = ref.merge(
    meta,
    left_on="ref_inchikey",
    right_on="meta_inchikey",
    how="left",
    suffixes=("", "_meta"),
)

ref_to_meta14 = ref.merge(
    meta,
    left_on="ref_inchikey14",
    right_on="meta_inchikey14",
    how="left",
    suffixes=("", "_meta14"),
)

ref_to_matched = ref.merge(
    matched,
    left_on="ref_inchikey14",
    right_on="matched_inchikey14",
    how="left",
    suffixes=("", "_matched"),
)

ref_to_cand_true = ref.merge(
    cand_true,
    left_on="ref_inchikey14",
    right_on="true_inchikey14",
    how="left",
    suffixes=("", "_cand"),
)

def count_nonnull(df, col):
    return int(df[col].notna().sum()) if col in df.columns else 0

print("=" * 100)
print("REF rows:", len(ref))
print("MetaboBase metadata rows:", len(meta))
print("matched metadata rows:", len(matched))
print("candidate true rows:", len(cand_true))
print("-" * 100)
print("REF matched to metadata by full InChIKey:", count_nonnull(ref_to_meta_full, "name"))
print("REF matched to metadata by InChIKey14:", count_nonnull(ref_to_meta14, "name"))
print("REF matched to matched_metadata by InChIKey14:", count_nonnull(ref_to_matched, "name"))
print("REF matched to candidate true by InChIKey14:", count_nonnull(ref_to_cand_true, "true_name"))
print("=" * 100)

audit = ref[["ref_inchikey", "ref_inchikey14"]].copy()
audit = audit.merge(
    meta[["source_row", "name", "inchikey", "meta_inchikey14", "smiles", "rt_sec"]],
    left_on="ref_inchikey14",
    right_on="meta_inchikey14",
    how="left",
)
audit = audit.rename(columns={
    "source_row": "meta_source_row",
    "name": "meta_name",
    "inchikey": "meta_inchikey",
    "smiles": "meta_smiles",
    "rt_sec": "meta_rt_sec",
})
audit = audit.merge(
    matched[["source_row", "name", "inchikey", "matched_inchikey14"]],
    left_on="ref_inchikey14",
    right_on="matched_inchikey14",
    how="left",
    suffixes=("", "_in_matched_file"),
)
audit = audit.rename(columns={
    "source_row": "matched_source_row",
    "name": "matched_name",
    "inchikey": "matched_inchikey",
})
audit["has_meta_match14"] = audit["meta_name"].notna()
audit["has_matched_file_match14"] = audit["matched_name"].notna()

audit.to_csv(OUT / "metabobase_s10_reference_match_audit.csv", index=False)

missing_meta = audit[~audit["has_meta_match14"]].copy()
missing_matched = audit[~audit["has_matched_file_match14"]].copy()

missing_meta.to_csv(OUT / "missing_from_metabobase_metadata_by_inchikey14.csv", index=False)
missing_matched.to_csv(OUT / "missing_from_current_matched_file_by_inchikey14.csv", index=False)

print("\nMissing from metadata by InChIKey14:")
print(missing_meta[["ref_inchikey", "ref_inchikey14"]].to_string(index=False))

print("\nMissing from current matched file by InChIKey14:")
print(missing_matched[["ref_inchikey", "ref_inchikey14", "meta_name", "meta_source_row"]].to_string(index=False))

print("\nSaved dir:", OUT)
print("=" * 100)

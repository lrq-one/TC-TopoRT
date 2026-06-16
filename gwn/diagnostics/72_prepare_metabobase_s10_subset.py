#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


REF = Path("experiments_candidate_filtering/abcort_reference/metabobase_abcort_s10_test_inchikeys.csv")
META = Path("experiments_candidate_filtering/msfinder_queries/msfinder_query_metadata.csv")
CAND = Path("experiments_candidate_filtering/parsed_candidates/msfinder_candidates_valid.csv")
OUT = Path("experiments_candidate_filtering/metabobase_s10_subset")
OUT.mkdir(parents=True, exist_ok=True)


def ik14(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    return s.split("-")[0]


ref = pd.read_csv(REF)
ref["inchikey"] = ref["inchikey"].astype(str).str.strip()
ref["inchikey14"] = ref["inchikey"].map(ik14)

meta = pd.read_csv(META)
meta = meta[meta["dataset"].eq("MetaboBase")].copy()
meta["true_inchikey"] = meta["inchikey"].astype(str).str.strip()
meta["true_inchikey14"] = meta["true_inchikey"].map(ik14)

cand = pd.read_csv(CAND)
cand = cand[cand["dataset"].eq("MetaboBase")].copy()
cand["true_inchikey"] = cand["true_inchikey"].astype(str).str.strip()
cand["true_inchikey14"] = cand["true_inchikey"].map(ik14)

# Match S10 reference to metadata by full InChIKey first, then by first block.
matched_meta = meta.merge(
    ref[["s10_row", "inchikey", "inchikey14"]],
    left_on="true_inchikey",
    right_on="inchikey",
    how="inner",
)

if matched_meta["s10_row"].nunique() < len(ref):
    matched14 = meta.merge(
        ref[["s10_row", "inchikey", "inchikey14"]],
        left_on="true_inchikey14",
        right_on="inchikey14",
        how="inner",
        suffixes=("", "_ref"),
    )
    matched_meta = matched14.copy()

matched_meta = matched_meta.sort_values("s10_row").drop_duplicates("s10_row")

matched_rows = set(matched_meta["source_row"].astype(int).tolist())
matched_ref_rows = set(matched_meta["s10_row"].astype(int).tolist())
missing_ref = ref[~ref["s10_row"].isin(matched_ref_rows)].copy()

s10_cand = cand[cand["source_row"].astype(int).isin(matched_rows)].copy()
s10_cand = s10_cand.merge(
    matched_meta[["source_row", "s10_row"]],
    on="source_row",
    how="left",
)
s10_cand = s10_cand.sort_values(["s10_row", "candidate_rank"])

# Per-query stats before RT filtering.
query_stats = []
for _, q in matched_meta.sort_values("s10_row").iterrows():
    source_row = int(q["source_row"])
    sub = s10_cand[s10_cand["source_row"].astype(int).eq(source_row)].copy()
    n = len(sub)
    hit_any = bool(sub["is_true"].any()) if n else False

    def topk(k):
        if n == 0:
            return False
        return bool(sub[sub["candidate_rank"] <= k]["is_true"].any())

    true_rank = None
    if hit_any:
        true_rank = int(sub[sub["is_true"]]["candidate_rank"].min())

    query_stats.append({
        "s10_row": int(q["s10_row"]),
        "query_id": q["query_id"],
        "source_row": source_row,
        "true_name": q["name"],
        "true_formula": q["formula"],
        "true_inchikey": q["true_inchikey"],
        "rt_sec": q["rt_sec"],
        "n_valid_candidates": n,
        "true_in_candidates": hit_any,
        "true_candidate_rank": true_rank,
        "top1_before": topk(1),
        "top5_before": topk(5),
        "top10_before": topk(10),
    })

query_stats = pd.DataFrame(query_stats)

matched_meta.to_csv(OUT / "metabobase_s10_test_queries_matched_metadata.csv", index=False)
missing_ref.to_csv(OUT / "metabobase_s10_unmatched_reference_inchikeys.csv", index=False)
s10_cand.to_csv(OUT / "metabobase_s10_candidates_valid.csv", index=False)
query_stats.to_csv(OUT / "metabobase_s10_msfinder_before_filter_query_stats.csv", index=False)

print("=" * 100)
print("S10 reference keys:", len(ref))
print("Matched S10 queries in MetaboBase metadata:", matched_meta['s10_row'].nunique())
print("Missing S10 reference keys:", len(missing_ref))
print("Valid MS-FINDER candidates for matched S10 queries:", len(s10_cand))
print("Queries with >=1 valid candidate:", int((query_stats['n_valid_candidates'] > 0).sum()))
print("Queries with true molecule in candidates:", int(query_stats["true_in_candidates"].sum()))
print()
print("MS-FINDER before-filter top-k on matched S10 queries:")
for k in [1, 5, 10]:
    col = f"top{k}_before"
    print(f"{col}: {query_stats[col].mean() * 100:.2f}%")
print()
print("Candidate count summary:")
print(query_stats["n_valid_candidates"].describe())
print("=" * 100)

if len(missing_ref):
    print("\n[WARNING] Missing S10 keys:")
    print(missing_ref.to_string(index=False))

print("\nSaved:")
for p in [
    OUT / "metabobase_s10_test_queries_matched_metadata.csv",
    OUT / "metabobase_s10_unmatched_reference_inchikeys.csv",
    OUT / "metabobase_s10_candidates_valid.csv",
    OUT / "metabobase_s10_msfinder_before_filter_query_stats.csv",
]:
    print(p)

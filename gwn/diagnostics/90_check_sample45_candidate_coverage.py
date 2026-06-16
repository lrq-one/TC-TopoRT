#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

SPLIT = Path("experiments_candidate_filtering/metabobase_tl_sample45/seed42/metabobase_test45_sample45_metadata.csv")
OUT = Path("experiments_candidate_filtering/metabobase_tl_sample45/seed42")
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATE_DIRS = [
    Path("experiments_candidate_filtering"),
]

def find_candidate_files():
    files = []
    for root in CANDIDATE_DIRS:
        for p in root.rglob("*.csv"):
            name = p.name.lower()
            if any(k in name for k in ["candidate", "msfinder", "structure"]):
                files.append(p)
    return sorted(set(files))


def main():
    meta = pd.read_csv(SPLIT)
    query_ids = set(meta["query_id"].astype(str))
    inchikeys = set(meta["inchikey"].astype(str))
    names = set(meta["name"].astype(str))

    print("=" * 100)
    print("sample45 test rows:", len(meta))
    print("query_ids:", sorted(list(query_ids))[:10], "...")
    print("=" * 100)

    rows = []
    matched_detail = []

    files = find_candidate_files()
    print("candidate-like csv files:", len(files))

    for p in files:
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False).fillna("")
        except Exception:
            continue

        cols = set(df.columns)
        joined = df.astype(str).agg(" | ".join, axis=1)

        hit_query = joined.apply(lambda x: any(q in x for q in query_ids))
        hit_ikey = joined.apply(lambda x: any(k in x for k in inchikeys))
        n_hit = int((hit_query | hit_ikey).sum())

        if n_hit > 0:
            rows.append({
                "file": str(p),
                "n_rows": len(df),
                "n_hit_rows": n_hit,
                "columns": "|".join(df.columns),
            })

            sub = df[hit_query | hit_ikey].copy()
            sub.insert(0, "__source_file", str(p))
            matched_detail.append(sub.head(200))

    summary = pd.DataFrame(rows).sort_values("n_hit_rows", ascending=False) if rows else pd.DataFrame()
    summary.to_csv(OUT / "sample45_candidate_file_coverage_summary.csv", index=False)

    if matched_detail:
        detail = pd.concat(matched_detail, axis=0, ignore_index=True)
        detail.to_csv(OUT / "sample45_candidate_file_coverage_detail.csv", index=False)
    else:
        detail = pd.DataFrame()
        detail.to_csv(OUT / "sample45_candidate_file_coverage_detail.csv", index=False)

    print("\n[coverage summary]")
    if len(summary):
        print(summary.head(40).to_string(index=False))
    else:
        print("NO candidate-like files hit sample45 query IDs or InChIKeys")

    # query-level coverage from best-looking file
    if len(summary):
        best_file = Path(summary.iloc[0]["file"])
        df = pd.read_csv(best_file, dtype=str, low_memory=False).fillna("")
        joined = df.astype(str).agg(" | ".join, axis=1)

        qrows = []
        for _, r in meta.iterrows():
            qid = str(r["query_id"])
            ik = str(r["inchikey"])
            nm = str(r["name"])
            mask = joined.str.contains(qid, regex=False) | joined.str.contains(ik, regex=False)
            qrows.append({
                "query_id": qid,
                "name": nm,
                "inchikey": ik,
                "sample45_role": r.get("sample45_role", ""),
                "n_candidate_like_rows_in_best_file": int(mask.sum()),
                "best_file": str(best_file),
            })
        qcov = pd.DataFrame(qrows)
        qcov.to_csv(OUT / "sample45_query_level_candidate_coverage.csv", index=False)

        print("\n[query-level coverage from best file]")
        print(qcov.to_string(index=False))

        print("\nmissing candidate rows:")
        print(qcov[qcov["n_candidate_like_rows_in_best_file"].eq(0)].to_string(index=False))

    print("\nSaved:")
    print(OUT / "sample45_candidate_file_coverage_summary.csv")
    print(OUT / "sample45_candidate_file_coverage_detail.csv")
    print(OUT / "sample45_query_level_candidate_coverage.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

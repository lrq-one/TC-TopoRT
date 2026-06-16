#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("experiments_candidate_filtering/riken_audit")
OUT.mkdir(parents=True, exist_ok=True)

ALL = Path("experiments_candidate_filtering/parsed_candidates/msfinder_structure_rows_all.csv")
VALID = Path("experiments_candidate_filtering/parsed_candidates/msfinder_candidates_valid.csv")
META = Path("experiments_candidate_filtering/msfinder_queries/msfinder_query_metadata.csv")


def to_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def dataset_is_riken(s):
    return s.astype(str).str.lower().str.contains("riken")


def summarize_candidates(df, label):
    if len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    if "is_true" in df.columns:
        df["is_true_bool"] = to_bool_series(df["is_true"])
    else:
        df["is_true_bool"] = False

    if "candidate_rank" in df.columns:
        df["candidate_rank_num"] = pd.to_numeric(df["candidate_rank"], errors="coerce")
    elif "candidate_rank_all" in df.columns:
        df["candidate_rank_num"] = pd.to_numeric(df["candidate_rank_all"], errors="coerce")
    else:
        df["candidate_rank_num"] = np.nan

    rows = []
    for qid, sub in df.groupby("query_id"):
        tr = sub[sub["is_true_bool"]].sort_values("candidate_rank_num")
        rows.append({
            "query_id": qid,
            f"n_rows_{label}": len(sub),
            f"true_in_{label}": bool(len(tr)),
            f"true_rank_{label}": float(tr.iloc[0]["candidate_rank_num"]) if len(tr) else np.nan,
            f"top1_before_{label}": bool(len(tr) and tr.iloc[0]["candidate_rank_num"] <= 1),
            f"top5_before_{label}": bool(len(tr) and tr.iloc[0]["candidate_rank_num"] <= 5),
            f"top10_before_{label}": bool(len(tr) and tr.iloc[0]["candidate_rank_num"] <= 10),
        })

    return pd.DataFrame(rows)


def main():
    all_df = pd.read_csv(ALL, dtype=str, low_memory=False).fillna("")
    valid_df = pd.read_csv(VALID, dtype=str, low_memory=False).fillna("")
    meta = pd.read_csv(META, dtype=str, low_memory=False).fillna("")

    print("=" * 100)
    print("[available datasets in query metadata]")
    print(meta["dataset"].astype(str).value_counts().to_string())
    print("=" * 100)

    riken_meta = meta[dataset_is_riken(meta["dataset"])].copy()
    riken_all = all_df[dataset_is_riken(all_df["dataset"])].copy()
    riken_valid = valid_df[dataset_is_riken(valid_df["dataset"])].copy()

    print("[RIKEN raw counts]")
    print("metadata rows:", len(riken_meta), "unique query:", riken_meta["query_id"].nunique() if len(riken_meta) else 0)
    print("all candidate rows:", len(riken_all), "unique query:", riken_all["query_id"].nunique() if len(riken_all) else 0)
    print("valid candidate rows:", len(riken_valid), "unique query:", riken_valid["query_id"].nunique() if len(riken_valid) else 0)

    all_q = summarize_candidates(riken_all, "all")
    valid_q = summarize_candidates(riken_valid, "valid")

    q = riken_meta[[
        "dataset", "query_id", "source_file", "source_row",
        "name", "precursor_mz", "precursor_type", "ion_mode",
        "formula", "smiles", "inchi", "inchikey", "rt_sec",
        "rt_min_written", "n_peaks", "msp_file",
    ]].drop_duplicates("query_id").copy()

    q = q.merge(all_q, on="query_id", how="left")
    q = q.merge(valid_q, on="query_id", how="left")

    for c in ["n_rows_all", "n_rows_valid"]:
        q[c] = q[c].fillna(0).astype(int)

    for c in ["true_in_all", "true_in_valid", "top1_before_valid", "top5_before_valid", "top10_before_valid"]:
        if c in q.columns:
            q[c] = q[c].fillna(False).astype(bool)

    q["rt_sec"] = pd.to_numeric(q["rt_sec"], errors="coerce")
    q["candidate_evaluable"] = q["true_in_valid"].astype(bool)

    q.to_csv(OUT / "riken_query_candidate_coverage_summary.csv", index=False)
    riken_meta.to_csv(OUT / "riken_query_metadata.csv", index=False)
    riken_valid.to_csv(OUT / "riken_candidates_valid.csv", index=False)

    print("\n[RIKEN query-level summary]")
    print("queries in metadata:", len(q))
    print("queries with valid candidate rows:", int((q["n_rows_valid"] > 0).sum()))
    print("queries with true in valid:", int(q["true_in_valid"].sum()))
    print("candidate-evaluable queries:", int(q["candidate_evaluable"].sum()))

    if int(q["true_in_valid"].sum()) > 0:
        sub = q[q["true_in_valid"]].copy()
        print("\n[MS-FINDER before filtering on candidate-evaluable RIKEN]")
        print("N:", len(sub))
        print("Top1:", round(100 * sub["top1_before_valid"].mean(), 4))
        print("Top5:", round(100 * sub["top5_before_valid"].mean(), 4))
        print("Top10:", round(100 * sub["top10_before_valid"].mean(), 4))
        print("median true rank:", sub["true_rank_valid"].median())
        print("mean candidates/query:", sub["n_rows_valid"].mean())

    print("\n[head]")
    keep = ["query_id", "name", "rt_sec", "n_rows_valid", "true_in_valid", "true_rank_valid"]
    keep = [c for c in keep if c in q.columns]
    print(q[keep].head(80).to_string(index=False))

    print("\nSaved:")
    print(OUT / "riken_query_candidate_coverage_summary.csv")
    print(OUT / "riken_query_metadata.csv")
    print(OUT / "riken_candidates_valid.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

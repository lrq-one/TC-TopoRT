#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def inchikey14(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    return s.split("-")[0].strip()


def infer_dataset_from_path(x):
    s = str(x)
    if "MetaboBase" in s or "metabo" in s.lower():
        return "MetaboBase"
    if "RIKEN_MONA" in s or "RIKEN" in s or "MONA" in s:
        return "RIKEN_MONA"
    return "UNKNOWN"


def parse_source_row(file_name):
    """
    File name examples:
    00001_Hesperetin
    00000_3-Glu-28-Xyl...
    """
    s = str(file_name)
    m = re.match(r"^(\d+)_", s)
    if m:
        return int(m.group(1))
    return np.nan


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def read_structure_file(path):
    path = Path(path)
    df = pd.read_csv(path, sep="\t", dtype=str, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df["source_structure_file"] = str(path)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exports_dir",
        default="experiments_candidate_filtering/msfinder_exports",
        help="Directory containing MS-FINDER Structure result*.txt files",
    )
    ap.add_argument(
        "--metadata",
        default="experiments_candidate_filtering/msfinder_queries/msfinder_query_metadata.csv",
        help="Metadata generated when converting raw libraries to MSP",
    )
    ap.add_argument(
        "--out_dir",
        default="experiments_candidate_filtering/parsed_candidates",
    )
    args = ap.parse_args()

    exports_dir = Path(args.exports_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(exports_dir.rglob("Structure result*.txt"))
    if not files:
        raise FileNotFoundError(f"No Structure result*.txt found under {exports_dir}")

    print("Found structure result files:")
    for f in files:
        print(" -", f)

    dfs = []
    for f in files:
        df = read_structure_file(f)
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)

    # Normalize expected columns.
    required = [
        "File path", "File name", "Title", "Precursor mz", "Precursor type",
        "Structure", "Total score", "Databases", "Formula", "Ontology",
        "InChIKey", "SMILES"
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        print("[WARNING] Missing expected columns:", missing)
        print("Available columns:", raw.columns.tolist())

    raw["dataset"] = raw.get("File path", "").map(infer_dataset_from_path)
    raw["source_row"] = raw.get("File name", "").map(parse_source_row)
    raw["query_file_name"] = raw.get("File name", "")
    raw["query_title"] = raw.get("Title", "")
    raw["candidate_name"] = raw.get("Structure", "").fillna("").astype(str).str.strip()
    raw["candidate_score"] = raw.get("Total score", np.nan).map(to_float)
    raw["candidate_formula"] = raw.get("Formula", "")
    raw["candidate_ontology"] = raw.get("Ontology", "")
    raw["candidate_inchikey"] = raw.get("InChIKey", "")
    raw["candidate_smiles"] = raw.get("SMILES", "")

    # Keep all rows, but mark valid structure candidates.
    raw["has_valid_score"] = raw["candidate_score"].fillna(-1) >= 0
    raw["has_structure_name"] = raw["candidate_name"].astype(str).str.strip().ne("")
    raw["has_inchikey"] = raw["candidate_inchikey"].astype(str).str.strip().ne("")
    raw["has_smiles"] = raw["candidate_smiles"].astype(str).str.strip().ne("")

    raw["is_valid_candidate"] = (
        raw["has_valid_score"] &
        raw["has_structure_name"] &
        (raw["has_inchikey"] | raw["has_smiles"])
    )

    raw = raw.sort_values(
        ["dataset", "source_row", "query_file_name", "candidate_score"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    raw["candidate_rank_all"] = raw.groupby(
        ["dataset", "source_row", "query_file_name"]
    ).cumcount() + 1

    valid = raw[raw["is_valid_candidate"]].copy()
    valid["candidate_rank"] = valid.groupby(
        ["dataset", "source_row", "query_file_name"]
    ).cumcount() + 1

    # Join true metadata.
    meta = pd.read_csv(args.metadata)
    meta["source_row"] = meta["source_row"].astype(int)
    meta_keep = meta[[
        "dataset", "source_row", "query_id", "name", "smiles", "inchi",
        "inchikey", "rt_sec", "precursor_mz", "precursor_type", "ion_mode",
        "formula", "n_peaks"
    ]].copy()

    meta_keep = meta_keep.rename(columns={
        "name": "true_name",
        "smiles": "true_smiles",
        "inchi": "true_inchi",
        "inchikey": "true_inchikey",
        "formula": "true_formula",
        "precursor_mz": "query_precursor_mz",
        "precursor_type": "query_precursor_type",
        "ion_mode": "query_ion_mode",
        "n_peaks": "query_n_peaks",
    })

    valid = valid.merge(meta_keep, on=["dataset", "source_row"], how="left")
    raw = raw.merge(meta_keep, on=["dataset", "source_row"], how="left")

    valid["candidate_inchikey14"] = valid["candidate_inchikey"].map(inchikey14)
    valid["true_inchikey14"] = valid["true_inchikey"].map(inchikey14)
    valid["is_true"] = (
        valid["candidate_inchikey14"].ne("") &
        valid["candidate_inchikey14"].eq(valid["true_inchikey14"])
    )

    raw["candidate_inchikey14"] = raw["candidate_inchikey"].map(inchikey14)
    raw["true_inchikey14"] = raw["true_inchikey"].map(inchikey14)
    raw["is_true"] = (
        raw["candidate_inchikey14"].ne("") &
        raw["candidate_inchikey14"].eq(raw["true_inchikey14"])
    )

    # Save.
    raw_out = out_dir / "msfinder_structure_rows_all.csv"
    valid_out = out_dir / "msfinder_candidates_valid.csv"
    summary_out = out_dir / "msfinder_candidate_summary_by_dataset.csv"

    raw.to_csv(raw_out, index=False)
    valid.to_csv(valid_out, index=False)

    summaries = []
    for dataset, m in meta_keep.groupby("dataset"):
        v = valid[valid["dataset"] == dataset].copy()
        n_queries_total = m[["dataset", "source_row"]].drop_duplicates().shape[0]
        n_queries_with_candidate = v[["dataset", "source_row"]].drop_duplicates().shape[0]
        n_valid_candidates = len(v)
        n_with_smiles = int(v["has_smiles"].sum()) if len(v) else 0

        cand_counts = v.groupby("source_row").size() if len(v) else pd.Series(dtype=float)
        true_by_query = v.groupby("source_row")["is_true"].any() if len(v) else pd.Series(dtype=bool)

        def topk_acc(k):
            if len(v) == 0:
                return np.nan
            topk = v[v["candidate_rank"] <= k]
            hit = topk.groupby("source_row")["is_true"].any()
            # denominator uses all queries in metadata, missing true = false.
            return float(hit.reindex(m["source_row"].unique(), fill_value=False).mean() * 100.0)

        summaries.append({
            "dataset": dataset,
            "n_queries_total": int(n_queries_total),
            "n_queries_with_candidate": int(n_queries_with_candidate),
            "query_candidate_coverage_pct": float(n_queries_with_candidate / n_queries_total * 100.0) if n_queries_total else np.nan,
            "n_valid_candidates": int(n_valid_candidates),
            "n_candidates_with_smiles": int(n_with_smiles),
            "smiles_coverage_pct": float(n_with_smiles / n_valid_candidates * 100.0) if n_valid_candidates else np.nan,
            "mean_candidates_per_query_with_candidates": float(cand_counts.mean()) if len(cand_counts) else 0.0,
            "median_candidates_per_query_with_candidates": float(cand_counts.median()) if len(cand_counts) else 0.0,
            "max_candidates_per_query": int(cand_counts.max()) if len(cand_counts) else 0,
            "n_queries_true_in_candidates": int(true_by_query.sum()) if len(true_by_query) else 0,
            "true_in_candidates_pct_all_queries": float(true_by_query.reindex(m["source_row"].unique(), fill_value=False).mean() * 100.0) if n_queries_total else np.nan,
            "msfinder_top1_acc_pct_all_queries": topk_acc(1),
            "msfinder_top5_acc_pct_all_queries": topk_acc(5),
            "msfinder_top10_acc_pct_all_queries": topk_acc(10),
        })

    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_out, index=False)

    print("=" * 80)
    print("Saved:", raw_out, raw.shape)
    print("Saved:", valid_out, valid.shape)
    print("Saved:", summary_out, summary.shape)
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\n[Valid candidate examples]")
    cols = [
        "dataset", "query_file_name", "true_name", "candidate_rank",
        "candidate_name", "candidate_score", "candidate_formula",
        "candidate_inchikey", "candidate_smiles", "is_true"
    ]
    show_cols = [c for c in cols if c in valid.columns]
    print(valid[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

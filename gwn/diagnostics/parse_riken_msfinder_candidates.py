#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

from rdkit import Chem

STRUCT = Path("experiments_candidate_filtering/msfinder_exports/riken/Structure result-2080.txt")
FORMULA = Path("experiments_candidate_filtering/msfinder_exports/riken/Formula result-2080.txt")
META = Path("experiments_candidate_filtering/msfinder_queries/msfinder_query_metadata.csv")

OUT = Path("experiments_candidate_filtering/riken_parsed")
OUT.mkdir(parents=True, exist_ok=True)


def norm_text(x):
    return str(x).strip()


def norm_lower(x):
    return str(x).strip().lower()


def file_stem(x):
    s = str(x).replace("\\", "/").strip()
    base = s.split("/")[-1]
    if base.lower().endswith(".msp"):
        base = base[:-4]
    return base


def inchikey14(x):
    s = str(x).strip()
    if len(s) >= 14:
        return s[:14]
    return ""


def smiles_ok(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return m is not None
    except Exception:
        return False


def make_meta_stem(row):
    for c in ["msp_file", "source_file", "name"]:
        if c in row and str(row[c]).strip():
            v = str(row[c]).strip()
            if c in ["msp_file", "source_file"]:
                return file_stem(v)
    return str(row.get("name", "")).strip()


def main():
    print("=" * 100)
    print("[input files]")
    print("STRUCT:", STRUCT, "exists:", STRUCT.exists())
    print("FORMULA:", FORMULA, "exists:", FORMULA.exists())
    print("META:", META, "exists:", META.exists())
    print("=" * 100)

    struct = pd.read_csv(STRUCT, sep="\t", dtype=str, low_memory=False, encoding="utf-8-sig").fillna("")
    meta = pd.read_csv(META, dtype=str, low_memory=False).fillna("")

    meta = meta[meta["dataset"].astype(str).str.lower().str.contains("riken")].copy()

    print("[raw counts]")
    print("structure rows:", len(struct))
    print("structure unique File name:", struct["File name"].nunique())
    print("metadata rows:", len(meta))
    print("metadata unique query_id:", meta["query_id"].nunique())
    print("=" * 100)

    struct = struct.reset_index(drop=False).rename(columns={"index": "raw_row_index"})
    struct["raw_file_stem"] = struct["File name"].map(file_stem)
    struct["raw_title_norm"] = struct["Title"].map(norm_lower)

    meta = meta.reset_index(drop=True)
    meta["meta_file_stem"] = meta.apply(make_meta_stem, axis=1)
    meta["meta_name_norm"] = meta["name"].map(norm_lower)
    meta["true_inchikey"] = meta["inchikey"].map(norm_text)
    meta["true_inchikey14"] = meta["true_inchikey"].map(inchikey14)

    # 优先按 msp file stem 匹配
    meta_stem_cols = [
        "query_id", "dataset", "name", "smiles", "inchi", "inchikey",
        "formula", "rt_sec", "rt_min_written", "precursor_mz",
        "precursor_type", "ion_mode", "n_peaks", "msp_file",
        "meta_file_stem", "meta_name_norm", "true_inchikey", "true_inchikey14"
    ]
    meta_stem_cols = [c for c in meta_stem_cols if c in meta.columns]

    merged = struct.merge(
        meta[meta_stem_cols],
        left_on="raw_file_stem",
        right_on="meta_file_stem",
        how="left",
        suffixes=("", "_meta")
    )

    # 对未匹配的，再按 Title/name 小写匹配兜底
    unmatched = merged["query_id"].isna()
    if unmatched.any():
        title_map = meta.drop_duplicates("meta_name_norm").set_index("meta_name_norm")
        fill_cols = [c for c in meta_stem_cols if c not in ["meta_file_stem"]]
        for idx in merged[unmatched].index:
            key = merged.at[idx, "raw_title_norm"]
            if key in title_map.index:
                for c in fill_cols:
                    if c in title_map.columns:
                        merged.at[idx, c] = title_map.at[key, c]

    print("[matching]")
    print("matched rows:", int(merged["query_id"].notna().sum()), "/", len(merged))
    print("matched unique queries:", merged.loc[merged["query_id"].notna(), "query_id"].nunique())
    print("unmatched raw File name:", merged.loc[merged["query_id"].isna(), "raw_file_stem"].nunique())

    unmatched_names = sorted(merged.loc[merged["query_id"].isna(), "raw_file_stem"].dropna().unique().tolist())[:50]
    if unmatched_names:
        print("unmatched examples:", unmatched_names)

    # 标准化候选列
    out = pd.DataFrame()
    out["dataset"] = "RIKEN_MONA"
    out["query_id"] = merged["query_id"].astype(str)
    out["query_name"] = merged["name"].astype(str)
    out["query_rt_sec"] = pd.to_numeric(merged["rt_sec"], errors="coerce")
    out["query_smiles"] = merged["smiles"].astype(str)
    out["query_formula"] = merged["formula"].astype(str)
    out["query_inchikey"] = merged["true_inchikey"].astype(str)
    out["query_inchikey14"] = merged["true_inchikey14"].astype(str)

    out["raw_file_stem"] = merged["raw_file_stem"].astype(str)
    out["raw_title"] = merged["Title"].astype(str)
    out["precursor_mz"] = merged["Precursor mz"].astype(str)
    out["precursor_type"] = merged["Precursor type"].astype(str)

    out["candidate_rank_raw"] = merged.groupby("raw_file_stem").cumcount() + 1
    out["candidate_name"] = merged["Structure"].astype(str)
    out["candidate_score"] = pd.to_numeric(merged["Total score"], errors="coerce")
    out["candidate_formula"] = merged["Formula"].astype(str)
    out["candidate_ontology"] = merged["Ontology"].astype(str)
    out["candidate_inchikey"] = merged["InChIKey"].astype(str)
    out["candidate_inchikey14"] = out["candidate_inchikey"].map(inchikey14)
    out["candidate_smiles"] = merged["SMILES"].astype(str)
    out["candidate_databases"] = merged["Databases"].astype(str)
    out["raw_row_index"] = merged["raw_row_index"]

    out["has_query_match"] = out["query_id"].astype(str).ne("") & out["query_id"].astype(str).ne("nan")
    out["has_score"] = out["candidate_score"].notna()
    out["has_smiles"] = out["candidate_smiles"].astype(str).str.len().gt(0)
    out["has_inchikey"] = out["candidate_inchikey"].astype(str).str.len().gt(0)
    out["rdkit_smiles_ok"] = out["candidate_smiles"].map(smiles_ok)

    out["is_true_full_inchikey"] = (
        out["candidate_inchikey"].astype(str).eq(out["query_inchikey"].astype(str))
        & out["query_inchikey"].astype(str).str.len().gt(0)
    )
    out["is_true_inchikey14"] = (
        out["candidate_inchikey14"].astype(str).eq(out["query_inchikey14"].astype(str))
        & out["query_inchikey14"].astype(str).str.len().gt(0)
    )

    # 用 InChIKey 前14位作为主判定，避免立体/盐型差异导致真实结构漏判
    out["is_true"] = out["is_true_inchikey14"]

    all_csv = OUT / "riken_candidates_all_from_structure_result.csv"
    out.to_csv(all_csv, index=False)

    valid = out[
        out["has_query_match"]
        & out["has_score"]
        & out["has_smiles"]
        & out["has_inchikey"]
        & out["rdkit_smiles_ok"]
    ].copy()

    # 按每个 query 内 MS-FINDER score 重新排序
    valid = valid.sort_values(
        ["query_id", "candidate_score", "raw_row_index"],
        ascending=[True, False, True]
    ).reset_index(drop=True)
    valid["candidate_rank"] = valid.groupby("query_id").cumcount() + 1

    valid_csv = OUT / "riken_candidates_valid.csv"
    valid.to_csv(valid_csv, index=False)

    # query-level coverage
    rows = []
    for qid, sub in valid.groupby("query_id"):
        true_sub = sub[sub["is_true"]].sort_values("candidate_rank")
        rows.append({
            "query_id": qid,
            "query_name": sub["query_name"].iloc[0],
            "query_rt_sec": sub["query_rt_sec"].iloc[0],
            "query_formula": sub["query_formula"].iloc[0],
            "query_inchikey": sub["query_inchikey"].iloc[0],
            "n_valid_candidates": len(sub),
            "true_in_valid": bool(len(true_sub)),
            "true_rank": int(true_sub["candidate_rank"].iloc[0]) if len(true_sub) else np.nan,
            "top1_before": bool(len(true_sub) and true_sub["candidate_rank"].iloc[0] <= 1),
            "top5_before": bool(len(true_sub) and true_sub["candidate_rank"].iloc[0] <= 5),
            "top10_before": bool(len(true_sub) and true_sub["candidate_rank"].iloc[0] <= 10),
        })

    qsum = pd.DataFrame(rows)
    qsum_csv = OUT / "riken_query_candidate_coverage_summary.csv"
    qsum.to_csv(qsum_csv, index=False)

    # metadata matched
    meta_out = meta.copy()
    meta_out["candidate_evaluable"] = meta_out["query_id"].isin(
        set(qsum.loc[qsum["true_in_valid"], "query_id"])
    )
    meta_out.to_csv(OUT / "riken_metadata_with_candidate_evaluable_flag.csv", index=False)

    print("=" * 100)
    print("[RIKEN parsed candidate summary]")
    print("all candidate rows:", len(out))
    print("valid candidate rows:", len(valid))
    print("valid unique queries:", valid["query_id"].nunique())
    print("queries with true candidate:", int(qsum["true_in_valid"].sum()), "/", len(qsum))

    if len(qsum):
        eval_q = qsum[qsum["true_in_valid"]].copy()
        print("\n[MS-FINDER original on true-in-valid RIKEN queries]")
        print("N:", len(eval_q))
        print("Top1:", round(100 * eval_q["top1_before"].mean(), 4))
        print("Top5:", round(100 * eval_q["top5_before"].mean(), 4))
        print("Top10:", round(100 * eval_q["top10_before"].mean(), 4))
        print("median true rank:", eval_q["true_rank"].median())
        print("mean candidates/query:", round(eval_q["n_valid_candidates"].mean(), 4))

    print("\n[coverage head]")
    show_cols = ["query_id", "query_name", "query_rt_sec", "n_valid_candidates", "true_in_valid", "true_rank"]
    print(qsum[show_cols].head(80).to_string(index=False))

    print("\n[problem queries: no true in valid]")
    print(qsum[~qsum["true_in_valid"]][show_cols].head(80).to_string(index=False))

    print("\nSaved:")
    print(all_csv)
    print(valid_csv)
    print(qsum_csv)
    print(OUT / "riken_metadata_with_candidate_evaluable_flag.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

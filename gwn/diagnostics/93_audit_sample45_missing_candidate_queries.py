#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from rdkit import Chem

SPLIT = Path("experiments_candidate_filtering/metabobase_tl_sample45/seed42/metabobase_test45_sample45_metadata.csv")
ALL = Path("experiments_candidate_filtering/parsed_candidates/msfinder_structure_rows_all.csv")
VALID = Path("experiments_candidate_filtering/parsed_candidates/msfinder_candidates_valid.csv")
PRED = Path("experiments_candidate_filtering/metabobase_sample45_predictions_tl_seed42/sample45_candidate_predictions_tl_seed42.csv")

OUT = Path("experiments_candidate_filtering/metabobase_sample45_missing_candidate_audit")
OUT.mkdir(parents=True, exist_ok=True)


def canon_ok(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return m is not None
    except Exception:
        return False


def main():
    meta = pd.read_csv(SPLIT, dtype=str).fillna("")
    meta["rt_sec_float"] = pd.to_numeric(meta["rt_sec"], errors="coerce")

    all_df = pd.read_csv(ALL, dtype=str, low_memory=False).fillna("")
    valid_df = pd.read_csv(VALID, dtype=str, low_memory=False).fillna("")
    pred_df = pd.read_csv(PRED, dtype=str, low_memory=False).fillna("")

    rows = []
    detail_rows = []

    for _, r in meta.iterrows():
        qid = str(r["query_id"])
        ik = str(r["inchikey"])
        rt = float(r["rt_sec_float"])

        a = all_df[all_df["query_id"].astype(str).eq(qid)].copy()
        v = valid_df[valid_df["query_id"].astype(str).eq(qid)].copy()
        p = pred_df[pred_df["query_id"].astype(str).eq(qid)].copy()

        if len(a):
            a["candidate_smiles_rdkit_ok"] = a["candidate_smiles"].map(canon_ok) if "candidate_smiles" in a.columns else False
            a["is_valid_candidate_str"] = a["is_valid_candidate"].astype(str) if "is_valid_candidate" in a.columns else ""
            a["has_valid_score_str"] = a["has_valid_score"].astype(str) if "has_valid_score" in a.columns else ""
            a["has_structure_name_str"] = a["has_structure_name"].astype(str) if "has_structure_name" in a.columns else ""
            a["has_inchikey_str"] = a["has_inchikey"].astype(str) if "has_inchikey" in a.columns else ""
            a["has_smiles_str"] = a["has_smiles"].astype(str) if "has_smiles" in a.columns else ""
        else:
            a["candidate_smiles_rdkit_ok"] = []

        rows.append({
            "query_id": qid,
            "name": r.get("name", ""),
            "inchikey": ik,
            "rt_sec": rt,
            "rt_lt_300": rt < 300,
            "sample45_role": r.get("sample45_role", ""),
            "n_rows_all": len(a),
            "n_rows_valid": len(v),
            "n_rows_final_pred": len(p),
            "true_in_all": bool(a["is_true"].astype(str).str.lower().isin(["true", "1"]).any()) if len(a) and "is_true" in a.columns else False,
            "true_in_valid": bool(v["is_true"].astype(str).str.lower().isin(["true", "1"]).any()) if len(v) and "is_true" in v.columns else False,
            "true_in_final_pred": bool(p["is_true"].astype(str).str.lower().isin(["true", "1"]).any()) if len(p) and "is_true" in p.columns else False,
            "n_rdkit_ok_in_all": int(a["candidate_smiles_rdkit_ok"].sum()) if len(a) else 0,
            "n_is_valid_candidate_true_in_all": int(a["is_valid_candidate"].astype(str).str.lower().isin(["true", "1"]).sum()) if len(a) and "is_valid_candidate" in a.columns else 0,
        })

        if len(a) and (len(v) == 0 or len(p) == 0 or not bool(p["is_true"].astype(str).str.lower().isin(["true", "1"]).any())):
            keep_cols = [
                "query_id", "true_name", "rt_sec",
                "candidate_rank_all", "candidate_rank",
                "candidate_name", "candidate_score",
                "candidate_inchikey", "candidate_smiles",
                "has_valid_score", "has_structure_name", "has_inchikey", "has_smiles",
                "is_valid_candidate", "is_true",
                "candidate_smiles_rdkit_ok",
            ]
            keep_cols = [c for c in keep_cols if c in a.columns]
            tmp = a[keep_cols].head(50).copy()
            tmp.insert(0, "__audit_query_name", r.get("name", ""))
            tmp.insert(1, "__audit_query_rt_sec", rt)
            detail_rows.append(tmp)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "sample45_candidate_coverage_audit_summary.csv", index=False)

    if detail_rows:
        detail = pd.concat(detail_rows, axis=0, ignore_index=True)
    else:
        detail = pd.DataFrame()
    detail.to_csv(OUT / "sample45_problem_query_candidate_rows_from_all.csv", index=False)

    print("=" * 100)
    print("[coverage audit summary]")
    print(summary.to_string(index=False))
    print("=" * 100)

    print("\n[queries missing from final prediction]")
    print(summary[summary["n_rows_final_pred"].eq(0)].to_string(index=False))

    print("\n[queries without true in final prediction]")
    print(summary[~summary["true_in_final_pred"]].to_string(index=False))

    print("\n[problem candidate rows from ALL]")
    if len(detail):
        print(detail.head(120).to_string(index=False))
    else:
        print("NO DETAIL ROWS")

    print("\nSaved:")
    print(OUT / "sample45_candidate_coverage_audit_summary.csv")
    print(OUT / "sample45_problem_query_candidate_rows_from_all.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

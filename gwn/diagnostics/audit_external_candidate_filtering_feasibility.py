#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

try:
    from rdkit.Chem import inchi
    HAS_INCHI = True
except Exception:
    HAS_INCHI = False


OUT_DIR = Path("diagnostics/external_candidate_filtering_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def mol_from_inchi(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if not x or x.lower() == "nan":
        return None
    try:
        return Chem.MolFromInchi(x, sanitize=True)
    except Exception:
        return None


def mol_from_smiles(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if not x or x.lower() == "nan":
        return None
    try:
        return Chem.MolFromSmiles(x)
    except Exception:
        return None


def formula_from_mol(mol):
    if mol is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


def inchikey_from_mol(mol):
    if mol is None or not HAS_INCHI:
        return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def load_massbank():
    rows = []
    for name in ["MassBank1", "MassBank2"]:
        p = Path(f"external_data/raw_libraries/{name}.csv")
        df = pd.read_csv(p)
        df["query_source"] = name
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df = df.rename(columns={"RT": "query_rt"})
    mols = [mol_from_inchi(x) for x in df["InChI"]]
    df["query_formula"] = [formula_from_mol(m) for m in mols]
    df["query_inchikey"] = [inchikey_from_mol(m) for m in mols]
    df["query_id"] = [f"Q{i:05d}" for i in range(len(df))]
    return df


def load_riken():
    p = Path("external_data/raw_libraries/RIKEN_MONA.xlsx")
    xl = pd.ExcelFile(p)
    df = pd.read_excel(p, sheet_name=xl.sheet_names[0])
    out = pd.DataFrame()
    out["candidate_source"] = "RIKEN_MONA"
    out["candidate_formula"] = df.get("molecular formula")
    out["candidate_inchikey"] = df.get("InChIKey")
    out["candidate_smiles"] = df.get("smiles")
    out["candidate_inchi"] = df.get("InChI")
    out["candidate_rt"] = df.get("RT")
    return out


def load_metabobase():
    p = Path("external_data/raw_libraries/MetaboBase.csv")
    df = pd.read_csv(p)
    out = pd.DataFrame()
    out["candidate_source"] = "MetaboBase"
    out["candidate_formula"] = df.get("Formula")
    out["candidate_inchikey"] = df.get("InChIKey")
    out["candidate_smiles"] = df.get("computedSMILES").fillna(df.get("SMILES"))
    out["candidate_inchi"] = df.get("InChI")
    out["candidate_rt"] = df.get("RT")
    return out


def clean_key(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return None
    return x


def audit_one(query_df, cand_df, label):
    q = query_df.copy()
    c = cand_df.copy()

    q["query_formula"] = q["query_formula"].map(clean_key)
    q["query_inchikey"] = q["query_inchikey"].map(clean_key)
    c["candidate_formula"] = c["candidate_formula"].map(clean_key)
    c["candidate_inchikey"] = c["candidate_inchikey"].map(clean_key)

    q_valid = q.dropna(subset=["query_formula", "query_inchikey"]).copy()
    c_valid = c.dropna(subset=["candidate_formula", "candidate_inchikey"]).copy()

    # Candidate pool by formula
    pool = q_valid[["query_id", "query_source", "query_formula", "query_inchikey", "query_rt"]].merge(
        c_valid,
        left_on="query_formula",
        right_on="candidate_formula",
        how="left",
    )

    pool["is_true_candidate"] = pool["query_inchikey"] == pool["candidate_inchikey"]

    per_query = (
        pool.groupby(["query_id", "query_source", "query_formula", "query_inchikey"], dropna=False)
        .agg(
            pool_n=("candidate_inchikey", lambda x: x.notna().sum()),
            true_in_pool=("is_true_candidate", "max"),
        )
        .reset_index()
    )

    rows = []
    for min_pool in [1, 2, 5, 10]:
        sub = per_query[per_query["pool_n"] >= min_pool]
        if len(sub) == 0:
            rows.append({
                "candidate_library": label,
                "min_pool_size": min_pool,
                "n_queries_valid": len(q_valid),
                "eligible_queries": 0,
                "eligible_ratio": 0.0,
                "true_in_pool_queries": 0,
                "true_in_pool_rate_among_eligible": np.nan,
                "pool_n_mean": np.nan,
                "pool_n_median": np.nan,
                "pool_n_max": np.nan,
            })
        else:
            rows.append({
                "candidate_library": label,
                "min_pool_size": min_pool,
                "n_queries_valid": len(q_valid),
                "eligible_queries": len(sub),
                "eligible_ratio": len(sub) / len(q_valid),
                "true_in_pool_queries": int(sub["true_in_pool"].sum()),
                "true_in_pool_rate_among_eligible": float(sub["true_in_pool"].mean()),
                "pool_n_mean": float(sub["pool_n"].mean()),
                "pool_n_median": float(sub["pool_n"].median()),
                "pool_n_max": int(sub["pool_n"].max()),
            })

    summary = pd.DataFrame(rows)

    pool.to_csv(OUT_DIR / f"{label}_formula_matched_pool.csv", index=False)
    per_query.to_csv(OUT_DIR / f"{label}_per_query_pool_audit.csv", index=False)

    return summary


def main():
    q = load_massbank()
    riken = load_riken()
    metabo = load_metabobase()
    merged = pd.concat([riken, metabo], ignore_index=True)

    q.to_csv(OUT_DIR / "massbank_queries_with_formula_inchikey.csv", index=False)
    riken.to_csv(OUT_DIR / "riken_candidates_normalized.csv", index=False)
    metabo.to_csv(OUT_DIR / "metabobase_candidates_normalized.csv", index=False)
    merged.to_csv(OUT_DIR / "merged_candidates_normalized.csv", index=False)

    summaries = []
    summaries.append(audit_one(q, riken, "RIKEN_MONA"))
    summaries.append(audit_one(q, metabo, "MetaboBase"))
    summaries.append(audit_one(q, merged, "RIKEN_MONA_plus_MetaboBase"))

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT_DIR / "external_candidate_filtering_feasibility_summary.csv", index=False)

    print("\n=== MassBank query parsing ===")
    print("n_queries:", len(q))
    print("formula parsed:", q["query_formula"].notna().sum())
    print("inchikey parsed:", q["query_inchikey"].notna().sum())

    print("\n=== Candidate filtering feasibility ===")
    print(summary.to_string(index=False))
    print("\n[SAVE]", OUT_DIR / "external_candidate_filtering_feasibility_summary.csv")


if __name__ == "__main__":
    main()

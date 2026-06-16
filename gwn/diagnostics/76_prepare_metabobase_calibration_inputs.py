#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception as e:
    Chem = None
    rdMolStandardize = None
    print("[WARNING] RDKit unavailable:", e)


META = Path("experiments_candidate_filtering/msfinder_queries/msfinder_query_metadata.csv")
S10 = Path("experiments_candidate_filtering/metabobase_s10_subset/metabobase_s10_test_queries_matched_metadata.csv")
OUT = Path("experiments_candidate_filtering/metabobase_calibration_prediction_inputs")
OUT.mkdir(parents=True, exist_ok=True)


def canon_smiles(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return ""
    if Chem is None:
        return s
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def tautomer_strict_smiles(s):
    s = canon_smiles(s)
    if not s or Chem is None or rdMolStandardize is None:
        return s
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s
    try:
        te = rdMolStandardize.TautomerEnumerator()
        tmol = te.Canonicalize(mol)
        return Chem.MolToSmiles(tmol, isomericSmiles=True)
    except Exception:
        return s


meta = pd.read_csv(META)
meta = meta[meta["dataset"].eq("MetaboBase")].copy()

s10 = pd.read_csv(S10)
test_source_rows = set(s10["source_row"].astype(int).tolist())

meta["source_row"] = meta["source_row"].astype(int)
meta["split"] = meta["source_row"].map(lambda x: "s10_test_matched" if x in test_source_rows else "calib_train")

meta["true_smiles_canon"] = meta["smiles"].map(canon_smiles)
bad = meta["true_smiles_canon"].eq("")
if bad.any():
    print("[WARNING] invalid true smiles:", int(bad.sum()))
meta = meta[~bad].copy()

uniq = (
    meta[["true_smiles_canon"]]
    .drop_duplicates()
    .sort_values("true_smiles_canon")
    .reset_index(drop=True)
)
uniq["candidate_uid"] = range(len(uniq))
uniq["candidate_smiles_canon"] = uniq["true_smiles_canon"]
uniq["taut_smiles"] = uniq["candidate_smiles_canon"].map(tautomer_strict_smiles)
uniq["taut_changed"] = (uniq["candidate_smiles_canon"] != uniq["taut_smiles"]).astype(int)

row_map = meta.merge(
    uniq[["true_smiles_canon", "candidate_uid", "candidate_smiles_canon", "taut_smiles", "taut_changed"]],
    on="true_smiles_canon",
    how="left",
)

# 兼容 74 脚本后处理字段
row_map["s10_row"] = row_map["source_row"]
row_map["candidate_rank"] = 1
row_map["candidate_score"] = 999.0
row_map["candidate_name"] = row_map["name"]
row_map["is_true"] = True
row_map["true_name"] = row_map["name"]
row_map["true_formula"] = row_map["formula"]
row_map["true_inchikey"] = row_map["inchikey"]

origin = pd.DataFrame({
    "smiles": uniq["candidate_smiles_canon"],
    "rt": 999.0,
})
taut = pd.DataFrame({
    "smiles": uniq["taut_smiles"],
    "rt": 999.0,
})

uniq.to_csv(OUT / "metabobase_calibration_true_unique_smiles.csv", index=False)
origin.to_csv(OUT / "metabobase_calibration_true_origin.csv", index=False)
taut.to_csv(OUT / "metabobase_calibration_true_taut_strict.csv", index=False)
row_map.to_csv(OUT / "metabobase_calibration_true_row_map.csv", index=False)

print("=" * 100)
print("MetaboBase metadata rows:", len(meta))
print("calib_train rows:", int((meta["split"] == "calib_train").sum()))
print("s10_test_matched rows:", int((meta["split"] == "s10_test_matched").sum()))
print("unique true molecules:", len(uniq))
print("tautomer changed unique:", int(uniq["taut_changed"].sum()))
print("saved dir:", OUT)
print("=" * 100)

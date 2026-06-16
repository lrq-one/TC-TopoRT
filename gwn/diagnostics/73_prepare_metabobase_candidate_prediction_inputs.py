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
    print("[WARNING] RDKit tautomer canonicalization unavailable:", e)


IN_CAND = Path("experiments_candidate_filtering/metabobase_s10_subset/metabobase_s10_candidates_valid.csv")
OUT_DIR = Path("experiments_candidate_filtering/metabobase_s10_prediction_inputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_UNIQUE = OUT_DIR / "metabobase_s10_unique_candidate_smiles.csv"
OUT_ORIGIN = OUT_DIR / "metabobase_s10_candidate_origin.csv"
OUT_TAUT = OUT_DIR / "metabobase_s10_candidate_taut_strict.csv"
OUT_MAP = OUT_DIR / "metabobase_s10_candidate_row_map.csv"


def canon_smiles(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    if not s:
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


df = pd.read_csv(IN_CAND)

# 只保留有 smiles 的候选；你的结果里 has_smiles=100%，这里再保险一次
df["candidate_smiles"] = df["candidate_smiles"].astype(str).str.strip()
df = df[df["candidate_smiles"].ne("") & df["candidate_smiles"].ne("nan")].copy()

df["candidate_smiles_canon"] = df["candidate_smiles"].map(canon_smiles)
bad = df["candidate_smiles_canon"].eq("")
if bad.any():
    print("[WARNING] invalid candidate SMILES:", int(bad.sum()))
df = df[~bad].copy()

# 去重：同一个候选 SMILES 只预测一次，后面再 merge 回所有 query-candidate 行
uniq = (
    df[["candidate_smiles_canon"]]
    .drop_duplicates()
    .sort_values("candidate_smiles_canon")
    .reset_index(drop=True)
)
uniq["candidate_uid"] = range(len(uniq))

df = df.merge(uniq, on="candidate_smiles_canon", how="left")

# 注意：这里 RT 只是占位，不参与预测评价。
# 你的 SMRT dataset 里之前有 rt>300 过滤，所以 dummy RT 设 999.0，避免被过滤掉。
origin = pd.DataFrame({
    "smiles": uniq["candidate_smiles_canon"],
    "rt": 999.0,
})
origin.to_csv(OUT_ORIGIN, index=False)

taut_smiles = uniq["candidate_smiles_canon"].map(tautomer_strict_smiles)
taut = pd.DataFrame({
    "smiles": taut_smiles,
    "rt": 999.0,
})
taut.to_csv(OUT_TAUT, index=False)

uniq["taut_smiles"] = taut_smiles
uniq["taut_changed"] = (uniq["candidate_smiles_canon"] != uniq["taut_smiles"]).astype(int)
uniq.to_csv(OUT_UNIQUE, index=False)

df.to_csv(OUT_MAP, index=False)

print("=" * 100)
print("input candidate rows:", len(pd.read_csv(IN_CAND)))
print("valid candidate rows:", len(df))
print("unique candidate molecules:", len(uniq))
print("tautomer changed unique:", int(uniq["taut_changed"].sum()))
print("saved:", OUT_UNIQUE)
print("saved:", OUT_ORIGIN)
print("saved:", OUT_TAUT)
print("saved:", OUT_MAP)
print("=" * 100)
print(uniq.head(10).to_string(index=False))

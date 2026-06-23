#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path.cwd()
OUT = ROOT / "manuscript_figures_final"
OUT.mkdir(parents=True, exist_ok=True)

def read_csv(p):
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            pass
    return None

def infer_dataset(path):
    s = str(path).lower()
    if "riken" in s or "plasma" in s:
        return "RIKEN-PlaSMA"
    if "metabo" in s:
        return "MetaboBase"
    return "Unknown"

def norm(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())

rows = []

for p in ROOT.rglob("*.csv"):
    s = str(p).lower()
    if not any(k in s for k in ["candidate", "filter", "rerank", "metabobase", "riken", "plasma"]):
        continue
    if p.stat().st_size > 200 * 1024 * 1024:
        continue

    df = read_csv(p)
    if df is None:
        continue

    cols = list(df.columns)
    ncols = [norm(c) for c in cols]
    dataset = infer_dataset(p)

    info = {
        "path": str(p),
        "dataset": dataset,
        "shape": str(df.shape),
        "columns": " | ".join(cols),
        "has_query_summary_counts": False,
        "reduction_percent": np.nan,
        "zero_after": np.nan,
        "zero_after_frac": np.nan,
        "top1": np.nan,
        "top5": np.nan,
        "top10": np.nan,
    }

    if {"n_candidates_before", "n_candidates_after"}.issubset(set(cols)):
        before = pd.to_numeric(df["n_candidates_before"], errors="coerce")
        after = pd.to_numeric(df["n_candidates_after"], errors="coerce")
        if before.sum() > 0:
            info["has_query_summary_counts"] = True
            info["reduction_percent"] = float((1 - after.sum() / before.sum()) * 100)
            info["zero_after"] = int((after.fillna(0) == 0).sum())
            info["zero_after_frac"] = float((after.fillna(0) == 0).mean())

    # Try to detect summary metric columns
    for c in cols:
        cn = norm(c)
        if cn in ["top1", "top1percent", "top1accuracy", "top1acc"]:
            info["top1"] = pd.to_numeric(df[c], errors="coerce").dropna().iloc[0] if pd.to_numeric(df[c], errors="coerce").dropna().shape[0] else np.nan
        if cn in ["top5", "top5percent", "top5accuracy", "top5acc"]:
            info["top5"] = pd.to_numeric(df[c], errors="coerce").dropna().iloc[0] if pd.to_numeric(df[c], errors="coerce").dropna().shape[0] else np.nan
        if cn in ["top10", "top10percent", "top10accuracy", "top10acc"]:
            info["top10"] = pd.to_numeric(df[c], errors="coerce").dropna().iloc[0] if pd.to_numeric(df[c], errors="coerce").dropna().shape[0] else np.nan

    rows.append(info)

res = pd.DataFrame(rows)
res.to_csv(OUT / "audit_candidate_filtering_all_csv.csv", index=False)

print("\n========== QUERY SUMMARY FILES WITH before/after ==========")
q = res[res["has_query_summary_counts"] == True].copy()
if len(q) == 0:
    print("No query_summary count files found.")
else:
    q["target_dist_metabo"] = (q["reduction_percent"] - 69.14).abs()
    q["target_dist_riken"] = (q["reduction_percent"] - 46.23).abs()
    show = q[[
        "dataset", "shape", "reduction_percent", "zero_after",
        "zero_after_frac", "path"
    ]].sort_values(["dataset", "reduction_percent"])
    print(show.to_string(index=False))

print("\n========== FILES POSSIBLY CONTAINING TOP-K SUMMARY ==========")
metric_like = res[
    res["columns"].str.lower().str.contains("top1|top_1|top-1|top5|top_5|top-5|top10|top_10|top-10|reduction", regex=True)
].copy()
if len(metric_like) == 0:
    print("No metric-like files found.")
else:
    print(metric_like[["dataset", "shape", "top1", "top5", "top10", "path", "columns"]].head(80).to_string(index=False))

print("\nSaved full audit to:")
print(OUT / "audit_candidate_filtering_all_csv.csv")

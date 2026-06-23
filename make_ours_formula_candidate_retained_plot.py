#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path.cwd()
OUT = ROOT / "manuscript_figures_abcort_style"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())

def read_csv_robust(path):
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1", "utf-16", "utf-16-le", "utf-16-be"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return None

def find_formula_col(df):
    candidates = []
    for c in df.columns:
        n = norm_col(c)
        if any(k in n for k in [
            "formula",
            "molecularformula",
            "queryformula",
            "mf",
        ]):
            candidates.append(c)
    return candidates[0] if candidates else None

def find_dataset_col(df):
    candidates = []
    for c in df.columns:
        n = norm_col(c)
        if n in ["dataset", "datasetname", "source", "database", "db", "bench", "split"]:
            candidates.append(c)
    return candidates[0] if candidates else None

def find_total_retained_cols(df):
    total_cols = []
    retained_cols = []

    for c in df.columns:
        n = norm_col(c)

        if (
            ("total" in n and "candidate" in n)
            or ("num" in n and "candidate" in n and "retain" not in n and "remain" not in n and "after" not in n)
            or n in ["total", "before", "beforefilter", "numbefore", "candidatebefore", "totalnum", "totaln"]
        ):
            total_cols.append(c)

        if (
            ("retained" in n and "candidate" in n)
            or ("retain" in n and "candidate" in n)
            or ("remaining" in n and "candidate" in n)
            or ("remain" in n and "candidate" in n)
            or ("after" in n and "candidate" in n)
            or n in ["retained", "remain", "remaining", "after", "afterfilter", "numafter", "candidateafter", "retainedn"]
        ):
            retained_cols.append(c)

    total_col = total_cols[0] if total_cols else None
    retained_col = retained_cols[0] if retained_cols else None
    return total_col, retained_col

def find_retained_flag_col(df):
    flag_cols = []
    for c in df.columns:
        n = norm_col(c)
        if any(k in n for k in [
            "retained",
            "retain",
            "isretained",
            "kept",
            "keep",
            "selected",
            "pass",
            "passed",
            "survive",
            "survived",
            "afterfilter",
            "infilter",
        ]):
            flag_cols.append(c)
    return flag_cols[0] if flag_cols else None

def to_bool_series(s):
    # Robust conversion for retained flags.
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0) > 0

    x = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y", "keep", "kept", "retained", "selected", "pass", "passed", "survived"}
    false_set = {"0", "false", "f", "no", "n", "drop", "dropped", "removed", "fail", "failed", "nan", "none", ""}

    out = x.map(lambda v: True if v in true_set else (False if v in false_set else np.nan))
    return out.fillna(False).astype(bool)

def dataset_group_from_text(x):
    s = str(x).lower()
    if "riken" in s or "plasma" in s:
        return "RIKEN-PlaSMA"
    if "metabo" in s or "monadb" in s or "mona" in s:
        return "MetaboBase"
    return "CandidateSet"

def try_direct_count_table(path, df):
    formula_col = find_formula_col(df)
    dataset_col = find_dataset_col(df)
    total_col, retained_col = find_total_retained_cols(df)

    if not formula_col or not total_col or not retained_col:
        return None

    out = df[[formula_col, total_col, retained_col] + ([dataset_col] if dataset_col else [])].copy()
    out = out.rename(columns={
        formula_col: "formula",
        total_col: "total_candidates",
        retained_col: "retained_candidates",
    })
    if dataset_col:
        out = out.rename(columns={dataset_col: "dataset"})
    else:
        out["dataset"] = str(path)

    out["formula"] = out["formula"].astype(str)
    out["dataset"] = out["dataset"].astype(str).map(dataset_group_from_text)
    out["total_candidates"] = pd.to_numeric(out["total_candidates"], errors="coerce")
    out["retained_candidates"] = pd.to_numeric(out["retained_candidates"], errors="coerce")

    out = out.dropna(subset=["formula", "total_candidates", "retained_candidates"])
    out = out[(out["total_candidates"] > 0) & (out["retained_candidates"] >= 0)]
    out = out[out["total_candidates"] >= out["retained_candidates"]]

    if len(out) < 5:
        return None

    out["source_file"] = str(path)
    out["source_mode"] = "direct_total_retained_columns"
    return out

def try_candidate_row_table(path, df):
    formula_col = find_formula_col(df)
    dataset_col = find_dataset_col(df)
    flag_col = find_retained_flag_col(df)

    if not formula_col or not flag_col:
        return None

    tmp = df[[formula_col, flag_col] + ([dataset_col] if dataset_col else [])].copy()
    tmp = tmp.rename(columns={formula_col: "formula", flag_col: "retained_flag"})

    if dataset_col:
        tmp = tmp.rename(columns={dataset_col: "dataset"})
    else:
        tmp["dataset"] = str(path)

    tmp["formula"] = tmp["formula"].astype(str)
    tmp["dataset"] = tmp["dataset"].astype(str).map(dataset_group_from_text)
    tmp["is_retained"] = to_bool_series(tmp["retained_flag"])

    out = (
        tmp.groupby(["dataset", "formula"], as_index=False)
           .agg(total_candidates=("formula", "size"),
                retained_candidates=("is_retained", "sum"))
    )

    out = out[(out["total_candidates"] > 0) & (out["retained_candidates"] >= 0)]
    out = out[out["total_candidates"] >= out["retained_candidates"]]

    if len(out) < 5:
        return None

    out["source_file"] = str(path)
    out["source_mode"] = f"row_level_candidates_flag_col={flag_col}"
    return out

def collect_candidate_tables():
    # 1. Prefer manually prepared clean table.
    preferred = [
        ROOT / "candidate_formula_counts.csv",
        ROOT / "paper_results_TCDV_TopoRT" / "candidate_filtering" / "candidate_formula_counts.csv",
        ROOT / "candidate_filtering" / "candidate_formula_counts.csv",
    ]

    results = []
    checked = []

    for p in preferred:
        if p.exists():
            df = read_csv_robust(p)
            if df is not None:
                checked.append((p, list(df.columns), df.shape))
                t = try_direct_count_table(p, df)
                if t is None:
                    t = try_candidate_row_table(p, df)
                if t is not None:
                    print("[USE preferred]", p)
                    results.append(t)

    # 2. Search all relevant CSV files.
    keywords = ["candidate", "filter", "rerank", "metabobase", "riken", "plasma", "mona"]
    csv_files = []
    for p in ROOT.rglob("*.csv"):
        s = str(p).lower()
        if any(k in s for k in keywords):
            if p.stat().st_size < 200 * 1024 * 1024:
                csv_files.append(p)

    for p in sorted(set(csv_files)):
        df = read_csv_robust(p)
        if df is None:
            continue

        checked.append((p, list(df.columns), df.shape))

        t = try_direct_count_table(p, df)
        if t is None:
            t = try_candidate_row_table(p, df)

        if t is not None:
            print("[FOUND usable candidate formula table]", p)
            results.append(t)

    # Save scanned columns for debugging.
    debug_rows = []
    for p, cols, shape in checked:
        debug_rows.append({
            "path": str(p),
            "shape": str(shape),
            "columns": " | ".join(map(str, cols)),
        })
    pd.DataFrame(debug_rows).to_csv(OUT / "debug_scanned_candidate_csv_columns.csv", index=False)

    return results

def make_plot(df):
    # Normalize dataset labels.
    df["dataset"] = df["dataset"].map(dataset_group_from_text)

    # Keep only the two target groups if present.
    if df["dataset"].isin(["RIKEN-PlaSMA", "MetaboBase"]).any():
        df = df[df["dataset"].isin(["RIKEN-PlaSMA", "MetaboBase"])].copy()

    # Aggregate duplicates from multiple files.
    df = (
        df.groupby(["dataset", "formula"], as_index=False)
          .agg(total_candidates=("total_candidates", "max"),
               retained_candidates=("retained_candidates", "min"))
    )

    df = df[(df["total_candidates"] > 0) & (df["retained_candidates"] >= 0)]
    df = df[df["total_candidates"] >= df["retained_candidates"]]

    if len(df) == 0:
        raise RuntimeError("No valid rows after aggregation.")

    # Sort similar to the ABCoRT figure:
    # RIKEN first, then MetaboBase; within each group sort by total candidate count.
    order = {"RIKEN-PlaSMA": 0, "MetaboBase": 1, "CandidateSet": 2}
    df["dataset_order"] = df["dataset"].map(order).fillna(9)
    df = df.sort_values(["dataset_order", "total_candidates"], ascending=[True, True]).reset_index(drop=True)

    # Limit only if extremely large.
    max_rows = 180
    if len(df) > max_rows:
        print(f"[WARN] too many formulas: {len(df)}. Keeping last {max_rows} rows for readability.")
        df = df.tail(max_rows).copy().reset_index(drop=True)

    y = np.arange(len(df))
    height = max(8.0, 0.075 * len(df))

    plt.figure(figsize=(8.2, height))

    is_riken = df["dataset"].eq("RIKEN-PlaSMA")
    total_colors = np.where(is_riken, "#cdecc8", "#d8e6f7")
    retained_colors = np.where(is_riken, "#79c77a", "#8c9eb7")

    plt.barh(y, df["total_candidates"], color=total_colors, edgecolor="none")
    plt.barh(y, df["retained_candidates"], color=retained_colors, edgecolor="none")

    plt.yticks(y, df["formula"], fontsize=5)
    plt.xlabel("Number of Candidates")
    plt.ylabel("Formula")

    handles = [
        Patch(facecolor="#cdecc8", label="Total Number of Candidates in RIKEN-PlaSMA"),
        Patch(facecolor="#79c77a", label="Retained Candidates in RIKEN-PlaSMA"),
        Patch(facecolor="#d8e6f7", label="Total Number of Candidates in MetaboBase"),
        Patch(facecolor="#8c9eb7", label="Retained Candidates in MetaboBase"),
    ]
    plt.legend(handles=handles, frameon=True, fontsize=8, loc="upper right")

    plt.tight_layout()

    pdf = OUT / "fig_candidate_formula_level_retained_ours.pdf"
    png = OUT / "fig_candidate_formula_level_retained_ours.png"
    src = OUT / "source_candidate_formula_level_retained_ours.csv"

    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    df.to_csv(src, index=False)

    print("\n[SAVE]", pdf)
    print("[SAVE]", png)
    print("[SAVE]", src)

def main():
    tables = collect_candidate_tables()

    if not tables:
        print("\n============================================================")
        print("没有生成这张 formula-level 长条图。")
        print("原因：当前目录没有检测到可用的真实 formula-level candidate 明细。")
        print("")
        print("我已经把扫描过的 CSV 列名保存到：")
        print(OUT / "debug_scanned_candidate_csv_columns.csv")
        print("")
        print("你需要找一个 CSV，至少包含下面两种格式之一：")
        print("")
        print("格式 A：")
        print("dataset,formula,total_candidates,retained_candidates")
        print("")
        print("格式 B：")
        print("dataset,formula,retained_flag")
        print("每一行是一个 candidate，retained_flag 表示是否过滤后保留。")
        print("")
        print("找到后命名为 candidate_formula_counts.csv 放到项目根目录，然后重新运行本脚本。")
        print("============================================================")
        raise SystemExit(1)

    df = pd.concat(tables, ignore_index=True)
    print("\n=== Usable rows collected ===")
    print(df[["dataset", "formula", "total_candidates", "retained_candidates", "source_mode"]].head(20).to_string(index=False))
    print("rows:", len(df))

    make_plot(df)

if __name__ == "__main__":
    main()

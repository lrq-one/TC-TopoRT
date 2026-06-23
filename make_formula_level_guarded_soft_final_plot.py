#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path.cwd()
OUT = ROOT / "manuscript_figures_final"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FINAL_SUMMARY = {
    "MetaboBase": ROOT / "ablations/gwn_cwn_structural_ablation/experiments_candidate_filtering/metabobase_evaluable45_rank_guard_soft_eval_seed42/rank_guard_soft_rerank_summary.csv",
    "RIKEN-PlaSMA": ROOT / "ablations/gwn_cwn_structural_ablation/experiments_candidate_filtering/riken_exact85_rank_guard_soft_eval_seed42/rank_guard_soft_rerank_summary.csv",
}

TARGET = {
    "MetaboBase": {
        "candidate_reduction_pct": 69.14,
        "top1_after_pct": 55.56,
        "top5_after_pct": 82.22,
        "top10_after_pct": 88.89,
    },
    "RIKEN-PlaSMA": {
        "candidate_reduction_pct": 46.23,
        "top1_after_pct": 54.12,
        "top5_after_pct": 77.65,
        "top10_after_pct": 89.41,
    },
}

def read_csv(path):
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Cannot read CSV: {path}")

def norm(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())

def infer_dataset(path):
    s = str(path).lower()
    if "riken" in s or "plasma" in s:
        return "RIKEN-PlaSMA"
    if "metabo" in s:
        return "MetaboBase"
    return "Unknown"

def select_final_summary_row(dataset):
    p = FINAL_SUMMARY[dataset]
    if not p.exists():
        raise FileNotFoundError(p)

    df = read_csv(p)

    need = [
        "candidate_reduction_pct",
        "top1_after_pct",
        "top5_after_pct",
        "top10_after_pct",
        "n_candidate_rows_before",
        "n_candidate_rows_after",
        "threshold_sec",
        "guard_k",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"{p} missing columns: {missing}")

    dist = np.zeros(len(df), dtype=float)
    for c, tv in TARGET[dataset].items():
        dist += np.abs(pd.to_numeric(df[c], errors="coerce").to_numpy(float) - tv)

    df = df.copy()
    df["target_distance"] = dist
    sel = df.sort_values("target_distance", ascending=True).iloc[0].copy()
    sel["dataset"] = dataset
    sel["summary_file"] = str(p)

    return sel, df.sort_values("target_distance").head(20)

def find_column(cols, candidates, contains_any=None):
    cols = list(cols)
    nmap = {c: norm(c) for c in cols}

    for cand in candidates:
        if cand in cols:
            return cand

    if contains_any:
        for c, n in nmap.items():
            if all(k in n for k in contains_any):
                return c

    return None

def candidate_file_score(path, df, dataset):
    cols = set(df.columns)
    score = 0

    # 必须是候选行级，不是 summary
    if df.shape[0] < 100:
        return -999

    s = str(path).lower()
    if dataset == "MetaboBase" and "metabo" in s:
        score += 20
    if dataset == "RIKEN-PlaSMA" and ("riken" in s or "plasma" in s):
        score += 20

    # 候选级关键列
    for c in ["candidate_rank", "candidate_rank_all", "candidate_score", "candidate_smiles", "candidate_formula"]:
        if c in cols:
            score += 5

    # query/formula/key
    for c in ["s10_row", "query_id", "true_name", "true_formula", "query_formula", "formula"]:
        if c in cols:
            score += 4

    # RT 误差列
    for c in ["abs_rt_delta", "abs_rt_delta_structure_consistent", "candidate_true_abs_delta", "true_abs_rt_delta"]:
        if c in cols:
            score += 6

    # 排除明显 summary 文件
    bad_words = ["summary", "query_hard_rt_filter", "query_original_msfinder_rank", "rank_guard_soft_rerank_summary", "rows_beating"]
    if any(w in s for w in bad_words):
        score -= 80

    return score

def find_candidate_row_files(dataset):
    files = []
    for p in ROOT.rglob("*.csv"):
        s = str(p).lower()

        if not any(k in s for k in ["candidate", "coverage", "detail", "riken", "metabo", "plasma"]):
            continue
        if p.stat().st_size > 300 * 1024 * 1024:
            continue

        try:
            df = read_csv(p)
        except Exception:
            continue

        sc = candidate_file_score(p, df, dataset)
        if sc > 0:
            files.append((sc, p, df.shape, list(df.columns)))

    files = sorted(files, key=lambda x: x[0], reverse=True)
    return files

def prepare_candidate_df(dataset, path):
    df = read_csv(path).copy()

    # 找 query key
    query_col = find_column(df.columns, ["s10_row", "query_id", "query_name", "true_name"])
    if query_col is None:
        raise RuntimeError(f"Cannot find query column in {path}")

    # 找 formula：优先 true/query formula，不要 candidate_formula
    formula_col = find_column(df.columns, ["true_formula", "query_formula", "formula", "Formula"])
    if formula_col is None:
        raise RuntimeError(f"Cannot find query/true formula column in {path}")

    # 找 rank
    rank_col = find_column(df.columns, ["candidate_rank", "candidate_rank_all", "rank", "msfinder_rank"])
    if rank_col is None:
        raise RuntimeError(f"Cannot find candidate rank column in {path}")

    # 找 RT delta
    delta_priority = [
        "abs_rt_delta",
        "abs_rt_delta_structure_consistent",
        "candidate_true_abs_delta",
        "candidate_delta_minus_official_err",
        "true_abs_rt_delta",
    ]
    delta_col = find_column(df.columns, delta_priority)
    if delta_col is None:
        # 模糊匹配
        for c in df.columns:
            n = norm(c)
            if "abs" in n and "rt" in n and ("delta" in n or "err" in n):
                delta_col = c
                break
    if delta_col is None:
        raise RuntimeError(f"Cannot find abs RT delta column in {path}")

    out = df[[query_col, formula_col, rank_col, delta_col]].copy()
    out = out.rename(columns={
        query_col: "query_id_internal",
        formula_col: "formula",
        rank_col: "candidate_rank",
        delta_col: "abs_rt_delta",
    })

    out["dataset"] = dataset
    out["formula"] = out["formula"].astype(str)
    out["candidate_rank"] = pd.to_numeric(out["candidate_rank"], errors="coerce")
    out["abs_rt_delta"] = pd.to_numeric(out["abs_rt_delta"], errors="coerce")

    out = out.dropna(subset=["query_id_internal", "formula", "candidate_rank", "abs_rt_delta"])
    out = out[out["candidate_rank"] > 0]

    return out, {
        "query_col": query_col,
        "formula_col": formula_col,
        "rank_col": rank_col,
        "delta_col": delta_col,
    }

def reconstruct_for_dataset(dataset, sel):
    threshold = float(sel["threshold_sec"])
    guard_k = int(float(sel["guard_k"]))

    target_before = int(round(float(sel["n_candidate_rows_before"])))
    target_after = int(round(float(sel["n_candidate_rows_after"])))
    target_reduction = float(sel["candidate_reduction_pct"])

    candidates = find_candidate_row_files(dataset)

    debug = []
    best = None

    for sc, p, shape, cols in candidates[:80]:
        try:
            cdf, colinfo = prepare_candidate_df(dataset, p)
        except Exception as e:
            debug.append({
                "dataset": dataset,
                "path": str(p),
                "score": sc,
                "shape": str(shape),
                "status": f"prepare_failed: {e}",
            })
            continue

        if len(cdf) == 0:
            continue

        # final guarded-soft retained rule:
        # keep top guard_k original candidates plus candidates within RT threshold.
        cdf["retained"] = (cdf["abs_rt_delta"] <= threshold) | (cdf["candidate_rank"] <= guard_k)

        before = len(cdf)
        after = int(cdf["retained"].sum())
        reduction = (1 - after / before) * 100 if before else np.nan

        diff_before = abs(before - target_before)
        diff_after = abs(after - target_after)
        diff_red = abs(reduction - target_reduction)

        status = "checked"
        if diff_before == 0 and diff_after == 0:
            status = "MATCH_EXACT"

        debug.append({
            "dataset": dataset,
            "path": str(p),
            "score": sc,
            "shape": str(shape),
            "before": before,
            "after": after,
            "reduction": reduction,
            "target_before": target_before,
            "target_after": target_after,
            "target_reduction": target_reduction,
            "diff_before": diff_before,
            "diff_after": diff_after,
            "diff_reduction": diff_red,
            "status": status,
            **colinfo,
        })

        metric = diff_before + diff_after + diff_red
        if best is None or metric < best[0]:
            best = (metric, p, cdf, colinfo, before, after, reduction, status)

        if status == "MATCH_EXACT":
            break

    debug_df = pd.DataFrame(debug)
    debug_df.to_csv(OUT / f"debug_reconstruct_candidate_files_{dataset.replace('-', '_')}.csv", index=False)

    if best is None:
        raise RuntimeError(f"No usable candidate row file found for {dataset}")

    metric, p, cdf, colinfo, before, after, reduction, status = best

    print("\n" + "=" * 90)
    print(dataset)
    print("selected summary:")
    print(" threshold_sec =", threshold, "guard_k =", guard_k)
    print(" target before/after/reduction =", target_before, target_after, target_reduction)
    print("best candidate row file:", p)
    print("columns used:", colinfo)
    print("reconstructed before/after/reduction =", before, after, reduction)
    print("status:", status)

    if status != "MATCH_EXACT":
        print("\n[WARNING] No exact match.")
        print("Open debug file:")
        print(OUT / f"debug_reconstruct_candidate_files_{dataset.replace('-', '_')}.csv")
        print("I will still stop here to avoid drawing a wrong formula-level figure.")
        raise SystemExit(1)

    # per query/formula retained count
    q = (
        cdf.groupby(["dataset", "query_id_internal", "formula"], as_index=False)
           .agg(
               total_candidates=("retained", "size"),
               retained_candidates=("retained", "sum"),
           )
    )

    # 加重复 formula 后缀，避免 y 轴重复
    seen = {}
    labels = []
    for f in q["formula"].astype(str):
        seen[f] = seen.get(f, 0) + 1
        labels.append(f if seen[f] == 1 else f"{f}-{seen[f]}")
    q["formula_label"] = labels

    q["source_candidate_file"] = str(p)
    q["threshold_sec"] = threshold
    q["guard_k"] = guard_k
    q["summary_file"] = sel["summary_file"]

    return q

def plot_formula_level(allq):
    df = pd.concat(allq, ignore_index=True)

    order = {"RIKEN-PlaSMA": 0, "MetaboBase": 1}
    df["dataset_order"] = df["dataset"].map(order)
    df = df.sort_values(["dataset_order", "total_candidates"], ascending=[True, True]).reset_index(drop=True)

    src = OUT / "source_formula_level_guarded_soft_final.csv"
    df.to_csv(src, index=False)

    print("\n" + "=" * 90)
    print("PLOTTED DATA SUMMARY")
    g = df.groupby("dataset")[["total_candidates", "retained_candidates"]].sum()
    g["reduction_percent"] = (1 - g["retained_candidates"] / g["total_candidates"]) * 100
    print(g.to_string())
    print("zero retained query count:")
    print(df.groupby("dataset").apply(lambda x: int((x["retained_candidates"] == 0).sum())).to_string())

    y = np.arange(len(df))
    height = max(8.5, 0.085 * len(df))

    plt.figure(figsize=(8.2, height))

    is_riken = df["dataset"].eq("RIKEN-PlaSMA")
    total_colors = np.where(is_riken, "#cdecc8", "#d8e6f7")
    retained_colors = np.where(is_riken, "#7fc97f", "#8da0b6")

    plt.barh(y, df["total_candidates"], color=total_colors, edgecolor="none")
    plt.barh(y, df["retained_candidates"], color=retained_colors, edgecolor="none")

    plt.yticks(y, df["formula_label"], fontsize=5.2)
    plt.xlabel("Number of Candidates")
    plt.ylabel("Formula")

    handles = [
        Patch(facecolor="#cdecc8", label="Total candidates in RIKEN-PlaSMA"),
        Patch(facecolor="#7fc97f", label="Retained candidates in RIKEN-PlaSMA"),
        Patch(facecolor="#d8e6f7", label="Total candidates in MetaboBase"),
        Patch(facecolor="#8da0b6", label="Retained candidates in MetaboBase"),
    ]
    plt.legend(handles=handles, frameon=True, fontsize=8, loc="upper right")

    plt.tight_layout()

    pdf = OUT / "fig_formula_level_guarded_soft_final.pdf"
    png = OUT / "fig_formula_level_guarded_soft_final.png"

    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n[SAVE]", pdf)
    print("[SAVE]", png)
    print("[SAVE]", src)

def main():
    selected = {}
    for dataset in ["MetaboBase", "RIKEN-PlaSMA"]:
        sel, debug = select_final_summary_row(dataset)
        selected[dataset] = sel

        debug.to_csv(OUT / f"debug_selected_summary_rows_{dataset.replace('-', '_')}.csv", index=False)

        print("\n" + "=" * 90)
        print("FINAL SUMMARY SELECTED:", dataset)
        cols = [
            "dataset", "method", "n_queries",
            "n_candidate_rows_before", "n_candidate_rows_after",
            "candidate_reduction_pct", "true_retention_pct",
            "top1_before_pct", "top1_after_pct",
            "top5_before_pct", "top5_after_pct",
            "top10_before_pct", "top10_after_pct",
            "threshold_sec", "guard_k", "tau", "alpha",
            "target_distance", "summary_file",
        ]
        cols = [c for c in cols if c in sel.index]
        print(sel[cols].to_string())

    allq = []
    for dataset in ["RIKEN-PlaSMA", "MetaboBase"]:
        allq.append(reconstruct_for_dataset(dataset, selected[dataset]))

    plot_formula_level(allq)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
RES = ROOT / "paper_results_TCDV_TopoRT"
OUT = ROOT / "manuscript_figures_jcim_extra"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[SAVE]", OUT / f"{name}.pdf")
    print("[SAVE]", OUT / f"{name}.png")


# ============================================================
# 1. Radar: TCDV vs ABCoRT on SMRT metrics
#    Similar to ABCoRT ablation radar, but scientifically safer:
#    each axis is normalized so higher is better.
# ============================================================

def make_smrt_radar():
    # Raw values from Table 1
    raw = pd.DataFrame([
        {
            "method": "ABCoRT",
            "MAE": 25.75,
            "MRE": 3.24,
            "MedAE": 11.78,
            "MedRE": 1.50,
            "R2": 0.895,
        },
        {
            "method": "TCDV-S",
            "MAE": 25.055,
            "MRE": 3.162,
            "MedAE": 11.317,
            "MedRE": 1.453,
            "R2": 0.8983,
        },
        {
            "method": "TCDV-E",
            "MAE": 24.920,
            "MRE": 3.145,
            "MedAE": 11.164,
            "MedRE": 1.439,
            "R2": 0.8988,
        },
    ])

    metrics = ["MAE", "MRE", "MedAE", "MedRE", "R2"]
    lower_better = {"MAE", "MRE", "MedAE", "MedRE"}

    # Convert to normalized score: higher is better.
    score = raw.copy()
    for m in metrics:
        if m in lower_better:
            best = raw[m].min()
            score[m] = best / raw[m]
        else:
            best = raw[m].max()
            score[m] = raw[m] / best

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5.8, 5.2))
    ax = plt.subplot(111, polar=True)

    labels = ["MAE", "MRE", "MedAE", "MedRE", r"$R^2$"]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0.96, 1.002)
    ax.set_yticks([0.96, 0.97, 0.98, 0.99, 1.00])
    ax.set_yticklabels(["0.96", "0.97", "0.98", "0.99", "1.00"], fontsize=8)

    for _, row in score.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.6, label=row["method"])
        ax.fill(angles, values, alpha=0.15)

    ax.set_title("Normalized SMRT performance radar", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), frameon=False, fontsize=8)

    # Save raw + normalized values
    raw.to_csv(OUT / "source_smrt_radar_raw_metrics.csv", index=False)
    score.to_csv(OUT / "source_smrt_radar_normalized_scores.csv", index=False)

    savefig("fig_smrt_radar_tcdv_vs_abcort")


# ============================================================
# 2. Radar: structural ablation
#    Full / no explicit ring 2-cells / no CWN message passing.
# ============================================================

def make_structural_radar():
    path = RES / "tables" / "04_structural_ablation_seed5.csv"
    if not path.exists():
        print("[MISS]", path)
        return

    df = pd.read_csv(path)
    df = df[df["exists"] == True].copy()

    # Use metrics available in final_metrics.
    metrics = ["mae", "medae", "rmse", "p95", "r2"]
    label_map = {
        "mae": "MAE",
        "medae": "MedAE",
        "rmse": "RMSE",
        "p95": "p95",
        "r2": r"$R^2$",
    }
    lower_better = {"mae", "medae", "rmse", "p95"}

    score = df[["variant"] + metrics].copy()
    for m in metrics:
        if m in lower_better:
            best = df[m].min()
            score[m] = best / df[m]
        else:
            best = df[m].max()
            score[m] = df[m] / best

    name_map = {
        "Full TCDV-TopoRT seed5": "Full",
        "w/o explicit ring 2-cells": "w/o ring 2-cells",
        "w/o CWN message passing": "w/o CWN",
    }
    score["short_name"] = score["variant"].map(name_map).fillna(score["variant"])

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5.8, 5.2))
    ax = plt.subplot(111, polar=True)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label_map[m] for m in metrics])

    # CWN0 is very bad, so use wider range.
    ax.set_ylim(0.55, 1.02)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=8)

    for _, row in score.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.6, label=row["short_name"])
        ax.fill(angles, values, alpha=0.15)

    ax.set_title("Structural ablation radar", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), frameon=False, fontsize=8)

    score.to_csv(OUT / "source_structural_ablation_radar_normalized_scores.csv", index=False)

    savefig("fig_structural_ablation_radar")


# ============================================================
# 3. Candidate filtering summary, similar visual logic to ABCoRT Fig. 5
#    This can always be generated from your summary table.
# ============================================================

def make_candidate_summary():
    path = RES / "tables" / "07_candidate_filtering_reranking_summary.csv"
    if not path.exists():
        print("[MISS]", path)
        return

    cf = pd.read_csv(path)

    metrics = ["reduction_percent", "top1_percent", "top5_percent", "top10_percent"]
    metric_labels = ["Reduction", "Top-1", "Top-5", "Top-10"]

    datasets = ["MetaboBase", "RIKEN_PlaSMA"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True)

    for ax, ds in zip(axes, datasets):
        sub = cf[cf["dataset"] == ds].copy()
        if len(sub) == 0:
            # try hyphen version if present
            sub = cf[cf["dataset"].astype(str).str.replace("_", "-", regex=False) == ds.replace("_", "-")].copy()

        abc = sub[sub["method"].str.contains("ABCoRT", case=False, na=False)].iloc[0]
        ours = sub[sub["method"].str.contains("Ours|TCDV", case=False, na=False)].iloc[0]

        x = np.arange(len(metrics))
        w = 0.36

        ax.bar(x - w/2, [abc[m] for m in metrics], width=w, label="ABCoRT-TL", alpha=0.55, edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, [ours[m] for m in metrics], width=w, label="TCDV-TopoRT", alpha=0.75, edgecolor="black", linewidth=0.5)

        ax.set_title(ds.replace("_", "-"))
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=25, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)")

        for i, m in enumerate(metrics):
            ax.text(i - w/2, abc[m] + 1.2, f"{abc[m]:.1f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + w/2, ours[m] + 1.2, f"{ours[m]:.1f}", ha="center", va="bottom", fontsize=7)

    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("RT-aware candidate filtering and reranking", y=1.03)

    savefig("fig_candidate_filtering_summary_like_abcort")


# ============================================================
# 4. Candidate formula-level long horizontal bar, ABCoRT Fig. 5 style.
#    Requires per-formula/query total and retained candidate counts.
#    The script automatically searches CSV files with suitable columns.
# ============================================================

def normalize_colname(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())


def detect_candidate_count_table(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if df.shape[0] < 5:
        return None

    norm = {c: normalize_colname(c) for c in df.columns}

    formula_candidates = []
    total_candidates = []
    retained_candidates = []
    dataset_candidates = []

    for c, n in norm.items():
        if any(k in n for k in ["formula", "queryformula", "molecularformula", "compoundformula"]):
            formula_candidates.append(c)
        if ("total" in n and "candidate" in n) or n in ["ntotal", "before", "numcandidates", "totalnumcandidates"]:
            total_candidates.append(c)
        if any(k in n for k in ["retainedcandidate", "retainednum", "nretained", "after", "remainingcandidate"]):
            retained_candidates.append(c)
        if n in ["dataset", "datasetname", "source"]:
            dataset_candidates.append(c)

    if not formula_candidates or not total_candidates or not retained_candidates:
        return None

    formula_col = formula_candidates[0]
    total_col = total_candidates[0]
    retained_col = retained_candidates[0]
    dataset_col = dataset_candidates[0] if dataset_candidates else None

    out = df[[formula_col, total_col, retained_col] + ([dataset_col] if dataset_col else [])].copy()
    rename = {
        formula_col: "formula",
        total_col: "total_candidates",
        retained_col: "retained_candidates",
    }
    if dataset_col:
        rename[dataset_col] = "dataset"
    else:
        rename["dataset"] = None

    out = out.rename(columns=rename)
    if "dataset" not in out.columns:
        out["dataset"] = "CandidateSet"

    out["total_candidates"] = pd.to_numeric(out["total_candidates"], errors="coerce")
    out["retained_candidates"] = pd.to_numeric(out["retained_candidates"], errors="coerce")
    out = out.dropna(subset=["formula", "total_candidates", "retained_candidates"])

    if len(out) < 5:
        return None

    out["source_file"] = str(csv_path)
    return out


def make_candidate_formula_longbar():
    # First, check if the user already created a clean file.
    preferred = [
        ROOT / "candidate_formula_counts.csv",
        ROOT / "paper_results_TCDV_TopoRT" / "candidate_filtering" / "candidate_formula_counts.csv",
        ROOT / "paper_results_TCDV_TopoRT" / "candidate_filtering" / "raw_candidate_files_found" / "candidate_formula_counts.csv",
    ]

    found = []
    for p in preferred:
        if p.exists():
            t = detect_candidate_count_table(p)
            if t is not None:
                found.append(t)

    # If not found, search project CSV files.
    if not found:
        search_roots = [
            ROOT / "paper_results_TCDV_TopoRT" / "candidate_filtering",
            ROOT,
        ]

        candidates = []
        for sr in search_roots:
            if sr.exists():
                candidates.extend(sr.rglob("*.csv"))

        # avoid scanning huge prediction files first
        candidates = [p for p in candidates if p.stat().st_size < 20 * 1024 * 1024]
        candidates = sorted(set(candidates), key=lambda x: str(x))

        for p in candidates:
            low = str(p).lower()
            if not any(k in low for k in ["candidate", "filter", "rerank", "riken", "plasma", "metabobase"]):
                continue
            t = detect_candidate_count_table(p)
            if t is not None:
                found.append(t)
                print("[FOUND candidate formula table]", p)
                break

    if not found:
        print("")
        print("[NO FORMULA-LEVEL CANDIDATE COUNT TABLE FOUND]")
        print("To generate the ABCoRT Fig.5-style long horizontal retained-candidate plot,")
        print("create a CSV at:")
        print("  candidate_formula_counts.csv")
        print("with columns:")
        print("  dataset,formula,total_candidates,retained_candidates")
        print("Then rerun:")
        print("  python make_abcort_like_extra_figures.py")
        print("")
        return

    df = pd.concat(found, ignore_index=True)
    df["dataset"] = df["dataset"].astype(str)
    df["formula"] = df["formula"].astype(str)
    df["total_candidates"] = df["total_candidates"].astype(float)
    df["retained_candidates"] = df["retained_candidates"].astype(float)

    # Keep sane values
    df = df[(df["total_candidates"] >= df["retained_candidates"]) & (df["total_candidates"] > 0)].copy()
    if len(df) == 0:
        print("[WARN] formula table exists but no valid rows")
        return

    # Sort like ABCoRT: grouped dataset, total candidate ascending.
    df["dataset_order"] = df["dataset"].map(lambda x: 0 if "riken" in x.lower() or "plasma" in x.lower() else 1)
    df = df.sort_values(["dataset_order", "total_candidates"], ascending=[True, True]).reset_index(drop=True)

    # If too many rows, keep top 140 for readability.
    max_rows = 140
    if len(df) > max_rows:
        df = df.tail(max_rows).copy()

    y = np.arange(len(df))
    fig_h = max(7.5, len(df) * 0.085)

    plt.figure(figsize=(8.0, fig_h))

    # Colors by dataset group
    is_riken = df["dataset"].str.lower().str.contains("riken|plasma")
    total_colors = np.where(is_riken, "#cdecc8", "#cfe0f5")
    retained_colors = np.where(is_riken, "#7bc77a", "#7f9dbd")

    plt.barh(y, df["total_candidates"], color=total_colors, edgecolor="none", label="Total candidates")
    plt.barh(y, df["retained_candidates"], color=retained_colors, edgecolor="none", label="Retained candidates")

    plt.yticks(y, df["formula"], fontsize=5.5)
    plt.xlabel("Number of candidates")
    plt.ylabel("Formula")
    plt.title("Formula-level candidate filtering")

    # Custom legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#cdecc8", label="Total candidates in RIKEN-PlaSMA"),
        Patch(facecolor="#7bc77a", label="Retained candidates in RIKEN-PlaSMA"),
        Patch(facecolor="#cfe0f5", label="Total candidates in MetaboBase"),
        Patch(facecolor="#7f9dbd", label="Retained candidates in MetaboBase"),
    ]
    plt.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")

    df.to_csv(OUT / "source_candidate_formula_retained_like_abcort.csv", index=False)

    savefig("fig_candidate_formula_retained_like_abcort")


def main():
    make_smrt_radar()
    make_structural_radar()
    make_candidate_summary()
    make_candidate_formula_longbar()

    print("")
    print("DONE. Output directory:")
    print(OUT)
    print("")
    print("Important:")
    print("If fig_candidate_formula_retained_like_abcort.pdf was not generated,")
    print("you need formula-level candidate count data with columns:")
    print("dataset,formula,total_candidates,retained_candidates")


if __name__ == "__main__":
    main()

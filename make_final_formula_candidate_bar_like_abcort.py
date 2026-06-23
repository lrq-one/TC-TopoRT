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


def read_csv(path):
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return None


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def infer_dataset(path):
    s = str(path).lower()
    if "riken" in s or "plasma" in s or "mona" in s:
        return "RIKEN-PlaSMA"
    if "metabo" in s:
        return "MetaboBase"
    return "Unknown"


def find_query_summaries():
    rows = []

    for p in ROOT.rglob("*.csv"):
        name = str(p).lower()
        if "query_summary" not in name:
            continue

        df = read_csv(p)
        if df is None:
            continue

        cols = set(df.columns)
        if not {"n_candidates_before", "n_candidates_after"}.issubset(cols):
            continue

        before = pd.to_numeric(df["n_candidates_before"], errors="coerce")
        after = pd.to_numeric(df["n_candidates_after"], errors="coerce")

        if before.notna().sum() == 0 or after.notna().sum() == 0:
            continue

        reduction = (1.0 - after.sum() / before.sum()) * 100.0
        dataset = infer_dataset(p)

        rows.append({
            "path": str(p),
            "dataset": dataset,
            "n_queries": len(df),
            "before_sum": float(before.sum()),
            "after_sum": float(after.sum()),
            "reduction_pct": float(reduction),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "debug_query_summary_candidates.csv", index=False)
    return out


def choose_best_summary(cands, dataset, target_reduction):
    sub = cands[cands["dataset"] == dataset].copy()
    if len(sub) == 0:
        return None

    sub["dist"] = (sub["reduction_pct"] - target_reduction).abs()

    # 优先接近最终 reduction，其次 query 数更多
    sub = sub.sort_values(["dist", "n_queries"], ascending=[True, False])
    return Path(sub.iloc[0]["path"])


def find_formula_mapping(summary_path, summary_df):
    """
    找 formula 映射。
    优先同目录下 true_predictions / candidate_predictions / audit 文件。
    用 s10_row 或 query_id join。
    """
    keys = []
    if "s10_row" in summary_df.columns:
        keys.append("s10_row")
    if "query_id" in summary_df.columns:
        keys.append("query_id")
    if "true_name" in summary_df.columns:
        keys.append("true_name")

    search_roots = [
        summary_path.parent,
        summary_path.parent.parent,
        ROOT,
    ]

    candidate_files = []
    for sr in search_roots:
        if sr.exists():
            for p in sr.rglob("*.csv"):
                s = str(p).lower()
                if any(k in s for k in ["true", "candidate", "audit", "formula", "query"]):
                    if p.stat().st_size < 200 * 1024 * 1024:
                        candidate_files.append(p)

    for p in sorted(set(candidate_files)):
        df = read_csv(p)
        if df is None:
            continue

        formula_cols = [c for c in df.columns if norm(c) in ["formula", "trueformula", "queryformula"] or "formula" in norm(c)]
        if not formula_cols:
            continue

        for key in keys:
            if key in df.columns and key in summary_df.columns:
                formula_col = None

                # 优先 query_formula / true_formula，再 formula
                for cand in ["query_formula", "true_formula", "Formula", "formula"]:
                    if cand in df.columns:
                        formula_col = cand
                        break
                if formula_col is None:
                    formula_col = formula_cols[0]

                map_df = df[[key, formula_col]].dropna().drop_duplicates()
                map_df = map_df.rename(columns={formula_col: "formula"})
                if len(map_df) > 0:
                    return key, map_df, str(p)

    return None, None, None


def build_dataset_rows(dataset, summary_path):
    df = read_csv(summary_path)
    if df is None:
        raise RuntimeError(f"Cannot read {summary_path}")

    before = pd.to_numeric(df["n_candidates_before"], errors="coerce")
    after = pd.to_numeric(df["n_candidates_after"], errors="coerce")
    df = df.copy()
    df["total_candidates"] = before
    df["retained_candidates"] = after

    # 如果 summary 自己有 formula，直接用
    formula_col = None
    for c in ["query_formula", "true_formula", "formula", "Formula"]:
        if c in df.columns:
            formula_col = c
            break

    if formula_col is not None:
        df["formula"] = df[formula_col].astype(str)
        source_map = "formula column in query_summary"
    else:
        key, map_df, source_map = find_formula_mapping(summary_path, df)
        if key is None:
            # 最后保底：没有 formula 就用 query id，但这里会明确警告
            print("[WARN] Cannot find formula mapping for", summary_path)
            print("       Using query index as formula labels. This is not ideal.")
            df["formula"] = [f"query_{i+1}" for i in range(len(df))]
            source_map = "fallback query index"
        else:
            df = df.merge(map_df, on=key, how="left")
            df["formula"] = df["formula"].fillna(df[key].astype(str))

    out = df[["formula", "total_candidates", "retained_candidates"]].copy()
    out["dataset"] = dataset
    out["summary_file"] = str(summary_path)
    out["formula_mapping_source"] = source_map

    out = out.dropna(subset=["total_candidates", "retained_candidates"])
    out = out[out["total_candidates"] > 0]
    out = out[out["retained_candidates"] >= 0]
    out = out[out["retained_candidates"] <= out["total_candidates"]]

    # 相同 formula 可能出现多个 query；为了画图，保留每条 query 作为一行。
    # 如果重复 formula 太多，加后缀避免 ytick 重名。
    counts = {}
    labels = []
    for f in out["formula"].astype(str):
        counts[f] = counts.get(f, 0) + 1
        labels.append(f if counts[f] == 1 else f"{f}-{counts[f]}")
    out["formula_label"] = labels

    return out


def make_plot(rows):
    df = pd.concat(rows, ignore_index=True)

    order = {"RIKEN-PlaSMA": 0, "MetaboBase": 1}
    df["dataset_order"] = df["dataset"].map(order)
    df = df.sort_values(["dataset_order", "total_candidates"], ascending=[True, True]).reset_index(drop=True)

    df.to_csv(OUT / "source_formula_candidate_bar_final.csv", index=False)

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
        Patch(facecolor="#cdecc8", label="Total Number of Candidates in RIKEN-PlaSMA"),
        Patch(facecolor="#7fc97f", label="Retained Candidates in RIKEN-PlaSMA"),
        Patch(facecolor="#d8e6f7", label="Total Number of Candidates in MetaboBase"),
        Patch(facecolor="#8da0b6", label="Retained Candidates in MetaboBase"),
    ]
    plt.legend(handles=handles, frameon=True, fontsize=8, loc="upper right")

    plt.tight_layout()

    pdf = OUT / "fig_formula_candidate_bar_like_abcort_final.pdf"
    png = OUT / "fig_formula_candidate_bar_like_abcort_final.png"

    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()

    print("[SAVE]", pdf)
    print("[SAVE]", png)
    print("[SAVE]", OUT / "source_formula_candidate_bar_final.csv")


def main():
    cands = find_query_summaries()

    print("\n=== candidate query_summary files ===")
    if len(cands) == 0:
        print("No query_summary files with n_candidates_before/after found.")
        raise SystemExit(1)

    print(cands.sort_values(["dataset", "reduction_pct"]).head(80).to_string(index=False))

    metabop = choose_best_summary(cands, "MetaboBase", 69.14)
    rikenp = choose_best_summary(cands, "RIKEN-PlaSMA", 46.23)

    selected = []
    if rikenp is not None:
        print("\n[SELECT RIKEN-PlaSMA]", rikenp)
        selected.append(build_dataset_rows("RIKEN-PlaSMA", rikenp))
    else:
        print("\n[WARN] No RIKEN-PlaSMA query_summary selected.")

    if metabop is not None:
        print("\n[SELECT MetaboBase]", metabop)
        selected.append(build_dataset_rows("MetaboBase", metabop))
    else:
        print("\n[WARN] No MetaboBase query_summary selected.")

    if not selected:
        raise SystemExit("No selected summaries.")

    make_plot(selected)


if __name__ == "__main__":
    main()

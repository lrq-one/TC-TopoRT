#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


IN_PRED = Path("experiments_candidate_filtering/metabobase_s10_predictions/metabobase_s10_candidate_predictions_final.csv")
OUT_DIR = Path("experiments_candidate_filtering/metabobase_s10_filtering_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_thresholds(s):
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def rank_after_filter(sub):
    kept = sub[sub["kept"]].copy()
    if kept.empty:
        return None
    kept = kept.sort_values(["candidate_rank", "candidate_score"], ascending=[True, False]).reset_index(drop=True)
    kept["rank_after_filter"] = np.arange(1, len(kept) + 1)
    true_rows = kept[kept["is_true"].astype(bool)]
    if true_rows.empty:
        return None
    return int(true_rows["rank_after_filter"].min())


def true_rank_before(sub):
    true_rows = sub[sub["is_true"].astype(bool)]
    if true_rows.empty:
        return None
    return int(true_rows["candidate_rank"].min())


def topk_from_rank(rank, k):
    if rank is None or pd.isna(rank):
        return False
    return int(rank) <= k


def eval_threshold(df, threshold):
    d = df.copy()
    d["kept"] = d["abs_rt_delta"].astype(float) <= float(threshold)

    rows = []
    for s10_row, sub in d.groupby("s10_row", sort=True):
        sub = sub.sort_values("candidate_rank").copy()

        n_before = len(sub)
        n_after = int(sub["kept"].sum())

        r_before = true_rank_before(sub)
        r_after = rank_after_filter(sub)

        true_in_before = r_before is not None
        true_in_after = r_after is not None

        true_delta = np.nan
        true_pred = np.nan
        if true_in_before:
            tr = sub[sub["is_true"].astype(bool)].sort_values("candidate_rank").iloc[0]
            true_delta = float(tr["abs_rt_delta"])
            true_pred = float(tr["candidate_pred_rt"])

        rows.append({
            "s10_row": int(s10_row),
            "query_id": sub["query_id"].iloc[0] if "query_id" in sub.columns else "",
            "true_name": sub["true_name"].iloc[0],
            "true_formula": sub["true_formula"].iloc[0] if "true_formula" in sub.columns else "",
            "true_inchikey": sub["true_inchikey"].iloc[0] if "true_inchikey" in sub.columns else "",
            "rt_sec": float(sub["rt_sec"].iloc[0]),
            "n_candidates_before": n_before,
            "n_candidates_after": n_after,
            "n_filtered": n_before - n_after,
            "filter_rate_pct": 100.0 * (n_before - n_after) / max(n_before, 1),
            "true_rank_before": r_before,
            "true_rank_after": r_after,
            "true_in_candidates_before": true_in_before,
            "true_retained_after_filter": true_in_after,
            "true_candidate_pred_rt": true_pred,
            "true_candidate_abs_rt_delta": true_delta,
            "top1_before": topk_from_rank(r_before, 1),
            "top5_before": topk_from_rank(r_before, 5),
            "top10_before": topk_from_rank(r_before, 10),
            "top1_after": topk_from_rank(r_after, 1),
            "top5_after": topk_from_rank(r_after, 5),
            "top10_after": topk_from_rank(r_after, 10),
        })

    q = pd.DataFrame(rows)

    n_queries = len(q)
    total_before = int(q["n_candidates_before"].sum())
    total_after = int(q["n_candidates_after"].sum())
    true_before = int(q["true_in_candidates_before"].sum())
    true_after = int(q["true_retained_after_filter"].sum())

    summary = {
        "threshold_sec": float(threshold),
        "n_queries": n_queries,
        "n_candidate_rows_before": total_before,
        "n_candidate_rows_after": total_after,
        "n_filtered": total_before - total_after,
        "candidate_reduction_pct": 100.0 * (total_before - total_after) / max(total_before, 1),
        "mean_candidates_before": float(q["n_candidates_before"].mean()),
        "mean_candidates_after": float(q["n_candidates_after"].mean()),
        "median_candidates_before": float(q["n_candidates_before"].median()),
        "median_candidates_after": float(q["n_candidates_after"].median()),
        "true_in_candidates_before": true_before,
        "true_retained_after_filter": true_after,
        "true_retention_pct_among_found": 100.0 * true_after / max(true_before, 1),
        "top1_before_pct": 100.0 * q["top1_before"].mean(),
        "top5_before_pct": 100.0 * q["top5_before"].mean(),
        "top10_before_pct": 100.0 * q["top10_before"].mean(),
        "top1_after_pct": 100.0 * q["top1_after"].mean(),
        "top5_after_pct": 100.0 * q["top5_after"].mean(),
        "top10_after_pct": 100.0 * q["top10_after"].mean(),
        "mean_true_abs_rt_delta": float(q["true_candidate_abs_rt_delta"].mean()),
        "median_true_abs_rt_delta": float(q["true_candidate_abs_rt_delta"].median()),
        "max_true_abs_rt_delta": float(q["true_candidate_abs_rt_delta"].max()),
    }

    return d, q, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--thresholds",
        default="75.17,100,150,185.31,200,250,300,400,500",
        help="comma-separated RT thresholds in seconds",
    )
    args = ap.parse_args()

    df = pd.read_csv(IN_PRED)

    required = ["s10_row", "candidate_rank", "candidate_score", "is_true", "rt_sec", "candidate_pred_rt", "abs_rt_delta"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing columns: {missing}")

    df["is_true"] = df["is_true"].astype(bool)

    print("=" * 100)
    print("Input:", IN_PRED)
    print("candidate rows:", len(df))
    print("queries:", df["s10_row"].nunique())
    print("prediction NaN:", int(df["candidate_pred_rt"].isna().sum()))
    print("abs_rt_delta summary:")
    print(df["abs_rt_delta"].describe().to_string())
    print("=" * 100)

    thresholds = parse_thresholds(args.thresholds)
    summaries = []

    for th in thresholds:
        d, q, s = eval_threshold(df, th)
        summaries.append(s)

        tag = str(th).replace(".", "p")
        d.to_csv(OUT_DIR / f"metabobase_s10_candidate_rows_threshold_{tag}.csv", index=False)
        q.to_csv(OUT_DIR / f"metabobase_s10_query_summary_threshold_{tag}.csv", index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT_DIR / "metabobase_s10_filtering_summary_by_threshold.csv", index=False)

    print("\nFiltering summary by threshold:")
    show_cols = [
        "threshold_sec",
        "n_queries",
        "n_candidate_rows_before",
        "n_candidate_rows_after",
        "candidate_reduction_pct",
        "true_retention_pct_among_found",
        "top1_before_pct",
        "top1_after_pct",
        "top5_before_pct",
        "top5_after_pct",
        "top10_before_pct",
        "top10_after_pct",
        "mean_true_abs_rt_delta",
        "median_true_abs_rt_delta",
    ]
    print(summary[show_cols].to_string(index=False))

    print("\nSaved:")
    print(OUT_DIR / "metabobase_s10_filtering_summary_by_threshold.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

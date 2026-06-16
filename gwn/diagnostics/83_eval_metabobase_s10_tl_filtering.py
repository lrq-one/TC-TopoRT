#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_thresholds(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def true_rank_before(sub):
    t = sub[sub["is_true"].astype(bool)]
    if t.empty:
        return None
    return int(t["candidate_rank"].min())


def rank_after_filter(sub):
    kept = sub[sub["kept"]].copy()
    if kept.empty:
        return None
    kept = kept.sort_values(["candidate_rank", "candidate_score"], ascending=[True, False]).reset_index(drop=True)
    kept["rank_after_filter"] = np.arange(1, len(kept) + 1)
    t = kept[kept["is_true"].astype(bool)]
    if t.empty:
        return None
    return int(t["rank_after_filter"].min())


def topk(rank, k):
    if rank is None or pd.isna(rank):
        return False
    return int(rank) <= k


def eval_threshold(df, threshold):
    d = df.copy()
    d["kept"] = d["abs_rt_delta"].astype(float) <= float(threshold)

    rows = []
    for s10_row, sub in d.groupby("s10_row", sort=True):
        sub = sub.sort_values("candidate_rank").copy()
        rb = true_rank_before(sub)
        ra = rank_after_filter(sub)

        true_delta = np.nan
        true_pred = np.nan
        if rb is not None:
            tr = sub[sub["is_true"].astype(bool)].sort_values("candidate_rank").iloc[0]
            true_delta = float(tr["abs_rt_delta"])
            true_pred = float(tr["candidate_pred_rt"])

        rows.append({
            "s10_row": int(s10_row),
            "true_name": sub["true_name"].iloc[0],
            "rt_sec": float(sub["rt_sec"].iloc[0]),
            "n_candidates_before": len(sub),
            "n_candidates_after": int(sub["kept"].sum()),
            "n_filtered": len(sub) - int(sub["kept"].sum()),
            "filter_rate_pct": 100.0 * (len(sub) - int(sub["kept"].sum())) / max(len(sub), 1),
            "true_rank_before": rb,
            "true_rank_after": ra,
            "true_in_before": rb is not None,
            "true_retained_after": ra is not None,
            "true_candidate_pred_rt": true_pred,
            "true_abs_rt_delta": true_delta,
            "top1_before": topk(rb, 1),
            "top5_before": topk(rb, 5),
            "top10_before": topk(rb, 10),
            "top1_after": topk(ra, 1),
            "top5_after": topk(ra, 5),
            "top10_after": topk(ra, 10),
        })

    q = pd.DataFrame(rows)
    total_before = int(q["n_candidates_before"].sum())
    total_after = int(q["n_candidates_after"].sum())
    true_before = int(q["true_in_before"].sum())
    true_after = int(q["true_retained_after"].sum())

    summary = {
        "threshold_sec": float(threshold),
        "n_queries": len(q),
        "n_candidate_rows_before": total_before,
        "n_candidate_rows_after": total_after,
        "candidate_reduction_pct": 100.0 * (total_before - total_after) / max(total_before, 1),
        "true_retention_pct_among_found": 100.0 * true_after / max(true_before, 1),
        "top1_before_pct": 100.0 * q["top1_before"].mean(),
        "top1_after_pct": 100.0 * q["top1_after"].mean(),
        "top5_before_pct": 100.0 * q["top5_before"].mean(),
        "top5_after_pct": 100.0 * q["top5_after"].mean(),
        "top10_before_pct": 100.0 * q["top10_before"].mean(),
        "top10_after_pct": 100.0 * q["top10_after"].mean(),
        "mean_true_abs_rt_delta": float(q["true_abs_rt_delta"].mean()),
        "median_true_abs_rt_delta": float(q["true_abs_rt_delta"].median()),
        "max_true_abs_rt_delta": float(q["true_abs_rt_delta"].max()),
    }
    return d, q, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred_csv",
        default="experiments_candidate_filtering/metabobase_s10_predictions_tl_exact39/metabobase_s10_candidate_predictions_tl_seed42.csv",
    )
    ap.add_argument(
        "--out_dir",
        default="experiments_candidate_filtering/metabobase_s10_tl_filtering_eval_seed42",
    )
    ap.add_argument(
        "--thresholds",
        default="75.17,100,136.34,150,185.31,214.28,250,300,400,500",
    )
    args = ap.parse_args()

    pred_csv = Path(args.pred_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)
    df["is_true"] = df["is_true"].astype(bool)

    print("=" * 100)
    print("Input:", pred_csv)
    print("candidate rows:", len(df))
    print("queries:", df["s10_row"].nunique())
    print("prediction NaN:", int(df["candidate_pred_rt"].isna().sum()))
    print("candidate_pred_rt summary:")
    print(df["candidate_pred_rt"].describe().to_string())
    print("abs_rt_delta summary:")
    print(df["abs_rt_delta"].describe().to_string())
    print("=" * 100)

    summaries = []
    for th in parse_thresholds(args.thresholds):
        d, q, s = eval_threshold(df, th)
        summaries.append(s)

        tag = str(th).replace(".", "p")
        d.to_csv(out_dir / f"candidate_rows_threshold_{tag}.csv", index=False)
        q.to_csv(out_dir / f"query_summary_threshold_{tag}.csv", index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "tl_filtering_summary_by_threshold.csv", index=False)

    print("\nTL filtering summary:")
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
        "max_true_abs_rt_delta",
    ]
    print(summary[show_cols].to_string(index=False))

    print("\nSaved:")
    print(out_dir / "tl_filtering_summary_by_threshold.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

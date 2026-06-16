#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def to_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def topk(rank, k):
    if rank is None or pd.isna(rank):
        return False
    return int(rank) <= k


def parse_float_list(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def eval_ranked(df, method, keep_mask, sort_cols, ascending):
    d = df.copy()
    d["kept"] = keep_mask.astype(bool)

    rows = []
    for s10_row, sub in d.groupby("s10_row", sort=True):
        sub = sub.copy()

        true_before_rows = sub[sub["is_true"]]
        if true_before_rows.empty:
            rb = None
        else:
            rb = int(true_before_rows["candidate_rank"].min())

        kept = sub[sub["kept"]].copy()
        if kept.empty:
            ra = None
        else:
            kept = kept.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
            kept["rank_after"] = np.arange(1, len(kept) + 1)
            true_after_rows = kept[kept["is_true"]]
            if true_after_rows.empty:
                ra = None
            else:
                ra = int(true_after_rows["rank_after"].min())

        true_delta = np.nan
        if not true_before_rows.empty:
            true_delta = float(true_before_rows.sort_values("candidate_rank").iloc[0]["abs_rt_delta"])

        rows.append({
            "s10_row": int(s10_row),
            "true_name": sub["true_name"].iloc[0] if "true_name" in sub.columns else "",
            "rt_sec": float(sub["rt_sec"].iloc[0]),
            "n_candidates_before": len(sub),
            "n_candidates_after": int(kept.shape[0]),
            "true_rank_before": rb,
            "true_rank_after": ra,
            "true_retained_after": ra is not None,
            "true_abs_rt_delta": true_delta,
            "top1_before": topk(rb, 1),
            "top5_before": topk(rb, 5),
            "top10_before": topk(rb, 10),
            "top1_after": topk(ra, 1),
            "top5_after": topk(ra, 5),
            "top10_after": topk(ra, 10),
        })

    q = pd.DataFrame(rows)
    before = int(q["n_candidates_before"].sum())
    after = int(q["n_candidates_after"].sum())

    return q, {
        "method": method,
        "n_queries": len(q),
        "n_candidate_rows_before": before,
        "n_candidate_rows_after": after,
        "candidate_reduction_pct": 100.0 * (before - after) / max(before, 1),
        "true_retention_pct": 100.0 * q["true_retained_after"].mean(),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred_csv",
        default="experiments_candidate_filtering/metabobase_s10_predictions_tl_exact39/metabobase_s10_candidate_predictions_tl_seed42.csv",
    )
    ap.add_argument(
        "--out_dir",
        default="experiments_candidate_filtering/metabobase_s10_tl_rank_guard_soft_eval_seed42",
    )
    ap.add_argument("--thresholds", default="75.17,100,136.34,150,185.31,214.28,250,300,400,500")
    ap.add_argument("--guard_ks", default="1,3,5,10")
    ap.add_argument("--taus", default="100,136.34,185.31,214.28,300")
    ap.add_argument("--alphas", default="0.1,0.25,0.5,1,2,4")
    args = ap.parse_args()

    pred_csv = Path(args.pred_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)
    df["is_true"] = to_bool_series(df["is_true"])

    if "candidate_score" not in df.columns:
        df["candidate_score"] = 0.0

    df["candidate_rank"] = df["candidate_rank"].astype(float)
    df["candidate_score"] = pd.to_numeric(df["candidate_score"], errors="coerce").fillna(0.0)
    df["abs_rt_delta"] = df["abs_rt_delta"].astype(float)

    thresholds = parse_float_list(args.thresholds)
    guard_ks = parse_int_list(args.guard_ks)
    taus = parse_float_list(args.taus)
    alphas = parse_float_list(args.alphas)

    summaries = []

    # 0. original MS-FINDER rank
    q, s = eval_ranked(
        df,
        method="original_msfinder_rank",
        keep_mask=pd.Series(True, index=df.index),
        sort_cols=["candidate_rank", "candidate_score"],
        ascending=[True, False],
    )
    summaries.append(s)
    q.to_csv(out_dir / "query_original_msfinder_rank.csv", index=False)

    # 1. hard RT filter, original order
    for th in thresholds:
        keep = df["abs_rt_delta"] <= th
        q, s = eval_ranked(
            df,
            method=f"hard_rt_filter_th{th}",
            keep_mask=keep,
            sort_cols=["candidate_rank", "candidate_score"],
            ascending=[True, False],
        )
        s["threshold_sec"] = th
        s["guard_k"] = np.nan
        s["tau"] = np.nan
        s["alpha"] = np.nan
        summaries.append(s)
        q.to_csv(out_dir / f"query_hard_rt_filter_th{str(th).replace('.', 'p')}.csv", index=False)

    # 2. rank-guard RT filter, original order
    for th in thresholds:
        for g in guard_ks:
            keep = (df["abs_rt_delta"] <= th) | (df["candidate_rank"] <= g)
            q, s = eval_ranked(
                df,
                method=f"rank_guard_filter_th{th}_g{g}",
                keep_mask=keep,
                sort_cols=["candidate_rank", "candidate_score"],
                ascending=[True, False],
            )
            s["threshold_sec"] = th
            s["guard_k"] = g
            s["tau"] = np.nan
            s["alpha"] = np.nan
            summaries.append(s)

    # 3. soft reranking without filtering
    for tau in taus:
        for alpha in alphas:
            d = df.copy()
            d["hybrid_score"] = d["candidate_rank"] + alpha * d["abs_rt_delta"] / tau
            q, s = eval_ranked(
                d,
                method=f"soft_rerank_tau{tau}_alpha{alpha}",
                keep_mask=pd.Series(True, index=d.index),
                sort_cols=["hybrid_score", "candidate_rank", "candidate_score"],
                ascending=[True, True, False],
            )
            s["threshold_sec"] = np.nan
            s["guard_k"] = np.nan
            s["tau"] = tau
            s["alpha"] = alpha
            summaries.append(s)

    # 4. rank-guard filter + soft reranking
    for th in thresholds:
        for g in guard_ks:
            for tau in [185.31, 214.28]:
                for alpha in [0.25, 0.5, 1.0, 2.0]:
                    d = df.copy()
                    d["hybrid_score"] = d["candidate_rank"] + alpha * d["abs_rt_delta"] / tau
                    keep = (d["abs_rt_delta"] <= th) | (d["candidate_rank"] <= g)
                    q, s = eval_ranked(
                        d,
                        method=f"rank_guard_filter_soft_th{th}_g{g}_tau{tau}_alpha{alpha}",
                        keep_mask=keep,
                        sort_cols=["hybrid_score", "candidate_rank", "candidate_score"],
                        ascending=[True, True, False],
                    )
                    s["threshold_sec"] = th
                    s["guard_k"] = g
                    s["tau"] = tau
                    s["alpha"] = alpha
                    summaries.append(s)

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "rank_guard_soft_rerank_summary.csv", index=False)

    print("=" * 100)
    print("input:", pred_csv)
    print("rows:", len(df), "queries:", df["s10_row"].nunique())
    print("saved:", out_dir / "rank_guard_soft_rerank_summary.csv")
    print("=" * 100)

    print("\n[Original]")
    print(summary[summary["method"].eq("original_msfinder_rank")].to_string(index=False))

    print("\n[Hard filter key rows]")
    show_hard = summary[
        summary["method"].str.startswith("hard_rt_filter")
        & summary["threshold_sec"].isin([75.17, 100.0, 185.31, 214.28, 300.0, 500.0])
    ].copy()
    print(show_hard.to_string(index=False))

    print("\n[Rank-guard filter key rows: guard_k=5 or 10]")
    show_guard = summary[
        summary["method"].str.startswith("rank_guard_filter_th")
        & summary["threshold_sec"].isin([100.0, 185.31, 214.28, 300.0])
        & summary["guard_k"].isin([5, 10])
    ].copy()
    print(show_guard.to_string(index=False))

    print("\n[Best soft rerank by top1/top5/top10]")
    soft = summary[summary["method"].str.startswith("soft_rerank")].copy()
    for metric in ["top1_after_pct", "top5_after_pct", "top10_after_pct"]:
        print(f"\nBest by {metric}:")
        print(
            soft.sort_values(
                [metric, "top1_after_pct", "top5_after_pct", "top10_after_pct"],
                ascending=[False, False, False, False],
            ).head(10).to_string(index=False)
        )

    print("\n[Best rank-guard + soft by top1/top5/top10]")
    gs = summary[summary["method"].str.startswith("rank_guard_filter_soft")].copy()
    for metric in ["top1_after_pct", "top5_after_pct", "top10_after_pct"]:
        print(f"\nBest by {metric}:")
        print(
            gs.sort_values(
                [metric, "candidate_reduction_pct", "top1_after_pct", "top5_after_pct", "top10_after_pct"],
                ascending=[False, False, False, False, False],
            ).head(10).to_string(index=False)
        )

    print("=" * 100)


if __name__ == "__main__":
    main()

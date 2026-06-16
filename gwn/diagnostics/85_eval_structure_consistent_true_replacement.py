#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd

CAND = Path("experiments_candidate_filtering/metabobase_s10_predictions_tl_exact39/metabobase_s10_candidate_predictions_tl_seed42.csv")
META = Path("experiments_candidate_filtering/metabobase_tl_exact39/metabobase_test_exact39_metadata.csv")
OFFICIAL_PRED = Path("experiments_candidate_filtering/metabobase_tl_exact39_training/seed42/dualview_avg/test_predictions_avg.csv")

OUT = Path("experiments_candidate_filtering/metabobase_s10_structure_consistent_audit_seed42")
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [75.17, 100.0, 136.34, 150.0, 185.31, 214.28, 250.0, 300.0, 400.0, 500.0]


def true_rank_before(sub):
    t = sub[sub["is_true"].astype(bool)]
    if t.empty:
        return None
    return int(t["candidate_rank"].min())


def rank_after_filter(sub, delta_col):
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


def eval_threshold(df, delta_col, threshold):
    d = df.copy()
    d["kept"] = d[delta_col].astype(float) <= float(threshold)

    rows = []
    for s10_row, sub in d.groupby("s10_row", sort=True):
        sub = sub.sort_values("candidate_rank").copy()
        rb = true_rank_before(sub)
        ra = rank_after_filter(sub, delta_col)

        true_delta = np.nan
        if rb is not None:
            tr = sub[sub["is_true"].astype(bool)].sort_values("candidate_rank").iloc[0]
            true_delta = float(tr[delta_col])

        rows.append({
            "s10_row": int(s10_row),
            "true_name": sub["true_name"].iloc[0],
            "rt_sec": float(sub["rt_sec"].iloc[0]),
            "n_candidates_before": len(sub),
            "n_candidates_after": int(sub["kept"].sum()),
            "n_filtered": len(sub) - int(sub["kept"].sum()),
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

    total_before = int(q["n_candidates_before"].sum())
    total_after = int(q["n_candidates_after"].sum())

    return {
        "threshold_sec": float(threshold),
        "n_queries": len(q),
        "n_candidate_rows_before": total_before,
        "n_candidate_rows_after": total_after,
        "candidate_reduction_pct": 100.0 * (total_before - total_after) / max(total_before, 1),
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
    cand = pd.read_csv(CAND)
    meta = pd.read_csv(META).reset_index(drop=True)
    off = pd.read_csv(OFFICIAL_PRED).reset_index(drop=True)

    cand["is_true"] = cand["is_true"].astype(bool)

    official = pd.DataFrame({
        "official_inchikey": meta["inchikey"].astype(str),
        "official_name": meta["name"].astype(str),
        "official_rt_sec": meta["rt_sec"].astype(float),
        "official_pred_rt": off["avg_pred"].astype(float),
        "official_abs_err": off["abs_err"].astype(float),
    })

    d = cand.copy()
    d["candidate_pred_rt_original"] = d["candidate_pred_rt"].astype(float)
    d["abs_rt_delta_original"] = d["abs_rt_delta"].astype(float)

    # 只对 true row 做诊断替换
    d = d.merge(
        official,
        left_on="true_inchikey",
        right_on="official_inchikey",
        how="left",
    )

    replace_mask = d["is_true"] & d["official_pred_rt"].notna()
    d["candidate_pred_rt_structure_consistent"] = d["candidate_pred_rt_original"]
    d.loc[replace_mask, "candidate_pred_rt_structure_consistent"] = d.loc[replace_mask, "official_pred_rt"]

    d["abs_rt_delta_structure_consistent"] = (
        d["candidate_pred_rt_structure_consistent"].astype(float) - d["rt_sec"].astype(float)
    ).abs()

    d.to_csv(OUT / "candidate_predictions_true_replaced_by_official_prediction.csv", index=False)

    summaries = []
    for mode, delta_col in [
        ("original_tl_candidate_prediction", "abs_rt_delta_original"),
        ("true_replaced_by_official_prediction_AUDIT_ONLY", "abs_rt_delta_structure_consistent"),
    ]:
        for th in THRESHOLDS:
            s = eval_threshold(d, delta_col, th)
            s["mode"] = mode
            summaries.append(s)

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "structure_consistency_filtering_summary.csv", index=False)

    true_rows = d[d["is_true"]].copy()
    true_rows["delta_improvement"] = true_rows["abs_rt_delta_original"] - true_rows["abs_rt_delta_structure_consistent"]
    true_rows = true_rows.sort_values("delta_improvement", ascending=False)
    true_rows.to_csv(OUT / "true_rows_delta_improvement.csv", index=False)

    print("=" * 100)
    print("candidate rows:", len(d))
    print("queries:", d["s10_row"].nunique())
    print("true rows:", int(d["is_true"].sum()))
    print("true rows replaced by official prediction:", int(replace_mask.sum()))
    print("=" * 100)

    print("\nTrue-row delta improvement top:")
    cols = [
        "s10_row", "true_name", "candidate_rank", "candidate_name", "rt_sec",
        "candidate_pred_rt_original", "abs_rt_delta_original",
        "official_pred_rt", "official_abs_err",
        "candidate_pred_rt_structure_consistent",
        "abs_rt_delta_structure_consistent",
        "delta_improvement",
        "true_inchikey", "candidate_inchikey", "official_inchikey",
    ]
    cols = [c for c in cols if c in true_rows.columns]
    print(true_rows[cols].head(20).to_string(index=False))

    print("\nSummary:")
    show = summary[
        summary["threshold_sec"].isin([75.17, 100.0, 136.34, 185.31, 214.28, 300.0, 500.0])
    ].copy()
    show_cols = [
        "mode", "threshold_sec", "candidate_reduction_pct", "true_retention_pct",
        "top1_before_pct", "top1_after_pct",
        "top5_before_pct", "top5_after_pct",
        "top10_before_pct", "top10_after_pct",
        "mean_true_abs_rt_delta", "median_true_abs_rt_delta", "max_true_abs_rt_delta",
    ]
    print(show[show_cols].to_string(index=False))

    print("\nSaved:")
    print(OUT / "structure_consistency_filtering_summary.csv")
    print(OUT / "true_rows_delta_improvement.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

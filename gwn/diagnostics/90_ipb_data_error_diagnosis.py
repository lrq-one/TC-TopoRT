from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


DATASET = "IPB_Halle_82"
META = Path("paper_analysis_stage4_external/external_predret10_stage4_meta.csv")
OUT = Path("paper_analysis_stage4S_IPB_data_diagnosis")
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "old_zscore_rtfull": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_IPB_Halle_82_src0",
    "zscore_rtfull_lr1e4_wd5e2": "paper_analysis_stage4P_IPB_rtfull_lr1e4_wd5e2_src0",
    "zscore_rtfull_lr5e5_wd1e1": "paper_analysis_stage4P_IPB_rtfull_lr5e5_wd1e1_src0",
    "zscore_rtfull_lr2e5_wd5e2": "paper_analysis_stage4P_IPB_rtfull_lr2e5_wd5e2_src0",
    "raw_rtfull_lr1e4_wd5e2_huber10": "paper_analysis_stage4Q_IPB_raw_rtfull_lr1e4_wd5e2_huber10_src0",
    "zscore_outlin_lr1e3_wd1e2": "paper_analysis_stage4R_IPB_zscore_outlin_lr1e3_wd1e2_src0",
}


def safe_read_csv(p):
    p = Path(p)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print("[BAD CSV]", p, e)
        return None


def assign_cv_fold(meta):
    sub = meta[meta["dataset_name"] == DATASET].copy().reset_index(drop=True)
    sub["cv_fold"] = -1
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for fold, (_, te) in enumerate(kf.split(np.zeros(len(sub)))):
        sub.loc[te, "cv_fold"] = fold
    return sub


def mae(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(y - p)))


def main():
    meta = pd.read_csv(META)
    ipb_meta = assign_cv_fold(meta)

    metric_rows = []
    pred_tables = {}

    for name, d in CANDIDATES.items():
        d = Path(d)
        m = safe_read_csv(d / "external_tl_metrics_by_run.csv")
        p = safe_read_csv(d / "external_tl_predictions.csv")

        if m is None or p is None:
            print("[SKIP missing/incomplete]", name, d)
            continue

        for _, r in m.iterrows():
            metric_rows.append({
                "candidate": name,
                "dir": str(d),
                "method": r.get("method", ""),
                "mae": r.get("mae", np.nan),
                "rmse": r.get("rmse", np.nan),
                "r2": r.get("r2", np.nan),
                "spearman": r.get("spearman", np.nan),
            })

        p = p.merge(
            ipb_meta[["stage4_index", "cv_fold"]],
            on="stage4_index",
            how="left"
        )
        pred_tables[name] = p

    metric_df = pd.DataFrame(metric_rows)
    metric_df = metric_df.sort_values(["method", "mae"])
    metric_df.to_csv(OUT / "ipb_candidate_metrics.csv", index=False)

    print("\n=== CANDIDATE METRICS ===")
    print(metric_df.to_string(index=False))

    mean_metrics = metric_df[metric_df["method"] == "mean_tl"].copy()
    if len(mean_metrics) == 0:
        raise RuntimeError("No mean_tl metrics found")

    best_name = mean_metrics.sort_values("mae").iloc[0]["candidate"]
    print("\n=== BEST CANDIDATE ===")
    print(best_name)

    best = pred_tables[best_name].copy()

    # 1) fold 诊断
    fold_rows = []
    for fold, sub in best.groupby("cv_fold"):
        y = sub["rt"].values
        row = {
            "candidate": best_name,
            "cv_fold": int(fold),
            "n": len(sub),
            "rt_min": float(np.min(y)),
            "rt_max": float(np.max(y)),
            "rt_mean": float(np.mean(y)),
            "rt_std": float(np.std(y)),
            "origin_mae": mae(sub["rt"], sub["origin_tl_pred"]),
            "taut_mae": mae(sub["rt"], sub["taut_tl_pred"]),
            "mean_mae": mae(sub["rt"], sub["mean_tl_pred"]),
            "max_mean_abs_error": float(sub["mean_tl_abs_error"].max()),
            "num_err_gt_20": int((sub["mean_tl_abs_error"] > 20).sum()),
            "num_err_gt_30": int((sub["mean_tl_abs_error"] > 30).sum()),
            "num_err_gt_40": int((sub["mean_tl_abs_error"] > 40).sum()),
        }
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows).sort_values("mean_mae", ascending=False)
    fold_df.to_csv(OUT / "ipb_best_fold_diagnosis.csv", index=False)

    print("\n=== BEST FOLD DIAGNOSIS ===")
    print(fold_df.to_string(index=False))

    # 2) best run top error samples
    keep_cols = [
        "cv_fold", "stage4_index", "record_id", "name", "formula",
        "inchikey", "rt", "origin_tl_pred", "taut_tl_pred", "mean_tl_pred",
        "origin_tl_abs_error", "taut_tl_abs_error", "mean_tl_abs_error",
        "taut_changed", "smrt_exact_overlap", "origin_smiles", "taut_smiles",
    ]
    keep_cols = [c for c in keep_cols if c in best.columns]

    top_err = best[keep_cols].sort_values("mean_tl_abs_error", ascending=False)
    top_err.to_csv(OUT / "ipb_best_top_errors.csv", index=False)

    print("\n=== TOP 20 ERRORS IN BEST CANDIDATE ===")
    print(top_err.head(20).to_string(index=False))

    # 3) 多个候选共同 hard sample
    common = ipb_meta.copy()
    common = common[["stage4_index", "cv_fold", "record_id", "name", "formula", "inchikey", "rt"] + 
                    [c for c in ["taut_changed", "smrt_exact_overlap", "origin_smiles", "taut_smiles"] if c in ipb_meta.columns]]

    for cand, p in pred_tables.items():
        small = p[["stage4_index", "mean_tl_pred", "mean_tl_abs_error"]].copy()
        small = small.rename(columns={
            "mean_tl_pred": f"{cand}__pred",
            "mean_tl_abs_error": f"{cand}__abs_err",
        })
        common = common.merge(small, on="stage4_index", how="left")

    err_cols = [c for c in common.columns if c.endswith("__abs_err")]
    common["hard_count_gt20"] = common[err_cols].gt(20).sum(axis=1)
    common["hard_count_gt30"] = common[err_cols].gt(30).sum(axis=1)
    common["mean_abs_err_across_candidates"] = common[err_cols].mean(axis=1)

    common = common.sort_values(
        ["hard_count_gt30", "hard_count_gt20", "mean_abs_err_across_candidates"],
        ascending=False
    )
    common.to_csv(OUT / "ipb_common_hard_samples.csv", index=False)

    print("\n=== COMMON HARD SAMPLES TOP 20 ===")
    show_cols = ["cv_fold", "stage4_index", "record_id", "name", "formula", "rt",
                 "hard_count_gt20", "hard_count_gt30", "mean_abs_err_across_candidates"]
    show_cols = [c for c in show_cols if c in common.columns]
    print(common[show_cols].head(20).to_string(index=False))

    # 4) origin/taut bias pattern
    b = best.copy()
    b["origin_signed_err"] = b["origin_tl_pred"] - b["rt"]
    b["taut_signed_err"] = b["taut_tl_pred"] - b["rt"]
    b["mean_signed_err"] = b["mean_tl_pred"] - b["rt"]
    b["view_gap_abs"] = np.abs(b["origin_tl_pred"] - b["taut_tl_pred"])

    def category(r):
        if r["origin_signed_err"] > 0 and r["taut_signed_err"] > 0:
            return "both_over"
        if r["origin_signed_err"] < 0 and r["taut_signed_err"] < 0:
            return "both_under"
        return "view_disagree"

    b["view_error_category"] = b.apply(category, axis=1)

    cat = b.groupby("view_error_category").agg(
        n=("stage4_index", "count"),
        mean_mae=("mean_tl_abs_error", "mean"),
        max_mae=("mean_tl_abs_error", "max"),
        mean_view_gap=("view_gap_abs", "mean"),
    ).reset_index()
    cat.to_csv(OUT / "ipb_view_bias_categories.csv", index=False)

    print("\n=== VIEW ERROR CATEGORIES ===")
    print(cat.to_string(index=False))

    print("\n=== SAVED ===")
    for f in [
        "ipb_candidate_metrics.csv",
        "ipb_best_fold_diagnosis.csv",
        "ipb_best_top_errors.csv",
        "ipb_common_hard_samples.csv",
        "ipb_view_bias_categories.csv",
    ]:
        print(OUT / f)


if __name__ == "__main__":
    main()

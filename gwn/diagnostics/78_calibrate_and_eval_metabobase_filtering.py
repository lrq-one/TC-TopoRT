#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, HuberRegressor, RidgeCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CALIB = Path("experiments_candidate_filtering/metabobase_calibration_predictions/metabobase_calibration_true_predictions_final.csv")
CAND = Path("experiments_candidate_filtering/metabobase_s10_predictions/metabobase_s10_candidate_predictions_final.csv")
OUT = Path("experiments_candidate_filtering/metabobase_calibrated_filtering_eval")
OUT.mkdir(parents=True, exist_ok=True)


def get_feature_matrix(df):
    cols = [
        "pred_origin_mean",
        "pred_origin_std",
        "pred_taut_mean",
        "pred_taut_std",
        "pred_final_mean",
        "pred_final_std",
    ]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = ["candidate_pred_rt"]
    X = df[cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X.to_numpy(float), cols


def add_metrics(name, y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "method": name,
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": r2_score(y, pred),
        "bias": float(np.mean(pred - y)),
        "MedAE": float(np.median(np.abs(pred - y))),
        "P90_AE": float(np.percentile(np.abs(pred - y), 90)),
        "P95_AE": float(np.percentile(np.abs(pred - y), 95)),
    }


def cv_mae_estimator(model, X, y, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pred = np.zeros_like(y, dtype=float)
    for tr, va in kf.split(X):
        m = clone(model)
        m.fit(X[tr], y[tr])
        pred[va] = m.predict(X[va])
    return mean_absolute_error(y, pred), pred


def fit_predict_isotonic(x_train, y_train, x_all):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x_train.reshape(-1), y_train)
    return iso.predict(x_all.reshape(-1)), iso


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


def topk(rank, k):
    if rank is None or pd.isna(rank):
        return False
    return int(rank) <= k


def eval_filter(df, pred_col, threshold):
    d = df.copy()
    d["candidate_pred_rt_eval"] = d[pred_col].astype(float)
    d["abs_rt_delta_eval"] = (d["candidate_pred_rt_eval"] - d["rt_sec"].astype(float)).abs()
    d["kept"] = d["abs_rt_delta_eval"] <= float(threshold)

    rows = []
    for s10_row, sub in d.groupby("s10_row", sort=True):
        sub = sub.sort_values("candidate_rank").copy()
        rb = true_rank_before(sub)
        ra = rank_after_filter(sub)

        true_delta = np.nan
        if rb is not None:
            true_delta = float(sub[sub["is_true"].astype(bool)].sort_values("candidate_rank").iloc[0]["abs_rt_delta_eval"])

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
            "true_abs_rt_delta_eval": true_delta,
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
        "pred_col": pred_col,
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
        "mean_true_abs_rt_delta_eval": float(q["true_abs_rt_delta_eval"].mean()),
        "median_true_abs_rt_delta_eval": float(q["true_abs_rt_delta_eval"].median()),
    }
    return d, q, summary


def main():
    calib = pd.read_csv(CALIB)
    cand = pd.read_csv(CAND)

    train = calib[calib["split"].eq("calib_train")].copy()
    test = calib[calib["split"].eq("s10_test_matched")].copy()

    X_train, feat_cols = get_feature_matrix(train)
    X_all_calib, _ = get_feature_matrix(calib)
    X_test, _ = get_feature_matrix(test)
    X_cand, _ = get_feature_matrix(cand)

    y_train = train["rt_sec"].to_numpy(float)
    y_calib_all = calib["rt_sec"].to_numpy(float)
    y_test = test["rt_sec"].to_numpy(float)

    print("=" * 100)
    print("Calibration train rows:", len(train))
    print("S10 matched true rows:", len(test))
    print("Candidate rows:", len(cand))
    print("Feature columns:", feat_cols)
    print("=" * 100)

    models = {
        "linear_1d_final": LinearRegression(),
        "huber_1d_final": HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=5000),
        "ridge_multifeat": make_pipeline(StandardScaler(), RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
        "huber_multifeat": make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=5000)),
    }

    # 1D final feature for simple models
    x1_train = train["candidate_pred_rt"].to_numpy(float).reshape(-1, 1)
    x1_all = calib["candidate_pred_rt"].to_numpy(float).reshape(-1, 1)
    x1_test = test["candidate_pred_rt"].to_numpy(float).reshape(-1, 1)
    x1_cand = cand["candidate_pred_rt"].to_numpy(float).reshape(-1, 1)

    calib_metrics = []
    cand_out = cand.copy()

    fitted = {}

    # raw baseline
    calib["pred_raw"] = calib["candidate_pred_rt"].astype(float)
    cand_out["pred_raw"] = cand_out["candidate_pred_rt"].astype(float)
    calib_metrics.append(add_metrics("raw_smrt_direct_train", y_train, train["candidate_pred_rt"].to_numpy(float)))
    calib_metrics.append(add_metrics("raw_smrt_direct_s10", y_test, test["candidate_pred_rt"].to_numpy(float)))

    for name, model in models.items():
        if "1d" in name:
            Xtr, Xall, Xt, Xc = x1_train, x1_all, x1_test, x1_cand
        else:
            Xtr, Xall, Xt, Xc = X_train, X_all_calib, X_test, X_cand

        cv_mae, cv_pred = cv_mae_estimator(model, Xtr, y_train)
        m = clone(model)
        m.fit(Xtr, y_train)

        calib[f"pred_{name}"] = m.predict(Xall)
        cand_out[f"pred_{name}"] = m.predict(Xc)
        fitted[name] = m

        calib_metrics.append({
            **add_metrics(f"{name}_train_fit", y_train, m.predict(Xtr)),
            "cv_mae_train": cv_mae,
        })
        calib_metrics.append({
            **add_metrics(f"{name}_s10_eval", y_test, m.predict(Xt)),
            "cv_mae_train": cv_mae,
        })

    # isotonic on 1D final
    iso_pred_all, iso = fit_predict_isotonic(
        train["candidate_pred_rt"].to_numpy(float),
        y_train,
        calib["candidate_pred_rt"].to_numpy(float),
    )
    iso_pred_cand = iso.predict(cand["candidate_pred_rt"].to_numpy(float))
    calib["pred_isotonic_1d"] = iso_pred_all
    cand_out["pred_isotonic_1d"] = iso_pred_cand

    # KFold CV for isotonic
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    iso_cv_pred = np.zeros_like(y_train, dtype=float)
    raw_train_pred = train["candidate_pred_rt"].to_numpy(float)
    for tr, va in kf.split(raw_train_pred):
        pred_va, _ = fit_predict_isotonic(raw_train_pred[tr], y_train[tr], raw_train_pred[va])
        iso_cv_pred[va] = pred_va
    iso_cv_mae = mean_absolute_error(y_train, iso_cv_pred)

    calib_metrics.append({
        **add_metrics("isotonic_1d_train_fit", y_train, iso.predict(raw_train_pred)),
        "cv_mae_train": iso_cv_mae,
    })
    calib_metrics.append({
        **add_metrics("isotonic_1d_s10_eval", y_test, iso.predict(test["candidate_pred_rt"].to_numpy(float))),
        "cv_mae_train": iso_cv_mae,
    })

    metrics_df = pd.DataFrame(calib_metrics)
    metrics_df.to_csv(OUT / "metabobase_calibration_metrics.csv", index=False)

    calib.to_csv(OUT / "metabobase_true_predictions_calibrated.csv", index=False)
    cand_out.to_csv(OUT / "metabobase_s10_candidate_predictions_calibrated.csv", index=False)

    print("\nCalibration metrics:")
    print(metrics_df.to_string(index=False))

    # choose best model by train CV MAE among calibrated models
    candidates = metrics_df[metrics_df["method"].str.endswith("_train_fit")].copy()
    candidates = candidates.dropna(subset=["cv_mae_train"])
    best_train_row = candidates.sort_values("cv_mae_train").iloc[0]
    best_name = best_train_row["method"].replace("_train_fit", "")
    best_col = f"pred_{best_name}"

    print("\nBest calibration by calib_train CV MAE:")
    print(best_name)
    print("best_col:", best_col)
    print("cv_mae_train:", float(best_train_row["cv_mae_train"]))

    # thresholds: fixed ABCoRT threshold + 3*CV MAE + broad sweep
    best_cv = float(best_train_row["cv_mae_train"])
    thresholds = sorted(set([
        75.17, 100.0, 150.0, 185.31, 200.0, 250.0, 300.0, 400.0, 500.0,
        round(3.0 * best_cv, 2),
    ]))

    pred_cols = ["pred_raw", best_col, "pred_linear_1d_final", "pred_huber_1d_final", "pred_ridge_multifeat", "pred_huber_multifeat", "pred_isotonic_1d"]
    pred_cols = [c for c in pred_cols if c in cand_out.columns]
    pred_cols = list(dict.fromkeys(pred_cols))

    summaries = []
    for pc in pred_cols:
        for th in thresholds:
            d, q, s = eval_filter(cand_out.rename(columns={pc: "TEMP_PRED"}), "TEMP_PRED", th)
            s["calibration_method"] = pc.replace("pred_", "")
            summaries.append(s)

            tag = f"{pc}_th_{str(th).replace('.', 'p')}"
            q.to_csv(OUT / f"query_summary_{tag}.csv", index=False)

    filt_summary = pd.DataFrame(summaries)
    filt_summary = filt_summary[
        [
            "calibration_method", "pred_col", "threshold_sec", "n_queries",
            "n_candidate_rows_before", "n_candidate_rows_after",
            "candidate_reduction_pct", "true_retention_pct_among_found",
            "top1_before_pct", "top1_after_pct",
            "top5_before_pct", "top5_after_pct",
            "top10_before_pct", "top10_after_pct",
            "mean_true_abs_rt_delta_eval", "median_true_abs_rt_delta_eval",
        ]
    ]
    filt_summary.to_csv(OUT / "metabobase_calibrated_filtering_summary.csv", index=False)

    print("\nFiltering summary:")
    show = filt_summary[
        (filt_summary["calibration_method"].isin(["raw", best_name, "huber_1d_final", "ridge_multifeat", "isotonic_1d"]))
        & (filt_summary["threshold_sec"].isin([185.31, round(3.0 * best_cv, 2), 300.0, 400.0, 500.0]))
    ].copy()
    print(show.to_string(index=False))

    print("\nSaved:")
    print(OUT / "metabobase_calibration_metrics.csv")
    print(OUT / "metabobase_s10_candidate_predictions_calibrated.csv")
    print(OUT / "metabobase_calibrated_filtering_summary.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

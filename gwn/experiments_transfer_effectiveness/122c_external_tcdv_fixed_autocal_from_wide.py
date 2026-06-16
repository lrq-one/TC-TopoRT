#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
122c_external_tcdv_fixed_autocal_from_wide.py

No-leak fixed AutoCal / AutoSelect for external TCDV transfer predictions.

Input:
  external_tl_predictions.csv from 119_external_tcdv_scratch_vs_tl.py

This script:
1. Reconstructs external cv_fold using the same KFold(cv_seed).
2. Builds source-fold/view prediction candidates.
3. For each outer held-out fold:
   - uses only other folds to select candidate + calibrator
   - applies the selected calibrator to held-out fold
4. Reports no-leak results.

Important:
No test-fold labels are used for selecting/calibrating that test fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression


def metric_row(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n": 0, "mae": np.nan, "medae": np.nan, "rmse": np.nan,
            "r2": np.nan, "pearson": np.nan, "spearman": np.nan, "bias": np.nan,
        }

    err = np.abs(y_true - y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(err)),
        "medae": float(np.median(err)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "pearson": float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")) if len(y_true) > 1 else np.nan,
        "spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")) if len(y_true) > 1 else np.nan,
        "bias": float(np.mean(y_pred - y_true)),
    }


def reconstruct_cv_fold(df: pd.DataFrame, cv_seed: int, cv_folds: int) -> pd.DataFrame:
    df = df.copy()
    maps = []

    for ds, sub_all in df.groupby("dataset_name"):
        unique = (
            sub_all[["dataset_name", "stage4_index"]]
            .drop_duplicates()
            .sort_values("stage4_index")
            .reset_index(drop=True)
        )

        k = min(int(cv_folds), len(unique))
        cv = KFold(n_splits=k, shuffle=True, random_state=int(cv_seed))
        fold_id = np.full(len(unique), -1, dtype=int)

        for f, (_, te_idx) in enumerate(cv.split(np.zeros(len(unique)))):
            fold_id[te_idx] = f

        unique["cv_fold"] = fold_id
        maps.append(unique)

    fmap = pd.concat(maps, ignore_index=True)
    out = df.merge(fmap, on=["dataset_name", "stage4_index"], how="left")

    if out["cv_fold"].isna().any():
        raise RuntimeError("cv_fold reconstruction failed.")

    out["cv_fold"] = out["cv_fold"].astype(int)
    return out


def trim_mean(arr, axis=1):
    arr = np.asarray(arr, dtype=float)
    if arr.shape[axis] <= 2:
        return np.mean(arr, axis=axis)
    s = np.sort(arr, axis=axis)
    if axis == 1:
        return np.mean(s[:, 1:-1], axis=1)
    return np.mean(s[1:-1], axis=0)


def build_bank(df: pd.DataFrame, source_folds):
    required = [
        "dataset_name", "stage4_index", "rt", "run_key", "source_fold", "cv_fold",
        "origin_tl_pred", "taut_tl_pred",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df[df["source_fold"].astype(int).isin([int(x) for x in source_folds])].copy()
    df["source_fold"] = df["source_fold"].astype(int)
    df["run_key"] = df["run_key"].astype(str)

    long_rows = []
    for pred_col, prefix in [
        ("origin_tl_pred", "origin"),
        ("taut_tl_pred", "taut"),
    ]:
        tmp = df[["dataset_name", "stage4_index", "rt", "cv_fold", "run_key", "source_fold", pred_col]].copy()
        tmp["feat_name"] = prefix + "_" + tmp["run_key"].astype(str) + "_src" + tmp["source_fold"].astype(str)
        tmp = tmp.rename(columns={pred_col: "pred"})
        long_rows.append(tmp)

    long_df = pd.concat(long_rows, ignore_index=True)

    bank = (
        long_df.pivot_table(
            index=["dataset_name", "stage4_index", "rt", "cv_fold"],
            columns="feat_name",
            values="pred",
            aggfunc="mean",
        )
        .reset_index()
    )
    bank.columns.name = None

    origin_cols = sorted([c for c in bank.columns if c.startswith("origin_")])
    taut_cols = sorted([c for c in bank.columns if c.startswith("taut_")])

    taut_by_suffix = {c.replace("taut_", ""): c for c in taut_cols}
    matched_origin, matched_taut = [], []
    for oc in origin_cols:
        suf = oc.replace("origin_", "")
        tc = taut_by_suffix.get(suf)
        if tc is not None:
            matched_origin.append(oc)
            matched_taut.append(tc)

    if not matched_origin:
        raise RuntimeError("No matched origin/taut source-fold predictions.")

    O = bank[matched_origin].astype(float).values
    T = bank[matched_taut].astype(float).values
    P = 0.5 * (O + T)

    # 固定候选，不按数据集手动更换
    bank["cand_origin_mean"] = O.mean(axis=1)
    bank["cand_taut_mean"] = T.mean(axis=1)
    bank["cand_pair_mean"] = P.mean(axis=1)

    bank["cand_origin_median"] = np.median(O, axis=1)
    bank["cand_taut_median"] = np.median(T, axis=1)
    bank["cand_pair_median"] = np.median(P, axis=1)

    bank["cand_origin_trimmean"] = trim_mean(O, axis=1)
    bank["cand_taut_trimmean"] = trim_mean(T, axis=1)
    bank["cand_pair_trimmean"] = trim_mean(P, axis=1)

    # view disagreement 有时候说明 taut view 不可靠，但这里不作为学习特征，只保存诊断
    bank["diag_abs_origin_taut_delta"] = np.abs(bank["cand_origin_mean"] - bank["cand_taut_mean"])

    cand_cols = [
        "cand_origin_mean",
        "cand_taut_mean",
        "cand_pair_mean",
        "cand_origin_median",
        "cand_taut_median",
        "cand_pair_median",
        "cand_origin_trimmean",
        "cand_taut_trimmean",
        "cand_pair_trimmean",
    ]

    if bank[cand_cols].isna().any().any():
        bad = bank[cand_cols].columns[bank[cand_cols].isna().any()].tolist()
        raise RuntimeError(f"NaN in candidate cols: {bad}")

    return bank, cand_cols, matched_origin, matched_taut


def fit_predict_calibrator(calib_mode, p_train, y_train, p_test):
    p_train = np.asarray(p_train, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    p_test = np.asarray(p_test, dtype=float).reshape(-1)

    if calib_mode == "raw":
        return p_train.copy(), p_test.copy(), {"a": 1.0, "b": 0.0}

    if calib_mode == "bias":
        b = float(np.mean(y_train - p_train))
        return p_train + b, p_test + b, {"a": 1.0, "b": b}

    X_train = p_train.reshape(-1, 1)
    X_test = p_test.reshape(-1, 1)

    if calib_mode == "linear":
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model.predict(X_train), model.predict(X_test), {
            "a": float(model.coef_[0]),
            "b": float(model.intercept_),
        }

    if calib_mode == "ridge":
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        return model.predict(X_train), model.predict(X_test), {
            "a": float(model.coef_[0]),
            "b": float(model.intercept_),
        }

    if calib_mode == "huber":
        model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=5000)
        model.fit(X_train, y_train)
        return model.predict(X_train), model.predict(X_test), {
            "a": float(model.coef_[0]),
            "b": float(model.intercept_),
        }

    raise ValueError(f"Unknown calib_mode={calib_mode}")


def noleak_autocal_one_dataset(ddf, cand_cols, calib_modes, selection_metric="mae"):
    pred_rows = []
    fold_rows = []

    for f in sorted(ddf["cv_fold"].unique()):
        train = ddf[ddf["cv_fold"] != f].copy()
        test = ddf[ddf["cv_fold"] == f].copy()

        y_train = train["rt"].values.astype(float)
        y_test = test["rt"].values.astype(float)

        candidates = []

        for cand in cand_cols:
            p_train_raw = train[cand].values.astype(float)
            p_test_raw = test[cand].values.astype(float)

            for mode in calib_modes:
                try:
                    p_train_cal, p_test_cal, params = fit_predict_calibrator(
                        mode, p_train_raw, y_train, p_test_raw
                    )
                except Exception:
                    continue

                m_train = metric_row(y_train, p_train_cal)
                score = m_train[selection_metric]

                candidates.append({
                    "cand": cand,
                    "calib_mode": mode,
                    "train_score": score,
                    "train_mae": m_train["mae"],
                    "train_medae": m_train["medae"],
                    "p_test": p_test_cal,
                    "a": params.get("a", np.nan),
                    "b": params.get("b", np.nan),
                })

        if not candidates:
            raise RuntimeError(f"No candidates for dataset={test['dataset_name'].iloc[0]} fold={f}")

        candidates = sorted(candidates, key=lambda x: x["train_score"])
        best = candidates[0]

        tmp = test[["dataset_name", "stage4_index", "cv_fold", "rt"] + cand_cols].copy()
        tmp["y_pred_autocal"] = best["p_test"]
        tmp["selected_candidate"] = best["cand"]
        tmp["selected_calib"] = best["calib_mode"]
        tmp["selected_train_score"] = best["train_score"]
        tmp["selected_a"] = best["a"]
        tmp["selected_b"] = best["b"]
        pred_rows.append(tmp)

        m_test = metric_row(y_test, best["p_test"])
        m_test.update({
            "dataset_name": test["dataset_name"].iloc[0],
            "cv_fold": int(f),
            "selected_candidate": best["cand"],
            "selected_calib": best["calib_mode"],
            "selected_train_mae": best["train_mae"],
            "selected_train_medae": best["train_medae"],
            "selected_a": best["a"],
            "selected_b": best["b"],
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        })
        fold_rows.append(m_test)

    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(fold_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cv_seed", type=int, required=True)
    ap.add_argument("--cv_folds", type=int, default=10)
    ap.add_argument("--source_folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--calib_modes", nargs="+", default=["raw", "bias", "ridge", "huber"])
    ap.add_argument("--selection_metric", choices=["mae", "medae"], default="mae")
    args = ap.parse_args()

    pred_csv = Path(args.pred_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)
    print("=== Load ===")
    print("pred_csv:", pred_csv)
    print("shape:", df.shape)

    df = reconstruct_cv_fold(df, args.cv_seed, args.cv_folds)
    bank, cand_cols, origin_cols, taut_cols = build_bank(df, args.source_folds)

    print("\n=== Bank ===")
    print("bank:", bank.shape)
    print("datasets:", sorted(bank["dataset_name"].unique().tolist()))
    print("origin_cols:", origin_cols)
    print("taut_cols:", taut_cols)
    print("cand_cols:", cand_cols)
    print("calib_modes:", args.calib_modes)
    print("selection_metric:", args.selection_metric)

    bank.to_csv(out_dir / "tcdv_autocal_prediction_bank.csv", index=False)

    all_preds = []
    all_folds = []
    summary_rows = []

    for ds, ddf in bank.groupby("dataset_name"):
        ddf = ddf.copy()

        pred_df, fold_df = noleak_autocal_one_dataset(
            ddf,
            cand_cols=cand_cols,
            calib_modes=args.calib_modes,
            selection_metric=args.selection_metric,
        )

        all_preds.append(pred_df)
        all_folds.append(fold_df)

        # auto result
        m = metric_row(pred_df["rt"].values, pred_df["y_pred_autocal"].values)
        m.update({"dataset_name": ds, "method": "tcdv_fixed_noleak_autocal"})
        summary_rows.append(m)

        # raw fixed candidates
        for col in cand_cols:
            mb = metric_row(ddf["rt"].values, ddf[col].values)
            mb.update({"dataset_name": ds, "method": col})
            summary_rows.append(mb)

    all_preds = pd.concat(all_preds, ignore_index=True)
    all_folds = pd.concat(all_folds, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    order = {"tcdv_fixed_noleak_autocal": 0}
    summary["method_order"] = summary["method"].map(order).fillna(1)
    summary = summary.sort_values(["dataset_name", "method_order", "mae"]).drop(columns=["method_order"])

    all_preds.to_csv(out_dir / "tcdv_fixed_noleak_autocal_predictions.csv", index=False)
    all_folds.to_csv(out_dir / "tcdv_fixed_noleak_autocal_fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "tcdv_fixed_noleak_autocal_summary.csv", index=False)

    # selection count
    sel = (
        all_folds.groupby(["dataset_name", "selected_candidate", "selected_calib"])
        .size()
        .reset_index(name="n_folds")
        .sort_values(["dataset_name", "n_folds"], ascending=[True, False])
    )
    sel.to_csv(out_dir / "tcdv_fixed_noleak_autocal_selection_counts.csv", index=False)

    meta = {
        "pred_csv": str(pred_csv),
        "cv_seed": int(args.cv_seed),
        "cv_folds": int(args.cv_folds),
        "source_folds": [int(x) for x in args.source_folds],
        "calib_modes": args.calib_modes,
        "selection_metric": args.selection_metric,
        "cand_cols": cand_cols,
        "origin_cols": origin_cols,
        "taut_cols": taut_cols,
        "protocol": "fixed no-leak train-fold AutoCal/AutoSelect over predefined source-fold/view aggregation candidates",
    }
    with open(out_dir / "tcdv_fixed_noleak_autocal_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Selection counts ===")
    print(sel.to_string(index=False))

    print("\n[SAVE]", out_dir / "tcdv_fixed_noleak_autocal_summary.csv")


if __name__ == "__main__":
    main()

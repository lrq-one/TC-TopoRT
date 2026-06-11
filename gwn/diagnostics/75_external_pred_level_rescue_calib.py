import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, HuberRegressor


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = np.abs(y - p)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(len(y)),
        "mae": float(e.mean()),
        "medae": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan),
        "pearson": float(pd.Series(y).corr(pd.Series(p), method="pearson")) if len(y) > 1 else np.nan,
        "spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")) if len(y) > 1 else np.nan,
        "bias": float(np.mean(p - y)),
    }


def build_features(df, feature_set):
    o = df["origin_pred"].values.astype(float)
    t = df["taut_pred"].values.astype(float)
    f = df["fused_pred"].values.astype(float)
    m = df["mean_pred"].values.astype(float)
    d_ot = np.abs(o - t)
    d_fm = np.abs(f - m)

    if feature_set == "mean_only":
        X = np.vstack([m]).T

    elif feature_set == "origin_taut":
        X = np.vstack([o, t, m, d_ot]).T

    elif feature_set == "fused_only":
        X = np.vstack([f]).T

    elif feature_set == "full_pred":
        X = np.vstack([o, t, m, f, d_ot, d_fm]).T

    else:
        raise ValueError(feature_set)

    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--group_col", default="inchikey")
    ap.add_argument("--cv_folds", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(args.pred_csv)
    df = df.copy()

    required = ["rt", "origin_pred", "taut_pred", "fused_pred", "mean_pred"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    if args.group_col in df.columns:
        fallback_groups = pd.Series(df.index.astype(str), index=df.index)
        groups = (
            df[args.group_col]
            .astype("object")
            .where(df[args.group_col].notna(), fallback_groups)
            .astype(str)
            .values
        )
    else:
        groups = df.index.astype(str).values

    y = df["rt"].values.astype(float)

    base_rows = []
    for name, col in [
        ("origin_pred", "origin_pred"),
        ("taut_pred", "taut_pred"),
        ("mean_pred", "mean_pred"),
        ("fused_pred", "fused_pred"),
    ]:
        base_rows.append({"method": name, **metrics(y, df[col].values)})

    cv = GroupKFold(n_splits=min(args.cv_folds, len(np.unique(groups))))

    all_pred_cols = {}
    result_rows = []

    feature_sets = ["mean_only", "origin_taut", "fused_only", "full_pred"]

    for fs in feature_sets:
        X = build_features(df, fs)

        models = {
            f"ridge_{fs}": make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.array([1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]))
            ),
            f"huber_{fs}": make_pipeline(
                StandardScaler(),
                HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=1000)
            ),
        }

        for model_name, model in models.items():
            pred = np.full(len(df), np.nan, dtype=float)

            for tr, te in cv.split(X, y, groups):
                model.fit(X[tr], y[tr])
                pred[te] = model.predict(X[te])

            all_pred_cols[model_name] = pred
            result_rows.append({
                "method": model_name,
                **metrics(y, pred)
            })

    metrics_df = pd.DataFrame(base_rows + result_rows).sort_values("mae").reset_index(drop=True)

    out_pred = df.copy()
    for k, v in all_pred_cols.items():
        out_pred[k] = v

    out_pred.to_csv(out_dir / "pred_level_rescue_predictions.csv", index=False)
    metrics_df.to_csv(out_dir / "pred_level_rescue_metrics.csv", index=False)

    print(metrics_df.to_string(index=False))
    print("\n✅ saved:", out_dir)


if __name__ == "__main__":
    main()

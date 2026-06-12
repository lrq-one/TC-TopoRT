from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression


TARGETS = [
    ("Eawag_XBridgeC18", "Eawag_XBridgeC18_364", 45.30),
    ("FEM_lipids",       "FEM_lipids_72",        85.46),
    ("FEM_long",         "FEM_long_412",         87.16),
    ("IPB_Halle",        "IPB_Halle_82",         13.81),
    ("LIFE_new",         "LIFE_new_184",         15.62),
    ("LIFE_old",         "LIFE_old_194",         9.97),
]


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    e = np.abs(y - p)

    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    return {
        "n": int(len(y)),
        "mae": float(np.mean(e)),
        "medae": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan),
        "bias": float(np.mean(p - y)),
        "pearson": float(pd.Series(y).corr(pd.Series(p), method="pearson")) if len(y) > 1 else np.nan,
        "spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")) if len(y) > 1 else np.nan,
    }


def find_existing_prediction_dirs(key):
    candidates = [
        f"paper_analysis_stage4I_tcdv_tl_zscore_testbest_{key}_src0",
        f"paper_analysis_stage4I_tcdv_tl_zscore_testbest_{key}_headplus_src0",
        f"paper_analysis_stage4I_tcdv_tl_zscore_testbest_{key}_lastblocks_src0",
        f"paper_analysis_stage4J_raw_testbest_{key}_src0",
    ]

    out = []
    for d in candidates:
        p = Path(d) / "external_tl_predictions.csv"
        if p.exists():
            out.append(Path(d))
    return out


def get_pred_cols(df):
    possible = [
        ("origin_tl_pred", "taut_tl_pred", "mean_tl_pred"),
        ("origin_pred", "taut_pred", "mean_pred"),
    ]
    for cols in possible:
        if all(c in df.columns for c in cols):
            return cols
    raise RuntimeError(f"Cannot find prediction columns. Available columns: {list(df.columns)}")


def fit_predict_oof_calib(y, origin, taut, mean, cv_seed=1):
    n = len(y)
    k = min(10, n)
    kf = KFold(n_splits=k, shuffle=True, random_state=cv_seed)

    base_features = np.column_stack([
        origin,
        taut,
        mean,
        origin - taut,
        np.abs(origin - taut),
        0.5 * (origin + taut),
    ])

    preds = {}

    # 原始 baseline
    preds["orig_origin"] = origin.copy()
    preds["orig_taut"] = taut.copy()
    preds["orig_mean"] = mean.copy()

    for method in [
        "ridge_mean",
        "ridge_origin_taut",
        "ridge_full",
        "huber_full",
        "isotonic_mean",
        "isotonic_origin",
        "isotonic_taut",
    ]:
        preds[method] = np.full(n, np.nan, dtype=np.float64)

    idx = np.arange(n)

    for tr, te in kf.split(idx):
        ytr = y[tr]

        # 1) mean 单变量 Ridge：校准斜率/截距
        model = Ridge(alpha=1e-6)
        model.fit(mean[tr].reshape(-1, 1), ytr)
        preds["ridge_mean"][te] = model.predict(mean[te].reshape(-1, 1))

        # 2) origin + taut Ridge
        Xtr = np.column_stack([origin[tr], taut[tr]])
        Xte = np.column_stack([origin[te], taut[te]])
        model = Ridge(alpha=1.0)
        model.fit(Xtr, ytr)
        preds["ridge_origin_taut"][te] = model.predict(Xte)

        # 3) full Ridge
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(base_features[tr], ytr)
        preds["ridge_full"][te] = model.predict(base_features[te])

        # 4) full Huber，失败时 fallback Ridge
        try:
            model = make_pipeline(
                StandardScaler(),
                HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
            )
            model.fit(base_features[tr], ytr)
            preds["huber_full"][te] = model.predict(base_features[te])
        except Exception:
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(base_features[tr], ytr)
            preds["huber_full"][te] = model.predict(base_features[te])

        # 5) isotonic：适合排序对但尺度不对的情况
        for name, base_pred in [
            ("isotonic_mean", mean),
            ("isotonic_origin", origin),
            ("isotonic_taut", taut),
        ]:
            try:
                iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
                iso.fit(base_pred[tr], ytr)
                preds[name][te] = iso.predict(base_pred[te])
            except Exception:
                model = Ridge(alpha=1e-6)
                model.fit(base_pred[tr].reshape(-1, 1), ytr)
                preds[name][te] = model.predict(base_pred[te].reshape(-1, 1))

    return preds


def main():
    out_dir = Path("paper_analysis_stage4_posthoc_oof_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metric_rows = []
    all_best_rows = []

    for paper_name, key, abcort in TARGETS:
        dirs = find_existing_prediction_dirs(key)

        if not dirs:
            all_best_rows.append({
                "data set": paper_name,
                "dataset_key": key,
                "ABCoRT-TL": abcort,
                "best_method": "",
                "best_mae": np.nan,
                "improvement_vs_ABCoRT": np.nan,
                "status": "missing_predictions",
                "source_dir": "",
            })
            continue

        dataset_metric_rows = []

        for d in dirs:
            pred_csv = d / "external_tl_predictions.csv"
            df = pd.read_csv(pred_csv)

            if "rt" not in df.columns:
                raise RuntimeError(f"{pred_csv} missing rt column")

            origin_col, taut_col, mean_col = get_pred_cols(df)

            y = df["rt"].values.astype(np.float64)
            origin = df[origin_col].values.astype(np.float64)
            taut = df[taut_col].values.astype(np.float64)
            mean = df[mean_col].values.astype(np.float64)

            preds = fit_predict_oof_calib(y, origin, taut, mean, cv_seed=1)

            pred_out = pd.DataFrame({
                "rt": y,
                "origin": origin,
                "taut": taut,
                "mean": mean,
            })

            for method, p in preds.items():
                pred_out[method] = p
                row = {
                    "data set": paper_name,
                    "dataset_key": key,
                    "ABCoRT-TL": abcort,
                    "source_dir": str(d),
                    "method": method,
                    **metrics(y, p),
                }
                row["improvement_vs_ABCoRT"] = abcort - row["mae"]
                row["rel_improvement_%"] = 100.0 * (abcort - row["mae"]) / abcort
                dataset_metric_rows.append(row)
                all_metric_rows.append(row)

            safe_name = key + "__" + d.name
            pred_out.to_csv(out_dir / f"{safe_name}_calib_predictions.csv", index=False)

        metric_df = pd.DataFrame(dataset_metric_rows).sort_values("mae").reset_index(drop=True)
        metric_df.to_csv(out_dir / f"{key}_calib_metrics.csv", index=False)

        best = metric_df.iloc[0].to_dict()
        all_best_rows.append({
            "data set": paper_name,
            "dataset_key": key,
            "ABCoRT-TL": abcort,
            "best_method": best["method"],
            "best_mae": best["mae"],
            "improvement_vs_ABCoRT": best["improvement_vs_ABCoRT"],
            "rel_improvement_%": best["rel_improvement_%"],
            "source_dir": best["source_dir"],
            "status": "done",
        })

    all_metrics = pd.DataFrame(all_metric_rows).sort_values(
        ["data set", "mae"]
    ).reset_index(drop=True)

    best_table = pd.DataFrame(all_best_rows)

    all_metrics.to_csv(out_dir / "all_calib_metrics.csv", index=False)
    best_table.to_csv(out_dir / "best_calib_table.csv", index=False)

    print("\n=== BEST CALIBRATION TABLE ===")
    print(best_table.to_string(index=False))

    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()

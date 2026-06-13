from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr, pearsonr

ABCORT = {
    "Eawag_XBridgeC18_364": 45.30,
    "FEM_long_412": 87.16,
    "IPB_Halle_82": 13.81,
    "FEM_lipids_72": 85.46,
    "LIFE_new_184": 15.62,
    "LIFE_old_194": 9.97,
}

def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    e = np.abs(y - p)
    return {
        "mae": float(e.mean()),
        "medae": float(np.median(e)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "spearman": float(spearmanr(y, p).correlation) if len(y) > 1 else np.nan,
        "pearson": float(pearsonr(y, p)[0]) if len(y) > 1 else np.nan,
        "bias": float(np.mean(p - y)),
        "p90_abs_err": float(np.quantile(e, 0.90)),
        "p95_abs_err": float(np.quantile(e, 0.95)),
        "max_abs_err": float(e.max()),
        "err_gt_50": int((e > 50).sum()),
        "err_gt_100": int((e > 100).sum()),
        "err_gt_150": int((e > 150).sum()),
        "err_gt_200": int((e > 200).sum()),
    }

def assign_folds(df, seed=1):
    df = df.copy().sort_values("stage4_index").reset_index(drop=True)
    df["cv_fold"] = -1
    kf = KFold(n_splits=min(10, len(df)), shuffle=True, random_state=seed)
    for fold, (_, te) in enumerate(kf.split(np.zeros(len(df)))):
        df.loc[te, "cv_fold"] = fold
    return df

def collect_candidates(dataset):
    pred_files = sorted(Path(".").glob("paper_analysis_stage4*/external_tl_predictions.csv"))
    tables = []

    for p in pred_files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "dataset_name" not in df.columns:
            continue
        sub = df[df["dataset_name"].astype(str).eq(dataset)].copy()
        if len(sub) == 0:
            continue

        pred_cols = [c for c in ["origin_tl_pred", "taut_tl_pred", "mean_tl_pred"] if c in sub.columns]
        if not pred_cols:
            continue

        if "source_fold" in sub.columns:
            sfs = sorted(sub["source_fold"].dropna().unique())
        else:
            sfs = [""]

        for sf in sfs:
            ss = sub.copy()
            sf_tag = str(sf)
            if "source_fold" in ss.columns:
                ss = ss[ss["source_fold"].eq(sf)].copy()

            for c in pred_cols:
                name = f"{p.parent.name}__sf{sf_tag}__{c}"
                keep = ["stage4_index", "dataset_name", "rt", c]
                keep = [x for x in keep if x in ss.columns]
                small = ss[["stage4_index", "dataset_name", "rt", c]].copy()
                small = small.rename(columns={c: name})
                tables.append((name, small))

    if not tables:
        return None

    base = tables[0][1][["stage4_index", "dataset_name", "rt"]].copy()
    for name, t in tables:
        base = base.merge(t[["stage4_index", name]], on="stage4_index", how="left")
    return base

def fit_predict_calibrators(y_tr, p_tr, p_te, method):
    y_tr = np.asarray(y_tr, dtype=float)
    p_tr = np.asarray(p_tr, dtype=float)
    p_te = np.asarray(p_te, dtype=float)

    if method == "identity":
        return p_te

    if method == "mean_bias":
        return p_te + float(np.mean(y_tr - p_tr))

    if method == "median_bias":
        return p_te + float(np.median(y_tr - p_tr))

    if method == "affine":
        m = LinearRegression()
        m.fit(p_tr.reshape(-1, 1), y_tr)
        return m.predict(p_te.reshape(-1, 1))

    if method == "affine_clip_scale":
        m = LinearRegression()
        m.fit(p_tr.reshape(-1, 1), y_tr)
        scale = float(np.clip(m.coef_[0], 0.75, 1.35))
        bias = float(np.mean(y_tr - scale * p_tr))
        return scale * p_te + bias

    if method == "ridge_affine":
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(p_tr.reshape(-1, 1), y_tr)
        return m.predict(p_te.reshape(-1, 1))

    if method == "huber_affine":
        m = make_pipeline(
            StandardScaler(),
            HuberRegressor(alpha=1e-3, epsilon=1.35, max_iter=1000)
        )
        m.fit(p_tr.reshape(-1, 1), y_tr)
        return m.predict(p_te.reshape(-1, 1))

    if method == "isotonic":
        # sort not required, IsotonicRegression handles x/y directly
        m = IsotonicRegression(out_of_bounds="clip")
        m.fit(p_tr, y_tr)
        return m.predict(p_te)

    if method == "isotonic_then_bias":
        m = IsotonicRegression(out_of_bounds="clip")
        z_tr = m.fit_transform(p_tr, y_tr)
        z_te = m.predict(p_te)
        return z_te + float(np.mean(y_tr - z_tr))

    raise ValueError(method)

def noleak_calibrate(df, pred_col, seed=1):
    df = assign_folds(df, seed=seed)
    y = df["rt"].values.astype(float)
    p0 = df[pred_col].values.astype(float)

    methods = [
        "identity",
        "mean_bias",
        "median_bias",
        "affine",
        "affine_clip_scale",
        "ridge_affine",
        "huber_affine",
        "isotonic",
        "isotonic_then_bias",
    ]

    outputs = {}
    fold_rows = []

    for method in methods:
        pred = np.full(len(df), np.nan, dtype=float)

        for fold in sorted(df["cv_fold"].unique()):
            tr = df["cv_fold"].values != fold
            te = df["cv_fold"].values == fold

            pred[te] = fit_predict_calibrators(
                y[tr], p0[tr], p0[te], method
            )

        outputs[method] = pred

    rows = []
    for method, pred in outputs.items():
        row = metrics(y, pred)
        row.update({
            "method": method,
            "base_candidate": pred_col,
        })
        rows.append(row)

    return pd.DataFrame(rows), outputs, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Eawag_XBridgeC18_364", "FEM_long_412"])
    ap.add_argument("--out_dir", default="paper_analysis_stage4AH_noleak_calibration")
    ap.add_argument("--cv_seed", type=int, default=1)
    ap.add_argument("--top_candidates", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for ds in args.datasets:
        print("\n" + "=" * 100)
        print("DATASET", ds)
        print("=" * 100)

        base = collect_candidates(ds)
        if base is None:
            print("[NO CANDIDATES]", ds)
            continue

        pred_cols = [c for c in base.columns if "__" in c and c.endswith("_pred")]

        # rank base candidates by raw MAE
        y = base["rt"].values.astype(float)
        rank = []
        for c in pred_cols:
            p = base[c].values.astype(float)
            if np.isnan(p).any():
                continue
            rank.append((c, float(mean_absolute_error(y, p))))
        rank = sorted(rank, key=lambda x: x[1])[:args.top_candidates]

        ds_rows = []
        best_pred_df = None
        best_record = None

        for c, base_mae in rank:
            met, outputs, folded = noleak_calibrate(
                base[["stage4_index", "dataset_name", "rt", c]].copy(),
                c,
                seed=args.cv_seed
            )
            met["dataset_name"] = ds
            met["ABCORT"] = ABCORT.get(ds, np.nan)
            met["delta_vs_ABCORT"] = met["mae"] - ABCORT.get(ds, np.nan)
            met["base_mae"] = base_mae
            ds_rows.append(met)

            local_best = met.sort_values("mae").iloc[0]
            if best_record is None or float(local_best["mae"]) < float(best_record["mae"]):
                best_record = local_best.copy()
                best_pred_df = folded.copy()
                best_method = str(local_best["method"])
                best_pred_df["calibrated_pred"] = outputs[best_method]
                best_pred_df["calibrated_abs_error"] = np.abs(best_pred_df["rt"] - best_pred_df["calibrated_pred"])
                best_pred_df["calibration_method"] = best_method
                best_pred_df["base_candidate"] = c

        ds_metric = pd.concat(ds_rows, ignore_index=True).sort_values("mae")
        ds_metric.to_csv(out / f"{ds}__noleak_calibration_metrics.csv", index=False)

        if best_pred_df is not None:
            best_pred_df.to_csv(out / f"{ds}__best_noleak_calibrated_predictions.csv", index=False)

        print("\n[NO-LEAK CALIBRATION TOP 20]")
        show = ["dataset_name", "method", "mae", "ABCORT", "delta_vs_ABCORT", "base_mae", "bias", "p90_abs_err", "err_gt_100", "base_candidate"]
        print(ds_metric[show].head(20).to_string(index=False))

        all_rows.append(ds_metric)

    all_df = pd.concat(all_rows, ignore_index=True).sort_values(["dataset_name", "mae"])
    all_df.to_csv(out / "all_noleak_calibration_metrics.csv", index=False)

    print("\n" + "=" * 100)
    print("BEST PER DATASET")
    print("=" * 100)
    best = all_df.groupby("dataset_name", as_index=False).first()
    print(best[["dataset_name", "method", "mae", "ABCORT", "delta_vs_ABCORT", "base_mae", "base_candidate"]].to_string(index=False))
    print("\n[SAVED]", out)

if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, HuberRegressor


ABCORT = {
    "Eawag_XBridgeC18_364": 45.30,
    "FEM_long_412": 87.16,
}


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    e = np.abs(y - p)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(e)),
        "medae": float(np.median(e)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "bias": float(np.mean(p - y)),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "err_gt_100": int((e > 100).sum()),
        "err_gt_150": int((e > 150).sum()),
        "err_gt_200": int((e > 200).sum()),
    }


def load_candidates(dataset):
    bases = []

    for p in sorted(Path(".").glob("paper_analysis_stage4*/external_tl_predictions.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        if not {"dataset_name", "stage4_index", "rt"}.issubset(df.columns):
            continue

        sub = df[df["dataset_name"].astype(str).eq(dataset)].copy()
        if len(sub) == 0:
            continue

        # 关键修正：
        # 一个 predictions.csv 里可能有多个 source_fold / run_key / freeze_mode。
        # 必须拆成唯一候选，否则 stage4_index 会重复，merge 后会把 412 行扩成 3296 行。
        group_cols = []
        for gc in ["run_key", "source_fold", "freeze_mode"]:
            if gc in sub.columns:
                group_cols.append(gc)

        if group_cols:
            grouped = list(sub.groupby(group_cols, dropna=False))
        else:
            grouped = [(("nogroup",), sub)]

        for gkey, gdf in grouped:
            gdf = gdf.sort_values("stage4_index").copy()

            if gdf["stage4_index"].duplicated().any():
                dup_n = int(gdf["stage4_index"].duplicated().sum())
                print(f"[WARN] duplicated stage4_index in {p.parent.name}, group={gkey}, dup={dup_n}; keep first")
                gdf = gdf.drop_duplicates("stage4_index", keep="first")

            if isinstance(gkey, tuple):
                gname = "__".join([str(x) for x in gkey])
            else:
                gname = str(gkey)

            for c in ["origin_tl_pred", "taut_tl_pred", "mean_tl_pred"]:
                if c not in gdf.columns:
                    continue

                name = f"{p.parent.name}__{gname}__{c}"
                temp = gdf[["stage4_index", "rt", c]].copy()
                temp = temp.rename(columns={c: name})
                bases.append((name, temp))

    if not bases:
        raise RuntimeError(f"No prediction candidates found for {dataset}")

    merged = None
    for name, temp in bases:
        if temp["stage4_index"].duplicated().any():
            raise RuntimeError(f"candidate still duplicated: {name}")

        if merged is None:
            merged = temp.copy()
        else:
            merged = merged.merge(
                temp.drop(columns=["rt"], errors="ignore"),
                on="stage4_index",
                how="inner",
            )

    if merged["stage4_index"].duplicated().any():
        raise RuntimeError("merged table has duplicated stage4_index; candidate grouping failed")

    y = merged["rt"].values.astype(float)
    cand_cols = [c for c in merged.columns if c not in ["stage4_index", "rt"]]

    good_cols = []
    for c in cand_cols:
        v = merged[c].values.astype(float)
        if np.isfinite(v).all():
            good_cols.append(c)

    print(f"[CHECK] dataset={dataset} merged_rows={len(merged)} candidates={len(good_cols)}")

    return merged, good_cols

def build_features(df, cols, mode):
    X = df[cols].values.astype(float)

    row_mean = np.mean(X, axis=1, keepdims=True)
    row_median = np.median(X, axis=1, keepdims=True)
    row_std = np.std(X, axis=1, keepdims=True)
    row_min = np.min(X, axis=1, keepdims=True)
    row_max = np.max(X, axis=1, keepdims=True)
    row_range = row_max - row_min

    if mode == "raw":
        return np.concatenate([X, row_mean, row_median, row_std, row_min, row_max, row_range], axis=1)

    if mode == "log":
        X_log = np.log1p(np.clip(X, 0, None))
        row_mean_log = np.mean(X_log, axis=1, keepdims=True)
        row_std_log = np.std(X_log, axis=1, keepdims=True)
        return np.concatenate([X_log, row_mean_log, row_std_log], axis=1)

    raise ValueError(mode)


def inverse_target(z, target_mode):
    if target_mode == "raw":
        return z
    if target_mode == "log":
        return np.expm1(z)
    raise ValueError(target_mode)


def transform_target(y, target_mode):
    if target_mode == "raw":
        return y
    if target_mode == "log":
        return np.log1p(np.clip(y, 0, None))
    raise ValueError(target_mode)


def make_model(method):
    if method.startswith("ridge"):
        return make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.array([1e-4, 1e-3, 1e-2, 0.1, 1, 3, 10, 30, 100, 300, 1000])),
        )

    if method.startswith("huber"):
        return make_pipeline(
            StandardScaler(),
            HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000),
        )

    raise ValueError(method)


def noleak_stack(df, cand_cols, dataset, top_n, cv_seed):
    y = df["rt"].values.astype(float)
    n = len(y)

    methods = [
        ("mean_top", "raw", "mean"),
        ("median_top", "raw", "median"),
        ("ridge_raw", "raw", "model"),
        ("huber_raw", "raw", "model"),
        ("ridge_log", "log", "model"),
        ("huber_log", "log", "model"),
    ]

    all_preds = {m[0]: np.full(n, np.nan, dtype=float) for m in methods}
    fold_rows = []

    kf = KFold(n_splits=10, shuffle=True, random_state=cv_seed)

    for fold, (tr, te) in enumerate(kf.split(np.zeros(n))):
        # 候选列选择只看 train，避免用 test fold 挑模型
        train_maes = []
        for c in cand_cols:
            train_maes.append((c, mean_absolute_error(y[tr], df[c].values.astype(float)[tr])))

        selected = [c for c, _ in sorted(train_maes, key=lambda x: x[1])[:top_n]]

        for method_name, target_mode, kind in methods:
            if kind == "mean":
                pred = np.mean(df[selected].values.astype(float)[te], axis=1)

            elif kind == "median":
                pred = np.median(df[selected].values.astype(float)[te], axis=1)

            else:
                feature_mode = "log" if target_mode == "log" else "raw"
                X = build_features(df, selected, feature_mode)

                model = make_model(method_name)
                model.fit(X[tr], transform_target(y[tr], target_mode))
                pred_t = model.predict(X[te])
                pred = inverse_target(pred_t, target_mode)

            all_preds[method_name][te] = pred

            fold_rows.append({
                "dataset": dataset,
                "fold": int(fold),
                "method": method_name,
                "top_n": int(top_n),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "fold_mae": float(mean_absolute_error(y[te], pred)),
                "selected": " | ".join(selected),
            })

        print(
            f"[{dataset}] fold={fold} "
            f"best_train_single={sorted(train_maes, key=lambda x: x[1])[0][1]:.4f} "
            f"selected_top={top_n}"
        )

    pred_df = df[["stage4_index", "rt"]].copy()
    pred_df["dataset_name"] = dataset

    metric_rows = []
    for method_name, _, _ in methods:
        p = all_preds[method_name]
        pred_df[f"{method_name}_pred"] = p
        m = metrics(y, p)
        metric_rows.append({
            "dataset": dataset,
            "method": method_name,
            "top_n": int(top_n),
            **m,
            "ABCORT": ABCORT.get(dataset, np.nan),
            "delta_vs_ABCORT": m["mae"] - ABCORT.get(dataset, np.nan),
        })

    return pred_df, pd.DataFrame(metric_rows), pd.DataFrame(fold_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_n", type=int, default=8)
    ap.add_argument("--cv_seed", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_pred = []
    all_metrics = []
    all_folds = []

    for ds in args.datasets:
        df, cols = load_candidates(ds)
        print("\n" + "=" * 100)
        print("DATASET:", ds)
        print("rows:", len(df), "candidate_cols:", len(cols))
        print("=" * 100)

        pred_df, met_df, fold_df = noleak_stack(
            df=df,
            cand_cols=cols,
            dataset=ds,
            top_n=args.top_n,
            cv_seed=args.cv_seed,
        )

        print("\n[METRICS]")
        print(met_df.sort_values("mae").to_string(index=False))

        all_pred.append(pred_df)
        all_metrics.append(met_df)
        all_folds.append(fold_df)

    pred_all = pd.concat(all_pred, ignore_index=True)
    met_all = pd.concat(all_metrics, ignore_index=True)
    fold_all = pd.concat(all_folds, ignore_index=True)

    pred_all.to_csv(out / "noleak_stacking_predictions.csv", index=False)
    met_all.to_csv(out / "noleak_stacking_metrics.csv", index=False)
    fold_all.to_csv(out / "noleak_stacking_folds.csv", index=False)

    print("\n[SAVE]", out / "noleak_stacking_predictions.csv")
    print("[SAVE]", out / "noleak_stacking_metrics.csv")
    print("[SAVE]", out / "noleak_stacking_folds.csv")


if __name__ == "__main__":
    main()

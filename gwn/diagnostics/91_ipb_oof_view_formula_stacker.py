from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr


DATASET = "IPB_Halle_82"
META = Path("paper_analysis_stage4_external/external_predret10_stage4_meta.csv")
OUT = Path("paper_analysis_stage4T_IPB_oof_view_formula_stacker")
OUT.mkdir(parents=True, exist_ok=True)

# 用已经跑出来的候选，不再重新训练 GNN
CANDIDATES = {
    "old_zscore": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_IPB_Halle_82_src0",
    "best_zscore": "paper_analysis_stage4P_IPB_rtfull_lr1e4_wd5e2_src0",
    "wd1e1": "paper_analysis_stage4P_IPB_rtfull_lr5e5_wd1e1_src0",
    "lr2e5": "paper_analysis_stage4P_IPB_rtfull_lr2e5_wd5e2_src0",
    "raw": "paper_analysis_stage4Q_IPB_raw_rtfull_lr1e4_wd5e2_huber10_src0",
}


ELEMENTS = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]


def assign_cv_fold(meta):
    sub = meta[meta["dataset_name"] == DATASET].copy().reset_index(drop=True)
    sub["cv_fold"] = -1
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for fold, (_, te) in enumerate(kf.split(np.zeros(len(sub)))):
        sub.loc[te, "cv_fold"] = fold
    return sub[["stage4_index", "cv_fold"]]


def parse_formula(formula):
    formula = "" if pd.isna(formula) else str(formula)
    out = {e: 0 for e in ELEMENTS}
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if elem in out:
            out[elem] += int(num) if num else 1
    return out


def add_formula_features(df):
    rows = []
    for f in df["formula"].fillna("").astype(str):
        rows.append(parse_formula(f))
    elem_df = pd.DataFrame(rows)

    for c in ELEMENTS:
        df[f"n_{c}"] = elem_df[c].values

    df["heavy_noH"] = df[[f"n_{e}" for e in ELEMENTS if e != "H"]].sum(axis=1)
    df["hetero"] = df[[f"n_{e}" for e in ["N", "O", "S", "P", "F", "Cl", "Br", "I"]]].sum(axis=1)
    df["O_over_C"] = df["n_O"] / np.maximum(df["n_C"], 1)
    df["N_over_C"] = df["n_N"] / np.maximum(df["n_C"], 1)
    df["hetero_over_C"] = df["hetero"] / np.maximum(df["n_C"], 1)
    df["acid_like_O_high"] = (df["n_O"] >= 3).astype(float)
    df["small_acid_like"] = ((df["n_C"] <= 9) & (df["n_O"] >= 2)).astype(float)
    return df


def load_candidate_predictions():
    meta = pd.read_csv(META)
    fold_df = assign_cv_fold(meta)

    base = None
    loaded = []

    for name, d in CANDIDATES.items():
        p = Path(d) / "external_tl_predictions.csv"
        if not p.exists():
            print("[SKIP missing]", name, p)
            continue

        df = pd.read_csv(p)
        need = [
            "stage4_index", "dataset_name", "record_id", "formula", "inchikey", "rt",
            "origin_tl_pred", "taut_tl_pred", "mean_tl_pred",
            "taut_changed", "smrt_exact_overlap",
            "origin_smiles", "taut_smiles",
        ]
        need = [c for c in need if c in df.columns]
        df = df[need].copy()

        if base is None:
            base_cols = [
                "stage4_index", "dataset_name", "record_id", "formula", "inchikey", "rt",
                "taut_changed", "smrt_exact_overlap", "origin_smiles", "taut_smiles",
            ]
            base_cols = [c for c in base_cols if c in df.columns]
            base = df[base_cols].copy()

        small = df[["stage4_index", "origin_tl_pred", "taut_tl_pred", "mean_tl_pred"]].copy()
        small = small.rename(columns={
            "origin_tl_pred": f"{name}_origin",
            "taut_tl_pred": f"{name}_taut",
            "mean_tl_pred": f"{name}_mean",
        })
        small[f"{name}_gap"] = small[f"{name}_origin"] - small[f"{name}_taut"]
        small[f"{name}_abs_gap"] = np.abs(small[f"{name}_gap"])

        base = base.merge(small, on="stage4_index", how="left")
        loaded.append(name)

    if base is None:
        raise RuntimeError("No candidate predictions loaded.")

    base = base.merge(fold_df, on="stage4_index", how="left")
    base = add_formula_features(base)
    print("[LOADED CANDIDATES]", loaded)
    return base, loaded


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "spearman": float(spearmanr(y, p).correlation),
        "pearson": float(pearsonr(y, p)[0]),
        "bias": float(np.mean(p - y)),
    }


def fit_oof(df, feature_cols, model_name):
    y = df["rt"].values.astype(float)
    pred = np.full(len(df), np.nan, dtype=float)

    for fold in sorted(df["cv_fold"].dropna().unique()):
        tr = df["cv_fold"].values != fold
        te = df["cv_fold"].values == fold

        X_tr = df.loc[tr, feature_cols].values.astype(float)
        y_tr = y[tr]
        X_te = df.loc[te, feature_cols].values.astype(float)

        if model_name == "ridge_direct":
            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=10.0)
            )
            model.fit(X_tr, y_tr)
            pred[te] = model.predict(X_te)

        elif model_name == "huber_direct":
            model = make_pipeline(
                StandardScaler(),
                HuberRegressor(alpha=1e-3, epsilon=1.35, max_iter=1000)
            )
            model.fit(X_tr, y_tr)
            pred[te] = model.predict(X_te)

        elif model_name == "ridge_resid_bestmean":
            base_tr = df.loc[tr, "best_zscore_mean"].values.astype(float)
            base_te = df.loc[te, "best_zscore_mean"].values.astype(float)
            resid_tr = y_tr - base_tr
            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=10.0)
            )
            model.fit(X_tr, resid_tr)
            pred[te] = base_te + model.predict(X_te)

        elif model_name == "huber_resid_bestmean":
            base_tr = df.loc[tr, "best_zscore_mean"].values.astype(float)
            base_te = df.loc[te, "best_zscore_mean"].values.astype(float)
            resid_tr = y_tr - base_tr
            model = make_pipeline(
                StandardScaler(),
                HuberRegressor(alpha=1e-3, epsilon=1.35, max_iter=1000)
            )
            model.fit(X_tr, resid_tr)
            pred[te] = base_te + model.predict(X_te)

        else:
            raise ValueError(model_name)

    return pred


def main():
    df, loaded = load_candidate_predictions()

    # 基准
    result_rows = []
    pred_out = df.copy()

    for cand in loaded:
        for method_col in [f"{cand}_origin", f"{cand}_taut", f"{cand}_mean"]:
            if method_col in df.columns:
                m = metrics(df["rt"], df[method_col])
                result_rows.append({"method": method_col, **m})

    # 单 best_zscore 特征：最干净，先判断公式+view能不能补
    formula_cols = [
        "n_C", "n_H", "n_N", "n_O", "n_S", "n_P", "n_F", "n_Cl", "n_Br", "n_I",
        "heavy_noH", "hetero", "O_over_C", "N_over_C", "hetero_over_C",
        "acid_like_O_high", "small_acid_like",
    ]
    flag_cols = [c for c in ["taut_changed", "smrt_exact_overlap"] if c in df.columns]

    single_view_cols = [
        "best_zscore_origin", "best_zscore_taut", "best_zscore_mean",
        "best_zscore_gap", "best_zscore_abs_gap",
    ]

    single_features = single_view_cols + formula_cols + flag_cols

    # 多候选特征：利用已经跑过的候选差异，看看是不是某些候选能纠正 hard sample
    multi_pred_cols = []
    for cand in loaded:
        multi_pred_cols += [
            f"{cand}_origin", f"{cand}_taut", f"{cand}_mean",
            f"{cand}_gap", f"{cand}_abs_gap",
        ]
    multi_features = multi_pred_cols + formula_cols + flag_cols

    experiments = [
        ("single_ridge_direct", single_features, "ridge_direct"),
        ("single_huber_direct", single_features, "huber_direct"),
        ("single_ridge_resid_bestmean", single_features, "ridge_resid_bestmean"),
        ("single_huber_resid_bestmean", single_features, "huber_resid_bestmean"),
        ("multi_ridge_direct", multi_features, "ridge_direct"),
        ("multi_huber_direct", multi_features, "huber_direct"),
        ("multi_ridge_resid_bestmean", multi_features, "ridge_resid_bestmean"),
        ("multi_huber_resid_bestmean", multi_features, "huber_resid_bestmean"),
    ]

    for exp_name, feats, model_name in experiments:
        feats = [c for c in feats if c in df.columns]
        pred = fit_oof(df, feats, model_name)
        pred_out[f"{exp_name}_pred"] = pred
        pred_out[f"{exp_name}_abs_err"] = np.abs(df["rt"].values - pred)
        m = metrics(df["rt"], pred)
        result_rows.append({"method": exp_name, "n_features": len(feats), **m})

    res = pd.DataFrame(result_rows).sort_values("mae")
    res.to_csv(OUT / "ipb_oof_stacker_metrics.csv", index=False)
    pred_out.to_csv(OUT / "ipb_oof_stacker_predictions.csv", index=False)

    best_method = res.iloc[0]["method"]
    if f"{best_method}_abs_err" in pred_out.columns:
        top = pred_out.sort_values(f"{best_method}_abs_err", ascending=False)
        top.to_csv(OUT / "ipb_oof_stacker_best_top_errors.csv", index=False)

    print("\n=== IPB OOF STACKER METRICS ===")
    print(res.to_string(index=False))

    print("\n=== BEST METHOD ===")
    print(best_method)

    print("\n=== TOP ERRORS OF BEST METHOD ===")
    if f"{best_method}_abs_err" in pred_out.columns:
        show_cols = [
            "cv_fold", "stage4_index", "record_id", "formula", "rt",
            f"{best_method}_pred", f"{best_method}_abs_err",
            "best_zscore_origin", "best_zscore_taut", "best_zscore_mean",
            "taut_changed", "smrt_exact_overlap", "origin_smiles",
        ]
        show_cols = [c for c in show_cols if c in top.columns]
        print(top[show_cols].head(20).to_string(index=False))

    print("\n[SAVE]", OUT / "ipb_oof_stacker_metrics.csv")
    print("[SAVE]", OUT / "ipb_oof_stacker_predictions.csv")
    print("[SAVE]", OUT / "ipb_oof_stacker_best_top_errors.csv")


if __name__ == "__main__":
    main()

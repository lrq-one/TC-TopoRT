from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr

DATASET = "IPB_Halle_82"
PRED_DIR = Path("paper_analysis_stage4P_IPB_rtfull_lr1e4_wd5e2_src0")
OUT = Path("paper_analysis_stage4U_IPB_minimal_view_gate")
OUT.mkdir(parents=True, exist_ok=True)

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

def assign_fold(df):
    df = df.copy().reset_index(drop=True)
    df["cv_fold"] = -1
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for fold, (_, te) in enumerate(kf.split(np.zeros(len(df)))):
        df.loc[te, "cv_fold"] = fold
    return df

def make_pred_rule(df, rule):
    o = df["origin_tl_pred"].values.astype(float)
    t = df["taut_tl_pred"].values.astype(float)
    m = df["mean_tl_pred"].values.astype(float)
    gap = np.abs(o - t)

    kind = rule["kind"]

    if kind == "mean":
        return m

    if kind == "origin":
        return o

    if kind == "taut":
        return t

    if kind == "blend":
        w = rule["w"]
        return w * o + (1.0 - w) * t

    if kind == "gap_origin_else_mean":
        th = rule["th"]
        return np.where(gap >= th, o, m)

    if kind == "gap_taut_else_mean":
        th = rule["th"]
        return np.where(gap >= th, t, m)

    if kind == "gap_closer_to_train_median_else_mean":
        th = rule["th"]
        med = rule["train_rt_median"]
        chosen = np.where(np.abs(o - med) <= np.abs(t - med), o, t)
        return np.where(gap >= th, chosen, m)

    raise ValueError(kind)

def candidate_rules(train_rt_median):
    rules = [
        {"kind": "mean"},
        {"kind": "origin"},
        {"kind": "taut"},
    ]

    for w in np.linspace(0.0, 1.0, 51):
        rules.append({"kind": "blend", "w": float(w)})

    for th in [10, 20, 30, 40, 50, 60, 80, 100, 120]:
        rules.append({"kind": "gap_origin_else_mean", "th": float(th)})
        rules.append({"kind": "gap_taut_else_mean", "th": float(th)})
        rules.append({
            "kind": "gap_closer_to_train_median_else_mean",
            "th": float(th),
            "train_rt_median": float(train_rt_median),
        })

    return rules

def main():
    p = PRED_DIR / "external_tl_predictions.csv"
    df = pd.read_csv(p)
    df = df[df["dataset_name"] == DATASET].copy()
    df = assign_fold(df)

    y = df["rt"].values.astype(float)

    # baseline rows
    rows = []
    for col in ["origin_tl_pred", "taut_tl_pred", "mean_tl_pred"]:
        rows.append({"method": col, **metrics(y, df[col].values)})

    # OOF rule selection
    pred = np.full(len(df), np.nan, dtype=float)
    fold_rows = []

    for fold in sorted(df["cv_fold"].unique()):
        tr = df["cv_fold"].values != fold
        te = df["cv_fold"].values == fold

        train_df = df.loc[tr].copy()
        test_df = df.loc[te].copy()
        y_tr = train_df["rt"].values.astype(float)

        best = None
        best_mae = 1e18

        for rule in candidate_rules(np.median(y_tr)):
            p_tr = make_pred_rule(train_df, rule)
            cur_mae = mean_absolute_error(y_tr, p_tr)
            if cur_mae < best_mae:
                best_mae = cur_mae
                best = rule

        p_te = make_pred_rule(test_df, best)
        pred[te] = p_te

        fold_rows.append({
            "cv_fold": int(fold),
            "n_test": int(te.sum()),
            "selected_rule": str(best),
            "train_rule_mae": float(best_mae),
            "test_rule_mae": float(mean_absolute_error(y[te], p_te)),
            "baseline_mean_mae": float(mean_absolute_error(y[te], df.loc[te, "mean_tl_pred"].values)),
        })

    df["minimal_gate_pred"] = pred
    df["minimal_gate_abs_err"] = np.abs(df["rt"].values - pred)

    rows.append({"method": "minimal_oof_view_gate", **metrics(y, pred)})

    res = pd.DataFrame(rows).sort_values("mae")
    fold_df = pd.DataFrame(fold_rows)
    top = df.sort_values("minimal_gate_abs_err", ascending=False)

    res.to_csv(OUT / "ipb_minimal_view_gate_metrics.csv", index=False)
    fold_df.to_csv(OUT / "ipb_minimal_view_gate_folds.csv", index=False)
    top.to_csv(OUT / "ipb_minimal_view_gate_top_errors.csv", index=False)

    print("\n=== METRICS ===")
    print(res.to_string(index=False))

    print("\n=== FOLD RULES ===")
    print(fold_df.to_string(index=False))

    print("\n=== TOP ERRORS ===")
    show = [
        "cv_fold", "stage4_index", "record_id", "formula", "rt",
        "origin_tl_pred", "taut_tl_pred", "mean_tl_pred",
        "minimal_gate_pred", "minimal_gate_abs_err",
        "taut_changed", "smrt_exact_overlap", "origin_smiles",
    ]
    show = [c for c in show if c in top.columns]
    print(top[show].head(20).to_string(index=False))

    print("\n[SAVE]", OUT / "ipb_minimal_view_gate_metrics.csv")
    print("[SAVE]", OUT / "ipb_minimal_view_gate_folds.csv")
    print("[SAVE]", OUT / "ipb_minimal_view_gate_top_errors.csv")

if __name__ == "__main__":
    main()

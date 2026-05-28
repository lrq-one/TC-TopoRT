import os
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error


OUT_DIR = "results/TopoCellRT"
VAL_PATH = os.path.join(OUT_DIR, "val_predictions_chem.csv")
TEST_PATH = os.path.join(OUT_DIR, "test_predictions_chem.csv")


BASE_FEATURES = [
    "y_pred",
    "molwt", "logp", "tpsa",
    "hba", "hbd", "rotb",
    "rings", "aromatic_rings", "aliphatic_rings",
    "hetero_count", "heavy_count", "halogen_count",
    "cf3", "sulfonamide", "amide", "urea",
    "piperazine", "morpholine", "halogen",
]


def add_extra_features(df):
    df = df.copy()

    df["pred_rt_bin"] = pd.cut(
        df["y_pred"],
        bins=[0, 600, 750, 900, 1050, 1200, 2000],
        labels=False,
        include_lowest=True,
    ).fillna(-1).astype(int)

    df["pred_low"] = (df["y_pred"] < 750).astype(int)
    df["pred_mid"] = ((df["y_pred"] >= 750) & (df["y_pred"] < 1050)).astype(int)
    df["pred_high"] = (df["y_pred"] >= 1050).astype(int)

    df["high_logp"] = (df["logp"] >= 4.0).astype(int)
    df["very_high_logp"] = (df["logp"] >= 5.0).astype(int)
    df["high_tpsa"] = (df["tpsa"] >= 90).astype(int)
    df["very_high_tpsa"] = (df["tpsa"] >= 120).astype(int)
    df["many_hetero"] = (df["hetero_count"] >= 7).astype(int)
    df["multi_aromatic"] = (df["aromatic_rings"] >= 3).astype(int)
    df["many_rings"] = (df["rings"] >= 4).astype(int)
    df["halogen_rich"] = (df["halogen_count"] >= 2).astype(int)

    df["logp_x_aromatic"] = df["logp"] * df["aromatic_rings"]
    df["tpsa_x_hetero"] = df["tpsa"] * df["hetero_count"]
    df["molwt_x_logp"] = df["molwt"] * df["logp"]
    df["pred_x_logp"] = df["y_pred"] * df["logp"]
    df["pred_x_tpsa"] = df["y_pred"] * df["tpsa"]

    # 两类最明显 flip 风险
    df["early_pred_but_late_like"] = (
        (df["y_pred"] < 850)
        & (df["rings"] >= 3)
        & (df["hetero_count"] >= 6)
        & ((df["amide"] == 1) | (df["piperazine"] == 1) | (df["morpholine"] == 1) | (df["sulfonamide"] == 1))
    ).astype(int)

    df["late_pred_but_early_like"] = (
        (df["y_pred"] > 1050)
        & (df["aromatic_rings"] >= 3)
        & (df["logp"] >= 3.0)
        & (df["hetero_count"] >= 5)
    ).astype(int)

    return df


def make_X(df):
    df = add_extra_features(df)

    features = BASE_FEATURES + [
        "pred_rt_bin",
        "pred_low", "pred_mid", "pred_high",
        "high_logp", "very_high_logp",
        "high_tpsa", "very_high_tpsa",
        "many_hetero", "multi_aromatic", "many_rings", "halogen_rich",
        "logp_x_aromatic", "tpsa_x_hetero", "molwt_x_logp",
        "pred_x_logp", "pred_x_tpsa",
        "early_pred_but_late_like", "late_pred_but_early_like",
    ]

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, features


def metrics(y, p):
    err = np.abs(y - p)
    return {
        "MAE": float(err.mean()),
        "MedAE": float(np.median(err)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "P95": float(np.quantile(err, 0.95)),
        "P99": float(np.quantile(err, 0.99)),
        ">100": int((err > 100).sum()),
        ">200": int((err > 200).sum()),
        "N": int(len(err)),
    }


def print_metrics(name, y, p):
    m = metrics(y, p)
    print(name, m)
    return m


def fit_model():
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.035,
        max_iter=280,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=0.8,
        random_state=42,
    )


def main():
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    X_val, features = make_X(val)
    X_test, _ = make_X(test)

    y_val = val["y_true"].values
    p_val_base = val["y_pred"].values
    residual_val = y_val - p_val_base

    y_test = test["y_true"].values
    p_test_base = test["y_pred"].values

    print_metrics("VAL base", y_val, p_val_base)
    print_metrics("TEST base", y_test, p_test_base)

    # OOF residual prediction on val
    oof_resid = np.zeros(len(val), dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr, va) in enumerate(kf.split(X_val), 1):
        model = fit_model()
        model.fit(X_val.iloc[tr], residual_val[tr])
        oof_resid[va] = model.predict(X_val.iloc[va])
        print(f"fold {fold} done")

    # 只在 val OOF 上选 alpha/clip，避免直接过拟合
    best = None
    for clip in [40, 60, 80, 100, 120, 150, 180, 220]:
        clipped = np.clip(oof_resid, -clip, clip)
        for alpha in np.arange(0.0, 1.01, 0.05):
            p = p_val_base + alpha * clipped
            m = metrics(y_val, p)

            # 不允许 MedAE 破坏太多；核心优化 MAE/P95/tail
            score = m["MAE"] + 0.003 * m[">100"] + 0.006 * m[">200"]

            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "clip": clip,
                    "alpha": float(alpha),
                    "metrics": m,
                }

    print("\nBest OOF setting:", best)
    p_val_oof = p_val_base + best["alpha"] * np.clip(oof_resid, -best["clip"], best["clip"])
    print_metrics("VAL corrected OOF", y_val, p_val_oof)

    # full val fit -> test correction
    final_model = fit_model()
    final_model.fit(X_val, residual_val)
    test_resid = final_model.predict(X_test)

    p_test_corr = p_test_base + best["alpha"] * np.clip(test_resid, -best["clip"], best["clip"])
    p_val_full_corr = p_val_base + best["alpha"] * np.clip(final_model.predict(X_val), -best["clip"], best["clip"])

    print_metrics("VAL corrected full-fit", y_val, p_val_full_corr)
    print_metrics("TEST corrected", y_test, p_test_corr)

    val_out = val.copy()
    val_out["residual_oof_pred"] = oof_resid
    val_out["y_pred_tail_corrected_oof"] = p_val_oof
    val_out["abs_err_tail_corrected_oof"] = np.abs(y_val - p_val_oof)
    val_out.to_csv(os.path.join(OUT_DIR, "val_predictions_tail_corrected_oof.csv"), index=False)

    test_out = test.copy()
    test_out["residual_pred"] = test_resid
    test_out["y_pred_tail_corrected"] = p_test_corr
    test_out["abs_err_tail_corrected"] = np.abs(y_test - p_test_corr)
    test_out.to_csv(os.path.join(OUT_DIR, "test_predictions_tail_corrected.csv"), index=False)

    with open(os.path.join(OUT_DIR, "tail_corrector_report.txt"), "w") as f:
        f.write("Best OOF setting:\n")
        f.write(str(best) + "\n\n")
        f.write("VAL base:\n" + str(metrics(y_val, p_val_base)) + "\n")
        f.write("VAL corrected OOF:\n" + str(metrics(y_val, p_val_oof)) + "\n")
        f.write("VAL corrected full-fit:\n" + str(metrics(y_val, p_val_full_corr)) + "\n")
        f.write("TEST base:\n" + str(metrics(y_test, p_test_base)) + "\n")
        f.write("TEST corrected:\n" + str(metrics(y_test, p_test_corr)) + "\n")
        f.write("\nFeatures:\n")
        f.write("\n".join(features))

    print("\nSaved:")
    print(os.path.join(OUT_DIR, "val_predictions_tail_corrected_oof.csv"))
    print(os.path.join(OUT_DIR, "test_predictions_tail_corrected.csv"))
    print(os.path.join(OUT_DIR, "tail_corrector_report.txt"))


if __name__ == "__main__":
    main()

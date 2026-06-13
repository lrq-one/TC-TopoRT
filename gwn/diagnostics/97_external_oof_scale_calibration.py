from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr


OUT = Path("paper_analysis_stage4Y_oof_scale_calibration")
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "Eawag_XBridgeC18_364": {
        "target": 45.30,
        "candidates": {
            "old_zscore_rtfull": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_Eawag_XBridgeC18_364_src0",
            "deep_cwn_last1": "paper_analysis_stage4N_Eawag_deep_cwn_last1_lr5e5_src0",
            "deep_cwn_last2": "paper_analysis_stage4N_Eawag_deep_cwn_last2_lr3e5_src0",
        },
    },
    "FEM_long_412": {
        "target": 87.16,
        "candidates": {
            "zscore_rtfull": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_FEM_long_412_src0",
        },
    },
}


PRED_COLS = ["origin_tl_pred", "taut_tl_pred", "mean_tl_pred"]


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


def fit_predict_oof(df, pred_col, model_name):
    y = df["rt"].values.astype(float)
    x = df[[pred_col]].values.astype(float)

    pred = np.full(len(df), np.nan, dtype=float)

    kf = KFold(n_splits=10, shuffle=True, random_state=1)

    for tr, te in kf.split(x):
        if model_name == "affine":
            model = LinearRegression()
        elif model_name == "ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "huber":
            model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
        else:
            raise ValueError(model_name)

        model.fit(x[tr], y[tr])
        pred[te] = model.predict(x[te])

    return pred


def main():
    all_rows = []
    all_preds = []

    for dataset_name, cfg in DATASETS.items():
        target = cfg["target"]

        for cand_name, cand_dir in cfg["candidates"].items():
            pred_path = Path(cand_dir) / "external_tl_predictions.csv"
            if not pred_path.exists():
                print("[SKIP missing]", dataset_name, cand_name, pred_path)
                continue

            df = pd.read_csv(pred_path)
            df = df[df["dataset_name"] == dataset_name].copy().reset_index(drop=True)

            if len(df) == 0:
                print("[SKIP empty]", dataset_name, cand_name)
                continue

            y = df["rt"].values.astype(float)

            for pred_col in PRED_COLS:
                if pred_col not in df.columns:
                    continue

                base_pred = df[pred_col].values.astype(float)
                row = {
                    "dataset_name": dataset_name,
                    "candidate": cand_name,
                    "base_pred_col": pred_col,
                    "calibration": "none",
                    "target_abcort": target,
                    **metrics(y, base_pred),
                }
                row["improve_vs_target"] = target - row["mae"]
                all_rows.append(row)

                for cal in ["affine", "ridge", "huber"]:
                    p_cal = fit_predict_oof(df, pred_col, cal)

                    row = {
                        "dataset_name": dataset_name,
                        "candidate": cand_name,
                        "base_pred_col": pred_col,
                        "calibration": cal,
                        "target_abcort": target,
                        **metrics(y, p_cal),
                    }
                    row["improve_vs_target"] = target - row["mae"]
                    all_rows.append(row)

                    tmp = df[["dataset_name", "stage4_index", "record_id", "rt"]].copy()
                    tmp["candidate"] = cand_name
                    tmp["base_pred_col"] = pred_col
                    tmp["calibration"] = cal
                    tmp["base_pred"] = base_pred
                    tmp["cal_pred"] = p_cal
                    tmp["base_abs_err"] = np.abs(base_pred - y)
                    tmp["cal_abs_err"] = np.abs(p_cal - y)
                    all_preds.append(tmp)

    res = pd.DataFrame(all_rows)
    res = res.sort_values(["dataset_name", "mae"]).reset_index(drop=True)
    res.to_csv(OUT / "external_oof_scale_calibration_metrics.csv", index=False)

    if all_preds:
        pd.concat(all_preds, ignore_index=True).to_csv(
            OUT / "external_oof_scale_calibration_predictions.csv",
            index=False,
        )

    print("\n=== BEST BY DATASET ===")
    for dataset_name, sub in res.groupby("dataset_name"):
        print("\n", dataset_name)
        print(sub.head(12).to_string(index=False))

    print("\n[SAVE]", OUT / "external_oof_scale_calibration_metrics.csv")
    print("[SAVE]", OUT / "external_oof_scale_calibration_predictions.csv")


if __name__ == "__main__":
    main()

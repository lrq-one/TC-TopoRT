import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error


def metrics(y, p):
    err = np.abs(y - p)
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "MedAE": float(median_absolute_error(y, p)),
        "RMSE": float(mean_squared_error(y, p, squared=False)),
        "P95": float(np.percentile(err, 95)),
        "P99": float(np.percentile(err, 99)),
        ">100": int((err > 100).sum()),
        ">200": int((err > 200).sum()),
        "N": int(len(y)),
    }


def load_feat(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    smiles = z["smiles"].astype(str)
    feat = z["diff_token_feat"].astype(np.float32)  # [N, 3, D]
    y = z["y"].astype(np.float32)

    # flatten 0/1/2 token features
    x = feat.reshape(feat.shape[0], -1)

    df = pd.DataFrame({"SMILES": smiles, "y_feat": y})
    return df, x


def attach_features(pred_path, feat_df, feat_x):
    pred = pd.read_csv(pred_path)

    cols = pred.columns.tolist()
    if "Predicted_RT" in cols:
        pred_col = "Predicted_RT"
        y_col = "Actual_RT"
    else:
        pred_col = "y_pred"
        y_col = "y_true"

    pred = pred[["SMILES", y_col, pred_col]].copy()
    pred = pred.rename(columns={y_col: "y", pred_col: "base_pred"})

    idx_map = {s: i for i, s in enumerate(feat_df["SMILES"].astype(str).tolist())}

    keep = []
    x_list = []
    for i, smi in enumerate(pred["SMILES"].astype(str)):
        if smi in idx_map:
            keep.append(i)
            x_list.append(feat_x[idx_map[smi]])

    pred = pred.iloc[keep].reset_index(drop=True)
    x = np.stack(x_list, axis=0).astype(np.float32)

    return pred, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_npz", required=True)
    ap.add_argument("--train_pred", default="results_TopoCellRT_CWNReplace_orig/base_train_predictions.csv")
    ap.add_argument("--val_pred", default="results_TopoCellRT_CWNReplace_orig/base_val_predictions.csv")
    ap.add_argument("--test_pred", default="results_TopoCellRT_CWNReplace_orig/base_test_predictions.csv")
    ap.add_argument("--out_csv", default="results_TopoCellRT_CWNReplace_orig/unimol_residual_probe_test.csv")
    args = ap.parse_args()

    feat_df, feat_x = load_feat(args.feat_npz)

    train_df, x_train = attach_features(args.train_pred, feat_df, feat_x)
    val_df, x_val = attach_features(args.val_pred, feat_df, feat_x)
    test_df, x_test = attach_features(args.test_pred, feat_df, feat_x)

    print("matched train/val/test:", len(train_df), len(val_df), len(test_df))

    y_train = train_df["y"].values.astype(np.float32)
    y_val = val_df["y"].values.astype(np.float32)
    y_test = test_df["y"].values.astype(np.float32)

    base_train = train_df["base_pred"].values.astype(np.float32)
    base_val = val_df["base_pred"].values.astype(np.float32)
    base_test = test_df["base_pred"].values.astype(np.float32)

    print("\nBASE VAL :", metrics(y_val, base_val))
    print("BASE TEST:", metrics(y_test, base_test))

    # D2-A: Uni-Mol-only
    reg_y = make_pipeline(
        StandardScaler(with_mean=True),
        RidgeCV(alphas=[0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    )
    reg_y.fit(x_train, y_train)

    uni_val = reg_y.predict(x_val)
    uni_test = reg_y.predict(x_test)

    print("\nUNIMOL-ONLY VAL :", metrics(y_val, uni_val))
    print("UNIMOL-ONLY TEST:", metrics(y_test, uni_test))

    # D2-B: Residual correction for 26.35 CWNReplace
    residual_train = y_train - base_train

    reg_r = make_pipeline(
        StandardScaler(with_mean=True),
        RidgeCV(alphas=[0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    )
    reg_r.fit(x_train, residual_train)

    r_val = reg_r.predict(x_val)
    r_test = reg_r.predict(x_test)

    # alpha 只用 val 选，不能看 test 选
    rows = []
    for alpha in np.linspace(0.0, 1.0, 101):
        p_val = base_val + alpha * r_val
        m = metrics(y_val, p_val)
        rows.append((alpha, m["MAE"], m["P95"], m["P99"], m[">100"], m[">200"]))

    scan = pd.DataFrame(rows, columns=["alpha", "val_MAE", "val_P95", "val_P99", "val_100", "val_200"])
    best = scan.sort_values(["val_MAE", "val_P95"]).iloc[0]
    alpha = float(best["alpha"])

    final_val = base_val + alpha * r_val
    final_test = base_test + alpha * r_test

    print("\nRESIDUAL best alpha by val:", alpha)
    print("RESIDUAL VAL :", metrics(y_val, final_val))
    print("RESIDUAL TEST:", metrics(y_test, final_test))

    out = test_df.copy()
    out["unimol_only_pred"] = uni_test
    out["residual_pred"] = r_test
    out["final_pred"] = final_test
    out["final_abs_err"] = np.abs(out["y"].values - out["final_pred"].values)
    out.to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)


if __name__ == "__main__":
    main()

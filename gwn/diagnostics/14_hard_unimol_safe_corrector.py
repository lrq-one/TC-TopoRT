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
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "P95": float(np.percentile(err, 95)),
        "P99": float(np.percentile(err, 99)),
        ">80": int((err > 80).sum()),
        ">100": int((err > 100).sum()),
        ">200": int((err > 200).sum()),
        "N": int(len(y)),
    }


def norm_pred(df):
    cols = {c.lower(): c for c in df.columns}
    smi = cols.get("smiles", "SMILES")
    y = cols.get("actual_rt", None) or cols.get("y", None) or cols.get("y_true", None)
    p = cols.get("predicted_rt", None) or cols.get("y_pred", None) or cols.get("pred", None)
    if y is None or p is None:
        raise ValueError(f"bad prediction columns: {df.columns.tolist()}")
    out = df[[smi, y, p]].copy()
    out.columns = ["SMILES", "y", "base_pred"]
    out["SMILES"] = out["SMILES"].astype(str)
    out["y"] = out["y"].astype(float)
    out["base_pred"] = out["base_pred"].astype(float)
    out["base_abs_err"] = (out["y"] - out["base_pred"]).abs()
    return out


def load_feat(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    smiles = z["smiles"].astype(str)
    x = z["diff_token_feat"].astype(np.float32)
    x = x.reshape(x.shape[0], -1)
    return {s: x[i] for i, s in enumerate(smiles)}


def attach(df, fmap):
    keep = []
    xs = []
    for i, smi in enumerate(df["SMILES"].astype(str)):
        if smi in fmap:
            keep.append(i)
            xs.append(fmap[smi])
    sub = df.iloc[keep].copy().reset_index(drop=True)
    if len(xs) == 0:
        return sub, None
    return sub, np.stack(xs, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_npz", required=True)
    ap.add_argument("--train_pred", default="results_TopoCellRT_CWNReplace_orig/base_train_predictions.csv")
    ap.add_argument("--val_pred", default="results_TopoCellRT_CWNReplace_orig/base_val_predictions.csv")
    ap.add_argument("--test_pred", default="results_TopoCellRT_CWNReplace_orig/base_test_predictions.csv")
    ap.add_argument("--out_csv", default="results_TopoCellRT_CWNReplace_orig/hard_unimol_corrected_test.csv")
    args = ap.parse_args()

    fmap = load_feat(args.feat_npz)

    train = norm_pred(pd.read_csv(args.train_pred))
    val = norm_pred(pd.read_csv(args.val_pred))
    test = norm_pred(pd.read_csv(args.test_pred))

    tr, xtr = attach(train, fmap)
    va, xva = attach(val, fmap)
    te, xte = attach(test, fmap)

    print("feature matched train/val/test:", len(tr), len(va), len(te))
    if xtr is None or xva is None or xte is None:
        raise RuntimeError("not enough matched features")

    print("\nBASE full test:", metrics(test["y"].values, test["base_pred"].values))
    print("BASE hard test:", metrics(te["y"].values, te["base_pred"].values))

    ytr = tr["y"].values
    ptr = tr["base_pred"].values
    yva = va["y"].values
    pva = va["base_pred"].values
    yte = te["y"].values
    pte = te["base_pred"].values

    reg = make_pipeline(
        StandardScaler(with_mean=True),
        RidgeCV(alphas=[0.1, 1, 3, 10, 30, 100, 300, 1000, 3000])
    )
    reg.fit(xtr, ytr - ptr)

    rva = reg.predict(xva)
    rte = reg.predict(xte)

    # alpha + cap 都只用 val 选，避免 correction 过猛
    base_val_m = metrics(yva, pva)

    rows = []
    for cap in [20, 30, 40, 50, 60, 80, 100, 120, 160]:
        rva_cap = np.clip(rva, -cap, cap)
        for a in np.linspace(0.0, 0.8, 81):
            pv = pva + a * rva_cap
            m = metrics(yva, pv)

            # score：主要看 MAE，同时惩罚 >100 上升和 P95 变差
            penalty_100 = max(0, m[">100"] - base_val_m[">100"])
            penalty_p95 = max(0.0, m["P95"] - base_val_m["P95"])
            score = m["MAE"] + 0.08 * penalty_100 + 0.02 * penalty_p95

            rows.append((a, cap, score, m["MAE"], m["MedAE"], m["P95"], m["P99"], m[">80"], m[">100"], m[">200"]))

    scan = pd.DataFrame(
        rows,
        columns=["alpha", "cap", "score", "val_MAE", "val_MedAE", "val_P95", "val_P99", "val_80", "val_100", "val_200"]
    )

    best = scan.sort_values(["score", "val_MAE", "val_P95"]).iloc[0]
    alpha = float(best["alpha"])
    cap = float(best["cap"])

    print("\nselected alpha/cap:", alpha, cap)
    print("VAL hard base:", base_val_m)
    print("VAL hard corr:", metrics(yva, pva + alpha * np.clip(rva, -cap, cap)))
    print("\nTop val scan:")
    print(scan.sort_values(["score", "val_MAE"]).head(10).to_string(index=False))

    # 只修正有 Uni-Mol 特征的 test hard 分子，其他 test 分子保持原预测
    final = test.copy()
    pred_map = dict(zip(te["SMILES"], pte + alpha * np.clip(rte, -cap, cap)))
    final["final_pred"] = final.apply(
        lambda row: pred_map.get(row["SMILES"], row["base_pred"]),
        axis=1
    )
    final["final_abs_err"] = (final["y"] - final["final_pred"]).abs()
    final["was_corrected"] = final["SMILES"].isin(pred_map)

    print("\nTEST hard corr:", metrics(yte, pte + alpha * np.clip(rte, -cap, cap)))
    print("TEST full corr:", metrics(final["y"].values, final["final_pred"].values))

    final.to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)

    worst = final.sort_values("final_abs_err", ascending=False).head(30)
    print("\nTop 30 final worst:")
    print(worst[["SMILES", "y", "base_pred", "final_pred", "base_abs_err", "final_abs_err", "was_corrected"]].to_string(index=False))


if __name__ == "__main__":
    main()

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error


def metrics(y, p):
    e = np.abs(y - p)
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "MedAE": float(median_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "P95": float(np.percentile(e, 95)),
        "P99": float(np.percentile(e, 99)),
        ">80": int((e > 80).sum()),
        ">100": int((e > 100).sum()),
        ">200": int((e > 200).sum()),
        "N": int(len(y)),
    }


def norm_pred(path, pred_name):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    smi = cols.get("smiles", "SMILES")
    y = cols.get("actual_rt", None) or cols.get("y", None) or cols.get("y_true", None)
    p = cols.get("predicted_rt", None) or cols.get("y_pred", None) or cols.get("pred", None)

    if y is None or p is None:
        raise ValueError(f"bad columns in {path}: {df.columns.tolist()}")

    out = df[[smi, y, p]].copy()
    out.columns = ["SMILES", "y", pred_name]
    out["SMILES"] = out["SMILES"].astype(str)
    out["y"] = out["y"].astype(float)
    out[pred_name] = out[pred_name].astype(float)
    return out


def merge_pair(base_path, aux_path):
    b = norm_pred(base_path, "base_pred")
    a = norm_pred(aux_path, "aux_pred")
    df = b.merge(a[["SMILES", "aux_pred"]], on="SMILES", how="inner")
    df["view_diff"] = (df["base_pred"] - df["aux_pred"]).abs()
    return df


def make_pred(df, mode, w, tau):
    base = df["base_pred"].values
    aux = df["aux_pred"].values
    diff = df["view_diff"].values

    cand = base + w * (aux - base)

    if mode == "global":
        return cand

    if mode == "high_diff":
        use = diff >= tau
        out = base.copy()
        out[use] = cand[use]
        return out

    if mode == "low_diff":
        use = diff <= tau
        out = base.copy()
        out[use] = cand[use]
        return out

    raise ValueError(mode)


def score_metric(m, base_m):
    # 以 MAE 为主，同时不要明显伤 tail
    return (
        m["MAE"]
        + 0.03 * max(0.0, m["P95"] - base_m["P95"])
        + 0.08 * max(0, m[">100"] - base_m[">100"])
        + 0.15 * max(0, m[">200"] - base_m[">200"])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_train", required=True)
    ap.add_argument("--base_val", required=True)
    ap.add_argument("--base_test", required=True)
    ap.add_argument("--aux_train", required=True)
    ap.add_argument("--aux_val", required=True)
    ap.add_argument("--aux_test", required=True)
    ap.add_argument("--out_dir", default="results_TopoCellRT_CWNReplace_orig/dualmodel_fusion")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train = merge_pair(args.base_train, args.aux_train)
    val = merge_pair(args.base_val, args.aux_val)
    test = merge_pair(args.base_test, args.aux_test)

    print("rows train/val/test:", len(train), len(val), len(test))

    yv = val["y"].values
    yt = test["y"].values

    base_val_m = metrics(yv, val["base_pred"].values)
    aux_val_m = metrics(yv, val["aux_pred"].values)
    base_test_m = metrics(yt, test["base_pred"].values)
    aux_test_m = metrics(yt, test["aux_pred"].values)

    print("\nVAL base:", base_val_m)
    print("VAL aux :", aux_val_m)
    print("\nTEST base:", base_test_m)
    print("TEST aux :", aux_test_m)

    rows = []

    weights = np.round(np.arange(-0.50, 1.501, 0.01), 2)
    taus = [0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200]

    for mode in ["global", "high_diff", "low_diff"]:
        for w in weights:
            if mode == "global":
                pred_v = make_pred(val, mode, w, 0)
                m = metrics(yv, pred_v)
                rows.append({
                    "mode": mode,
                    "w": float(w),
                    "tau": 0.0,
                    "score": score_metric(m, base_val_m),
                    **{f"val_{k}": v for k, v in m.items()}
                })
            else:
                for tau in taus:
                    pred_v = make_pred(val, mode, w, tau)
                    m = metrics(yv, pred_v)
                    rows.append({
                        "mode": mode,
                        "w": float(w),
                        "tau": float(tau),
                        "score": score_metric(m, base_val_m),
                        **{f"val_{k}": v for k, v in m.items()}
                    })

    scan = pd.DataFrame(rows)
    scan = scan.sort_values(["score", "val_MAE", "val_P95"]).reset_index(drop=True)

    print("\nTop val fusion candidates:")
    print(scan.head(30).to_string(index=False))

    best = scan.iloc[0]
    mode = best["mode"]
    w = float(best["w"])
    tau = float(best["tau"])

    print("\nSELECTED:", {"mode": mode, "w": w, "tau": tau})

    pred_val = make_pred(val, mode, w, tau)
    pred_test = make_pred(test, mode, w, tau)

    val_m = metrics(yv, pred_val)
    test_m = metrics(yt, pred_test)

    print("\nVAL selected:", val_m)
    print("TEST selected:", test_m)

    out = test.copy()
    out["final_pred"] = pred_test
    out["base_abs_err"] = np.abs(out["y"].values - out["base_pred"].values)
    out["aux_abs_err"] = np.abs(out["y"].values - out["aux_pred"].values)
    out["final_abs_err"] = np.abs(out["y"].values - out["final_pred"].values)
    out["gain_vs_base"] = out["base_abs_err"] - out["final_abs_err"]

    out.to_csv(f"{args.out_dir}/dualmodel_fused_test_predictions.csv", index=False)
    scan.to_csv(f"{args.out_dir}/dualmodel_fusion_val_scan.csv", index=False)

    print("\nGain summary test:")
    print("improved:", int((out["gain_vs_base"] > 0).sum()))
    print("worsened :", int((out["gain_vs_base"] < 0).sum()))
    print("net_gain:", float(out["gain_vs_base"].sum()))
    print("mean_gain:", float(out["gain_vs_base"].mean()))

    print("\nTop 30 final worst:")
    print(out.sort_values("final_abs_err", ascending=False).head(30)[
        ["SMILES", "y", "base_pred", "aux_pred", "final_pred", "base_abs_err", "aux_abs_err", "final_abs_err", "view_diff"]
    ].to_string(index=False))

    print("\nsaved:", args.out_dir)


if __name__ == "__main__":
    main()

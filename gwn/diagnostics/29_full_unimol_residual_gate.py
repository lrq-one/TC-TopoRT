import argparse
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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


def norm_pred(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    smi = cols.get("smiles", "SMILES")
    y = cols.get("actual_rt", None) or cols.get("y", None) or cols.get("y_true", None)
    p = cols.get("predicted_rt", None) or cols.get("y_pred", None) or cols.get("pred", None)

    if y is None or p is None:
        raise ValueError(f"bad columns in {path}: {df.columns.tolist()}")

    out = df[[smi, y, p]].copy()
    out.columns = ["SMILES", "y", "base_pred"]
    out["SMILES"] = out["SMILES"].astype(str)
    out["y"] = out["y"].astype(float)
    out["base_pred"] = out["base_pred"].astype(float)
    return out


def load_feat(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    smiles = z["smiles"].astype(str)
    feat = z["diff_token_feat"].astype(np.float32)

    if feat.ndim < 2:
        raise ValueError(f"bad feat shape: {feat.shape}")

    feat = feat.reshape(feat.shape[0], -1)
    mp = {str(s): feat[i] for i, s in enumerate(smiles)}
    dim = feat.shape[1]
    return mp, dim


def align_features(df, feat_map, dim):
    xs = np.zeros((len(df), dim), dtype=np.float32)
    mask = np.zeros(len(df), dtype=np.float32)

    for i, s in enumerate(df["SMILES"].astype(str).values):
        x = feat_map.get(s)
        if x is not None:
            xs[i] = x
            mask[i] = 1.0

    # 加上 base_pred 和 has_3d mask，避免模型不知道哪些是缺失 3D
    extra = np.stack([
        df["base_pred"].values.astype(np.float32),
        mask.astype(np.float32),
    ], axis=1)

    xs = np.concatenate([xs, extra], axis=1)
    return xs, mask


def apply_correction(base, raw_delta, blend, cap, min_abs):
    delta = raw_delta.copy()
    delta = np.clip(delta, -cap, cap)
    use = np.abs(delta) >= min_abs
    out = base.copy()
    out[use] = base[use] + blend * delta[use]
    return out, use


def score_metric(m, base_m):
    # 主看 MAE，同时惩罚 tail 变差
    return (
        m["MAE"]
        + 0.03 * max(0.0, m["P95"] - base_m["P95"])
        + 0.05 * max(0.0, m["P99"] - base_m["P99"])
        + 0.08 * max(0, m[">100"] - base_m[">100"])
        + 0.15 * max(0, m[">200"] - base_m[">200"])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_npz", required=True)
    ap.add_argument("--train_pred", required=True)
    ap.add_argument("--val_pred", required=True)
    ap.add_argument("--test_pred", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("loading predictions...")
    train = norm_pred(args.train_pred)
    val = norm_pred(args.val_pred)
    test = norm_pred(args.test_pred)

    print("loading 3D features...")
    feat_map, dim = load_feat(args.feat_npz)
    print("feature smiles:", len(feat_map), "flat dim:", dim)

    xtr, mtr = align_features(train, feat_map, dim)
    xva, mva = align_features(val, feat_map, dim)
    xte, mte = align_features(test, feat_map, dim)

    print("matched train/val/test:", int(mtr.sum()), int(mva.sum()), int(mte.sum()))

    ytr = train["y"].values
    yva = val["y"].values
    yte = test["y"].values

    btr = train["base_pred"].values
    bva = val["base_pred"].values
    bte = test["base_pred"].values

    rtr = ytr - btr

    base_val_m = metrics(yva, bva)
    base_test_m = metrics(yte, bte)

    print("\nBASE val :", base_val_m)
    print("BASE test:", base_test_m)

    print("\nstandardizing features...")
    scaler = StandardScaler()
    xtr_s = scaler.fit_transform(xtr).astype(np.float32)
    xva_s = scaler.transform(xva).astype(np.float32)
    xte_s = scaler.transform(xte).astype(np.float32)

    rows = []
    models = {}

    ridge_alphas = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    blends = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]
    caps = [10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0]
    min_abs_list = [0.0, 2.0, 5.0, 10.0, 20.0]

    for a in ridge_alphas:
        print("training ridge alpha =", a)
        model = Ridge(alpha=a, solver="lsqr", fit_intercept=True, max_iter=3000)
        model.fit(xtr_s, rtr)
        models[a] = model

        raw_va = model.predict(xva_s).astype(np.float32)

        for blend in blends:
            for cap in caps:
                for min_abs in min_abs_list:
                    pred_va, use_va = apply_correction(bva, raw_va, blend, cap, min_abs)
                    m = metrics(yva, pred_va)
                    rows.append({
                        "alpha": a,
                        "blend": blend,
                        "cap": cap,
                        "min_abs": min_abs,
                        "used_val": int(use_va.sum()),
                        "score": score_metric(m, base_val_m),
                        **{f"val_{k}": v for k, v in m.items()},
                    })

    scan = pd.DataFrame(rows)
    scan = scan.sort_values(["score", "val_MAE", "val_P95"]).reset_index(drop=True)
    scan.to_csv(f"{args.out_dir}/full_unimol_residual_val_scan.csv", index=False)

    print("\nTop val candidates:")
    print(scan.head(30).to_string(index=False))

    best = scan.iloc[0]
    alpha = float(best["alpha"])
    blend = float(best["blend"])
    cap = float(best["cap"])
    min_abs = float(best["min_abs"])

    print("\nSELECTED:", {
        "alpha": alpha,
        "blend": blend,
        "cap": cap,
        "min_abs": min_abs,
    })

    model = models[alpha]
    raw_va = model.predict(xva_s).astype(np.float32)
    raw_te = model.predict(xte_s).astype(np.float32)

    pred_va, use_va = apply_correction(bva, raw_va, blend, cap, min_abs)
    pred_te, use_te = apply_correction(bte, raw_te, blend, cap, min_abs)

    val_m = metrics(yva, pred_va)
    test_m = metrics(yte, pred_te)

    print("\nVAL corrected :", val_m)
    print("TEST corrected:", test_m)
    print("used val/test:", int(use_va.sum()), int(use_te.sum()))

    out = test.copy()
    out["raw_delta"] = raw_te
    out["used_3d_correction"] = use_te
    out["final_pred"] = pred_te
    out["base_abs_err"] = np.abs(out["y"].values - out["base_pred"].values)
    out["final_abs_err"] = np.abs(out["y"].values - out["final_pred"].values)
    out["gain_vs_base"] = out["base_abs_err"] - out["final_abs_err"]

    out.to_csv(f"{args.out_dir}/full_unimol_residual_corrected_test.csv", index=False)

    print("\nGain summary test:")
    print("improved:", int((out["gain_vs_base"] > 0).sum()))
    print("worsened :", int((out["gain_vs_base"] < 0).sum()))
    print("net_gain:", float(out["gain_vs_base"].sum()))
    print("mean_gain:", float(out["gain_vs_base"].mean()))

    print("\nTop 30 final worst:")
    print(out.sort_values("final_abs_err", ascending=False).head(30)[[
        "SMILES", "y", "base_pred", "final_pred", "base_abs_err", "final_abs_err", "raw_delta", "used_3d_correction"
    ]].to_string(index=False))

    print("\nsaved:", args.out_dir)


if __name__ == "__main__":
    main()

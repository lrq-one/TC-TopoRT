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
    feat = feat.reshape(feat.shape[0], -1)
    mp = {str(s): feat[i] for i, s in enumerate(smiles)}
    return mp, feat.shape[1]


def align_features(df, feat_map, dim):
    xs = np.zeros((len(df), dim), dtype=np.float32)
    mask = np.zeros(len(df), dtype=np.float32)

    for i, s in enumerate(df["SMILES"].astype(str).values):
        x = feat_map.get(s)
        if x is not None:
            xs[i] = x
            mask[i] = 1.0

    extra = np.stack([
        df["base_pred"].values.astype(np.float32),
        mask.astype(np.float32),
    ], axis=1)

    return np.concatenate([xs, extra], axis=1), mask


def make_weights(abs_res, mode, gamma):
    w = np.ones_like(abs_res, dtype=np.float32)

    if mode == "uniform":
        return w

    if mode == "abs":
        # 残差越大，权重越高；最多约 1 + gamma * 4
        return 1.0 + gamma * np.clip(abs_res / 80.0, 0.0, 4.0)

    if mode == "hard80":
        return 1.0 + gamma * (abs_res >= 80.0).astype(np.float32)

    if mode == "hard100":
        return 1.0 + gamma * (abs_res >= 100.0).astype(np.float32)

    if mode == "tail":
        w += gamma * (abs_res >= 80.0).astype(np.float32)
        w += gamma * 2.0 * (abs_res >= 150.0).astype(np.float32)
        w += gamma * 3.0 * (abs_res >= 250.0).astype(np.float32)
        return w

    raise ValueError(mode)


def apply_correction(base, raw_delta, blend, cap, min_abs):
    delta = np.clip(raw_delta.copy(), -cap, cap)
    use = np.abs(delta) >= min_abs
    out = base.copy()
    out[use] = base[use] + blend * delta[use]
    return out, use


def score_metric(m, base_m):
    return (
        m["MAE"]
        + 0.03 * max(0.0, m["P95"] - base_m["P95"])
        + 0.05 * max(0.0, m["P99"] - base_m["P99"])
        + 0.10 * max(0, m[">100"] - base_m[">100"])
        + 0.20 * max(0, m[">200"] - base_m[">200"])
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

    train = norm_pred(args.train_pred)
    val = norm_pred(args.val_pred)
    test = norm_pred(args.test_pred)

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
    abs_rtr = np.abs(rtr)

    base_val_m = metrics(yva, bva)
    base_test_m = metrics(yte, bte)

    print("\nBASE val :", base_val_m)
    print("BASE test:", base_test_m)

    print("\nstandardizing features...")
    scaler = StandardScaler()
    xtr_s = scaler.fit_transform(xtr).astype(np.float32)
    xva_s = scaler.transform(xva).astype(np.float32)
    xte_s = scaler.transform(xte).astype(np.float32)

    configs = [
        ("uniform", 0.0),
        ("abs", 2.0), ("abs", 5.0), ("abs", 10.0),
        ("hard80", 5.0), ("hard80", 10.0), ("hard80", 20.0), ("hard80", 50.0),
        ("hard100", 10.0), ("hard100", 20.0), ("hard100", 50.0),
        ("tail", 2.0), ("tail", 5.0), ("tail", 10.0),
    ]

    ridge_alphas = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    blends = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]
    caps = [20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 180.0]
    min_abs_list = [0.0, 5.0, 10.0, 20.0, 40.0]

    rows = []
    saved_models = {}

    for mode, gamma in configs:
        sw = make_weights(abs_rtr, mode, gamma)
        print(f"\nconfig mode={mode} gamma={gamma} weight_mean={sw.mean():.3f} max={sw.max():.1f}")

        for alpha in ridge_alphas:
            print(" training ridge alpha =", alpha)
            model = Ridge(alpha=alpha, solver="lsqr", fit_intercept=True, max_iter=3000)
            model.fit(xtr_s, rtr, sample_weight=sw)

            key = (mode, gamma, alpha)
            saved_models[key] = model

            raw_va = model.predict(xva_s).astype(np.float32)

            for blend in blends:
                for cap in caps:
                    for min_abs in min_abs_list:
                        pred_va, use_va = apply_correction(bva, raw_va, blend, cap, min_abs)
                        m = metrics(yva, pred_va)
                        rows.append({
                            "mode": mode,
                            "gamma": gamma,
                            "alpha": alpha,
                            "blend": blend,
                            "cap": cap,
                            "min_abs": min_abs,
                            "used_val": int(use_va.sum()),
                            "score": score_metric(m, base_val_m),
                            **{f"val_{k}": v for k, v in m.items()},
                        })

    scan = pd.DataFrame(rows)
    scan = scan.sort_values(["score", "val_MAE", "val_P95"]).reset_index(drop=True)
    scan.to_csv(f"{args.out_dir}/weighted_full_unimol_val_scan.csv", index=False)

    print("\nTop val candidates:")
    print(scan.head(40).to_string(index=False))

    best = scan.iloc[0]
    mode = str(best["mode"])
    gamma = float(best["gamma"])
    alpha = float(best["alpha"])
    blend = float(best["blend"])
    cap = float(best["cap"])
    min_abs = float(best["min_abs"])

    print("\nSELECTED:", {
        "mode": mode,
        "gamma": gamma,
        "alpha": alpha,
        "blend": blend,
        "cap": cap,
        "min_abs": min_abs,
    })

    model = saved_models[(mode, gamma, alpha)]
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

    out.to_csv(f"{args.out_dir}/weighted_full_unimol_corrected_test.csv", index=False)

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

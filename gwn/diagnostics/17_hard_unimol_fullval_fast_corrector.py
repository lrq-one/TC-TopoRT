import argparse
import time
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
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
    z = np.load(npz_path, allow_pickle=False)
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


def make_features(x, df):
    base = df[["base_pred"]].values.astype(np.float32)
    return np.concatenate([x, base], axis=1)


def build_full_indices(full_df, hard_df):
    mp = {}
    for i, smi in enumerate(full_df["SMILES"].astype(str).values):
        if smi not in mp:
            mp[smi] = i
    idx = []
    missing = 0
    for smi in hard_df["SMILES"].astype(str).values:
        if smi in mp:
            idx.append(mp[smi])
        else:
            idx.append(-1)
            missing += 1
    if missing:
        print("WARNING missing hard smiles in full df:", missing)
    return np.asarray(idx, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_npz", required=True)
    ap.add_argument("--train_pred", default="results_TopoCellRT_CWNReplace_orig/base_train_predictions.csv")
    ap.add_argument("--val_pred", default="results_TopoCellRT_CWNReplace_orig/base_val_predictions.csv")
    ap.add_argument("--test_pred", default="results_TopoCellRT_CWNReplace_orig/base_test_predictions.csv")
    ap.add_argument("--out_csv", default="results_TopoCellRT_CWNReplace_orig/hard_unimol_fullval_fast_corrected_test.csv")
    args = ap.parse_args()

    t0 = time.time()

    fmap = load_feat(args.feat_npz)

    train = norm_pred(pd.read_csv(args.train_pred))
    val = norm_pred(pd.read_csv(args.val_pred))
    test = norm_pred(pd.read_csv(args.test_pred))

    tr, xtr_raw = attach(train, fmap)
    va, xva_raw = attach(val, fmap)
    te, xte_raw = attach(test, fmap)

    print("feature matched train/val/test:", len(tr), len(va), len(te))

    xtr = make_features(xtr_raw, tr)
    xva = make_features(xva_raw, va)
    xte = make_features(xte_raw, te)

    ytr = tr["y"].values
    ptr = tr["base_pred"].values
    yva = va["y"].values
    pva = va["base_pred"].values
    yte = te["y"].values
    pte = te["base_pred"].values

    yval_full = val["y"].values
    pval_full = val["base_pred"].values
    ytest_full = test["y"].values
    ptest_full = test["base_pred"].values

    va_full_idx = build_full_indices(val, va)
    te_full_idx = build_full_indices(test, te)

    valid_va = va_full_idx >= 0
    valid_te = te_full_idx >= 0

    print("\nBASE full test:", metrics(ytest_full, ptest_full))
    print("BASE hard test:", metrics(yte, pte))

    residual_tr = ytr - ptr

    print("\nFitting residual Ridge...")
    reg = make_pipeline(
        StandardScaler(with_mean=True),
        RidgeCV(alphas=[0.1, 1, 3, 10, 30, 100, 300, 1000, 3000])
    )
    reg.fit(xtr, residual_tr)

    rva = reg.predict(xva)
    rte = reg.predict(xte)

    print("Fitting direction classifier...")
    sign_y = (residual_tr > 0).astype(int)

    clf = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(
            C=0.3,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
    )
    clf.fit(xtr, sign_y)

    ppos_va = clf.predict_proba(xva)[:, 1]
    ppos_te = clf.predict_proba(xte)[:, 1]

    sign_va = np.where(ppos_va >= 0.5, 1.0, -1.0)
    sign_te = np.where(ppos_te >= 0.5, 1.0, -1.0)

    conf_va = np.maximum(ppos_va, 1.0 - ppos_va)
    conf_te = np.maximum(ppos_te, 1.0 - ppos_te)

    mag_va = np.abs(rva)
    mag_te = np.abs(rte)

    base_val_full_m = metrics(yval_full, pval_full)
    base_val_hard_m = metrics(yva, pva)

    print("\nVAL full base:", base_val_full_m)
    print("VAL hard base:", base_val_hard_m)

    rows = []

    caps = [20, 30, 40, 50, 60, 80, 100, 120]
    alphas = np.linspace(0.0, 0.8, 41)
    taus = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    min_mags = [0, 20, 40, 60, 80]

    total = len(caps) * len(alphas) * len(taus) * len(min_mags)
    print("\nScanning combos:", total)

    scanned = 0
    for cap in caps:
        clip_mag_va = np.clip(mag_va, 0, cap)

        for alpha in alphas:
            corr_all = alpha * clip_mag_va * sign_va

            for tau in taus:
                conf_mask = conf_va >= tau

                for min_mag in min_mags:
                    use = conf_mask & (mag_va >= min_mag) & valid_va

                    pv_hard = pva.copy()
                    pv_hard[use] = pv_hard[use] + corr_all[use]
                    mh = metrics(yva, pv_hard)

                    # vectorized full-val update，替代慢的 pandas apply
                    pv_full = pval_full.copy()
                    idx = va_full_idx[use]
                    pv_full[idx] = pv_hard[use]
                    mf = metrics(yval_full, pv_full)

                    penalty_100 = max(0, mf[">100"] - base_val_full_m[">100"])
                    penalty_p95 = max(0.0, mf["P95"] - base_val_full_m["P95"])
                    penalty_200 = max(0, mf[">200"] - base_val_full_m[">200"])

                    # 允许 MAE 降，但强惩罚 full-val tail 变差
                    score = (
                        mf["MAE"]
                        + 0.25 * penalty_100
                        + 0.08 * penalty_p95
                        + 0.40 * penalty_200
                    )

                    rows.append((
                        alpha, cap, tau, min_mag, score,
                        mf["MAE"], mf["MedAE"], mf["P95"], mf["P99"],
                        mf[">80"], mf[">100"], mf[">200"],
                        mh["MAE"], mh["P95"], mh[">100"], mh[">200"],
                        int(use.sum())
                    ))

                    scanned += 1

    scan = pd.DataFrame(
        rows,
        columns=[
            "alpha", "cap", "tau", "min_mag", "score",
            "full_val_MAE", "full_val_MedAE", "full_val_P95", "full_val_P99",
            "full_val_80", "full_val_100", "full_val_200",
            "hard_val_MAE", "hard_val_P95", "hard_val_100", "hard_val_200",
            "val_used"
        ]
    )

    best = scan.sort_values(["score", "full_val_MAE", "full_val_P95"]).iloc[0]

    alpha = float(best["alpha"])
    cap = float(best["cap"])
    tau = float(best["tau"])
    min_mag = float(best["min_mag"])

    print("\nselected alpha/cap/tau/min_mag:", alpha, cap, tau, min_mag)
    print("\nTop val scan:")
    print(scan.sort_values(["score", "full_val_MAE"]).head(15).to_string(index=False))

    # apply to val for report
    use_va = (conf_va >= tau) & (mag_va >= min_mag) & valid_va
    corr_va = alpha * np.clip(mag_va, 0, cap) * sign_va
    pva_corr = pva.copy()
    pva_corr[use_va] = pva_corr[use_va] + corr_va[use_va]

    pv_full_val = pval_full.copy()
    pv_full_val[va_full_idx[use_va]] = pva_corr[use_va]

    print("\nVAL hard corr:", metrics(yva, pva_corr))
    print("VAL full corr:", metrics(yval_full, pv_full_val))
    print("corrected hard val:", int(use_va.sum()), "/", len(va))

    # apply to test
    use_te = (conf_te >= tau) & (mag_te >= min_mag) & valid_te
    corr_te = alpha * np.clip(mag_te, 0, cap) * sign_te
    pte_corr = pte.copy()
    pte_corr[use_te] = pte_corr[use_te] + corr_te[use_te]

    ptest_final = ptest_full.copy()
    ptest_final[te_full_idx[use_te]] = pte_corr[use_te]

    final = test.copy()
    final["final_pred"] = ptest_final
    final["final_abs_err"] = np.abs(final["y"].values - final["final_pred"].values)
    corrected_smiles = set(te.loc[use_te, "SMILES"].astype(str).values)
    final["was_corrected"] = final["SMILES"].astype(str).isin(corrected_smiles)

    print("\nTEST hard corr:", metrics(yte, pte_corr))
    print("TEST full corr:", metrics(ytest_full, ptest_final))
    print("corrected hard test:", int(use_te.sum()), "/", len(te))

    final.to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)

    worst = final.sort_values("final_abs_err", ascending=False).head(30)
    print("\nTop 30 final worst:")
    print(worst[["SMILES", "y", "base_pred", "final_pred", "base_abs_err", "final_abs_err", "was_corrected"]].to_string(index=False))

    print("\ntime_sec:", round(time.time() - t0, 2))


if __name__ == "__main__":
    main()

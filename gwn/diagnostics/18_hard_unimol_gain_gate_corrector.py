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
    keep, xs = [], []
    for i, smi in enumerate(df["SMILES"].astype(str)):
        if smi in fmap:
            keep.append(i)
            xs.append(fmap[smi])
    sub = df.iloc[keep].copy().reset_index(drop=True)
    if len(xs) == 0:
        return sub, None
    return sub, np.stack(xs, axis=0).astype(np.float32)


def make_x(x, df):
    base = df[["base_pred"]].values.astype(np.float32)
    return np.concatenate([x, base], axis=1)


def make_gate_x(x, corr):
    corr = corr.reshape(-1, 1).astype(np.float32)
    return np.concatenate(
        [
            x,
            corr,
            np.abs(corr),
            np.sign(corr),
        ],
        axis=1,
    )


def build_full_indices(full_df, hard_df):
    mp = {}
    for i, smi in enumerate(full_df["SMILES"].astype(str).values):
        if smi not in mp:
            mp[smi] = i
    return np.asarray([mp.get(smi, -1) for smi in hard_df["SMILES"].astype(str).values], dtype=np.int64)


def fit_gain_gate(xtr_gate, y_gain):
    if len(np.unique(y_gain)) < 2:
        return None
    clf = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(
            C=0.2,
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )
    )
    clf.fit(xtr_gate, y_gain)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_npz", required=True)
    ap.add_argument("--train_pred", default="results_TopoCellRT_CWNReplace_orig/base_train_predictions.csv")
    ap.add_argument("--val_pred", default="results_TopoCellRT_CWNReplace_orig/base_val_predictions.csv")
    ap.add_argument("--test_pred", default="results_TopoCellRT_CWNReplace_orig/base_test_predictions.csv")
    ap.add_argument("--out_csv", default="results_TopoCellRT_CWNReplace_orig/hard_unimol_gain_gate_corrected_test.csv")
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

    xtr = make_x(xtr_raw, tr)
    xva = make_x(xva_raw, va)
    xte = make_x(xte_raw, te)

    ytr, ptr = tr["y"].values, tr["base_pred"].values
    yva, pva = va["y"].values, va["base_pred"].values
    yte, pte = te["y"].values, te["base_pred"].values

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

    print("\nFitting residual Ridge...")
    reg = make_pipeline(
        StandardScaler(with_mean=True),
        RidgeCV(alphas=[0.1, 1, 3, 10, 30, 100, 300, 1000, 3000])
    )
    reg.fit(xtr, ytr - ptr)

    rtr = reg.predict(xtr)
    rva = reg.predict(xva)
    rte = reg.predict(xte)

    base_val_full = metrics(yval_full, pval_full)
    print("\nVAL full base:", base_val_full)

    rows = []
    caps = [40, 60, 80, 100, 120, 160]
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    taus = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    min_corrs = [0, 10, 20, 40, 60]

    print("\nScanning gain-gated candidates...")

    for cap in caps:
        for alpha in alphas:
            corr_tr = alpha * np.clip(rtr, -cap, cap)
            corr_va = alpha * np.clip(rva, -cap, cap)

            # gain label: 这个 correction 是否让 train 误差变小
            base_err_tr = np.abs(ytr - ptr)
            cand_err_tr = np.abs(ytr - (ptr + corr_tr))
            y_gain = (cand_err_tr < base_err_tr).astype(int)

            xtr_gate = make_gate_x(xtr, corr_tr)
            xva_gate = make_gate_x(xva, corr_va)

            clf = fit_gain_gate(xtr_gate, y_gain)
            if clf is None:
                continue

            prob_va = clf.predict_proba(xva_gate)[:, 1]

            for tau in taus:
                for min_corr in min_corrs:
                    use = (prob_va >= tau) & (np.abs(corr_va) >= min_corr) & valid_va

                    pva_corr = pva.copy()
                    pva_corr[use] = pva_corr[use] + corr_va[use]

                    pv_full = pval_full.copy()
                    pv_full[va_full_idx[use]] = pva_corr[use]

                    mf = metrics(yval_full, pv_full)
                    mh = metrics(yva, pva_corr)

                    penalty_100 = max(0, mf[">100"] - base_val_full[">100"])
                    penalty_p95 = max(0.0, mf["P95"] - base_val_full["P95"])
                    penalty_200 = max(0, mf[">200"] - base_val_full[">200"])

                    score = (
                        mf["MAE"]
                        + 0.15 * penalty_100
                        + 0.05 * penalty_p95
                        + 0.25 * penalty_200
                    )

                    rows.append((
                        alpha, cap, tau, min_corr, score,
                        mf["MAE"], mf["P95"], mf["P99"], mf[">100"], mf[">200"],
                        mh["MAE"], mh["P95"], mh[">100"], mh[">200"],
                        int(use.sum()),
                        float(y_gain.mean()),
                    ))

    scan = pd.DataFrame(
        rows,
        columns=[
            "alpha", "cap", "tau", "min_corr", "score",
            "full_val_MAE", "full_val_P95", "full_val_P99", "full_val_100", "full_val_200",
            "hard_val_MAE", "hard_val_P95", "hard_val_100", "hard_val_200",
            "val_used", "train_gain_rate",
        ]
    )

    best = scan.sort_values(["score", "full_val_MAE", "full_val_P95"]).iloc[0]
    alpha = float(best["alpha"])
    cap = float(best["cap"])
    tau = float(best["tau"])
    min_corr = float(best["min_corr"])

    print("\nselected alpha/cap/tau/min_corr:", alpha, cap, tau, min_corr)
    print("\nTop val scan:")
    print(scan.sort_values(["score", "full_val_MAE"]).head(15).to_string(index=False))

    # refit gain gate with best alpha/cap
    corr_tr = alpha * np.clip(rtr, -cap, cap)
    corr_va = alpha * np.clip(rva, -cap, cap)
    corr_te = alpha * np.clip(rte, -cap, cap)

    y_gain = (np.abs(ytr - (ptr + corr_tr)) < np.abs(ytr - ptr)).astype(int)

    clf = fit_gain_gate(make_gate_x(xtr, corr_tr), y_gain)

    prob_va = clf.predict_proba(make_gate_x(xva, corr_va))[:, 1]
    prob_te = clf.predict_proba(make_gate_x(xte, corr_te))[:, 1]

    use_va = (prob_va >= tau) & (np.abs(corr_va) >= min_corr) & valid_va
    use_te = (prob_te >= tau) & (np.abs(corr_te) >= min_corr) & valid_te

    pva_corr = pva.copy()
    pva_corr[use_va] = pva_corr[use_va] + corr_va[use_va]

    pv_full = pval_full.copy()
    pv_full[va_full_idx[use_va]] = pva_corr[use_va]

    print("\nVAL hard corr:", metrics(yva, pva_corr))
    print("VAL full corr:", metrics(yval_full, pv_full))
    print("corrected hard val:", int(use_va.sum()), "/", len(va))

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

    corr_df = final[final["was_corrected"]].copy()
    if len(corr_df):
        corr_df["gain"] = corr_df["base_abs_err"] - corr_df["final_abs_err"]
        print("\nGain summary on corrected test:")
        print("corrected:", len(corr_df))
        print("improved:", int((corr_df["gain"] > 0).sum()))
        print("worsened:", int((corr_df["gain"] < 0).sum()))
        print("net_gain:", float(corr_df["gain"].sum()))
        print("mean_gain:", float(corr_df["gain"].mean()))

    worst = final.sort_values("final_abs_err", ascending=False).head(30)
    print("\nTop 30 final worst:")
    print(worst[["SMILES", "y", "base_pred", "final_pred", "base_abs_err", "final_abs_err", "was_corrected"]].to_string(index=False))

    print("\ntime_sec:", round(time.time() - t0, 2))


if __name__ == "__main__":
    main()

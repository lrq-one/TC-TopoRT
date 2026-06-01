import argparse
import os
import numpy as np
import pandas as pd


def load_pred(path, name):
    df = pd.read_csv(path)

    required = ["Source_Index", "SMILES", "Actual_RT", "Predicted_RT"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns {miss}, columns={df.columns.tolist()}")

    out = df[required].copy()
    out = out.rename(columns={
        "SMILES": f"SMILES_{name}",
        "Predicted_RT": f"{name}_pred",
        "Actual_RT": "y",
    })
    out["Source_Index"] = out["Source_Index"].astype(int)
    out["y"] = out["y"].astype(float)
    out[f"{name}_pred"] = out[f"{name}_pred"].astype(float)
    return out


def metrics(y, p):
    e = np.abs(y - p)
    return {
        "MAE": float(e.mean()),
        "MedAE": float(np.median(e)),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "P95": float(np.percentile(e, 95)),
        "P99": float(np.percentile(e, 99)),
        ">80": int((e > 80).sum()),
        ">100": int((e > 100).sum()),
        ">200": int((e > 200).sum()),
        "N": int(len(e)),
    }


def merge_idx(origin_path, taut_path, split):
    o = load_pred(origin_path, "origin")
    t = load_pred(taut_path, "taut")

    m = o.merge(
        t[["Source_Index", "SMILES_taut", "y", "taut_pred"]],
        on="Source_Index",
        how="inner",
        suffixes=("_origin", "_taut"),
    )

    print(f"\n[{split}]")
    print("origin rows:", len(o))
    print("taut rows  :", len(t))
    print("matched    :", len(m))

    if len(m) != len(o) or len(m) != len(t):
        raise RuntimeError(f"{split}: Source_Index 没有完全匹配。")

    ydiff = np.abs(m["y_origin"].values - m["y_taut"].values)
    print("max y diff:", float(ydiff.max()))
    print("mean y diff:", float(ydiff.mean()))

    if ydiff.max() > 1e-4:
        bad = np.where(ydiff > 1e-4)[0][:10]
        print("bad examples:")
        print(m.iloc[bad][["Source_Index", "SMILES_origin", "SMILES_taut", "y_origin", "y_taut"]])
        raise RuntimeError(f"{split}: y mismatch after Source_Index merge.")

    out = pd.DataFrame({
        "Source_Index": m["Source_Index"].values,
        "SMILES_origin": m["SMILES_origin"].values,
        "SMILES_taut": m["SMILES_taut"].values,
        "y": m["y_origin"].values,
        "origin_pred": m["origin_pred"].values,
        "taut_pred": m["taut_pred"].values,
    })

    out["view_diff"] = np.abs(out["origin_pred"] - out["taut_pred"])
    out["origin_abs_err"] = np.abs(out["y"] - out["origin_pred"])
    out["taut_abs_err"] = np.abs(out["y"] - out["taut_pred"])
    out["taut_better"] = (out["taut_abs_err"] < out["origin_abs_err"]).astype(int)
    return out


def score_metric(m, base_m):
    return (
        m["MAE"]
        + 0.03 * max(0.0, m["P95"] - base_m["P95"])
        + 0.05 * max(0.0, m["P99"] - base_m["P99"])
        + 0.08 * max(0, m[">100"] - base_m[">100"])
        + 0.15 * max(0, m[">200"] - base_m[">200"])
    )


def apply_fusion(df, mode, alpha, tau):
    po = df["origin_pred"].values
    pt = df["taut_pred"].values
    diff = df["view_diff"].values

    cand = alpha * po + (1.0 - alpha) * pt

    if mode == "global":
        final = cand
        used = np.ones(len(df), dtype=bool)
    elif mode == "diff_gated":
        used = diff >= tau
        final = po.copy()
        final[used] = cand[used]
    else:
        raise ValueError(mode)

    return final, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin_val", required=True)
    ap.add_argument("--origin_test", required=True)
    ap.add_argument("--taut_val", required=True)
    ap.add_argument("--taut_test", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    val = merge_idx(args.origin_val, args.taut_val, "VAL")
    test = merge_idx(args.origin_test, args.taut_test, "TEST")

    print("\n=== Single-view ===")
    print("VAL origin :", metrics(val["y"].values, val["origin_pred"].values))
    print("VAL taut   :", metrics(val["y"].values, val["taut_pred"].values))
    print("TEST origin:", metrics(test["y"].values, test["origin_pred"].values))
    print("TEST taut  :", metrics(test["y"].values, test["taut_pred"].values))

    print("\n=== Complementarity ===")
    print("VAL taut better :", int(val["taut_better"].sum()), "/", len(val))
    print("TEST taut better:", int(test["taut_better"].sum()), "/", len(test))
    print("VAL view_diff mean/P95 :", float(val["view_diff"].mean()), float(val["view_diff"].quantile(0.95)))
    print("TEST view_diff mean/P95:", float(test["view_diff"].mean()), float(test["view_diff"].quantile(0.95)))

    base_val_m = metrics(val["y"].values, val["origin_pred"].values)

    rows = []
    alphas_global = np.linspace(0, 1, 1001)
    alphas_gated = np.linspace(0, 1, 501)
    taus = [0, 2, 5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200]

    for alpha in alphas_global:
        pv, uv = apply_fusion(val, "global", alpha, -1)
        pt, ut = apply_fusion(test, "global", alpha, -1)
        mv = metrics(val["y"].values, pv)
        mt = metrics(test["y"].values, pt)
        rows.append({
            "mode": "global",
            "alpha_origin": float(alpha),
            "tau": -1.0,
            "used_val": int(uv.sum()),
            "used_test": int(ut.sum()),
            "score": score_metric(mv, base_val_m),
            **{f"val_{k}": v for k, v in mv.items()},
            **{f"test_{k}": v for k, v in mt.items()},
        })

    for tau in taus:
        for alpha in alphas_gated:
            pv, uv = apply_fusion(val, "diff_gated", alpha, tau)
            pt, ut = apply_fusion(test, "diff_gated", alpha, tau)
            mv = metrics(val["y"].values, pv)
            mt = metrics(test["y"].values, pt)
            rows.append({
                "mode": "diff_gated",
                "alpha_origin": float(alpha),
                "tau": float(tau),
                "used_val": int(uv.sum()),
                "used_test": int(ut.sum()),
                "score": score_metric(mv, base_val_m),
                **{f"val_{k}": v for k, v in mv.items()},
                **{f"test_{k}": v for k, v in mt.items()},
            })

    scan = pd.DataFrame(rows)
    scan = scan.sort_values(["score", "val_MAE", "val_P95", "val_>100"]).reset_index(drop=True)
    scan.to_csv(os.path.join(args.out_dir, "source_idx_fusion_scan.csv"), index=False)

    print("\n=== Top val-selected candidates ===")
    print(scan.head(40).to_string(index=False))

    best = scan.iloc[0]
    mode = str(best["mode"])
    alpha = float(best["alpha_origin"])
    tau = float(best["tau"])

    print("\nSELECTED:", {"mode": mode, "alpha_origin": alpha, "tau": tau})

    final_val, used_val = apply_fusion(val, mode, alpha, tau)
    final_test, used_test = apply_fusion(test, mode, alpha, tau)

    print("\nVAL fused :", metrics(val["y"].values, final_val))
    print("TEST fused:", metrics(test["y"].values, final_test))
    print("used val/test:", int(used_val.sum()), int(used_test.sum()))

    val_out = val.copy()
    val_out["final_pred"] = final_val
    val_out["final_abs_err"] = np.abs(val_out["y"] - val_out["final_pred"])
    val_out["gain_vs_origin"] = val_out["origin_abs_err"] - val_out["final_abs_err"]
    val_out["used_fusion"] = used_val
    val_out["alpha_origin"] = alpha
    val_out["fusion_mode"] = mode
    val_out["tau"] = tau
    val_out.to_csv(os.path.join(args.out_dir, "val_origin_taut_fused.csv"), index=False)

    test_out = test.copy()
    test_out["final_pred"] = final_test
    test_out["final_abs_err"] = np.abs(test_out["y"] - test_out["final_pred"])
    test_out["gain_vs_origin"] = test_out["origin_abs_err"] - test_out["final_abs_err"]
    test_out["used_fusion"] = used_test
    test_out["alpha_origin"] = alpha
    test_out["fusion_mode"] = mode
    test_out["tau"] = tau
    test_out.to_csv(os.path.join(args.out_dir, "test_origin_taut_fused.csv"), index=False)

    print("\n=== TEST gain summary ===")
    print("improved:", int((test_out["gain_vs_origin"] > 0).sum()))
    print("worsened :", int((test_out["gain_vs_origin"] < 0).sum()))
    print("net_gain:", float(test_out["gain_vs_origin"].sum()))
    print("mean_gain:", float(test_out["gain_vs_origin"].mean()))

    print("\n=== Top 30 final worst ===")
    print(test_out.sort_values("final_abs_err", ascending=False).head(30)[[
        "Source_Index",
        "SMILES_origin",
        "SMILES_taut",
        "y",
        "origin_pred",
        "taut_pred",
        "final_pred",
        "origin_abs_err",
        "taut_abs_err",
        "final_abs_err",
        "view_diff",
    ]].to_string(index=False))

    print("\nsaved:", args.out_dir)


if __name__ == "__main__":
    main()

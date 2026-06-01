import argparse
import os
import numpy as np
import pandas as pd


def load_pred(path, name):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    smi = cols.get("smiles", "SMILES")
    y = cols.get("actual_rt", None) or cols.get("y_true", None) or cols.get("y", None)
    p = cols.get("predicted_rt", None) or cols.get("y_pred", None) or cols.get("pred", None)

    out = df[[smi, y, p]].copy()
    out.columns = [f"SMILES_{name}", "y", f"{name}_pred"]
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


def merge_by_row(origin_path, taut_path, split):
    o = load_pred(origin_path, "origin")
    t = load_pred(taut_path, "taut")

    if len(o) != len(t):
        raise RuntimeError(f"{split}: row count mismatch {len(o)} vs {len(t)}")

    ydiff = np.abs(o["y"].values - t["y"].values)
    print(f"\n[{split}] rows:", len(o))
    print("max y diff:", float(ydiff.max()))
    print("mean y diff:", float(ydiff.mean()))

    if ydiff.max() > 1e-4:
        bad = np.where(ydiff > 1e-4)[0][:10]
        print("bad rows:", bad)
        print(o.iloc[bad][["SMILES_origin", "y"]])
        print(t.iloc[bad][["SMILES_taut", "y"]])
        raise RuntimeError(f"{split}: y mismatch. 不能按行融合。")

    df = pd.DataFrame({
        "row_id": np.arange(len(o)),
        "SMILES_origin": o["SMILES_origin"].values,
        "SMILES_taut": t["SMILES_taut"].values,
        "y": o["y"].values,
        "origin_pred": o["origin_pred"].values,
        "taut_pred": t["taut_pred"].values,
    })
    df["view_diff"] = np.abs(df["origin_pred"] - df["taut_pred"])
    df["origin_abs_err"] = np.abs(df["y"] - df["origin_pred"])
    df["taut_abs_err"] = np.abs(df["y"] - df["taut_pred"])
    return df


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

    val = merge_by_row(args.origin_val, args.taut_val, "VAL")
    test = merge_by_row(args.origin_test, args.taut_test, "TEST")

    print("\n=== Single view ===")
    print("VAL origin :", metrics(val["y"].values, val["origin_pred"].values))
    print("VAL taut   :", metrics(val["y"].values, val["taut_pred"].values))
    print("TEST origin:", metrics(test["y"].values, test["origin_pred"].values))
    print("TEST taut  :", metrics(test["y"].values, test["taut_pred"].values))

    base_val_m = metrics(val["y"].values, val["origin_pred"].values)

    rows = []
    for alpha in np.linspace(0, 1, 1001):
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

    for tau in [0, 2, 5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200]:
        for alpha in np.linspace(0, 1, 501):
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

    scan = pd.DataFrame(rows).sort_values(["score", "val_MAE", "val_P95", "val_>100"]).reset_index(drop=True)
    scan.to_csv(os.path.join(args.out_dir, "fusion_scan.csv"), index=False)

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

    print("\nsaved:", args.out_dir)


if __name__ == "__main__":
    main()

import argparse
import os
import numpy as np
import pandas as pd


def load_pred(path, pred_name):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    smi = cols.get("smiles", "SMILES")
    y = cols.get("actual_rt", None) or cols.get("y_true", None) or cols.get("y", None)
    p = cols.get("predicted_rt", None) or cols.get("y_pred", None) or cols.get("pred", None)

    if y is None or p is None:
        raise ValueError(f"{path} bad columns: {df.columns.tolist()}")

    out = df[[smi, y, p]].copy()
    out.columns = ["SMILES", "y", pred_name]
    out["SMILES"] = out["SMILES"].astype(str)
    out["y"] = out["y"].astype(float)
    out[pred_name] = out[pred_name].astype(float)
    out["row_id"] = np.arange(len(out))
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


def merge_by_order(origin_path, taut_path, split):
    o = load_pred(origin_path, "origin_pred")
    t = load_pred(taut_path, "taut_pred")

    if len(o) != len(t):
        raise RuntimeError(
            f"{split}: row length mismatch origin={len(o)} taut={len(t)}. "
            f"检查 SPLIT_SEED、过滤逻辑、CSV 是否一致。"
        )

    ydiff = np.abs(o["y"].values - t["y"].values)

    print(f"\n[{split}] rows:", len(o))
    print("max y diff by row:", float(ydiff.max()))
    print("mean y diff by row:", float(ydiff.mean()))

    if ydiff.max() > 1e-4:
        bad = np.where(ydiff > 1e-4)[0][:20]
        print("bad row examples:", bad)
        for i in bad[:5]:
            print(
                "row", int(i),
                "origin:", o.iloc[i]["SMILES"], o.iloc[i]["y"],
                "taut:", t.iloc[i]["SMILES"], t.iloc[i]["y"],
            )
        raise RuntimeError(f"{split}: y mismatch by row. 不能融合，说明行顺序没对齐。")

    out = pd.DataFrame({
        "row_id": np.arange(len(o)),
        "SMILES_origin": o["SMILES"].values,
        "SMILES_taut": t["SMILES"].values,
        "y": o["y"].values,
        "origin_pred": o["origin_pred"].values,
        "taut_pred": t["taut_pred"].values,
    })

    out["view_diff"] = np.abs(out["origin_pred"] - out["taut_pred"])
    out["origin_abs_err"] = np.abs(out["y"] - out["origin_pred"])
    out["taut_abs_err"] = np.abs(out["y"] - out["taut_pred"])
    out["taut_better"] = (out["taut_abs_err"] < out["origin_abs_err"]).astype(int)
    out["origin_better"] = (out["origin_abs_err"] < out["taut_abs_err"]).astype(int)

    return out


def score_metric(m, base_m):
    # val 上选融合权重：主看 MAE，同时避免 P95/P99/>100/>200 变差
    return (
        m["MAE"]
        + 0.03 * max(0.0, m["P95"] - base_m["P95"])
        + 0.05 * max(0.0, m["P99"] - base_m["P99"])
        + 0.08 * max(0, m[">100"] - base_m[">100"])
        + 0.15 * max(0, m[">200"] - base_m[">200"])
    )


def scan_global_alpha(val, test):
    yv = val["y"].values
    po_v = val["origin_pred"].values
    pt_v = val["taut_pred"].values

    yt = test["y"].values
    po_t = test["origin_pred"].values
    pt_t = test["taut_pred"].values

    base_val_m = metrics(yv, po_v)

    rows = []

    for alpha in np.linspace(0, 1, 1001):
        # alpha 越大越偏 origin；alpha=0 是纯 taut；alpha=1 是纯 origin
        pred_v = alpha * po_v + (1.0 - alpha) * pt_v
        pred_t = alpha * po_t + (1.0 - alpha) * pt_t

        mv = metrics(yv, pred_v)
        mt = metrics(yt, pred_t)

        rows.append({
            "mode": "global_alpha",
            "alpha_origin": float(alpha),
            "tau": -1.0,
            "score": score_metric(mv, base_val_m),
            **{f"val_{k}": v for k, v in mv.items()},
            **{f"test_{k}": v for k, v in mt.items()},
        })

    return rows


def scan_diff_gated_alpha(val, test):
    yv = val["y"].values
    po_v = val["origin_pred"].values
    pt_v = val["taut_pred"].values
    diff_v = val["view_diff"].values

    yt = test["y"].values
    po_t = test["origin_pred"].values
    pt_t = test["taut_pred"].values
    diff_t = test["view_diff"].values

    base_val_m = metrics(yv, po_v)

    rows = []
    alphas = np.linspace(0, 1, 501)
    taus = [0, 2, 5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200]

    for tau in taus:
        use_v = diff_v >= tau
        use_t = diff_t >= tau

        for alpha in alphas:
            cand_v = alpha * po_v + (1.0 - alpha) * pt_v
            cand_t = alpha * po_t + (1.0 - alpha) * pt_t

            pred_v = po_v.copy()
            pred_t = po_t.copy()

            pred_v[use_v] = cand_v[use_v]
            pred_t[use_t] = cand_t[use_t]

            mv = metrics(yv, pred_v)
            mt = metrics(yt, pred_t)

            rows.append({
                "mode": "viewdiff_gated_alpha",
                "alpha_origin": float(alpha),
                "tau": float(tau),
                "used_val": int(use_v.sum()),
                "used_test": int(use_t.sum()),
                "score": score_metric(mv, base_val_m),
                **{f"val_{k}": v for k, v in mv.items()},
                **{f"test_{k}": v for k, v in mt.items()},
            })

    return rows


def apply_selected(test, mode, alpha, tau):
    po = test["origin_pred"].values
    pt = test["taut_pred"].values
    diff = test["view_diff"].values

    cand = alpha * po + (1.0 - alpha) * pt

    if mode == "global_alpha":
        final = cand
        used = np.ones(len(test), dtype=bool)
    elif mode == "viewdiff_gated_alpha":
        used = diff >= tau
        final = po.copy()
        final[used] = cand[used]
    else:
        raise ValueError(mode)

    return final, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin_train", required=True)
    ap.add_argument("--origin_val", required=True)
    ap.add_argument("--origin_test", required=True)
    ap.add_argument("--taut_train", required=True)
    ap.add_argument("--taut_val", required=True)
    ap.add_argument("--taut_test", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train = merge_by_order(args.origin_train, args.taut_train, "TRAIN")
    val = merge_by_order(args.origin_val, args.taut_val, "VAL")
    test = merge_by_order(args.origin_test, args.taut_test, "TEST")

    print("\n=== Single-view metrics ===")
    print("VAL origin:", metrics(val["y"].values, val["origin_pred"].values))
    print("VAL taut  :", metrics(val["y"].values, val["taut_pred"].values))
    print("\nTEST origin:", metrics(test["y"].values, test["origin_pred"].values))
    print("TEST taut  :", metrics(test["y"].values, test["taut_pred"].values))

    print("\n=== Complementarity ===")
    print("VAL taut better:", int(val["taut_better"].sum()), "/", len(val))
    print("TEST taut better:", int(test["taut_better"].sum()), "/", len(test))
    print("VAL mean view_diff:", float(val["view_diff"].mean()), "P95:", float(val["view_diff"].quantile(0.95)))
    print("TEST mean view_diff:", float(test["view_diff"].mean()), "P95:", float(test["view_diff"].quantile(0.95)))

    rows = []
    rows.extend(scan_global_alpha(val, test))
    rows.extend(scan_diff_gated_alpha(val, test))

    scan = pd.DataFrame(rows)
    scan = scan.sort_values(["score", "val_MAE", "val_P95", "val_>100"]).reset_index(drop=True)
    scan.to_csv(os.path.join(args.out_dir, "origin_taut_fusion_scan.csv"), index=False)

    print("\n=== Top val-selected fusion candidates ===")
    print(scan.head(40).to_string(index=False))

    best = scan.iloc[0]
    mode = str(best["mode"])
    alpha = float(best["alpha_origin"])
    tau = float(best["tau"])

    print("\nSELECTED:", {"mode": mode, "alpha_origin": alpha, "tau": tau})

    final_test, used_test = apply_selected(test, mode, alpha, tau)
    final_val, used_val = apply_selected(val, mode, alpha, tau)

    print("\nVAL fused :", metrics(val["y"].values, final_val))
    print("TEST fused:", metrics(test["y"].values, final_test))
    print("used val/test:", int(used_val.sum()), int(used_test.sum()))

    train.to_csv(os.path.join(args.out_dir, "train_origin_taut_by_order.csv"), index=False)

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

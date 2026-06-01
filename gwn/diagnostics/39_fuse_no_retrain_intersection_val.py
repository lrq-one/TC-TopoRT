import argparse
import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem


def canonical_smiles(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return str(s)
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return str(s)


def y_key(y):
    return f"{float(y):.4f}"


def make_key(smiles, y):
    return canonical_smiles(smiles) + "||" + y_key(y)


def add_dup_id(df, key_col="key"):
    df = df.copy()
    df["dup_id"] = df.groupby(key_col).cumcount()
    return df


def load_pred(path, name):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    smi = cols.get("smiles", "SMILES")
    y = cols.get("actual_rt", None) or cols.get("y_true", None) or cols.get("y", None)
    p = cols.get("predicted_rt", None) or cols.get("y_pred", None) or cols.get("pred", None)

    if y is None or p is None:
        raise ValueError(f"{path} bad columns: {df.columns.tolist()}")

    out = df[[smi, y, p]].copy()
    out.columns = [f"SMILES_{name}", "y", f"{name}_pred"]
    out["y"] = out["y"].astype(float)
    out[f"{name}_pred"] = out[f"{name}_pred"].astype(float)
    return out


def load_valid_taut_train_rows(taut_train_csv):
    df = pd.read_csv(taut_train_csv, engine="python")
    df.columns = [str(c).lower().strip() for c in df.columns]

    if "smile" in df.columns and "smiles" not in df.columns:
        df.rename(columns={"smile": "smiles"}, inplace=True)

    if "smiles" not in df.columns or "rt" not in df.columns:
        raise ValueError(f"{taut_train_csv} columns wrong: {df.columns.tolist()}")

    if "orig_smile" not in df.columns:
        raise ValueError(
            f"{taut_train_csv} 没有 orig_smile 列。"
            f"你这个 taut CSV 必须是 strict 脚本生成的版本。"
        )

    df = df[df["rt"] > 300.0].copy()

    rows = []
    for _, row in df.iterrows():
        taut_smi = str(row["smiles"])
        orig_smi = str(row["orig_smile"])
        rt = float(row["rt"])

        mol = Chem.MolFromSmiles(taut_smi)
        if mol is None:
            continue

        rows.append({
            "orig_smile": orig_smi,
            "taut_smile": taut_smi,
            "y": rt,
        })

    return pd.DataFrame(rows)


def reconstruct_taut_val_meta(taut_train_csv, taut_val_pred_path, split_seed):
    full = load_valid_taut_train_rows(taut_train_csv)
    taut_val_pred = load_pred(taut_val_pred_path, "taut")

    n = len(full)
    train_len = int(0.9 * n)

    gen = torch.Generator().manual_seed(int(split_seed))
    perm = torch.randperm(n, generator=gen).tolist()
    val_idx = perm[train_len:]

    if len(val_idx) != len(taut_val_pred):
        raise RuntimeError(
            f"taut val length mismatch: reconstructed={len(val_idx)}, "
            f"pred_file={len(taut_val_pred)}. split_seed 或 CSV 不一致。"
        )

    meta = full.iloc[val_idx].reset_index(drop=True)
    out = taut_val_pred.copy()
    out["orig_smile_from_taut_csv"] = meta["orig_smile"].values
    out["taut_smile_from_taut_csv"] = meta["taut_smile"].values

    ydiff = np.abs(out["y"].values - meta["y"].values)
    print("\n[taut val reconstruction]")
    print("rows:", len(out))
    print("max y diff:", float(ydiff.max()))
    print("mean y diff:", float(ydiff.mean()))

    if ydiff.max() > 1e-4:
        bad = np.where(ydiff > 1e-4)[0][:10]
        print(out.iloc[bad][["SMILES_taut", "y", "orig_smile_from_taut_csv"]])
        print(meta.iloc[bad])
        raise RuntimeError("taut val reconstruction failed: y mismatch")

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


def score_metric(m, base_m):
    return (
        m["MAE"]
        + 0.03 * max(0.0, m["P95"] - base_m["P95"])
        + 0.05 * max(0.0, m["P99"] - base_m["P99"])
        + 0.08 * max(0, m[">100"] - base_m[">100"])
        + 0.15 * max(0, m[">200"] - base_m[">200"])
    )


def build_intersection_val(origin_val_path, taut_val_path, taut_train_csv, split_seed):
    origin_val = load_pred(origin_val_path, "origin")
    taut_val = reconstruct_taut_val_meta(taut_train_csv, taut_val_path, split_seed)

    origin_val["key"] = [
        make_key(s, y) for s, y in zip(origin_val["SMILES_origin"], origin_val["y"])
    ]

    taut_val["key"] = [
        make_key(s, y) for s, y in zip(taut_val["orig_smile_from_taut_csv"], taut_val["y"])
    ]

    origin_val = add_dup_id(origin_val, "key")
    taut_val = add_dup_id(taut_val, "key")

    m = origin_val.merge(
        taut_val[["key", "dup_id", "SMILES_taut", "orig_smile_from_taut_csv", "taut_pred", "y"]],
        on=["key", "dup_id"],
        how="inner",
        suffixes=("_origin", "_taut"),
    )

    print("\n[intersection val]")
    print("origin val rows:", len(origin_val))
    print("taut val rows  :", len(taut_val))
    print("intersection   :", len(m))

    if len(m) < 100:
        print("WARNING: 交集太少，val 选融合权重会不稳定。")

    ydiff = np.abs(m["y_origin"].values - m["y_taut"].values)
    print("max y diff:", float(ydiff.max()) if len(m) else None)
    print("mean y diff:", float(ydiff.mean()) if len(m) else None)

    out = pd.DataFrame({
        "SMILES_origin": m["SMILES_origin"].values,
        "SMILES_taut": m["SMILES_taut"].values,
        "y": m["y_origin"].values,
        "origin_pred": m["origin_pred"].values,
        "taut_pred": m["taut_pred"].values,
    })

    out["view_diff"] = np.abs(out["origin_pred"] - out["taut_pred"])
    out["origin_abs_err"] = np.abs(out["y"] - out["origin_pred"])
    out["taut_abs_err"] = np.abs(out["y"] - out["taut_pred"])
    return out


def build_test(origin_test_path, taut_test_path):
    o = load_pred(origin_test_path, "origin")
    t = load_pred(taut_test_path, "taut")

    if len(o) != len(t):
        raise RuntimeError(f"test length mismatch: origin={len(o)}, taut={len(t)}")

    ydiff = np.abs(o["y"].values - t["y"].values)
    print("\n[test row alignment]")
    print("rows:", len(o))
    print("max y diff:", float(ydiff.max()))
    print("mean y diff:", float(ydiff.mean()))

    if ydiff.max() > 1e-4:
        raise RuntimeError("test y mismatch; test cannot be fused row-wise")

    out = pd.DataFrame({
        "row_id": np.arange(len(o)),
        "SMILES_origin": o["SMILES_origin"].values,
        "SMILES_taut": t["SMILES_taut"].values,
        "y": o["y"].values,
        "origin_pred": o["origin_pred"].values,
        "taut_pred": t["taut_pred"].values,
    })

    out["view_diff"] = np.abs(out["origin_pred"] - out["taut_pred"])
    out["origin_abs_err"] = np.abs(out["y"] - out["origin_pred"])
    out["taut_abs_err"] = np.abs(out["y"] - out["taut_pred"])
    return out


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
    ap.add_argument("--taut_train_csv", required=True)
    ap.add_argument("--taut_split_seed", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    val = build_intersection_val(
        args.origin_val,
        args.taut_val,
        args.taut_train_csv,
        args.taut_split_seed,
    )

    test = build_test(args.origin_test, args.taut_test)

    print("\n=== Single-view metrics ===")
    print("VAL-intersection origin:", metrics(val["y"].values, val["origin_pred"].values))
    print("VAL-intersection taut  :", metrics(val["y"].values, val["taut_pred"].values))
    print("TEST origin:", metrics(test["y"].values, test["origin_pred"].values))
    print("TEST taut  :", metrics(test["y"].values, test["taut_pred"].values))

    print("\n=== Complementarity ===")
    print("VAL taut better :", int((val["taut_abs_err"] < val["origin_abs_err"]).sum()), "/", len(val))
    print("TEST taut better:", int((test["taut_abs_err"] < test["origin_abs_err"]).sum()), "/", len(test))
    print("VAL view_diff mean/P95 :", float(val["view_diff"].mean()), float(val["view_diff"].quantile(0.95)))
    print("TEST view_diff mean/P95:", float(test["view_diff"].mean()), float(test["view_diff"].quantile(0.95)))

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

    scan = pd.DataFrame(rows)
    scan = scan.sort_values(["score", "val_MAE", "val_P95", "val_>100"]).reset_index(drop=True)
    scan.to_csv(os.path.join(args.out_dir, "fusion_scan_intersection_val.csv"), index=False)

    print("\n=== Top val-selected candidates ===")
    print(scan.head(40).to_string(index=False))

    best = scan.iloc[0]
    mode = str(best["mode"])
    alpha = float(best["alpha_origin"])
    tau = float(best["tau"])

    print("\nSELECTED:", {"mode": mode, "alpha_origin": alpha, "tau": tau})

    final_val, used_val = apply_fusion(val, mode, alpha, tau)
    final_test, used_test = apply_fusion(test, mode, alpha, tau)

    print("\nVAL-intersection fused:", metrics(val["y"].values, final_val))
    print("TEST fused            :", metrics(test["y"].values, final_test))
    print("used val/test:", int(used_val.sum()), int(used_test.sum()))

    val_out = val.copy()
    val_out["final_pred"] = final_val
    val_out["final_abs_err"] = np.abs(val_out["y"] - val_out["final_pred"])
    val_out["gain_vs_origin"] = val_out["origin_abs_err"] - val_out["final_abs_err"]
    val_out["used_fusion"] = used_val
    val_out["alpha_origin"] = alpha
    val_out["fusion_mode"] = mode
    val_out["tau"] = tau
    val_out.to_csv(os.path.join(args.out_dir, "val_intersection_origin_taut_fused.csv"), index=False)

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

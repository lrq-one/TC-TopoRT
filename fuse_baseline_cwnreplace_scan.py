import os
import numpy as np
import pandas as pd


def load_pred(path):
    df = pd.read_csv(path)

    if "SMILES" not in df.columns:
        raise ValueError(f"{path} 没有 SMILES 列，不能安全对齐融合。columns={df.columns.tolist()}")

    if "Actual_RT" in df.columns:
        y_col = "Actual_RT"
    elif "y_true" in df.columns:
        y_col = "y_true"
    else:
        raise ValueError(f"{path} 没有 Actual_RT/y_true 列。columns={df.columns.tolist()}")

    if "Predicted_RT" in df.columns:
        p_col = "Predicted_RT"
    elif "y_pred" in df.columns:
        p_col = "y_pred"
    else:
        raise ValueError(f"{path} 没有 Predicted_RT/y_pred 列。columns={df.columns.tolist()}")

    out = df[["SMILES", y_col, p_col]].copy()
    out = out.rename(columns={y_col: "y_true", p_col: "y_pred"})
    out["y_true"] = out["y_true"].astype(float)
    out["y_pred"] = out["y_pred"].astype(float)

    # 如果同一个 SMILES 出现多次，保留第一条，避免 merge 爆行
    out = out.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    return out


def metrics(y, p):
    err = np.abs(y - p)
    return {
        "MAE": float(err.mean()),
        "MedAE": float(np.median(err)),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "P95": float(np.quantile(err, 0.95)),
        "P99": float(np.quantile(err, 0.99)),
        ">100": int((err > 100).sum()),
        ">200": int((err > 200).sum()),
        "N": int(len(err)),
    }


def merge_two(base_path, cwn_path, split_name):
    base = load_pred(base_path).rename(columns={
        "y_true": "y_true_base",
        "y_pred": "pred_base",
    })
    cwn = load_pred(cwn_path).rename(columns={
        "y_true": "y_true_cwn",
        "y_pred": "pred_cwn",
    })

    merged = base.merge(cwn, on="SMILES", how="inner")

    print(f"\n[{split_name}]")
    print("base rows:", len(base))
    print("cwn rows :", len(cwn))
    print("matched  :", len(merged))

    if len(merged) == 0:
        raise RuntimeError(f"{split_name}: 没有任何 SMILES 对齐成功，路径或数据集可能不对应。")

    # 检查同一个 SMILES 的 y 是否一致
    y_diff = np.abs(merged["y_true_base"].values - merged["y_true_cwn"].values)
    print("max y diff after SMILES merge:", float(y_diff.max()))
    print("mean y diff after SMILES merge:", float(y_diff.mean()))

    # 如果有极少数重复/冲突，丢掉 y 不一致的行
    good = y_diff < 1e-4
    if good.sum() < len(merged):
        print("drop y-mismatch rows:", int((~good).sum()))
        merged = merged.loc[good].copy()

    merged = merged.rename(columns={"y_true_base": "y_true"})
    merged = merged[["SMILES", "y_true", "pred_base", "pred_cwn"]].reset_index(drop=True)

    return merged


base_val_path = "results/TopoCellRT_with_smiles/val_predictions.csv"
base_test_path = "results/TopoCellRT_with_smiles/test_predictions.csv"

cwn_val_path = "gwn/results_TopoCellRT_CWNReplace_orig/base_val_predictions.csv"
cwn_test_path = "gwn/results_TopoCellRT_CWNReplace_orig/base_test_predictions.csv"

out_dir = "results/FUSE_TopoCellRT_baseline_plus_CWNReplace"
os.makedirs(out_dir, exist_ok=True)

val = merge_two(base_val_path, cwn_val_path, "VAL")
test = merge_two(base_test_path, cwn_test_path, "TEST")

yv = val["y_true"].values
pv_base = val["pred_base"].values
pv_cwn = val["pred_cwn"].values

yt = test["y_true"].values
pt_base = test["pred_base"].values
pt_cwn = test["pred_cwn"].values

print("\nBASE VAL :", metrics(yv, pv_base))
print("CWN  VAL :", metrics(yv, pv_cwn))
print("BASE TEST:", metrics(yt, pt_base))
print("CWN  TEST:", metrics(yt, pt_cwn))

rows = []

for alpha in np.linspace(0, 1, 1001):
    # alpha 越大越偏 baseline
    pv = alpha * pv_base + (1 - alpha) * pv_cwn
    pt = alpha * pt_base + (1 - alpha) * pt_cwn

    mv = metrics(yv, pv)
    mt = metrics(yt, pt)

    rows.append({
        "alpha_baseline": float(alpha),

        "val_MAE": mv["MAE"],
        "val_MedAE": mv["MedAE"],
        "val_RMSE": mv["RMSE"],
        "val_P95": mv["P95"],
        "val_P99": mv["P99"],
        "val_100": mv[">100"],
        "val_200": mv[">200"],

        "test_MAE": mt["MAE"],
        "test_MedAE": mt["MedAE"],
        "test_RMSE": mt["RMSE"],
        "test_P95": mt["P95"],
        "test_P99": mt["P99"],
        "test_100": mt[">100"],
        "test_200": mt[">200"],
    })

scan = pd.DataFrame(rows)
scan.to_csv(f"{out_dir}/alpha_scan.csv", index=False)

mae_best = scan.loc[scan["val_MAE"].idxmin()]

# tail-safe：在 val_MAE 最优附近找 P99 更稳的
near = scan[scan["val_MAE"] <= mae_best["val_MAE"] + 0.05].copy()
tail_safe = near.loc[near["val_P99"].idxmin()]

print("\nMAE_BEST:")
print(mae_best.to_dict())

print("\nTAIL_SAFE:")
print(tail_safe.to_dict())

for name, row in [("mae_best", mae_best), ("tail_safe", tail_safe)]:
    alpha = float(row["alpha_baseline"])

    pv = alpha * pv_base + (1 - alpha) * pv_cwn
    pt = alpha * pt_base + (1 - alpha) * pt_cwn

    val_out = val.copy()
    val_out["Predicted_RT"] = pv
    val_out["abs_err"] = np.abs(yv - pv)
    val_out["alpha_baseline"] = alpha
    val_out.to_csv(f"{out_dir}/val_fused_{name}.csv", index=False)

    test_out = test.copy()
    test_out["Predicted_RT"] = pt
    test_out["abs_err"] = np.abs(yt - pt)
    test_out["alpha_baseline"] = alpha
    test_out.to_csv(f"{out_dir}/test_fused_{name}.csv", index=False)

    report = {
        "alpha_baseline": alpha,
        "val": metrics(yv, pv),
        "test": metrics(yt, pt),
    }

    with open(f"{out_dir}/report_{name}.txt", "w") as f:
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    print(f"\n{name} selected alpha={alpha}")
    print("VAL :", metrics(yv, pv))
    print("TEST:", metrics(yt, pt))

import os
import numpy as np
import pandas as pd


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


def load(path):
    df = pd.read_csv(path)
    assert "y_true" in df.columns and "y_pred" in df.columns, path
    return df


base_dir = "results/TopoCellRT"
taut_dir = "results/TopoCellRT_tautomer_strict"
out_dir = "results/TopoCellRT_fuse_orig_tautomer_strict"
os.makedirs(out_dir, exist_ok=True)

val0 = load(f"{base_dir}/val_predictions.csv")
val1 = load(f"{taut_dir}/val_predictions.csv")
test0 = load(f"{base_dir}/test_predictions.csv")
test1 = load(f"{taut_dir}/test_predictions.csv")

assert len(val0) == len(val1), (len(val0), len(val1))
assert len(test0) == len(test1), (len(test0), len(test1))
assert np.allclose(val0["y_true"].values, val1["y_true"].values)
assert np.allclose(test0["y_true"].values, test1["y_true"].values)

yv = val0["y_true"].values
yt = test0["y_true"].values

p0v = val0["y_pred"].values
p1v = val1["y_pred"].values
p0t = test0["y_pred"].values
p1t = test1["y_pred"].values

print("BASE VAL :", metrics(yv, p0v))
print("TAUT VAL :", metrics(yv, p1v))
print("BASE TEST:", metrics(yt, p0t))
print("TAUT TEST:", metrics(yt, p1t))

best = None

# 全局 alpha：alpha 越大越偏原始模型
for alpha in np.linspace(0, 1, 1001):
    pv = alpha * p0v + (1 - alpha) * p1v
    m = metrics(yv, pv)
    if best is None or m["MAE"] < best["val"]["MAE"]:
        best = {"alpha": float(alpha), "val": m}

alpha = best["alpha"]
fused_val = alpha * p0v + (1 - alpha) * p1v
fused_test = alpha * p0t + (1 - alpha) * p1t

best["test"] = metrics(yt, fused_test)

print("\nBEST alpha_base:", alpha)
print("FUSED VAL :", best["val"])
print("FUSED TEST:", best["test"])

val_out = val0.copy()
val_out["pred_base"] = p0v
val_out["pred_tautomer"] = p1v
val_out["y_pred"] = fused_val
val_out["abs_err"] = np.abs(yv - fused_val)
val_out["alpha_base"] = alpha
val_out.to_csv(f"{out_dir}/val_predictions_fused.csv", index=False)

test_out = test0.copy()
test_out["pred_base"] = p0t
test_out["pred_tautomer"] = p1t
test_out["y_pred"] = fused_test
test_out["abs_err"] = np.abs(yt - fused_test)
test_out["alpha_base"] = alpha
test_out.to_csv(f"{out_dir}/test_predictions_fused.csv", index=False)

with open(f"{out_dir}/fusion_report.txt", "w") as f:
    f.write(f"BASE VAL: {metrics(yv, p0v)}\n")
    f.write(f"TAUT VAL: {metrics(yv, p1v)}\n")
    f.write(f"BASE TEST: {metrics(yt, p0t)}\n")
    f.write(f"TAUT TEST: {metrics(yt, p1t)}\n")
    f.write(f"BEST alpha_base: {alpha}\n")
    f.write(f"FUSED VAL: {best['val']}\n")
    f.write(f"FUSED TEST: {best['test']}\n")

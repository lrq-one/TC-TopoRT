import os
import numpy as np
import pandas as pd

ALPHA = 0.582

base_dir = "results/TopoCellRT"
taut_dir = "results/TopoCellRT_tautomer_strict"
out_dir = "results/FINAL_multiview_tail_safe_alpha0582"
os.makedirs(out_dir, exist_ok=True)


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


def fuse(split):
    base = pd.read_csv(f"{base_dir}/{split}_predictions.csv")
    taut = pd.read_csv(f"{taut_dir}/{split}_predictions.csv")

    assert len(base) == len(taut)
    assert np.allclose(base["y_true"].values, taut["y_true"].values)

    y = base["y_true"].values
    p_base = base["y_pred"].values
    p_taut = taut["y_pred"].values
    p = ALPHA * p_base + (1.0 - ALPHA) * p_taut

    out = base.copy()
    out["pred_base"] = p_base
    out["pred_tautomer"] = p_taut
    out["y_pred"] = p
    out["abs_err"] = np.abs(y - p)
    out["alpha_base"] = ALPHA
    out.to_csv(f"{out_dir}/{split}_predictions_fused_alpha0582.csv", index=False)

    return metrics(y, p)


val_m = fuse("val")
test_m = fuse("test")

print("alpha_base:", ALPHA)
print("VAL :", val_m)
print("TEST:", test_m)

with open(f"{out_dir}/final_report_alpha0582.txt", "w") as f:
    f.write(f"alpha_base: {ALPHA}\n")
    f.write(f"VAL: {val_m}\n")
    f.write(f"TEST: {test_m}\n")

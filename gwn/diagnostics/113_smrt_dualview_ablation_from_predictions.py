from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, HuberRegressor

RUN_DIRS = [
    "results_OOF_DualView_Stack_v1",
    "results_OOF_DualView_Stack_seed5",
    "results_OOF_DualView_Stack_seed79",
    "results_OOF_DualView_Stack_seed123",
    "results_OOF_DualView_Stack_seed256",
]

OUT = Path("final_smrt_results")
OUT.mkdir(exist_ok=True)

def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = np.abs(y - p)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    rel = e / (np.abs(y) + 1e-8) * 100.0
    return {
        "n": int(len(y)),
        "mae": float(np.mean(e)),
        "medae": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mre": float(np.mean(rel)),
        "medre": float(np.median(rel)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "gt80": int((e > 80).sum()),
        "gt100": int((e > 100).sum()),
        "gt200": int((e > 200).sum()),
        "bias": float(np.mean(p - y)),
    }

def build_stack_features(origin_pred, taut_pred, changed):
    origin_pred = np.asarray(origin_pred, dtype=np.float64)
    taut_pred = np.asarray(taut_pred, dtype=np.float64)
    changed = np.asarray(changed, dtype=np.float64)

    diff = np.abs(origin_pred - taut_pred)
    mean_pred = 0.5 * (origin_pred + taut_pred)
    min_pred = np.minimum(origin_pred, taut_pred)
    max_pred = np.maximum(origin_pred, taut_pred)

    return np.vstack([
        origin_pred,
        taut_pred,
        diff,
        mean_pred,
        min_pred,
        max_pred,
        changed,
        diff * changed,
        origin_pred * changed / 1000.0,
        taut_pred * changed / 1000.0,
    ]).T

def disagreement_fusion(origin_pred, taut_pred, alpha, tau, temperature=5.0):
    origin_pred = np.asarray(origin_pred, dtype=np.float64)
    taut_pred = np.asarray(taut_pred, dtype=np.float64)
    diff = np.abs(origin_pred - taut_pred)
    soft_use = 1.0 / (1.0 + np.exp(-((diff - tau) / temperature)))
    mixed = alpha * origin_pred + (1.0 - alpha) * taut_pred
    return (1.0 - soft_use) * origin_pred + soft_use * mixed

def fit_candidates(oof_df, test_df):
    y_oof = oof_df["Actual_RT"].values.astype(np.float64)
    y_test = test_df["Actual_RT"].values.astype(np.float64)

    o_oof = oof_df["Origin_OOF_Pred"].values.astype(np.float64)
    t_oof = oof_df["Taut_OOF_Pred"].values.astype(np.float64)
    c_oof = oof_df["Taut_Changed"].values.astype(np.float64)

    o_test = test_df["Origin_Test_Pred"].values.astype(np.float64)
    t_test = test_df["Taut_Test_Pred"].values.astype(np.float64)
    c_test = test_df["Taut_Changed"].values.astype(np.float64)

    candidates = {}

    candidates["Origin only"] = {
        "oof_pred": o_oof,
        "test_pred": o_test,
        "params": "",
    }

    candidates["Tautomer only"] = {
        "oof_pred": t_oof,
        "test_pred": t_test,
        "params": "",
    }

    candidates["Mean fusion"] = {
        "oof_pred": 0.5 * (o_oof + t_oof),
        "test_pred": 0.5 * (o_test + t_test),
        "params": "",
    }

    best = None
    alpha_grid = np.linspace(0.0, 1.0, 101)
    tau_grid = np.array([0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0])

    for tau in tau_grid:
        for alpha in alpha_grid:
            p = disagreement_fusion(o_oof, t_oof, alpha=alpha, tau=tau, temperature=5.0)
            m = metrics(y_oof, p)
            if best is None or m["mae"] < best["mae"]:
                best = {
                    "alpha": float(alpha),
                    "tau": float(tau),
                    "mae": float(m["mae"]),
                    "oof_pred": p,
                    "test_pred": disagreement_fusion(o_test, t_test, alpha=alpha, tau=tau, temperature=5.0),
                }

    candidates["OOF fixed gate"] = {
        "oof_pred": best["oof_pred"],
        "test_pred": best["test_pred"],
        "params": f"alpha={best['alpha']}; tau={best['tau']}",
    }

    x_oof = build_stack_features(o_oof, t_oof, c_oof)
    x_test = build_stack_features(o_test, t_test, c_test)

    ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.array([1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0])),
    )
    ridge.fit(x_oof, y_oof)
    candidates["Ridge stack"] = {
        "oof_pred": ridge.predict(x_oof),
        "test_pred": ridge.predict(x_test),
        "params": "StandardScaler+RidgeCV",
    }

    huber = make_pipeline(
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000),
    )
    huber.fit(x_oof, y_oof)
    candidates["Huber stack"] = {
        "oof_pred": huber.predict(x_oof),
        "test_pred": huber.predict(x_test),
        "params": "StandardScaler+HuberRegressor",
    }

    rows = []
    for name, item in candidates.items():
        test_m = metrics(y_test, item["test_pred"])
        oof_m = metrics(y_oof, item["oof_pred"])
        row = {
            "method": name,
            "params": item["params"],
            **{f"test_{k}": v for k, v in test_m.items()},
            **{f"oof_{k}": v for k, v in oof_m.items()},
        }
        rows.append(row)

    return rows

all_rows = []

for run_dir in RUN_DIRS:
    root = Path(run_dir)
    oof_path = root / "oof_base_predictions.csv"
    test_path = root / "test_base_predictions.csv"

    print("\n" + "=" * 100)
    print("[RUN]", run_dir)
    print("[OOF]", oof_path, "exists=", oof_path.exists())
    print("[TEST]", test_path, "exists=", test_path.exists())

    if not oof_path.exists() or not test_path.exists():
        continue

    oof_df = pd.read_csv(oof_path)
    test_df = pd.read_csv(test_path)

    rows = fit_candidates(oof_df, test_df)
    for r in rows:
        r["run_dir"] = run_dir
        all_rows.append(r)

df = pd.DataFrame(all_rows)
if len(df) == 0:
    raise SystemExit("[ERROR] no rows generated")

df = df.sort_values(["method", "run_dir"])
df.to_csv(OUT / "smrt_dualview_ablation_by_run.csv", index=False)

summary_rows = []
for method, sub in df.groupby("method"):
    row = {
        "method": method,
        "runs": len(sub),
    }
    for c in [
        "test_mae", "test_medae", "test_rmse", "test_mre", "test_medre",
        "test_r2", "test_p95", "test_p99", "test_gt80",
        "test_gt100", "test_gt200", "test_bias",
        "oof_mae",
    ]:
        vals = pd.to_numeric(sub[c], errors="coerce").dropna()
        row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
        row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows).sort_values("test_mae_mean")
summary.to_csv(OUT / "smrt_dualview_ablation_summary_mean_std.csv", index=False)

print("\n=== Dual-view ablation by run ===")
print(df[[
    "run_dir", "method", "params",
    "test_mae", "test_medae", "test_rmse", "test_r2",
    "test_p95", "test_p99", "test_gt100", "test_gt200", "test_bias",
    "oof_mae"
]].to_string(index=False))

print("\n=== Dual-view ablation summary ===")
print(summary[[
    "method", "runs",
    "test_mae_mean", "test_mae_std",
    "test_medae_mean", "test_medae_std",
    "test_rmse_mean", "test_rmse_std",
    "test_r2_mean", "test_r2_std",
    "test_p95_mean", "test_p95_std",
    "test_p99_mean", "test_p99_std",
    "test_gt100_mean", "test_gt100_std",
    "test_gt200_mean", "test_gt200_std",
    "test_bias_mean", "test_bias_std",
    "oof_mae_mean", "oof_mae_std",
]].to_string(index=False))

print("\n[SAVE]", OUT / "smrt_dualview_ablation_by_run.csv")
print("[SAVE]", OUT / "smrt_dualview_ablation_summary_mean_std.csv")

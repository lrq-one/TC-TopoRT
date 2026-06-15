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
        "p90": float(np.percentile(e, 90)),
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

def fit_predictions(oof_df, test_df):
    y_oof = oof_df["Actual_RT"].values.astype(np.float64)

    o_oof = oof_df["Origin_OOF_Pred"].values.astype(np.float64)
    t_oof = oof_df["Taut_OOF_Pred"].values.astype(np.float64)
    c_oof = oof_df["Taut_Changed"].values.astype(np.float64)

    o_test = test_df["Origin_Test_Pred"].values.astype(np.float64)
    t_test = test_df["Taut_Test_Pred"].values.astype(np.float64)
    c_test = test_df["Taut_Changed"].values.astype(np.float64)

    preds = {
        "Origin only": o_test,
        "Tautomer only": t_test,
        "Mean fusion": 0.5 * (o_test + t_test),
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
                }

    preds["OOF fixed gate"] = disagreement_fusion(
        o_test, t_test,
        alpha=best["alpha"],
        tau=best["tau"],
        temperature=5.0,
    )

    x_oof = build_stack_features(o_oof, t_oof, c_oof)
    x_test = build_stack_features(o_test, t_test, c_test)

    ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.array([1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0])),
    )
    ridge.fit(x_oof, y_oof)
    preds["Ridge stack"] = ridge.predict(x_test)

    huber = make_pipeline(
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000),
    )
    huber.fit(x_oof, y_oof)
    preds["Huber stack"] = huber.predict(x_test)

    return preds, {"fixed_gate_alpha": best["alpha"], "fixed_gate_tau": best["tau"]}

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

    preds, params = fit_predictions(oof_df, test_df)

    y = test_df["Actual_RT"].values.astype(np.float64)
    changed = test_df["Taut_Changed"].values.astype(float)

    groups = {
        "all": np.ones(len(test_df), dtype=bool),
        "taut_changed_0": changed < 0.5,
        "taut_changed_1": changed >= 0.5,
    }

    for method, p in preds.items():
        for group_name, mask in groups.items():
            if mask.sum() == 0:
                continue

            m = metrics(y[mask], np.asarray(p)[mask])
            row = {
                "run_dir": run_dir,
                "method": method,
                "group": group_name,
                "fixed_gate_alpha": params["fixed_gate_alpha"],
                "fixed_gate_tau": params["fixed_gate_tau"],
                **m,
            }
            all_rows.append(row)

df = pd.DataFrame(all_rows)
if len(df) == 0:
    raise SystemExit("[ERROR] no subgroup rows generated")

df = df.sort_values(["group", "method", "run_dir"])
df.to_csv(OUT / "smrt_taut_changed_subgroup_by_run.csv", index=False)

summary_rows = []
for (group, method), sub in df.groupby(["group", "method"]):
    row = {
        "group": group,
        "method": method,
        "runs": len(sub),
    }
    for c in ["n", "mae", "medae", "rmse", "mre", "medre", "r2", "p90", "p95", "p99", "gt80", "gt100", "gt200", "bias"]:
        vals = pd.to_numeric(sub[c], errors="coerce").dropna()
        row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
        row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows).sort_values(["group", "mae_mean"])
summary.to_csv(OUT / "smrt_taut_changed_subgroup_summary_mean_std.csv", index=False)

print("\n=== subgroup by run ===")
print(df[[
    "run_dir", "group", "method", "n", "mae", "medae", "rmse", "r2",
    "p95", "p99", "gt100", "gt200", "bias"
]].to_string(index=False))

print("\n=== subgroup summary ===")
print(summary[[
    "group", "method", "runs", "n_mean",
    "mae_mean", "mae_std",
    "medae_mean", "medae_std",
    "rmse_mean", "rmse_std",
    "r2_mean", "r2_std",
    "p95_mean", "p95_std",
    "p99_mean", "p99_std",
    "gt100_mean", "gt100_std",
    "gt200_mean", "gt200_std",
    "bias_mean", "bias_std",
]].to_string(index=False))

print("\n[SAVE]", OUT / "smrt_taut_changed_subgroup_by_run.csv")
print("[SAVE]", OUT / "smrt_taut_changed_subgroup_summary_mean_std.csv")

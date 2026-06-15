from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor

RUN_DIRS = [
    "results_OOF_DualView_Stack_v1",
    "results_OOF_DualView_Stack_seed5",
    "results_OOF_DualView_Stack_seed79",
    "results_OOF_DualView_Stack_seed123",
    "results_OOF_DualView_Stack_seed256",
]

OUT = Path("final_smrt_results")
OUT.mkdir(exist_ok=True)

N_PERM = 50
BASE_SEED = 20260614

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

def fit_huber_predict(y_oof, x_oof, x_test):
    model = make_pipeline(
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000),
    )
    model.fit(x_oof, y_oof)
    return model.predict(x_test), model.predict(x_oof)

rows = []

for run_i, run_dir in enumerate(RUN_DIRS):
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

    y_oof = oof_df["Actual_RT"].values.astype(np.float64)
    y_test = test_df["Actual_RT"].values.astype(np.float64)

    o_oof = oof_df["Origin_OOF_Pred"].values.astype(np.float64)
    t_oof = oof_df["Taut_OOF_Pred"].values.astype(np.float64)
    c_oof = oof_df["Taut_Changed"].values.astype(np.float64)

    o_test = test_df["Origin_Test_Pred"].values.astype(np.float64)
    t_test = test_df["Taut_Test_Pred"].values.astype(np.float64)
    c_test = test_df["Taut_Changed"].values.astype(np.float64)

    # Correct pairing.
    x_oof = build_stack_features(o_oof, t_oof, c_oof)
    x_test = build_stack_features(o_test, t_test, c_test)
    p_test, p_oof = fit_huber_predict(y_oof, x_oof, x_test)

    m = metrics(y_test, p_test)
    rows.append({
        "run_dir": run_dir,
        "condition": "paired",
        "perm_id": -1,
        "seed": "",
        **m,
    })

    # Shuffled tautomer pairing.
    for perm_id in range(N_PERM):
        seed = BASE_SEED + run_i * 1000 + perm_id
        rng = np.random.default_rng(seed)

        idx_oof = rng.permutation(len(oof_df))
        idx_test = rng.permutation(len(test_df))

        t_oof_shuf = t_oof[idx_oof]
        t_test_shuf = t_test[idx_test]

        # Keep Taut_Changed fixed to the original molecule.
        # Only the paired tautomer prediction is broken.
        x_oof_shuf = build_stack_features(o_oof, t_oof_shuf, c_oof)
        x_test_shuf = build_stack_features(o_test, t_test_shuf, c_test)

        p_test_shuf, _ = fit_huber_predict(y_oof, x_oof_shuf, x_test_shuf)
        m = metrics(y_test, p_test_shuf)

        rows.append({
            "run_dir": run_dir,
            "condition": "shuffled_taut_pred",
            "perm_id": perm_id,
            "seed": seed,
            **m,
        })

df = pd.DataFrame(rows)
if len(df) == 0:
    raise SystemExit("[ERROR] no rows generated")

df.to_csv(OUT / "smrt_shuffle_taut_pairing_by_run.csv", index=False)

summary_rows = []
for condition, sub in df.groupby("condition"):
    row = {
        "condition": condition,
        "rows": len(sub),
        "run_count": sub["run_dir"].nunique(),
    }
    for c in ["mae", "medae", "rmse", "mre", "medre", "r2", "p95", "p99", "gt80", "gt100", "gt200", "bias"]:
        vals = pd.to_numeric(sub[c], errors="coerce").dropna()
        row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
        row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows).sort_values("mae_mean")
summary.to_csv(OUT / "smrt_shuffle_taut_pairing_summary.csv", index=False)

# Per-run paired vs shuffle mean.
paired = df[df["condition"].eq("paired")].set_index("run_dir")
shuf = df[df["condition"].eq("shuffled_taut_pred")].groupby("run_dir").agg({
    "mae": ["mean", "std", "min", "max"],
    "medae": ["mean", "std"],
    "rmse": ["mean", "std"],
    "r2": ["mean", "std"],
    "gt100": ["mean", "std"],
    "gt200": ["mean", "std"],
}).reset_index()

shuf.columns = ["_".join([x for x in c if x]) for c in shuf.columns.to_flat_index()]
comp_rows = []

for _, r in shuf.iterrows():
    run_dir = r["run_dir"]
    pr = paired.loc[run_dir]
    comp_rows.append({
        "run_dir": run_dir,
        "paired_mae": float(pr["mae"]),
        "shuffle_mae_mean": float(r["mae_mean"]),
        "shuffle_mae_std": float(r["mae_std"]),
        "shuffle_mae_min": float(r["mae_min"]),
        "shuffle_mae_max": float(r["mae_max"]),
        "delta_shuffle_minus_paired": float(r["mae_mean"] - pr["mae"]),
        "paired_rmse": float(pr["rmse"]),
        "shuffle_rmse_mean": float(r["rmse_mean"]),
        "paired_r2": float(pr["r2"]),
        "shuffle_r2_mean": float(r["r2_mean"]),
        "paired_gt100": float(pr["gt100"]),
        "shuffle_gt100_mean": float(r["gt100_mean"]),
        "paired_gt200": float(pr["gt200"]),
        "shuffle_gt200_mean": float(r["gt200_mean"]),
    })

comp = pd.DataFrame(comp_rows)
comp.to_csv(OUT / "smrt_shuffle_taut_pairing_paired_vs_shuffle.csv", index=False)

print("\n=== paired vs shuffled per run ===")
print(comp.to_string(index=False))

print("\n=== overall summary ===")
print(summary[[
    "condition", "rows", "run_count",
    "mae_mean", "mae_std",
    "medae_mean", "medae_std",
    "rmse_mean", "rmse_std",
    "r2_mean", "r2_std",
    "gt100_mean", "gt100_std",
    "gt200_mean", "gt200_std",
    "bias_mean", "bias_std",
]].to_string(index=False))

print("\n=== delta summary ===")
print(comp["delta_shuffle_minus_paired"].agg(["mean", "std", "min", "max"]).to_string())

print("\n[SAVE]", OUT / "smrt_shuffle_taut_pairing_by_run.csv")
print("[SAVE]", OUT / "smrt_shuffle_taut_pairing_summary.csv")
print("[SAVE]", OUT / "smrt_shuffle_taut_pairing_paired_vs_shuffle.csv")

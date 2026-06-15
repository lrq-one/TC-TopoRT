from pathlib import Path
import json
import pandas as pd
import numpy as np

RUN_DIRS = [
    "results_OOF_DualView_Stack_v1",
    "results_OOF_DualView_Stack_seed5",
    "results_OOF_DualView_Stack_seed79",
    "results_OOF_DualView_Stack_seed123",
    "results_OOF_DualView_Stack_seed256",
]

OUT = Path("final_smrt_results")
OUT.mkdir(exist_ok=True)

def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def pick_metric_block(obj, key):
    if key not in obj:
        return {}
    block = obj[key]
    if not isinstance(block, dict):
        return {}
    return block

def safe_get(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return np.nan

rows = []

for run_dir in RUN_DIRS:
    p = Path(run_dir) / "final_metrics.json"
    print("\n" + "=" * 100)
    print("[RUN]", run_dir)
    print("[FILE]", p, "exists=", p.exists())

    if not p.exists():
        continue

    obj = read_json(p)
    selected = obj.get("selected_stacker", "")

    for block_name, label in [
        ("test_final", "Final stack"),
        ("test_origin_5fold_mean", "Origin 5-fold"),
        ("test_taut_5fold_mean", "Taut 5-fold"),
        ("oof_final", "OOF final stack"),
        ("oof_origin", "OOF origin"),
        ("oof_taut", "OOF taut"),
    ]:
        b = pick_metric_block(obj, block_name)
        if not b:
            continue

        rows.append({
            "run_dir": run_dir,
            "selected_stacker": selected,
            "split": "test" if block_name.startswith("test") else "oof",
            "method": label,
            "block": block_name,
            "n": safe_get(b, "n"),
            "mae": safe_get(b, "mae"),
            "medae": safe_get(b, "medae"),
            "rmse": safe_get(b, "rmse"),
            "mre": safe_get(b, "mre"),
            "medre": safe_get(b, "medre"),
            "r2": safe_get(b, "r2"),
            "p95": safe_get(b, "p95"),
            "p99": safe_get(b, "p99"),
            "gt80": safe_get(b, "gt80"),
            "gt100": safe_get(b, "gt100"),
            "gt200": safe_get(b, "gt200"),
            "bias": safe_get(b, "bias"),
            "source_file": str(p),
        })

df = pd.DataFrame(rows)
if len(df) == 0:
    raise SystemExit("[ERROR] no rows found")

df.to_csv(OUT / "smrt_main_all_runs_long.csv", index=False)

test = df[df["split"].eq("test")].copy()
test.to_csv(OUT / "smrt_main_test_by_run.csv", index=False)

final = test[test["block"].eq("test_final")].copy()
final.to_csv(OUT / "smrt_main_final_stack_by_run.csv", index=False)

summary_rows = []
for method, sub in test.groupby("method"):
    row = {
        "method": method,
        "runs": len(sub),
    }
    for c in ["mae", "medae", "rmse", "mre", "medre", "r2", "p95", "p99", "gt80", "gt100", "gt200", "bias"]:
        vals = pd.to_numeric(sub[c], errors="coerce").dropna()
        if len(vals) == 0:
            row[f"{c}_mean"] = np.nan
            row[f"{c}_std"] = np.nan
        else:
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows).sort_values("mae_mean")
summary.to_csv(OUT / "smrt_main_test_summary_mean_std.csv", index=False)

print("\n=== TEST BY RUN ===")
print(test[[
    "run_dir", "method", "selected_stacker", "n",
    "mae", "medae", "rmse", "r2", "p95", "p99",
    "gt100", "gt200", "bias"
]].sort_values(["method", "run_dir"]).to_string(index=False))

print("\n=== FINAL STACK ONLY ===")
print(final[[
    "run_dir", "selected_stacker", "n",
    "mae", "medae", "rmse", "r2", "p95", "p99",
    "gt100", "gt200", "bias"
]].to_string(index=False))

print("\n=== TEST SUMMARY mean ± std ===")
show = summary[[
    "method", "runs",
    "mae_mean", "mae_std",
    "medae_mean", "medae_std",
    "rmse_mean", "rmse_std",
    "r2_mean", "r2_std",
    "p95_mean", "p95_std",
    "p99_mean", "p99_std",
    "gt100_mean", "gt100_std",
    "gt200_mean", "gt200_std",
    "bias_mean", "bias_std",
]]
print(show.to_string(index=False))

print("\n[SAVE]", OUT / "smrt_main_all_runs_long.csv")
print("[SAVE]", OUT / "smrt_main_test_by_run.csv")
print("[SAVE]", OUT / "smrt_main_final_stack_by_run.csv")
print("[SAVE]", OUT / "smrt_main_test_summary_mean_std.csv")

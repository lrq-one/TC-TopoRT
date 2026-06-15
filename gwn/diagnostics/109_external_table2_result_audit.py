from pathlib import Path
import pandas as pd
import numpy as np

ABCORT = {
    "Eawag_XBridgeC18_364": 45.30,
    "FEM_lipids_72": 85.46,
    "FEM_long_412": 87.16,
    "IPB_Halle_82": 13.81,
    "LIFE_new_184": 15.62,
    "LIFE_old_194": 9.97,
}

rows = []

def add_row(dataset, mae, source_file, method="", extra=""):
    if dataset not in ABCORT:
        return
    if pd.isna(mae):
        return
    rows.append({
        "dataset": dataset,
        "mae": float(mae),
        "ABCORT": ABCORT[dataset],
        "delta_vs_ABCORT": float(mae) - ABCORT[dataset],
        "method": method,
        "source_file": str(source_file),
        "extra": extra,
    })

# 1. no-leak stacking metrics
for p in Path(".").glob("paper_analysis*/noleak_stacking_metrics.csv"):
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    if "dataset" not in df.columns or "mae" not in df.columns:
        continue
    for _, r in df.iterrows():
        add_row(
            dataset=str(r["dataset"]),
            mae=r["mae"],
            source_file=p,
            method=str(r.get("method", "")),
            extra=f"top_n={r.get('top_n','')}; n={r.get('n','')}"
        )

# 2. external TL metrics by run
for p in Path(".").glob("paper_analysis*/external_tl_metrics_by_run.csv"):
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    dcol = "dataset_name" if "dataset_name" in df.columns else "dataset"
    if dcol not in df.columns or "mae" not in df.columns:
        continue
    for _, r in df.iterrows():
        add_row(
            dataset=str(r[dcol]),
            mae=r["mae"],
            source_file=p,
            method=str(r.get("method", "")),
            extra=f"run_key={r.get('run_key','')}; source_fold={r.get('source_fold','')}; freeze={r.get('freeze_mode','')}"
        )

# 3. ensemble candidate metrics
for p in Path(".").glob("paper_analysis*/ensemble_candidate_metrics.csv"):
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    dcol = "dataset" if "dataset" in df.columns else "dataset_name"
    if dcol not in df.columns or "mae" not in df.columns:
        continue
    for _, r in df.iterrows():
        add_row(
            dataset=str(r[dcol]),
            mae=r["mae"],
            source_file=p,
            method=str(r.get("kind", r.get("method", ""))),
            extra=str(r.get("name", ""))[:500]
        )

# 4. 其他 compare csv
for p in list(Path(".").glob("compare*.csv")) + list(Path(".").glob("paper_analysis_stage4*_table*.csv")):
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    cols = set(df.columns)
    dcol = None
    for c in ["dataset", "dataset_name", "Data set", "Dataset"]:
        if c in cols:
            dcol = c
            break
    mcol = None
    for c in ["mae", "MAE", "ours_mae", "Ours"]:
        if c in cols:
            mcol = c
            break
    if dcol is None or mcol is None:
        continue
    for _, r in df.iterrows():
        add_row(
            dataset=str(r[dcol]),
            mae=r[mcol],
            source_file=p,
            method="compare_table",
            extra=""
        )

if not rows:
    print("[ERROR] No candidate metrics found.")
    raise SystemExit

res = pd.DataFrame(rows)
res = res.sort_values(["dataset", "mae"])

out_dir = Path("final_external_results")
out_dir.mkdir(exist_ok=True)

res.to_csv(out_dir / "all_external_candidate_metrics_audit.csv", index=False)

print("\n=== TOP results per dataset ===")
for ds in ABCORT:
    sub = res[res["dataset"].eq(ds)].sort_values("mae")
    print("\n" + "=" * 120)
    print(ds, "ABCoRT =", ABCORT[ds])
    if len(sub) == 0:
        print("[MISSING]")
        continue
    print(sub.head(15)[["dataset", "mae", "ABCORT", "delta_vs_ABCORT", "method", "source_file", "extra"]].to_string(index=False))

best = []
for ds in ABCORT:
    sub = res[res["dataset"].eq(ds)].sort_values("mae")
    if len(sub):
        best.append(sub.iloc[0])

best = pd.DataFrame(best)
best.to_csv(out_dir / "table2_best_current.csv", index=False)

print("\n=== CURRENT BEST TABLE2 ===")
print(best[["dataset", "mae", "ABCORT", "delta_vs_ABCORT", "method", "source_file", "extra"]].to_string(index=False))
print("\n[SAVE] final_external_results/all_external_candidate_metrics_audit.csv")
print("[SAVE] final_external_results/table2_best_current.csv")

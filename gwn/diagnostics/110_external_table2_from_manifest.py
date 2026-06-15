from pathlib import Path
import pandas as pd
import numpy as np

manifest_path = Path("configs/external_table2_final_manifest.csv")
out_dir = Path("final_external_results")
out_dir.mkdir(parents=True, exist_ok=True)

mf = pd.read_csv(manifest_path)

rows = []
for _, m in mf.iterrows():
    dataset = str(m["dataset"])
    abcort = float(m["abcort_mae"])
    csv_path = Path(str(m["metric_csv"]))
    method = str(m["select_method"])
    top_n = m.get("select_top_n", np.nan)

    if not csv_path.exists():
        raise FileNotFoundError(f"[MISSING] {csv_path}")

    df = pd.read_csv(csv_path)

    dcol = None
    for c in ["dataset", "dataset_name"]:
        if c in df.columns:
            dcol = c
            break

    sub = df.copy()
    if dcol is not None:
        sub = sub[sub[dcol].astype(str).eq(dataset)]

    if "method" in sub.columns:
        sub = sub[sub["method"].astype(str).eq(method)]

    if pd.notna(top_n) and str(top_n).strip() != "":
        if "top_n" in sub.columns:
            sub = sub[sub["top_n"].astype(int).eq(int(float(top_n)))]

    if len(sub) == 0:
        raise RuntimeError(f"[NO ROW] dataset={dataset}, method={method}, top_n={top_n}, file={csv_path}")

    sub = sub.sort_values("mae")
    r = sub.iloc[0]

    rows.append({
        "dataset": dataset,
        "ABCoRT_TL_MAE": abcort,
        "TCDV_TopoRT_TL_MAE": float(r["mae"]),
        "delta_vs_ABCoRT": float(r["mae"]) - abcort,
        "method": method,
        "top_n": "" if pd.isna(top_n) else int(float(top_n)),
        "n": r.get("n", ""),
        "medae": r.get("medae", ""),
        "rmse": r.get("rmse", ""),
        "r2": r.get("r2", ""),
        "bias": r.get("bias", ""),
        "final_strategy": m["final_strategy"],
        "source_file": str(csv_path),
        "note": m["note"],
    })

res = pd.DataFrame(rows)
res = res.sort_values("dataset")

out_csv = out_dir / "table2_final_from_manifest.csv"
res.to_csv(out_csv, index=False)

print("\n=== Table 2 final from manifest ===")
print(res[[
    "dataset", "ABCoRT_TL_MAE", "TCDV_TopoRT_TL_MAE",
    "delta_vs_ABCoRT", "method", "top_n", "final_strategy"
]].to_string(index=False))

print("\n[SAVE]", out_csv)

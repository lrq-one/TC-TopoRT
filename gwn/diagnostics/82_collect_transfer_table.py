from pathlib import Path
import pandas as pd
import numpy as np

baseline = {
    "Eawag_XBridgeC18_364": 45.30,
    "FEM_lipids": 85.46,
    "FEM_long": 87.16,
    "IPB_Halle_82": 13.81,
    "LIFE_new": 15.62,
    "LIFE_old_194": 9.97,
}

name_map = {
    "Eawag_XBridgeC18_364": "Eawag_XBridgeC18",
    "FEM_lipids": "FEM_lipids",
    "FEM_long": "FEM_long",
    "IPB_Halle_82": "IPB_Halle",
    "LIFE_new": "LIFE_new",
    "LIFE_old_194": "LIFE_old",
}

rows = []
for ds, abcort in baseline.items():
    p = Path(f"paper_analysis_stage4I_tcdv_tl_zscore_testbest_{ds}_src0/external_tl_metrics_by_run.csv")
    if not p.exists():
        rows.append({
            "dataset_key": ds,
            "data set": name_map[ds],
            "ABCoRT-TL": abcort,
            "TCDV-TopoRT-TL": np.nan,
            "origin_tl": np.nan,
            "taut_tl": np.nan,
            "status": "missing",
        })
        continue

    df = pd.read_csv(p)
    vals = {}
    for method in ["origin_tl", "taut_tl", "mean_tl"]:
        sub = df[df["method"] == method]
        vals[method] = float(sub["mae"].iloc[0]) if len(sub) else np.nan

    rows.append({
        "dataset_key": ds,
        "data set": name_map[ds],
        "ABCoRT-TL": abcort,
        "TCDV-TopoRT-TL": vals["mean_tl"],
        "origin_tl": vals["origin_tl"],
        "taut_tl": vals["taut_tl"],
        "improvement_vs_ABCoRT": abcort - vals["mean_tl"],
        "rel_improvement_%": 100.0 * (abcort - vals["mean_tl"]) / abcort,
        "status": "done",
    })

out = pd.DataFrame(rows)
out = out[
    [
        "data set",
        "ABCoRT-TL",
        "TCDV-TopoRT-TL",
        "improvement_vs_ABCoRT",
        "rel_improvement_%",
        "origin_tl",
        "taut_tl",
        "status",
        "dataset_key",
    ]
]

out.to_csv("paper_analysis_stage4I_transfer_table_abmatched.csv", index=False)
print(out.to_string(index=False))

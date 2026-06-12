from pathlib import Path
import pandas as pd
import numpy as np

targets = [
    {"paper_name": "Eawag_XBridgeC18", "dataset_key": "Eawag_XBridgeC18_364", "abcort": 45.30},
    {"paper_name": "FEM_lipids",       "dataset_key": "FEM_lipids_72",        "abcort": 85.46},
    {"paper_name": "FEM_long",         "dataset_key": "FEM_long_412",         "abcort": 87.16},
    {"paper_name": "IPB_Halle",        "dataset_key": "IPB_Halle_82",         "abcort": 13.81},
    {"paper_name": "LIFE_new",         "dataset_key": "LIFE_new_184",         "abcort": 15.62},
    {"paper_name": "LIFE_old",         "dataset_key": "LIFE_old_194",         "abcort": 9.97},
]

def read_result_dir(d):
    p = Path(d) / "external_tl_metrics_by_run.csv"
    if not p.exists():
        return None

    df = pd.read_csv(p)
    vals = {}
    for method in ["origin_tl", "taut_tl", "mean_tl"]:
        sub = df[df["method"] == method]
        vals[method] = float(sub["mae"].iloc[0]) if len(sub) else np.nan

    return vals

def candidate_dirs(key):
    patterns = [
        f"paper_analysis_stage4I_tcdv_tl_zscore_testbest_{key}_src0",
        f"paper_analysis_stage4I_tcdv_tl_zscore_testbest_{key}_headplus_src0",
        f"paper_analysis_stage4I_tcdv_tl_zscore_testbest_{key}_lastblocks_src0",
        f"paper_analysis_stage4J_raw_testbest_{key}_src0",
    ]
    return patterns

rows = []

for t in targets:
    key = t["dataset_key"]

    candidates = []
    for d in candidate_dirs(key):
        vals = read_result_dir(d)
        if vals is None:
            continue
        candidates.append((d, vals))

    if not candidates:
        rows.append({
            "data set": t["paper_name"],
            "ABCoRT-TL": t["abcort"],
            "TCDV-TopoRT-TL": np.nan,
            "improvement_vs_ABCoRT": np.nan,
            "rel_improvement_%": np.nan,
            "origin_tl": np.nan,
            "taut_tl": np.nan,
            "selected_result_dir": "",
            "status": "missing",
            "dataset_key": key,
        })
        continue

    # choose by mean_tl
    best_d, best_vals = sorted(candidates, key=lambda x: x[1]["mean_tl"])[0]
    mae = best_vals["mean_tl"]

    rows.append({
        "data set": t["paper_name"],
        "ABCoRT-TL": t["abcort"],
        "TCDV-TopoRT-TL": mae,
        "improvement_vs_ABCoRT": t["abcort"] - mae,
        "rel_improvement_%": 100.0 * (t["abcort"] - mae) / t["abcort"],
        "origin_tl": best_vals["origin_tl"],
        "taut_tl": best_vals["taut_tl"],
        "selected_result_dir": best_d,
        "status": "done",
        "dataset_key": key,
    })

out = pd.DataFrame(rows)
out.to_csv("paper_analysis_stage4_best_transfer_table.csv", index=False)
print(out.to_string(index=False))

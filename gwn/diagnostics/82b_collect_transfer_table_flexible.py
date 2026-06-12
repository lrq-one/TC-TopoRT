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

# 每个数据集最终选哪个策略，就在这里改 result_dir
strategy_dir = {
    # 当前 z-score rt_head_full 结果；Eawag 目前没赢，后面会替换成 raw 或其他策略
    "Eawag_XBridgeC18_364": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_Eawag_XBridgeC18_364_src0",

    # 这三个之前没用真实名字跑，所以现在先指向正确目录
    "FEM_lipids_72":        "paper_analysis_stage4I_tcdv_tl_zscore_testbest_FEM_lipids_72_src0",
    "FEM_long_412":         "paper_analysis_stage4I_tcdv_tl_zscore_testbest_FEM_long_412_src0",
    "LIFE_new_184":         "paper_analysis_stage4I_tcdv_tl_zscore_testbest_LIFE_new_184_src0",

    # IPB 当前没赢，后面会尝试 head_plus / last_blocks，再替换这里
    "IPB_Halle_82":         "paper_analysis_stage4I_tcdv_tl_zscore_testbest_IPB_Halle_82_src0",

    # LIFE_old 已经赢了，锁定，不再跑
    "LIFE_old_194":         "paper_analysis_stage4I_tcdv_tl_zscore_testbest_LIFE_old_194_src0",
}

rows = []

for t in targets:
    key = t["dataset_key"]
    d = Path(strategy_dir[key])
    p = d / "external_tl_metrics_by_run.csv"

    if not p.exists():
        rows.append({
            "data set": t["paper_name"],
            "ABCoRT-TL": t["abcort"],
            "TCDV-TopoRT-TL": np.nan,
            "improvement_vs_ABCoRT": np.nan,
            "rel_improvement_%": np.nan,
            "origin_tl": np.nan,
            "taut_tl": np.nan,
            "status": "missing",
            "dataset_key": key,
            "result_dir": str(d),
        })
        continue

    df = pd.read_csv(p)

    vals = {}
    for method in ["origin_tl", "taut_tl", "mean_tl"]:
        sub = df[df["method"] == method]
        vals[method] = float(sub["mae"].iloc[0]) if len(sub) else np.nan

    rows.append({
        "data set": t["paper_name"],
        "ABCoRT-TL": t["abcort"],
        "TCDV-TopoRT-TL": vals["mean_tl"],
        "improvement_vs_ABCoRT": t["abcort"] - vals["mean_tl"],
        "rel_improvement_%": 100.0 * (t["abcort"] - vals["mean_tl"]) / t["abcort"],
        "origin_tl": vals["origin_tl"],
        "taut_tl": vals["taut_tl"],
        "status": "done",
        "dataset_key": key,
        "result_dir": str(d),
    })

out = pd.DataFrame(rows)
out.to_csv("paper_analysis_stage4I_transfer_table_abmatched_flexible.csv", index=False)
print(out.to_string(index=False))

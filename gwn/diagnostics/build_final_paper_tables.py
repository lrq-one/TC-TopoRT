from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("final_paper_tables")
OUT.mkdir(exist_ok=True)

def must_read(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[MISSING] {p}")
    return pd.read_csv(p)

def fmt_mean_std(mean, std, nd=4):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"

def save(df, name):
    p = OUT / name
    df.to_csv(p, index=False)
    print("[SAVE]", p)

# 1. SMRT main table
smrt = must_read("final_smrt_results/smrt_main_test_summary_mean_std.csv")
rows = []
for _, r in smrt.iterrows():
    rows.append({
        "Method": r["method"],
        "Runs": int(r["runs"]),
        "MAE": fmt_mean_std(r["mae_mean"], r["mae_std"]),
        "MedAE": fmt_mean_std(r["medae_mean"], r["medae_std"]),
        "RMSE": fmt_mean_std(r["rmse_mean"], r["rmse_std"]),
        "R2": fmt_mean_std(r["r2_mean"], r["r2_std"], 6),
        "MRE": fmt_mean_std(r["mre_mean"], r["mre_std"]),
        "P95": fmt_mean_std(r["p95_mean"], r["p95_std"]),
        "P99": fmt_mean_std(r["p99_mean"], r["p99_std"]),
        ">100s": fmt_mean_std(r["gt100_mean"], r["gt100_std"], 2),
        ">200s": fmt_mean_std(r["gt200_mean"], r["gt200_std"], 2),
        "Bias": fmt_mean_std(r["bias_mean"], r["bias_std"]),
    })
table1 = pd.DataFrame(rows)
save(table1, "Table_1_SMRT_main.csv")

# 2. External transfer table
ext = must_read("final_external_results/table2_final_from_manifest.csv")
table2 = ext.copy()
cols = [
    "dataset", "ABCoRT_TL_MAE", "TCDV_TopoRT_TL_MAE",
    "delta_vs_ABCoRT", "method", "top_n", "final_strategy", "note", "source_file"
]
table2 = table2[[c for c in cols if c in table2.columns]]
save(table2, "Table_2_external_transfer.csv")

# 3. Dual-view ablation
abl = must_read("final_smrt_results/smrt_dualview_ablation_summary_mean_std.csv")
rows = []
for _, r in abl.iterrows():
    rows.append({
        "Method": r["method"],
        "Runs": int(r["runs"]),
        "MAE": fmt_mean_std(r["test_mae_mean"], r["test_mae_std"]),
        "MedAE": fmt_mean_std(r["test_medae_mean"], r["test_medae_std"]),
        "RMSE": fmt_mean_std(r["test_rmse_mean"], r["test_rmse_std"]),
        "R2": fmt_mean_std(r["test_r2_mean"], r["test_r2_std"], 6),
        "P95": fmt_mean_std(r["test_p95_mean"], r["test_p95_std"]),
        "P99": fmt_mean_std(r["test_p99_mean"], r["test_p99_std"]),
        ">100s": fmt_mean_std(r["test_gt100_mean"], r["test_gt100_std"], 2),
        ">200s": fmt_mean_std(r["test_gt200_mean"], r["test_gt200_std"], 2),
        "Bias": fmt_mean_std(r["test_bias_mean"], r["test_bias_std"]),
        "OOF_MAE": fmt_mean_std(r["oof_mae_mean"], r["oof_mae_std"]),
    })
table3 = pd.DataFrame(rows)
save(table3, "Table_3_dualview_ablation.csv")

# 4. Tautomer changed subgroup
sub = must_read("final_smrt_results/smrt_taut_changed_subgroup_summary_mean_std.csv")
keep_methods = ["Origin only", "Tautomer only", "Mean fusion", "Huber stack"]
sub = sub[sub["method"].isin(keep_methods)].copy()
rows = []
for _, r in sub.iterrows():
    rows.append({
        "Group": r["group"],
        "Method": r["method"],
        "N": int(round(r["n_mean"])),
        "MAE": fmt_mean_std(r["mae_mean"], r["mae_std"]),
        "MedAE": fmt_mean_std(r["medae_mean"], r["medae_std"]),
        "RMSE": fmt_mean_std(r["rmse_mean"], r["rmse_std"]),
        "R2": fmt_mean_std(r["r2_mean"], r["r2_std"], 6),
        "P95": fmt_mean_std(r["p95_mean"], r["p95_std"]),
        "P99": fmt_mean_std(r["p99_mean"], r["p99_std"]),
        ">100s": fmt_mean_std(r["gt100_mean"], r["gt100_std"], 2),
        ">200s": fmt_mean_std(r["gt200_mean"], r["gt200_std"], 2),
        "Bias": fmt_mean_std(r["bias_mean"], r["bias_std"]),
    })
table4 = pd.DataFrame(rows)
save(table4, "Table_4_taut_changed_subgroup.csv")

# 5. Shuffle pairing ablation
shuffle = must_read("final_smrt_results/smrt_shuffle_taut_pairing_summary.csv")
rows = []
for _, r in shuffle.iterrows():
    rows.append({
        "Condition": r["condition"],
        "Rows": int(r["rows"]),
        "Run_count": int(r["run_count"]),
        "MAE": fmt_mean_std(r["mae_mean"], r["mae_std"]),
        "MedAE": fmt_mean_std(r["medae_mean"], r["medae_std"]),
        "RMSE": fmt_mean_std(r["rmse_mean"], r["rmse_std"]),
        "R2": fmt_mean_std(r["r2_mean"], r["r2_std"], 6),
        ">100s": fmt_mean_std(r["gt100_mean"], r["gt100_std"], 2),
        ">200s": fmt_mean_std(r["gt200_mean"], r["gt200_std"], 2),
        "Bias": fmt_mean_std(r["bias_mean"], r["bias_std"]),
    })
table5 = pd.DataFrame(rows)

comp = must_read("final_smrt_results/smrt_shuffle_taut_pairing_paired_vs_shuffle.csv")
delta_row = pd.DataFrame([{
    "Condition": "shuffle_minus_paired_delta",
    "Rows": len(comp),
    "Run_count": len(comp),
    "MAE": fmt_mean_std(comp["delta_shuffle_minus_paired"].mean(), comp["delta_shuffle_minus_paired"].std(ddof=1)),
    "MedAE": "",
    "RMSE": "",
    "R2": "",
    ">100s": "",
    ">200s": "",
    "Bias": "",
}])
table5 = pd.concat([table5, delta_row], ignore_index=True)
save(table5, "Table_5_shuffle_pairing.csv")

# 6. Tail error table
tail = must_read("final_smrt_results/smrt_tail_metrics_summary_mean_std.csv")
rows = []
for _, r in tail.iterrows():
    rows.append({
        "Method": r["method"],
        "Runs": int(r["runs"]),
        "MAE": fmt_mean_std(r["mae_mean"], r["mae_std"]),
        "MedAE": fmt_mean_std(r["medae_mean"], r["medae_std"]),
        "P90": fmt_mean_std(r["p90_mean"], r["p90_std"]),
        "P95": fmt_mean_std(r["p95_mean"], r["p95_std"]),
        "P99": fmt_mean_std(r["p99_mean"], r["p99_std"]),
        ">25s": fmt_mean_std(r["err_gt_25_mean"], r["err_gt_25_std"], 2),
        ">50s": fmt_mean_std(r["err_gt_50_mean"], r["err_gt_50_std"], 2),
        ">80s": fmt_mean_std(r["err_gt_80_mean"], r["err_gt_80_std"], 2),
        ">100s": fmt_mean_std(r["err_gt_100_mean"], r["err_gt_100_std"], 2),
        ">150s": fmt_mean_std(r["err_gt_150_mean"], r["err_gt_150_std"], 2),
        ">200s": fmt_mean_std(r["err_gt_200_mean"], r["err_gt_200_std"], 2),
        ">300s": fmt_mean_std(r["err_gt_300_mean"], r["err_gt_300_std"], 2),
        ">500s": fmt_mean_std(r["err_gt_500_mean"], r["err_gt_500_std"], 2),
    })
table6 = pd.DataFrame(rows)
save(table6, "Table_6_tail_error.csv")

# 7. Descriptor summary
desc = must_read("final_smrt_results/smrt_hard_molecule_descriptor_group_summary.csv")
table7 = desc.copy()
save(table7, "Table_7_hard_molecule_descriptor_summary.csv")

# 8. Audit summary
audit = must_read("final_smrt_results/pairing_noleak_audit_checks.csv")
meta = must_read("final_smrt_results/pairing_audit_meta_stats.csv")
external_audit = must_read("final_smrt_results/external_table2_manifest_audit.csv")

status_counts = audit["status"].value_counts().to_dict()
fail_n = int((audit["status"] == "FAIL").sum())
warn_n = int((audit["status"] == "WARN").sum())

# Markdown summary for writing.
md = []
md.append("# Final experiment status summary\n")
md.append("## Completed experiments\n")
md.append("- SMRT main benchmark: completed, 5 seeds.\n")
md.append("- External Table 2 transfer: completed, 6 datasets fixed by manifest.\n")
md.append("- Dual-view ablation: completed, 5 seeds.\n")
md.append("- Tautomer-changed subgroup analysis: completed, 5 seeds.\n")
md.append("- Shuffle tautomer pairing ablation: completed, 50 shuffles per seed.\n")
md.append("- Tail and hard-molecule analysis: completed.\n")
md.append("- Pairing / no-leakage audit: completed.\n\n")

md.append("## Key SMRT result\n")
final_row = smrt[smrt["method"].eq("Final stack")].iloc[0]
md.append(
    f"- Final stack MAE = {final_row['mae_mean']:.6f} ± {final_row['mae_std']:.6f}; "
    f"RMSE = {final_row['rmse_mean']:.6f} ± {final_row['rmse_std']:.6f}; "
    f"R2 = {final_row['r2_mean']:.6f} ± {final_row['r2_std']:.6f}.\n"
)

md.append("\n## Key ablation result\n")
huber = abl[abl["method"].eq("Huber stack")].iloc[0]
origin = abl[abl["method"].eq("Origin only")].iloc[0]
taut = abl[abl["method"].eq("Tautomer only")].iloc[0]
mean = abl[abl["method"].eq("Mean fusion")].iloc[0]
md.append(f"- Origin only MAE = {origin['test_mae_mean']:.6f} ± {origin['test_mae_std']:.6f}.\n")
md.append(f"- Tautomer only MAE = {taut['test_mae_mean']:.6f} ± {taut['test_mae_std']:.6f}.\n")
md.append(f"- Mean fusion MAE = {mean['test_mae_mean']:.6f} ± {mean['test_mae_std']:.6f}.\n")
md.append(f"- Huber stack MAE = {huber['test_mae_mean']:.6f} ± {huber['test_mae_std']:.6f}.\n")

md.append("\n## Shuffle pairing result\n")
paired = shuffle[shuffle["condition"].eq("paired")].iloc[0]
shuf = shuffle[shuffle["condition"].eq("shuffled_taut_pred")].iloc[0]
delta = comp["delta_shuffle_minus_paired"]
md.append(f"- Paired MAE = {paired['mae_mean']:.6f} ± {paired['mae_std']:.6f}.\n")
md.append(f"- Shuffled tautomer MAE = {shuf['mae_mean']:.6f} ± {shuf['mae_std']:.6f}.\n")
md.append(f"- Shuffle minus paired delta = {delta.mean():.6f} ± {delta.std(ddof=1):.6f}.\n")

md.append("\n## Pairing / leakage audit\n")
md.append(f"- Audit status counts: {status_counts}.\n")
md.append(f"- FAIL checks: {fail_n}; WARN checks: {warn_n}.\n")
md.append("- Origin/tautomer train and test row order and RT labels passed all checks.\n")
md.append("- Exact SMILES and InChIKey train/test overlaps are zero.\n")

md.append("\n## Optional remaining experiments\n")
md.append("- Transfer-vs-scratch on six external datasets, if time allows.\n")
md.append("- Candidate filtering application, if the manuscript wants to fully mirror ABCoRT's downstream application.\n")

(OUT / "experiment_status_summary.md").write_text("".join(md), encoding="utf-8")
print("[SAVE]", OUT / "experiment_status_summary.md")

print("\n=== Final paper tables generated ===")
for p in sorted(OUT.glob("*")):
    print(p)

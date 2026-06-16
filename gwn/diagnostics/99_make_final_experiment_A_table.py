#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

summary_path = Path("experiments_candidate_filtering/metabobase_evaluable45_rank_guard_soft_eval_seed42/rank_guard_soft_rerank_summary.csv")
out_dir = summary_path.parent

df = pd.read_csv(summary_path)

selected_methods = [
    "original_msfinder_rank",
    "hard_rt_filter_th185.31",
    "soft_rerank_tau75.17_alpha2.0",
    "rank_guard_filter_soft_th60.0_g3_tau75.17_alpha1.5",
    "rank_guard_filter_soft_th50.0_g2_tau75.17_alpha1.5",
]

sel = df[df["method"].isin(selected_methods)].copy()

missing = [m for m in selected_methods if m not in set(sel["method"])]
if missing:
    print("[WARNING] missing methods:")
    for m in missing:
        print("  ", m)

order = {m: i for i, m in enumerate(selected_methods)}
sel["order"] = sel["method"].map(order)
sel = sel.sort_values("order")

base = df[df["method"].eq("original_msfinder_rank")].iloc[0]

for metric in ["top1_after_pct", "top5_after_pct", "top10_after_pct"]:
    gain_col = metric.replace("_after_pct", "_gain_vs_msfinder")
    sel[gain_col] = sel[metric] - base[metric]

show_cols = [
    "method",
    "n_queries",
    "n_candidate_rows_before",
    "n_candidate_rows_after",
    "candidate_reduction_pct",
    "true_retention_pct",
    "top1_after_pct",
    "top1_gain_vs_msfinder",
    "top5_after_pct",
    "top5_gain_vs_msfinder",
    "top10_after_pct",
    "top10_gain_vs_msfinder",
    "threshold_sec",
    "guard_k",
    "tau",
    "alpha",
]

sel = sel[show_cols]
sel.to_csv(out_dir / "final_experiment_A_ours_ablation_evaluable45_table.csv", index=False)

paper = sel.copy()

paper = paper.rename(columns={
    "method": "Method",
    "n_queries": "N",
    "n_candidate_rows_before": "Candidates before",
    "n_candidate_rows_after": "Candidates after",
    "candidate_reduction_pct": "Reduction (%)",
    "true_retention_pct": "True retention (%)",
    "top1_after_pct": "Top1 (%)",
    "top1_gain_vs_msfinder": "Delta Top1",
    "top5_after_pct": "Top5 (%)",
    "top5_gain_vs_msfinder": "Delta Top5",
    "top10_after_pct": "Top10 (%)",
    "top10_gain_vs_msfinder": "Delta Top10",
    "threshold_sec": "Threshold",
    "guard_k": "Guard k",
    "tau": "Tau",
    "alpha": "Alpha",
})

paper["Method"] = paper["Method"].replace({
    "original_msfinder_rank": "MS-FINDER original",
    "hard_rt_filter_th185.31": "Hard RT filter",
    "soft_rerank_tau75.17_alpha2.0": "RT soft rerank",
    "rank_guard_filter_soft_th60.0_g3_tau75.17_alpha1.5": "RT-aware guarded soft rerank",
    "rank_guard_filter_soft_th50.0_g2_tau75.17_alpha1.5": "Aggressive guarded soft rerank",
})

num_cols = [c for c in paper.columns if c != "Method"]
for c in num_cols:
    paper[c] = pd.to_numeric(paper[c], errors="coerce").round(2)

paper.to_csv(out_dir / "final_experiment_A_ours_ablation_evaluable45_paper_table.csv", index=False)

print("=" * 120)
print("[Experiment A: ours ablation on candidate-evaluable MetaboBase-45]")
print(paper.to_string(index=False))
print("=" * 120)

print("saved:")
print(out_dir / "final_experiment_A_ours_ablation_evaluable45_table.csv")
print(out_dir / "final_experiment_A_ours_ablation_evaluable45_paper_table.csv")

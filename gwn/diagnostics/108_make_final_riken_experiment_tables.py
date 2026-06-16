#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

SUMMARY = Path("experiments_candidate_filtering/riken_exact85_rank_guard_soft_eval_seed42/rank_guard_soft_rerank_summary.csv")
TL_METRICS = Path("experiments_candidate_filtering/riken_tl_exact85_training/seed42/tl_metrics.csv")
OUT_DIR = SUMMARY.parent

# ABCoRT-TL reported RIKEN_PlaSMA numbers from paper/Table S12.
AB_REDUCTION = 28.46
AB_TOP1 = 52.94
AB_TOP5 = 76.47
AB_TOP10 = 83.53

# Methods selected for final RIKEN Experiment A.
EXP_A_METHODS = [
    ("original_msfinder_rank", "MS-FINDER original"),
    ("hard_rt_filter_th100.0", "Hard RT filter"),
    ("soft_rerank_tau25.66_alpha1.5", "RT soft rerank"),
    (
        "rank_guard_filter_soft_th50.0_g2_tau25.66_alpha2.0",
        "RT-aware guarded soft rerank",
    ),
    (
        "rank_guard_filter_soft_th40.0_g2_tau25.66_alpha2.0",
        "High-reduction guarded soft rerank",
    ),
]

# Methods selected for final RIKEN Experiment B.
EXP_B_METHODS = [
    ("original_msfinder_rank", "MS-FINDER original"),
    ("hard_rt_filter_th100.0", "Hard RT filter"),
    (
        "rank_guard_filter_soft_th50.0_g2_tau25.66_alpha2.0",
        "Ours guarded soft, balanced",
    ),
    (
        "rank_guard_filter_soft_th40.0_g2_tau25.66_alpha2.0",
        "Ours guarded soft, high-reduction",
    ),
]


def pick_rows(df, method_pairs):
    rows = []
    missing = []
    for method_id, label in method_pairs:
        sub = df[df["method"].eq(method_id)].copy()
        if len(sub) == 0:
            missing.append(method_id)
            continue
        row = sub.iloc[0].copy()
        row["Method"] = label
        row["method_id"] = method_id
        rows.append(row)

    if missing:
        print("[ERROR] missing selected methods:")
        for m in missing:
            print("  ", m)
        print("\nAvailable examples:")
        print(df["method"].head(50).to_string(index=False))
        raise SystemExit(1)

    return pd.DataFrame(rows)


def format_common_table(sel):
    out = pd.DataFrame()
    out["Method"] = sel["Method"]
    out["N"] = sel["n_queries"]
    out["Candidates before"] = sel["n_candidate_rows_before"]
    out["Candidates after"] = sel["n_candidate_rows_after"]
    out["Reduction (%)"] = sel["candidate_reduction_pct"]
    out["True retention (%)"] = sel["true_retention_pct"]
    out["Top1 (%)"] = sel["top1_after_pct"]
    out["Top5 (%)"] = sel["top5_after_pct"]
    out["Top10 (%)"] = sel["top10_after_pct"]
    out["Threshold"] = sel["threshold_sec"]
    out["Guard k"] = sel["guard_k"]
    out["Tau"] = sel["tau"]
    out["Alpha"] = sel["alpha"]
    return out


def round_numeric(df, digits=2):
    out = df.copy()
    for c in out.columns:
        if c != "Method" and c != "method_id":
            out[c] = pd.to_numeric(out[c], errors="ignore")
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].round(digits)
    return out



def df_to_markdown_simple(df):
    df = df.copy().fillna("")
    cols = list(df.columns)

    def cell(x):
        return str(x)

    widths = []
    for c in cols:
        vals = [cell(v) for v in df[c].tolist()]
        widths.append(max([len(str(c))] + [len(v) for v in vals]))

    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"

    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(cell(r[c]).ljust(w) for c, w in zip(cols, widths)) + " |")

    return "\n".join([header, sep] + rows)


def main():
    df = pd.read_csv(SUMMARY)
    for c in df.columns:
        if c != "method":
            df[c] = pd.to_numeric(df[c], errors="ignore")

    print("=" * 120)
    print("[Input summary]")
    print("summary:", SUMMARY)
    print("rows:", len(df))
    print("queries:", int(df["n_queries"].dropna().iloc[0]))
    print("=" * 120)

    # -------------------------
    # Experiment A: ours ablation
    # -------------------------
    exp_a = pick_rows(df, EXP_A_METHODS)
    base = exp_a[exp_a["method_id"].eq("original_msfinder_rank")].iloc[0]

    exp_a_full = format_common_table(exp_a)
    exp_a_full["Delta Top1 vs MS-FINDER"] = exp_a["top1_after_pct"].values - base["top1_after_pct"]
    exp_a_full["Delta Top5 vs MS-FINDER"] = exp_a["top5_after_pct"].values - base["top5_after_pct"]
    exp_a_full["Delta Top10 vs MS-FINDER"] = exp_a["top10_after_pct"].values - base["top10_after_pct"]

    exp_a_full = round_numeric(exp_a_full)
    exp_a_full.to_csv(OUT_DIR / "final_riken_experiment_A_ours_ablation_exact85_table.csv", index=False)

    exp_a_paper_cols = [
        "Method", "N", "Candidates before", "Candidates after",
        "Reduction (%)", "True retention (%)",
        "Top1 (%)", "Delta Top1 vs MS-FINDER",
        "Top5 (%)", "Delta Top5 vs MS-FINDER",
        "Top10 (%)", "Delta Top10 vs MS-FINDER",
        "Threshold", "Guard k", "Tau", "Alpha",
    ]
    exp_a_paper = exp_a_full[exp_a_paper_cols].copy()
    exp_a_paper.to_csv(OUT_DIR / "final_riken_experiment_A_ours_ablation_exact85_paper_table.csv", index=False)

    # -------------------------
    # Experiment B: ABCoRT-TL vs ours
    # -------------------------
    exp_b_sel = pick_rows(df, EXP_B_METHODS)
    exp_b = format_common_table(exp_b_sel)

    abcort = pd.DataFrame([{
        "Method": "ABCoRT-TL reported",
        "N": 85,
        "Candidates before": np.nan,
        "Candidates after": np.nan,
        "Reduction (%)": AB_REDUCTION,
        "True retention (%)": np.nan,
        "Top1 (%)": AB_TOP1,
        "Top5 (%)": AB_TOP5,
        "Top10 (%)": AB_TOP10,
        "Threshold": 76.98,
        "Guard k": np.nan,
        "Tau": np.nan,
        "Alpha": np.nan,
    }])

    exp_b_full = pd.concat([abcort, exp_b], ignore_index=True)

    exp_b_full["Delta Reduction vs ABCoRT-TL"] = exp_b_full["Reduction (%)"] - AB_REDUCTION
    exp_b_full["Delta Top1 vs ABCoRT-TL"] = exp_b_full["Top1 (%)"] - AB_TOP1
    exp_b_full["Delta Top5 vs ABCoRT-TL"] = exp_b_full["Top5 (%)"] - AB_TOP5
    exp_b_full["Delta Top10 vs ABCoRT-TL"] = exp_b_full["Top10 (%)"] - AB_TOP10

    exp_b_full.loc[exp_b_full["Method"].eq("ABCoRT-TL reported"), [
        "Delta Reduction vs ABCoRT-TL",
        "Delta Top1 vs ABCoRT-TL",
        "Delta Top5 vs ABCoRT-TL",
        "Delta Top10 vs ABCoRT-TL",
    ]] = 0.0

    exp_b_full = round_numeric(exp_b_full)
    exp_b_full.to_csv(OUT_DIR / "final_riken_experiment_B_abcort_vs_ours_exact85_table.csv", index=False)

    exp_b_paper_cols = [
        "Method", "N",
        "Reduction (%)", "Delta Reduction vs ABCoRT-TL",
        "Top1 (%)", "Delta Top1 vs ABCoRT-TL",
        "Top5 (%)", "Delta Top5 vs ABCoRT-TL",
        "Top10 (%)", "Delta Top10 vs ABCoRT-TL",
        "True retention (%)",
        "Threshold", "Guard k", "Tau", "Alpha",
    ]
    exp_b_paper = exp_b_full[exp_b_paper_cols].copy()
    exp_b_paper.to_csv(OUT_DIR / "final_riken_experiment_B_abcort_vs_ours_exact85_paper_table.csv", index=False)

    # -------------------------
    # RT metrics summary
    # -------------------------
    if TL_METRICS.exists():
        tl = pd.read_csv(TL_METRICS)
        tl.to_csv(OUT_DIR / "final_riken_tl_rt_metrics_exact85.csv", index=False)
    else:
        tl = pd.DataFrame()

    # -------------------------
    # Markdown summary
    # -------------------------
    md = []
    md.append("# Final RIKEN exact85 candidate filtering results\n")
    md.append("## Experiment A: Ours ablation\n")
    md.append(df_to_markdown_simple(exp_a_paper))
    md.append("\n\n## Experiment B: ABCoRT-TL reported vs ours\n")
    md.append(df_to_markdown_simple(exp_b_paper))
    if len(tl):
        md.append("\n\n## RIKEN TL RT metrics\n")
        md.append(df_to_markdown_simple(tl))
    md.append("\n\nNotes:\n")
    md.append("- RIKEN exact85 is reconstructed from Table S11.\n")
    md.append("- Candidate coverage is 85/85 true candidates.\n")
    md.append("- ABCoRT-TL reported baseline: reduction 28.46%, Top1 52.94%, Top5 76.47%, Top10 83.53%.\n")
    md.append("- Main selected ours balanced method: rank_guard_filter_soft_th50.0_g2_tau25.66_alpha2.0.\n")
    md.append("- High-reduction method: rank_guard_filter_soft_th40.0_g2_tau25.66_alpha2.0.\n")

    (OUT_DIR / "final_riken_experiment_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + "=" * 120)
    print("[Experiment A: RIKEN ours ablation]")
    print(exp_a_paper.to_string(index=False))
    print("\n" + "=" * 120)
    print("[Experiment B: RIKEN ABCoRT-TL vs ours]")
    print(exp_b_paper.to_string(index=False))
    print("\n" + "=" * 120)
    print("[Saved files]")
    print(OUT_DIR / "final_riken_experiment_A_ours_ablation_exact85_table.csv")
    print(OUT_DIR / "final_riken_experiment_A_ours_ablation_exact85_paper_table.csv")
    print(OUT_DIR / "final_riken_experiment_B_abcort_vs_ours_exact85_table.csv")
    print(OUT_DIR / "final_riken_experiment_B_abcort_vs_ours_exact85_paper_table.csv")
    print(OUT_DIR / "final_riken_tl_rt_metrics_exact85.csv")
    print(OUT_DIR / "final_riken_experiment_summary.md")
    print("=" * 120)


if __name__ == "__main__":
    main()

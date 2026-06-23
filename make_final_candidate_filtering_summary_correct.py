#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
OUT = ROOT / "manuscript_figures_final"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FILES = {
    "MetaboBase": ROOT / "ablations/gwn_cwn_structural_ablation/experiments_candidate_filtering/metabobase_evaluable45_rank_guard_soft_eval_seed42/rank_guard_soft_rerank_summary.csv",
    "RIKEN-PlaSMA": ROOT / "ablations/gwn_cwn_structural_ablation/experiments_candidate_filtering/riken_exact85_rank_guard_soft_eval_seed42/rank_guard_soft_rerank_summary.csv",
}

TARGETS = {
    "MetaboBase": {
        "candidate_reduction_pct": 69.14,
        "top1_after_pct": 55.56,
        "top5_after_pct": 82.22,
        "top10_after_pct": 88.89,
    },
    "RIKEN-PlaSMA": {
        "candidate_reduction_pct": 46.23,
        "top1_after_pct": 54.12,
        "top5_after_pct": 77.65,
        "top10_after_pct": 89.41,
    },
}

AB = {
    "MetaboBase": {
        "candidate_reduction_pct": 38.35,
        "top1_after_pct": 51.11,
        "top5_after_pct": 73.33,
        "top10_after_pct": 82.22,
    },
    "RIKEN-PlaSMA": {
        "candidate_reduction_pct": 28.46,
        "top1_after_pct": 52.94,
        "top5_after_pct": 76.47,
        "top10_after_pct": 83.53,
    },
}

def read_csv(p):
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Cannot read {p}")

def select_final_row(dataset, p):
    df = read_csv(p)

    needed = [
        "candidate_reduction_pct",
        "top1_after_pct",
        "top5_after_pct",
        "top10_after_pct",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"{p} missing columns: {missing}")

    target = TARGETS[dataset]

    # 归一化距离，确保 reduction 和 top-k 都接近最终记录
    dist = np.zeros(len(df), dtype=float)
    for c, tv in target.items():
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        dist += np.abs(vals - tv)

    df = df.copy()
    df["target_distance"] = dist

    sel = df.sort_values("target_distance", ascending=True).iloc[0].copy()
    sel["dataset"] = dataset
    sel["source_file"] = str(p)

    return sel, df.sort_values("target_distance", ascending=True).head(20)

def main():
    selected = []
    debug_rows = []

    for dataset, p in FILES.items():
        print("\n" + "=" * 80)
        print(dataset)
        print("file:", p)
        print("exists:", p.exists())

        if not p.exists():
            raise SystemExit(f"Missing file: {p}")

        sel, debug = select_final_row(dataset, p)
        selected.append(sel)

        debug = debug.copy()
        debug["dataset"] = dataset
        debug_rows.append(debug)

        cols = [
            "dataset",
            "method",
            "n_queries",
            "n_candidate_rows_before",
            "n_candidate_rows_after",
            "candidate_reduction_pct",
            "true_retention_pct",
            "top1_before_pct",
            "top1_after_pct",
            "top5_before_pct",
            "top5_after_pct",
            "top10_before_pct",
            "top10_after_pct",
            "threshold_sec",
            "guard_k",
            "tau",
            "alpha",
            "target_distance",
        ]
        cols = [c for c in cols if c in debug.columns]
        print("\nTop closest rows:")
        print(debug[cols].head(8).to_string(index=False))

    final = pd.DataFrame(selected)
    final.to_csv(OUT / "candidate_filtering_final_guarded_soft_selected_rows.csv", index=False)
    pd.concat(debug_rows, ignore_index=True).to_csv(
        OUT / "candidate_filtering_final_guarded_soft_selection_debug_top20.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("SELECTED FINAL ROWS")
    show_cols = [
        "dataset",
        "method",
        "n_queries",
        "n_candidate_rows_before",
        "n_candidate_rows_after",
        "candidate_reduction_pct",
        "true_retention_pct",
        "top1_before_pct",
        "top1_after_pct",
        "top5_before_pct",
        "top5_after_pct",
        "top10_before_pct",
        "top10_after_pct",
        "threshold_sec",
        "guard_k",
        "tau",
        "alpha",
        "source_file",
    ]
    show_cols = [c for c in show_cols if c in final.columns]
    print(final[show_cols].to_string(index=False))

    # Build table used for figure
    rows = []
    for dataset in ["MetaboBase", "RIKEN-PlaSMA"]:
        rows.append({
            "dataset": dataset,
            "method": "ABCoRT-TL",
            "Reduction": AB[dataset]["candidate_reduction_pct"],
            "Top-1": AB[dataset]["top1_after_pct"],
            "Top-5": AB[dataset]["top5_after_pct"],
            "Top-10": AB[dataset]["top10_after_pct"],
        })

        sel = final[final["dataset"] == dataset].iloc[0]
        rows.append({
            "dataset": dataset,
            "method": "TCDV-TopoRT",
            "Reduction": float(sel["candidate_reduction_pct"]),
            "Top-1": float(sel["top1_after_pct"]),
            "Top-5": float(sel["top5_after_pct"]),
            "Top-10": float(sel["top10_after_pct"]),
        })

    figdf = pd.DataFrame(rows)
    figdf.to_csv(OUT / "candidate_filtering_final_summary_for_paper.csv", index=False)

    # Plot
    metrics = ["Reduction", "Top-1", "Top-5", "Top-10"]
    datasets = ["MetaboBase", "RIKEN-PlaSMA"]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)

    for ax, dataset in zip(axes, datasets):
        sub = figdf[figdf["dataset"] == dataset]
        abc = sub[sub["method"] == "ABCoRT-TL"].iloc[0]
        ours = sub[sub["method"] == "TCDV-TopoRT"].iloc[0]

        x = np.arange(len(metrics))
        w = 0.35

        ax.bar(
            x - w / 2,
            [abc[m] for m in metrics],
            width=w,
            label="ABCoRT-TL",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.65,
        )
        ax.bar(
            x + w / 2,
            [ours[m] for m in metrics],
            width=w,
            label="TCDV-TopoRT",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)")

        for i, m in enumerate(metrics):
            ax.text(i - w / 2, abc[m] + 1.1, f"{abc[m]:.1f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + w / 2, ours[m] + 1.1, f"{ours[m]:.1f}", ha="center", va="bottom", fontsize=7)

    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("RT-aware candidate filtering and reranking", y=1.03)

    plt.tight_layout()

    pdf = OUT / "fig_candidate_filtering_final_guarded_soft_summary.pdf"
    png = OUT / "fig_candidate_filtering_final_guarded_soft_summary.png"

    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n[SAVE]", pdf)
    print("[SAVE]", png)
    print("[SAVE]", OUT / "candidate_filtering_final_summary_for_paper.csv")
    print("[SAVE]", OUT / "candidate_filtering_final_guarded_soft_selected_rows.csv")

if __name__ == "__main__":
    main()

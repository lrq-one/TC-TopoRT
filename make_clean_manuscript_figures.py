#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
OUT = ROOT / "manuscript_figures_clean"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def savefig(name):
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    plt.tight_layout()
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()
    print("[SAVE]", pdf)
    print("[SAVE]", png)

# ============================================================
# Figure A: dual-view and fusion ablation
# ============================================================

def fig_dualview_fusion():
    df = pd.DataFrame({
        "variant": [
            "Original view",
            "Tautomer view",
            "Mean fusion",
            "OOF Huber stack"
        ],
        "mae": [25.278, 25.217, 25.059, 25.055],
        "std": [0.054, 0.070, 0.038, 0.039],
    })
    df["delta_vs_final"] = df["mae"] - 25.055
    df.to_csv(OUT / "source_dualview_fusion_ablation.csv", index=False)

    x = np.arange(len(df))
    plt.figure(figsize=(6.2, 3.6))
    bars = plt.bar(x, df["mae"], yerr=df["std"], capsize=3, edgecolor="black", linewidth=0.6)
    plt.axhline(25.055, linestyle="--", linewidth=1.0)
    plt.xticks(x, df["variant"], rotation=18, ha="right")
    plt.ylabel("MAE (s)")
    plt.ylim(24.9, 25.38)
    plt.title("Dual-view and prediction-level fusion ablation")

    for i, row in df.iterrows():
        plt.text(i, row["mae"] + row["std"] + 0.015,
                 f'{row["mae"]:.3f}', ha="center", va="bottom", fontsize=8)

    savefig("fig_dualview_fusion_ablation_clean")


# ============================================================
# Figure B: structural ablation
# Use true MAE values, not normalized radar.
# ============================================================

def fig_structural_ablation():
    df = pd.DataFrame({
        "variant": [
            "Full TCDV-TopoRT",
            "w/o explicit ring 2-cells",
            "w/o CWN message passing"
        ],
        "mae": [25.012, 25.102, 39.726],
    })
    base = df.loc[0, "mae"]
    df["delta"] = df["mae"] - base
    df.to_csv(OUT / "source_structural_ablation_seed5.csv", index=False)

    y = np.arange(len(df))
    plt.figure(figsize=(6.6, 3.2))
    plt.barh(y, df["mae"], edgecolor="black", linewidth=0.6)
    plt.yticks(y, df["variant"])
    plt.xlabel("MAE (s)")
    plt.title("Structural ablation on SMRT test set, seed 5")
    plt.xlim(0, 43)

    for i, row in df.iterrows():
        label = f'{row["mae"]:.3f}'
        if i > 0:
            label += f'  (+{row["delta"]:.3f})'
        plt.text(row["mae"] + 0.35, i, label, va="center", fontsize=8)

    plt.gca().invert_yaxis()
    savefig("fig_structural_ablation_mae_clean")


# ============================================================
# Figure C: external transfer with all baselines
# TCDV placed first, matching the revised table.
# ============================================================

def fig_external_transfer():
    df = pd.DataFrame({
        "Dataset": [
            "Eawag-XBridgeC18",
            "FEM-lipids",
            "FEM-long",
            "IPB-Halle",
            "LIFE-new",
            "LIFE-old",
        ],
        "TCDV": [47.218, 51.907, 88.493, 13.340, 13.341, 8.105],
        "ABCoRT-TL": [45.30, 85.46, 87.16, 13.81, 15.62, 9.97],
        "DeepGNN-TL": [45.97, 74.48, 110.89, 21.20, 18.13, 12.07],
        "RT-Tr-TL": [69.80, np.nan, 176.53, 27.97, 22.12, 13.09],
        "GIN": [55.25, 97.57, 135.44, 25.64, 21.93, 13.93],
    })
    df.to_csv(OUT / "source_external_transfer_all_baselines.csv", index=False)

    methods = ["TCDV", "ABCoRT-TL", "DeepGNN-TL", "RT-Tr-TL", "GIN"]
    x = np.arange(len(df))
    w = 0.15

    plt.figure(figsize=(8.6, 4.2))
    for j, m in enumerate(methods):
        vals = df[m].values
        xpos = x + (j - 2) * w
        plt.bar(xpos, vals, width=w, label=m, edgecolor="black", linewidth=0.35)

    plt.xticks(x, df["Dataset"], rotation=22, ha="right")
    plt.ylabel("MAE (s)")
    plt.title("External transfer performance across chromatographic datasets")
    plt.legend(frameon=False, ncol=3, fontsize=8)
    plt.ylim(0, 190)

    # Mark best value in each dataset
    for i, row in df.iterrows():
        vals = row[methods].astype(float)
        best_m = vals.idxmin()
        best_v = vals.min()
        j = methods.index(best_m)
        xpos = x[i] + (j - 2) * w
        plt.text(xpos, best_v + 3, "*", ha="center", va="bottom", fontsize=12)

    savefig("fig_external_transfer_all_baselines_clean")


# ============================================================
# Figure D: candidate filtering summary
# Not formula-level. This is the clean main-text candidate figure.
# ============================================================

def fig_candidate_filtering_summary():
    rows = [
        ["MetaboBase", "ABCoRT-TL", 38.35, 51.11, 73.33, 82.22],
        ["MetaboBase", "TCDV-TopoRT", 69.14, 55.56, 82.22, 88.89],
        ["RIKEN-PlaSMA", "ABCoRT-TL", 28.46, 52.94, 76.47, 83.53],
        ["RIKEN-PlaSMA", "TCDV-TopoRT", 46.23, 54.12, 77.65, 89.41],
    ]
    df = pd.DataFrame(rows, columns=["dataset", "method", "Reduction", "Top-1", "Top-5", "Top-10"])
    df.to_csv(OUT / "source_candidate_filtering_summary.csv", index=False)

    metrics = ["Reduction", "Top-1", "Top-5", "Top-10"]
    datasets = ["MetaboBase", "RIKEN-PlaSMA"]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        abc = sub[sub["method"] == "ABCoRT-TL"].iloc[0]
        ours = sub[sub["method"] == "TCDV-TopoRT"].iloc[0]

        x = np.arange(len(metrics))
        w = 0.35

        ax.bar(x - w/2, [abc[m] for m in metrics], width=w,
               label="ABCoRT-TL", edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, [ours[m] for m in metrics], width=w,
               label="TCDV-TopoRT", edgecolor="black", linewidth=0.5)

        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)")

        for i, m in enumerate(metrics):
            ax.text(i - w/2, abc[m] + 1.2, f"{abc[m]:.1f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + w/2, ours[m] + 1.2, f"{ours[m]:.1f}", ha="center", va="bottom", fontsize=7)

    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("RT-aware candidate filtering and reranking", y=1.03)
    savefig("fig_candidate_filtering_summary_clean")


# ============================================================
# Figure E: combined ablation overview
# This is better than radar if you want one ablation figure in main text.
# ============================================================

def fig_ablation_overview():
    dual = pd.DataFrame({
        "variant": ["Orig.", "Taut.", "Mean", "Stack"],
        "mae": [25.278, 25.217, 25.059, 25.055],
    })
    structural = pd.DataFrame({
        "variant": ["Full", "w/o ring 2-cells", "w/o CWN"],
        "mae": [25.012, 25.102, 39.726],
    })

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

    axes[0].bar(np.arange(len(dual)), dual["mae"], edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(np.arange(len(dual)))
    axes[0].set_xticklabels(dual["variant"], rotation=20, ha="right")
    axes[0].set_ylabel("MAE (s)")
    axes[0].set_title("(A) Dual-view/fusion ablation")
    axes[0].set_ylim(24.9, 25.4)
    for i, row in dual.iterrows():
        axes[0].text(i, row["mae"] + 0.015, f'{row["mae"]:.3f}', ha="center", fontsize=8)

    axes[1].bar(np.arange(len(structural)), structural["mae"], edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(np.arange(len(structural)))
    axes[1].set_xticklabels(structural["variant"], rotation=20, ha="right")
    axes[1].set_ylabel("MAE (s)")
    axes[1].set_title("(B) Structural ablation")
    axes[1].set_ylim(0, 43)
    for i, row in structural.iterrows():
        axes[1].text(i, row["mae"] + 0.6, f'{row["mae"]:.3f}', ha="center", fontsize=8)

    plt.tight_layout()
    savefig("fig_core_ablation_overview_clean")


def main():
    fig_dualview_fusion()
    fig_structural_ablation()
    fig_external_transfer()
    fig_candidate_filtering_summary()
    fig_ablation_overview()

    print("")
    print("DONE. Clean manuscript figures saved to:")
    print(OUT)
    print("")
    print("Recommended main-text figures:")
    print("  fig_core_ablation_overview_clean.pdf")
    print("  fig_external_transfer_all_baselines_clean.pdf")
    print("  fig_candidate_filtering_summary_clean.pdf")
    print("")
    print("Do NOT use the radar plot.")

if __name__ == "__main__":
    main()

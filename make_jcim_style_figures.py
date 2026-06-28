#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
RES = ROOT / "paper_results_TCDV_TopoRT"
OUT = ROOT / "manuscript_figures_jcim"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[SAVE]", OUT / f"{name}.pdf")
    print("[SAVE]", OUT / f"{name}.png")

# ============================================================
# Figure: SMRT benchmark MAE
# ============================================================

smrt = pd.DataFrame([
    ("GNN-RT", 39.87),
    ("1D-CNN", 34.70),
    ("MPNN", 31.50),
    ("CNN", 32.71),
    ("RT-Transformer", 27.30),
    ("DeepGCN-RT", 26.55),
    ("ABCoRT", 25.75),
    ("TC-TopoRT\nsingle-seed mean", 25.055),
    ("TC-TopoRT\n5-seed ensemble", 24.920),
], columns=["method", "mae"])

plt.figure(figsize=(7.2, 3.8))
x = np.arange(len(smrt))
bars = plt.bar(x, smrt["mae"], edgecolor="black", linewidth=0.6)
plt.ylabel("MAE (s)")
plt.xticks(x, smrt["method"], rotation=35, ha="right")
plt.ylim(20, max(smrt["mae"]) + 4)
for i, v in enumerate(smrt["mae"]):
    plt.text(i, v + 0.35, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
plt.title("SMRT benchmark performance")
savefig("fig_smrt_benchmark_mae")

# ============================================================
# Figure: dual-view fusion ablation
# ============================================================

dual = pd.read_csv(RES / "tables" / "03_dualview_and_fusion_ablation.csv")
plt.figure(figsize=(5.5, 3.6))
x = np.arange(len(dual))
plt.bar(x, dual["mae_mean"], yerr=dual["mae_std"], capsize=3, edgecolor="black", linewidth=0.6)
plt.ylabel("MAE (s)")
plt.xticks(x, dual["variant"], rotation=25, ha="right")
plt.ylim(24.9, 25.4)
for i, (m, s) in enumerate(zip(dual["mae_mean"], dual["mae_std"])):
    plt.text(i, m + 0.015, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
plt.title("Dual-view and prediction-level fusion ablation")
savefig("fig_dualview_fusion_ablation")

# ============================================================
# Figure: structural ablation
# ============================================================

struct = pd.read_csv(RES / "tables" / "04_structural_ablation_seed5.csv")
plt.figure(figsize=(5.8, 3.8))
x = np.arange(len(struct))
plt.bar(x, struct["mae"], edgecolor="black", linewidth=0.6)
plt.ylabel("MAE (s)")
plt.xticks(x, struct["variant"], rotation=20, ha="right")
plt.ylim(20, max(struct["mae"]) + 5)
for i, v in enumerate(struct["mae"]):
    plt.text(i, v + 0.5, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
plt.title("Topology-aware structural ablation")
savefig("fig_structural_ablation")

# ============================================================
# Figure: external transfer
# ============================================================

ext = pd.read_csv(RES / "tables" / "05_external_transfer_fixed_raw_autoselect.csv")
# robust column guess
cols = list(ext.columns)
dataset_col = [c for c in cols if "dataset" in c.lower()][0]
ours_col = [c for c in cols if ("ours" in c.lower() or "tcdv" in c.lower()) and "mae" in c.lower()][0]
abc_col = [c for c in cols if "abcort" in c.lower() and "mae" in c.lower()][0]

plt.figure(figsize=(7.0, 3.8))
x = np.arange(len(ext))
w = 0.36
plt.bar(x - w/2, ext[ours_col], width=w, label="TC-TopoRT", edgecolor="black", linewidth=0.5)
plt.bar(x + w/2, ext[abc_col], width=w, label="ABCoRT-TL reported", edgecolor="black", linewidth=0.5)
plt.ylabel("MAE (s)")
plt.xticks(x, ext[dataset_col], rotation=30, ha="right")
plt.legend(frameon=False)
plt.title("External transfer performance")
savefig("fig_external_transfer")

# ============================================================
# Figure: TL vs scratch improvement
# ============================================================

tl = pd.read_csv(RES / "tables" / "06_tl_vs_scratch_summary.csv")
tl = tl.sort_values("MAE_improvement_s", ascending=True)
plt.figure(figsize=(7.2, 4.0))
y = np.arange(len(tl))
plt.barh(y, tl["MAE_improvement_s"], edgecolor="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.8)
plt.yticks(y, tl["dataset_name"])
plt.xlabel("MAE improvement from transfer learning (s)")
plt.title("Transfer learning versus scratch training")
savefig("fig_tl_vs_scratch_improvement")

# ============================================================
# Figure: candidate filtering/reranking
# ============================================================

cf = pd.read_csv(RES / "tables" / "07_candidate_filtering_reranking_summary.csv")
metrics = ["reduction_percent", "top1_percent", "top5_percent", "top10_percent"]
metric_labels = ["Reduction", "Top-1", "Top-5", "Top-10"]

datasets = ["MetaboBase", "RIKEN_PlaSMA"]
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
for ax, ds in zip(axes, datasets):
    sub = cf[cf["dataset"] == ds].copy()
    abc = sub[sub["method"].str.contains("ABCoRT", case=False)].iloc[0]
    ours = sub[sub["method"].str.contains("Ours", case=False)].iloc[0]
    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w/2, [abc[m] for m in metrics], width=w, label="ABCoRT-TL reported", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, [ours[m] for m in metrics], width=w, label="TC-TopoRT", edgecolor="black", linewidth=0.5)
    ax.set_title(ds)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
plt.suptitle("RT-aware candidate filtering and reranking", y=1.04)
plt.tight_layout()
plt.savefig(OUT / "fig_candidate_filtering.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_candidate_filtering.png", dpi=300, bbox_inches="tight")
plt.close()
print("[SAVE]", OUT / "fig_candidate_filtering.pdf")
print("[SAVE]", OUT / "fig_candidate_filtering.png")

print("\nDONE. Figures saved to:", OUT)

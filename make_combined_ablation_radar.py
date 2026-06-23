#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
OUT = ROOT / "manuscript_figures_abcort_style"
OUT.mkdir(parents=True, exist_ok=True)

GWN = ROOT / "gwn"
ABL = ROOT / "ablations" / "gwn_cwn_structural_ablation"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SEED_DIRS = {
    "seed1-v1": GWN / "results_OOF_DualView_Stack_v1",
    "seed5": GWN / "results_OOF_DualView_Stack_seed5",
    "seed79": GWN / "results_OOF_DualView_Stack_seed79",
    "seed123": GWN / "results_OOF_DualView_Stack_seed123",
    "seed256": GWN / "results_OOF_DualView_Stack_seed256",
}

def calc_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    ae = np.abs(err)
    return {
        "MAE": ae.mean(),
        "MRE": np.mean(ae / np.maximum(np.abs(y), 1e-12)) * 100,
        "MedAE": np.median(ae),
        "MedRE": np.median(ae / np.maximum(np.abs(y), 1e-12)) * 100,
        "R2": 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2),
    }

def read_dualview_ablation():
    rows = []
    for seed, d in SEED_DIRS.items():
        p = d / "test_predictions.csv"
        if not p.exists():
            print("[MISS]", p)
            continue

        df = pd.read_csv(p)
        df["Mean_Fusion_Pred"] = (df["Origin_Test_Pred"] + df["Taut_Test_Pred"]) / 2.0

        variants = {
            "Original": "Origin_Test_Pred",
            "Tautomer": "Taut_Test_Pred",
            "Mean fusion": "Mean_Fusion_Pred",
            "OOF stack": "Final_Pred",
        }

        for name, col in variants.items():
            m = calc_metrics(df["Actual_RT"], df[col])
            rows.append({"seed": seed, "variant": name, **m})

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "source_dualview_radar_per_seed_metrics.csv", index=False)

    agg_rows = []
    order = ["Original", "Tautomer", "Mean fusion", "OOF stack"]
    for v in order:
        sub = per_seed[per_seed["variant"] == v]
        row = {"variant": v}
        for m in ["MAE", "MRE", "MedAE", "MedRE", "R2"]:
            row[m] = sub[m].mean()
            row[m + "_std"] = sub[m].std(ddof=1)
        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows)
    agg.to_csv(OUT / "source_dualview_radar_mean_metrics.csv", index=False)
    return agg

def read_structural_ablation():
    items = [
        ("Full", GWN / "results_OOF_DualView_Stack_seed5" / "final_metrics.json"),
        ("w/o ring 2-cells", ABL / "results_Ablation_No2Cell_DualView_Stack_seed5" / "final_metrics.json"),
        ("w/o CWN", ABL / "results_Ablation_CWN0_DualView_Stack_seed5" / "final_metrics.json"),
    ]

    rows = []
    for name, p in items:
        if not p.exists():
            print("[MISS]", p)
            continue
        m = json.loads(p.read_text())
        t = m["test_final"]
        rows.append({
            "variant": name,
            "MAE": t["mae"],
            "MRE": t["mre"],
            "MedAE": t["medae"],
            "MedRE": t["medre"],
            "R2": t["r2"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "source_structural_radar_metrics.csv", index=False)
    return df

def normalize_for_radar(df):
    metrics = ["MAE", "MRE", "MedAE", "MedRE", "R2"]
    lower_better = {"MAE", "MRE", "MedAE", "MedRE"}

    out = df.copy()
    for m in metrics:
        vals = out[m].astype(float).values
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)

        if abs(vmax - vmin) < 1e-12:
            out[m + "_score"] = 1.0
            continue

        if m in lower_better:
            score = (vmax - vals) / (vmax - vmin)
        else:
            score = (vals - vmin) / (vmax - vmin)

        # 保留 0.2 下限，避免最差模型缩成一个点，看起来太难读
        out[m + "_score"] = 0.20 + 0.80 * score

    return out

def draw_radar(ax, df, title):
    metrics = ["MAE", "MRE", "MedAE", "MedRE", "R2"]
    labels = ["MAE", "MRE", "MedAE", "MedRE", r"$R^2$"]

    score = normalize_for_radar(df)
    score.to_csv(OUT / f"source_{title.replace(' ', '_').replace('/', '_')}_normalized_scores.csv", index=False)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.03)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.grid(True, linewidth=0.7, alpha=0.6)

    for _, row in score.iterrows():
        values = [row[m + "_score"] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.7, label=row["variant"])
        ax.fill(angles, values, alpha=0.14)

    ax.set_title(title, y=1.08, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.40, 1.14), frameon=False, fontsize=8)

def main():
    dual = read_dualview_ablation()
    structural = read_structural_ablation()

    print("\n=== Dual-view/fusion metrics ===")
    print(dual.to_string(index=False))

    print("\n=== Structural metrics ===")
    print(structural.to_string(index=False))

    fig = plt.figure(figsize=(11.5, 5.4))
    ax1 = fig.add_subplot(121, polar=True)
    ax2 = fig.add_subplot(122, polar=True)

    draw_radar(ax1, dual, "(A) Dual-view/fusion ablation")
    draw_radar(ax2, structural, "(B) Structural ablation")

    fig.suptitle("Ablation radar analysis of TCDV-TopoRT", fontsize=14, y=1.04)
    plt.tight_layout()

    pdf = OUT / "fig_radar_ablation_combined.pdf"
    png = OUT / "fig_radar_ablation_combined.png"

    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n[SAVE]", pdf)
    print("[SAVE]", png)

if __name__ == "__main__":
    main()

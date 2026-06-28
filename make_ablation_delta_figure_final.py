#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
OUT = ROOT / "manuscript_figures_final"
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

def load_full_seed5():
    p = GWN / "results_OOF_DualView_Stack_seed5" / "final_metrics.json"
    m = json.loads(p.read_text())
    return m["test_final"]

def load_no2cell():
    p = ABL / "results_Ablation_No2Cell_DualView_Stack_seed5" / "final_metrics.json"
    m = json.loads(p.read_text())
    return m["test_final"]

def load_cwn0():
    p = ABL / "results_Ablation_CWN0_DualView_Stack_seed5" / "final_metrics.json"
    m = json.loads(p.read_text())
    return m["test_final"]

def single_view_5seed_metrics():
    seed_dirs = [
        GWN / "results_OOF_DualView_Stack_v1",
        GWN / "results_OOF_DualView_Stack_seed5",
        GWN / "results_OOF_DualView_Stack_seed79",
        GWN / "results_OOF_DualView_Stack_seed123",
        GWN / "results_OOF_DualView_Stack_seed256",
    ]

    rows = []
    for d in seed_dirs:
        p = d / "test_predictions.csv"
        df = pd.read_csv(p)
        y = df["Actual_RT"].to_numpy(float)
        for name, col in [
            ("Original view only", "Origin_Test_Pred"),
            ("Tautomer view only", "Taut_Test_Pred"),
        ]:
            pred = df[col].to_numpy(float)
            mae = np.mean(np.abs(pred - y))
            medae = np.median(np.abs(pred - y))
            rmse = np.sqrt(np.mean((pred - y) ** 2))
            r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
            rows.append({
                "variant": name,
                "seed_dir": d.name,
                "mae": mae,
                "medae": medae,
                "rmse": rmse,
                "r2": r2,
            })

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("variant", as_index=False)
          .agg(
              mae=("mae", "mean"),
              mae_std=("mae", "std"),
              medae=("medae", "mean"),
              rmse=("rmse", "mean"),
              r2=("r2", "mean"),
          )
    )
    return agg

def main():
    sv = single_view_5seed_metrics()

    full = load_full_seed5()
    no2 = load_no2cell()
    cwn0 = load_cwn0()

    # 用 seed5 Full 作为结构消融同 seed 基准
    full_mae = float(full["mae"])

    rows = []

    # single-view 是 5 seed mean
    for _, r in sv.iterrows():
        rows.append({
            "variant": r["variant"],
            "mae": float(r["mae"]),
            "mae_std": float(r["mae_std"]),
            "group": "single-view/fusion",
            "note": "five-seed mean of single-view 5-fold predictions",
        })

    rows.extend([
        {
            "variant": "Full TC-TopoRT",
            "mae": full_mae,
            "mae_std": 0.0,
            "group": "full",
            "note": "seed5 final OOF-stacked model",
        },
        {
            "variant": "w/o ring 2-cells",
            "mae": float(no2["mae"]),
            "mae_std": 0.0,
            "group": "structural",
            "note": "seed5, max ring size set to 2",
        },
        {
            "variant": "w/o CWN",
            "mae": float(cwn0["mae"]),
            "mae_std": 0.0,
            "group": "structural",
            "note": "seed5, CWN layers set to 0",
        },
    ])

    df = pd.DataFrame(rows)

    order = [
        "Original view only",
        "Tautomer view only",
        "w/o ring 2-cells",
        "Full TC-TopoRT",
        "w/o CWN",
    ]
    df["order"] = df["variant"].map({v:i for i,v in enumerate(order)})
    df = df.sort_values("order").drop(columns=["order"])

    df["delta_mae_vs_full"] = df["mae"] - full_mae
    df.to_csv(OUT / "source_ablation_delta_final.csv", index=False)

    print("\n=== Ablation metrics ===")
    print(df[["variant", "mae", "mae_std", "delta_mae_vs_full", "note"]].to_string(index=False))

    # ---------------------------
    # Figure: two-panel
    # ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))

    # Panel A: zoomed MAE near 25
    small_df = df[df["variant"] != "w/o CWN"].copy()
    x = np.arange(len(small_df))

    axes[0].bar(
        x,
        small_df["mae"],
        yerr=small_df["mae_std"].replace(0, np.nan),
        capsize=3,
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0].axhline(full_mae, linestyle="--", linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        ["Original", "Tautomer", "w/o ring\n2-cells", "Full"],
        rotation=0,
        ha="center",
    )
    axes[0].set_ylabel("MAE (s)")
    axes[0].set_title("(A) Zoomed comparison near Full model")
    axes[0].set_ylim(24.95, 25.35)

    for i, r in small_df.reset_index(drop=True).iterrows():
        axes[0].text(i, r["mae"] + 0.018, f'{r["mae"]:.3f}', ha="center", va="bottom", fontsize=8)

    axes[0].text(
        3, full_mae - 0.035,
        "Full baseline",
        ha="center", va="top", fontsize=7
    )

    # Panel B: Delta MAE relative to Full
    delta_df = df[df["variant"] != "Full TC-TopoRT"].copy()
    y = np.arange(len(delta_df))

    axes[1].barh(
        y,
        delta_df["delta_mae_vs_full"],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(
        ["Original", "Tautomer", "w/o ring 2-cells", "w/o CWN"]
    )
    axes[1].set_xlabel(r"$\Delta$MAE vs Full (s)")
    axes[1].set_title("(B) Error increase after removing components")
    axes[1].set_xlim(0, max(delta_df["delta_mae_vs_full"]) * 1.12)

    for i, r in delta_df.reset_index(drop=True).iterrows():
        val = r["delta_mae_vs_full"]
        axes[1].text(val + 0.15, i, f'+{val:.3f}', va="center", fontsize=8)

    axes[1].invert_yaxis()

    plt.tight_layout()

    pdf = OUT / "fig_ablation_delta_mae_final.pdf"
    png = OUT / "fig_ablation_delta_mae_final.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n[SAVE]", pdf)
    print("[SAVE]", png)

if __name__ == "__main__":
    main()

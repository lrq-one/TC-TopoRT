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

SEED_DIRS = {
    "seed1-v1": GWN / "results_OOF_DualView_Stack_v1",
    "seed5": GWN / "results_OOF_DualView_Stack_seed5",
    "seed79": GWN / "results_OOF_DualView_Stack_seed79",
    "seed123": GWN / "results_OOF_DualView_Stack_seed123",
    "seed256": GWN / "results_OOF_DualView_Stack_seed256",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def calc_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    ae = np.abs(err)
    return {
        "MAE": float(ae.mean()),
        "MRE": float(np.mean(ae / np.maximum(np.abs(y), 1e-12)) * 100.0),
        "MedAE": float(np.median(ae)),
        "MedRE": float(np.median(ae / np.maximum(np.abs(y), 1e-12)) * 100.0),
        "R2": float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def read_dual_metrics():
    rows = []
    for seed, d in SEED_DIRS.items():
        p = d / "test_predictions.csv"
        if not p.exists():
            print("[MISS]", p)
            continue

        df = pd.read_csv(p)
        for variant, col in {
            "Original": "Origin_Test_Pred",
            "Tautomer": "Taut_Test_Pred",
            "Full": "Final_Pred",
        }.items():
            m = calc_metrics(df["Actual_RT"], df[col])
            rows.append({"seed": seed, "variant": variant, **m})

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "source_ablation_radar_dual_per_seed.csv", index=False)

    agg = []
    for variant in ["Original", "Tautomer", "Full"]:
        sub = per_seed[per_seed["variant"] == variant]
        row = {"variant": variant}
        for m in ["MAE", "MRE", "MedAE", "MedRE", "R2"]:
            row[m] = sub[m].mean()
            row[m + "_std"] = sub[m].std(ddof=1)
        agg.append(row)

    return pd.DataFrame(agg)


def read_structural_metrics():
    rows = []

    # w/o ring 2-cells
    p = ABL / "results_Ablation_No2Cell_DualView_Stack_seed5" / "final_metrics.json"
    if p.exists():
        m = json.loads(p.read_text())["test_final"]
        rows.append({
            "variant": "w/o ring 2-cells",
            "MAE": float(m["mae"]),
            "MRE": float(m["mre"]),
            "MedAE": float(m["medae"]),
            "MedRE": float(m["medre"]),
            "R2": float(m["r2"]),
        })
    else:
        print("[MISS]", p)

    # w/o CWN
    p = ABL / "results_Ablation_CWN0_DualView_Stack_seed5" / "final_metrics.json"
    if p.exists():
        m = json.loads(p.read_text())["test_final"]
        rows.append({
            "variant": "w/o CWN",
            "MAE": float(m["mae"]),
            "MRE": float(m["mre"]),
            "MedAE": float(m["medae"]),
            "MedRE": float(m["medre"]),
            "R2": float(m["r2"]),
        })
    else:
        print("[MISS]", p)

    return pd.DataFrame(rows)


def scale_value(v, metric):
    """
    Convert raw metric value to radius [0.15, 1.0].
    This is ABCoRT-like raw-axis radar:
    MAE/MRE/MedAE/MedRE are raw error magnitudes; larger means worse but appears farther out.
    R2 larger means better and appears farther out.
    """
    ranges = {
        "MAE":   (24.5, 40.5),
        "MRE":   (3.10, 5.10),
        "MedAE": (11.0, 23.0),
        "MedRE": (1.40, 2.90),
        "R2":    (0.84, 0.902),
    }
    lo, hi = ranges[metric]
    r = (float(v) - lo) / (hi - lo)
    r = max(0.0, min(1.0, r))
    return 0.15 + 0.85 * r


def make_radar(df):
    metrics = ["MAE", "MRE", "MedAE", "MedRE", "R2"]
    labels = ["MAE(s)", "MRE(%)", "MedAE(s)", "MedRE(%)", r"$R^2$"]

    # 图里保留这五条：不要 Mean fusion；Full = OOF stack
    order = ["Original", "Tautomer", "Full", "w/o ring 2-cells", "w/o CWN"]
    df["order"] = df["variant"].map({v: i for i, v in enumerate(order)})
    df = df.sort_values("order").drop(columns=["order"])
    df.to_csv(OUT / "source_ablation_radar_final_raw_metrics.csv", index=False)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    colors = {
        "Original": "#4C78A8",
        "Tautomer": "#72B7B2",
        "Full": "#E45756",
        "w/o ring 2-cells": "#F58518",
        "w/o CWN": "#54A24B",
    }

    plt.figure(figsize=(6.4, 5.8))
    ax = plt.subplot(111, polar=True)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)

    # 不显示 0.2/0.4 这种归一化刻度
    ax.set_yticks([0.25, 0.45, 0.65, 0.85, 1.0])
    ax.set_yticklabels([])
    ax.grid(True, linewidth=0.7, alpha=0.55)

    # 每个轴写自己的真实刻度，仿你师兄那种
    tick_values = {
        "MAE":   [25, 30, 35, 40],
        "MRE":   [3.2, 3.8, 4.4, 5.0],
        "MedAE": [12, 15, 19, 22],
        "MedRE": [1.5, 1.9, 2.4, 2.8],
        "R2":    [0.85, 0.87, 0.89, 0.90],
    }

    for i, metric in enumerate(metrics):
        theta = angles[i]
        for tv in tick_values[metric]:
            rr = scale_value(tv, metric)
            if metric == "R2":
                txt = f"{tv:.2f}"
            elif metric == "MRE" or metric == "MedRE":
                txt = f"{tv:.1f}"
            else:
                txt = f"{tv:g}"
            ax.text(theta, rr, txt, fontsize=7, ha="center", va="center", alpha=0.75)

    for _, row in df.iterrows():
        variant = row["variant"]
        radii = [scale_value(row[m], m) for m in metrics]
        radii += radii[:1]

        ax.plot(
            angles,
            radii,
            label=variant,
            color=colors.get(variant, None),
            linewidth=2.0 if variant == "Full" else 1.7,
        )
        ax.fill(
            angles,
            radii,
            color=colors.get(variant, None),
            alpha=0.13 if variant != "w/o CWN" else 0.18,
        )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.28, 1.10),
        frameon=True,
        fontsize=8,
    )

    ax.set_title("Ablation analysis of TCDV-TopoRT", y=1.10, fontsize=13)

    pdf = OUT / "fig_ablation_radar_like_abcort_final.pdf"
    png = OUT / "fig_ablation_radar_like_abcort_final.png"

    plt.tight_layout()
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()

    print("[SAVE]", pdf)
    print("[SAVE]", png)


def main():
    dual = read_dual_metrics()
    structural = read_structural_metrics()
    df = pd.concat([dual, structural], ignore_index=True)

    print("\n=== metrics used ===")
    print(df[["variant", "MAE", "MRE", "MedAE", "MedRE", "R2"]].to_string(index=False))

    make_radar(df)


if __name__ == "__main__":
    main()

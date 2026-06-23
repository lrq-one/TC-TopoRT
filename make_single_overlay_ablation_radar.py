#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path.cwd()
OUT = ROOT / "manuscript_figures_abcort_style"
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

METRICS = ["MAE", "MRE", "MedAE", "MedRE", "R2"]
LABELS = ["MAE", "MRE", "MedAE", "MedRE", r"$R^2$"]
LOWER_BETTER = {"MAE", "MRE", "MedAE", "MedRE"}


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


def read_dualview_fusion_metrics():
    rows = []

    for seed, d in SEED_DIRS.items():
        p = d / "test_predictions.csv"
        if not p.exists():
            print("[MISS]", p)
            continue

        df = pd.read_csv(p)
        required = ["Actual_RT", "Origin_Test_Pred", "Taut_Test_Pred", "Final_Pred"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print("[SKIP]", p, "missing columns:", missing)
            continue

        df["Mean_Fusion_Pred"] = (df["Origin_Test_Pred"] + df["Taut_Test_Pred"]) / 2.0

        variant_cols = {
            "Original": "Origin_Test_Pred",
            "Tautomer": "Taut_Test_Pred",
            "Mean fusion": "Mean_Fusion_Pred",
            "OOF stack": "Final_Pred",
        }

        for variant, col in variant_cols.items():
            m = calc_metrics(df["Actual_RT"], df[col])
            rows.append({
                "group": "Dual-view/fusion",
                "variant": variant,
                "seed": seed,
                **m,
            })

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "source_single_overlay_dualview_per_seed_metrics.csv", index=False)

    agg_rows = []
    order = ["Original", "Tautomer", "Mean fusion", "OOF stack"]
    for variant in order:
        sub = per_seed[per_seed["variant"] == variant]
        row = {
            "group": "Dual-view/fusion",
            "variant": variant,
        }
        for m in METRICS:
            row[m] = sub[m].mean()
            row[m + "_std"] = sub[m].std(ddof=1)
        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def read_structural_metrics():
    items = [
        ("Full", GWN / "results_OOF_DualView_Stack_seed5" / "final_metrics.json"),
        ("w/o ring 2-cells", ABL / "results_Ablation_No2Cell_DualView_Stack_seed5" / "final_metrics.json"),
        ("w/o CWN", ABL / "results_Ablation_CWN0_DualView_Stack_seed5" / "final_metrics.json"),
    ]

    rows = []
    for variant, p in items:
        if not p.exists():
            print("[MISS]", p)
            continue

        m = json.loads(p.read_text())
        t = m["test_final"]

        rows.append({
            "group": "Structural",
            "variant": variant,
            "MAE": float(t["mae"]),
            "MRE": float(t["mre"]),
            "MedAE": float(t["medae"]),
            "MedRE": float(t["medre"]),
            "R2": float(t["r2"]),
        })

    return pd.DataFrame(rows)


def normalize_scores(df, mode):
    """
    mode='global':
        all variants are normalized together.
        This is more mathematically direct, but w/o CWN is so poor that the other curves may cluster.
    mode='group':
        dual-view/fusion variants and structural variants are normalized within their own ablation family.
        This is closer to an ablation-visualization figure and easier to read.
    """
    out = df.copy()

    for m in METRICS:
        out[m + "_score"] = np.nan

    if mode == "global":
        groups = [("__ALL__", out.index)]
    elif mode == "group":
        groups = [(g, out[out["group"] == g].index) for g in out["group"].unique()]
    else:
        raise ValueError("mode must be 'global' or 'group'")

    for _, idx in groups:
        sub = out.loc[idx]

        for m in METRICS:
            vals = sub[m].astype(float).values
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)

            if abs(vmax - vmin) < 1e-12:
                score = np.ones_like(vals)
            else:
                if m in LOWER_BETTER:
                    score = (vmax - vals) / (vmax - vmin)
                else:
                    score = (vals - vmin) / (vmax - vmin)

            # 保留 0.20 下限，避免最差曲线完全缩成中心点
            score = 0.20 + 0.80 * score
            out.loc[idx, m + "_score"] = score

    return out


def plot_single_overlay(df, mode, out_name):
    score = normalize_scores(df, mode=mode)
    score.to_csv(OUT / f"source_{out_name}_normalized_scores.csv", index=False)

    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    # 颜色：dual/fusion 用蓝绿紫红；structural 用黑橙棕，线型 dashed
    style = {
        "Original":             {"color": "#4C78A8", "ls": "-",  "lw": 1.8},
        "Tautomer":             {"color": "#72B7B2", "ls": "-",  "lw": 1.8},
        "Mean fusion":          {"color": "#54A24B", "ls": "-",  "lw": 1.8},
        "OOF stack":            {"color": "#E45756", "ls": "-",  "lw": 2.2},
        "Full":                 {"color": "#111111", "ls": "--", "lw": 2.0},
        "w/o ring 2-cells":     {"color": "#F58518", "ls": "--", "lw": 2.0},
        "w/o CWN":              {"color": "#8C564B", "ls": "--", "lw": 2.0},
    }

    plt.figure(figsize=(7.2, 6.2))
    ax = plt.subplot(111, polar=True)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LABELS, fontsize=10)

    ax.set_ylim(0.0, 1.03)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(True, linewidth=0.7, alpha=0.65)

    for _, row in score.iterrows():
        variant = row["variant"]
        values = [row[m + "_score"] for m in METRICS]
        values += values[:1]

        st = style.get(variant, {"color": None, "ls": "-", "lw": 1.6})
        ax.plot(
            angles,
            values,
            label=variant,
            color=st["color"],
            linestyle=st["ls"],
            linewidth=st["lw"],
        )
        ax.fill(
            angles,
            values,
            color=st["color"],
            alpha=0.08 if row["group"] == "Dual-view/fusion" else 0.05,
        )

    if mode == "group":
        title = "Ablation radar analysis of TCDV-TopoRT"
        subtitle = "scores normalized within each ablation group"
    else:
        title = "Ablation radar analysis of TCDV-TopoRT"
        subtitle = "scores normalized across all variants"

    ax.set_title(title + "\n" + subtitle, y=1.12, fontsize=13)

    # 两个 legend：一个解释颜色曲线，一个解释线型分组
    leg1 = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.42, 1.12),
        frameon=False,
        fontsize=8,
        title="Variants",
        title_fontsize=9,
    )
    ax.add_artist(leg1)

    group_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="Dual-view/fusion"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Structural"),
    ]
    ax.legend(
        handles=group_handles,
        loc="lower right",
        bbox_to_anchor=(1.40, 0.04),
        frameon=False,
        fontsize=8,
        title="Ablation group",
        title_fontsize=9,
    )

    pdf = OUT / f"{out_name}.pdf"
    png = OUT / f"{out_name}.png"

    plt.tight_layout()
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()

    print("[SAVE]", pdf)
    print("[SAVE]", png)


def main():
    dual = read_dualview_fusion_metrics()
    structural = read_structural_metrics()

    df = pd.concat([dual, structural], ignore_index=True)
    df.to_csv(OUT / "source_ablation_single_overlay_raw_metrics.csv", index=False)

    print("\n=== Raw metrics used for single overlay radar ===")
    print(df[["group", "variant"] + METRICS].to_string(index=False))

    # 推荐：组内归一化，视觉上最像消融雷达图
    plot_single_overlay(
        df,
        mode="group",
        out_name="fig_radar_ablation_single_overlay_groupnorm",
    )

    # 备用：全局归一化，更严格，但 dual/fusion 差异会被 w/o CWN 压缩
    plot_single_overlay(
        df,
        mode="global",
        out_name="fig_radar_ablation_single_overlay_globalnorm",
    )

    print("\nDONE.")
    print("Recommended for manuscript:")
    print(OUT / "fig_radar_ablation_single_overlay_groupnorm.pdf")
    print("\nBackup global-normalized version:")
    print(OUT / "fig_radar_ablation_single_overlay_globalnorm.pdf")


if __name__ == "__main__":
    main()

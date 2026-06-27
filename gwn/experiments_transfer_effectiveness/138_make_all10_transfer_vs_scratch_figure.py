#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("experiments_transfer_effectiveness/all10_transfer_vs_scratch_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Final comparison values
# ============================================================

DATA = [
    ["FEM_short_73",            73,  99.439, 67.499],
    ["UniToyama_Atlantis_143", 143,  72.114, 57.395],
    ["FEM_long_412",           412, 117.154, 88.493],
    ["Eawag_XBridgeC18_364",   364,  58.512, 47.218],
    ["LIFE_old_194",           194,  11.604,  8.105],
    ["MTBLS87_147",            147,  69.444, 69.010],
    ["LIFE_new_184",           184,  14.701, 13.341],
    ["Cao_HILIC_116",          116,  67.761, 71.592],
    ["IPB_Halle_82",            82,  13.051, 13.340],
    ["FEM_lipids_72",           72,  55.761, 51.907],
]

COLS = ["dataset_name", "n", "scratch_mae", "transfer_mae"]

# ============================================================
# Color palette: matched to your architecture figure style
# ============================================================

COLOR_POS = "#6F93C6"      # muted steel blue
COLOR_NEG = "#9C88B8"      # muted purple
COLOR_EDGE = "#4F5D73"     # soft dark gray-blue
COLOR_GRID = "#D9DEE7"     # light gray-blue
COLOR_AXIS = "#666666"
COLOR_TEXT = "#2F2F2F"
COLOR_ZERO = "#7A7A7A"


def make_display_name(name: str) -> str:
    # cleaner display names for figure
    return name.replace("_", " ")


def save_plain_markdown_table(df: pd.DataFrame, path: Path):
    float_cols = {"scratch_mae", "transfer_mae", "mae_improvement"}
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(df.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(df.columns)) + " |\n")
        for _, r in df.iterrows():
            vals = []
            for c in df.columns:
                v = r[c]
                if c in float_cols:
                    vals.append(f"{float(v):.3f}")
                elif c == "n":
                    vals.append(str(int(v)))
                elif c == "transfer_better":
                    vals.append("True" if bool(v) else "False")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")


def main():
    df = pd.DataFrame(DATA, columns=COLS)
    df["mae_improvement"] = df["scratch_mae"] - df["transfer_mae"]
    df["transfer_better"] = df["mae_improvement"] > 0
    df["display_name"] = df["dataset_name"].map(make_display_name)

    out_df = df[[
        "dataset_name", "n", "scratch_mae", "transfer_mae",
        "mae_improvement", "transfer_better"
    ]].copy()

    csv_path = OUT_DIR / "tl_vs_scratch_summary.csv"
    md_path = OUT_DIR / "tl_vs_scratch_summary.md"
    txt_path = OUT_DIR / "tl_vs_scratch_summary.txt"
    overall_path = OUT_DIR / "tl_vs_scratch_overall.csv"

    out_df.to_csv(csv_path, index=False)
    save_plain_markdown_table(out_df, md_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(out_df.to_string(index=False))
        f.write("\n")

    overall = {
        "num_datasets": int(len(df)),
        "transfer_better_count": int(df["transfer_better"].sum()),
        "scratch_better_count": int((~df["transfer_better"]).sum()),
        "mean_mae_improvement": float(df["mae_improvement"].mean()),
        "median_mae_improvement": float(df["mae_improvement"].median()),
    }
    pd.DataFrame([overall]).to_csv(overall_path, index=False)

    # ========================================================
    # Plot
    # ========================================================
    plot_df = df.sort_values("mae_improvement", ascending=False).reset_index(drop=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": COLOR_AXIS,
        "xtick.color": COLOR_AXIS,
        "ytick.color": COLOR_AXIS,
        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(8.8, 5.4), facecolor="white")
    ax.set_facecolor("white")

    y = np.arange(len(plot_df))
    vals = plot_df["mae_improvement"].values
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in vals]

    bars = ax.barh(
        y,
        vals,
        color=colors,
        edgecolor=COLOR_EDGE,
        linewidth=0.7,
        height=0.66,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_name"], fontsize=9)
    ax.invert_yaxis()

    # zero line
    ax.axvline(0, color=COLOR_ZERO, linewidth=1.0, zorder=2)

    # x label only, no in-figure title
    ax.set_xlabel("MAE improvement from transfer learning (s)", fontsize=10)

    # subtle grid
    ax.grid(axis="x", linestyle="--", linewidth=0.6, color=COLOR_GRID, alpha=0.9, zorder=1)
    ax.grid(axis="y", visible=False)

    # cleaner spines
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(COLOR_AXIS)

    xmin = min(vals.min(), 0)
    xmax = max(vals.max(), 0)
    ax.set_xlim(xmin - 2.6, xmax + 4.0)

    # Value labels
    for bar, v in zip(bars, vals):
        ymid = bar.get_y() + bar.get_height() / 2
        if v >= 0:
            ax.text(
                v + 0.45, ymid, f"{v:+.3f}",
                va="center", ha="left", fontsize=8, color=COLOR_TEXT
            )
        else:
            ax.text(
                v - 0.45, ymid, f"{v:+.3f}",
                va="center", ha="right", fontsize=8, color=COLOR_TEXT
            )

    # ticks
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # layout
    plt.subplots_adjust(left=0.31, right=0.97, top=0.96, bottom=0.16)

    png_path = OUT_DIR / "fig_tl_vs_scratch_bar.png"
    pdf_path = OUT_DIR / "fig_tl_vs_scratch_bar.pdf"
    plt.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close()

    caption = (
        "Comparison between transfer learning and scratch training across the all10 external chromatographic datasets. "
        "Bars show MAE improvement, defined as scratch MAE minus transfer-learning MAE. "
        "Positive values indicate better performance from transfer learning."
    )
    (OUT_DIR / "figure_caption.txt").write_text(caption + "\n", encoding="utf-8")

    print("\n=== TL vs scratch summary ===")
    print(out_df.to_string(index=False))

    print("\n=== Overall ===")
    for k, v in overall.items():
        print(f"{k}: {v}")

    print("\n[SAVE]", csv_path)
    print("[SAVE]", md_path)
    print("[SAVE]", txt_path)
    print("[SAVE]", overall_path)
    print("[SAVE]", png_path)
    print("[SAVE]", pdf_path)
    print("[SAVE]", OUT_DIR / "figure_caption.txt")


if __name__ == "__main__":
    main()

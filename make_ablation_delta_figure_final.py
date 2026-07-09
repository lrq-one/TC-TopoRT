from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(".").resolve()
OUT_DIR = ROOT / "manuscript_figures_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "fig_ablation_delta_mae_final.png"
OUT_PDF = OUT_DIR / "fig_ablation_delta_mae_final.pdf"

CANDIDATES = [
    ROOT / "manuscript_figures_final/source_ablation_delta_final.csv",
    ROOT / "gwn/final_paper_tables/source_ablation_delta_final.csv",
]

# ------------------------------------------------------------
# visual style: keep close to your old blue-toned figure
# ------------------------------------------------------------
COLOR_MAP = {
    "Original view only": "#1f78b4",
    "Tautomer view only": "#1f78b4",
    "w/o ring 2-cells": "#1f78b4",
    "Full TC-TopoRT": "#1f78b4",
    "Full model": "#1f78b4",
    "w/o CWN": "#1f78b4",
}

EDGE = "#1c4e6e"
GRID = "#d9d9d9"

DISPLAY_MAP = {
    "Original view only": "Original",
    "Tautomer view only": "Tautomer",
    "w/o ring 2-cells": "w/o ring\n2-cells",
    "Full TC-TopoRT": "Full",
    "Full model": "Full",
    "w/o CWN": "w/o CWN",
}

LEFT_ORDER = [
    "Original view only",
    "Tautomer view only",
    "w/o ring 2-cells",
    "Full TC-TopoRT",
]

RIGHT_ORDER = [
    "Original view only",
    "Tautomer view only",
    "w/o ring 2-cells",
    "w/o CWN",
]


def find_input():
    for p in CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find ablation source CSV.\n"
        "Expected one of:\n" + "\n".join(str(p) for p in CANDIDATES)
    )


def load_df():
    p = find_input()
    df = pd.read_csv(p)

    if "variant" not in df.columns or "mae" not in df.columns:
        raise RuntimeError(f"Unexpected columns in {p}: {df.columns.tolist()}")

    # normalize one naming variant
    df["variant"] = df["variant"].replace({"Full model": "Full TC-TopoRT"})
    if "mae_std" not in df.columns:
        df["mae_std"] = 0.0

    return p, df


def get_row(df, name):
    sub = df[df["variant"] == name]
    if len(sub) == 0:
        raise KeyError(f"Missing variant: {name}")
    return sub.iloc[0]


def main():
    in_path, df = load_df()

    full_mae = float(get_row(df, "Full TC-TopoRT")["mae"])

    left_vals, left_std, left_labels, left_colors = [], [], [], []
    for name in LEFT_ORDER:
        row = get_row(df, name)
        left_vals.append(float(row["mae"]))
        left_std.append(float(row.get("mae_std", 0.0)))
        left_labels.append(DISPLAY_MAP[name])
        left_colors.append(COLOR_MAP[name])

    right_vals, right_labels, right_colors = [], [], []
    for name in RIGHT_ORDER:
        row = get_row(df, name)
        delta = float(row["mae"]) - full_mae
        right_vals.append(delta)
        right_labels.append(DISPLAY_MAP[name])
        right_colors.append(COLOR_MAP.get(name, "#1f78b4"))

    plt.close("all")
    fig = plt.figure(figsize=(7.0, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.33)

    # ============================================================
    # Panel A
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    x1 = np.arange(len(left_vals))
    bars1 = ax1.bar(
        x1,
        left_vals,
        yerr=left_std,
        color=left_colors,
        edgecolor=EDGE,
        linewidth=0.8,
        capsize=2.5,
        width=0.62,
        zorder=3,
    )

    baseline = full_mae
    ymin = min(min(left_vals), baseline) - 0.06
    ymax = max(left_vals) + max(left_std + [0]) + 0.06

    ax1.axhline(
        baseline,
        color="#8ebad3",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )

    ax1.set_ylim(ymin, ymax)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(left_labels, fontsize=7.2)
    ax1.set_ylabel("MAE (s)", fontsize=7.8)
    ax1.set_title("(A) Zoomed comparison near Full model", fontsize=7.8, pad=4)

    ax1.grid(axis="y", linestyle="--", linewidth=0.5, color=GRID, alpha=0.7, zorder=0)
    ax1.set_axisbelow(True)

    for rect, val in zip(bars1, left_vals):
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.008,
            f"{val:.3f}" if val < 25.1 else f"{val:.3f}".rstrip("0").rstrip("."),
            ha="center",
            va="bottom",
            fontsize=6.9,
            color="#222222",
        )

    # ============================================================
    # Panel B
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])
    y2 = np.arange(len(right_vals))

    bars2 = ax2.barh(
        y2,
        right_vals,
        color=right_colors,
        edgecolor=EDGE,
        linewidth=0.8,
        height=0.58,
        zorder=3,
    )

    ax2.set_yticks(y2)
    ax2.set_yticklabels(right_labels, fontsize=7.2)
    ax2.invert_yaxis()
    ax2.set_xlabel(r"$\Delta$MAE vs Full (s)", fontsize=7.8)
    ax2.set_title("(B) Error increase after removing components", fontsize=7.8, pad=4)

    xmax = max(right_vals) + 3.6
    ax2.set_xlim(0, xmax)
    ax2.set_xticks(np.arange(0, np.ceil(xmax) + 0.1, 2))
    ax2.grid(axis="x", linestyle="--", linewidth=0.5, color=GRID, alpha=0.7, zorder=0)
    ax2.set_axisbelow(True)

    # important: label stays INSIDE plotting area with visible margin
    right_margin = 1.8
    outside_offset = 0.14
    inside_offset = 0.22

    for rect, val in zip(bars2, right_vals):
        y = rect.get_y() + rect.get_height() / 2
        txt = f"+{val:.3f}"

        proposed = val + outside_offset
        if proposed <= xmax - right_margin:
            # normal case: put outside bar, but still inside frame
            ax2.text(
                proposed,
                y,
                txt,
                ha="left",
                va="center",
                fontsize=6.9,
                color="#222222",
                clip_on=True,
            )
        else:
            # if too close to border, place slightly inside the bar end
            ax2.text(
                max(val - inside_offset, 0.05),
                y,
                txt,
                ha="right",
                va="center",
                fontsize=6.9,
                color="white",
                bbox=dict(facecolor=COLOR_MAP.get("w/o CWN", "#1f78b4"), edgecolor="#1c567a", pad=0.18),
            )

    # spine / ticks
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#444444")
        ax.tick_params(axis="both", labelsize=7.0, length=3.0, width=0.8)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.18)

    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.02)

    print("[INPUT]", in_path.relative_to(ROOT))
    print("[SAVE]", OUT_PDF.relative_to(ROOT))
    print("[SAVE]", OUT_PNG.relative_to(ROOT))
    print()
    print(df[["variant", "mae", "mae_std"]].to_string(index=False))


if __name__ == "__main__":
    main()

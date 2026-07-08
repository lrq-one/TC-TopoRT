from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(".").resolve()
OUT_DIR = ROOT / "manuscript_figures_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "fig_threshold_sensitivity_candidate_filtering.pdf"
OUT_PNG = OUT_DIR / "fig_threshold_sensitivity_candidate_filtering.png"

DATASETS = [
    {
        "panel": "A",
        "name": "MetaboBase",
        "x": [30, 45, 60, 75, 90],
        "reduction": [81.48, 74.86, 69.14, 64.37, 60.01],
        "retention": [80.00, 88.89, 93.33, 93.33, 93.33],
        "top10": [77.78, 86.67, 88.89, 88.89, 86.67],
        "fn": [9, 5, 3, 3, 3],
        "selected_t": 60,
    },
    {
        "panel": "B",
        "name": "RIKEN-PlaSMA",
        "x": [30, 40, 50, 60, 70],
        "reduction": [63.16, 54.08, 46.23, 40.11, 35.05],
        "retention": [89.41, 95.29, 97.65, 97.65, 98.82],
        "top10": [82.35, 87.06, 89.41, 88.24, 88.24],
        "fn": [9, 4, 2, 2, 1],
        "selected_t": 50,
    },
]

# Percentage lines
C_REDUCTION = "#5B84C4"
C_RETENTION = "#35B7A4"
C_TOP10 = "#7B68C5"

# False negatives: deep orange
C_FN_BAR = "#E58A2B"
C_FN_EDGE = "#A85400"
C_FN_TEXT = "#8F4300"

C_VLINE = "#666666"
C_GRID = "#D9D9D9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color=C_GRID, alpha=0.80)
    ax.set_axisbelow(True)


def draw_one_dataset(fig, cell, ds, show_ylabels=False):
    # 上下等高
    sub = cell.subgridspec(2, 1, height_ratios=[1, 1], hspace=0.12)

    ax_top = fig.add_subplot(sub[0, 0])
    ax_bot = fig.add_subplot(sub[1, 0], sharex=ax_top)

    x = np.array(ds["x"], dtype=float)
    reduction = np.array(ds["reduction"], dtype=float)
    retention = np.array(ds["retention"], dtype=float)
    top10 = np.array(ds["top10"], dtype=float)
    fn = np.array(ds["fn"], dtype=float)
    t_sel = float(ds["selected_t"])

    step = np.min(np.diff(x))
    bar_width = step * 0.42
    selected_idx = list(x).index(t_sel)
    fn_sel = int(fn[selected_idx])

    # ---------- top: percentage lines ----------
    style_axis(ax_top)

    ax_top.plot(
        x, reduction,
        color=C_REDUCTION,
        marker="o",
        markersize=6.5,
        linewidth=2.2,
        zorder=3,
    )
    ax_top.plot(
        x, retention,
        color=C_RETENTION,
        marker="s",
        markersize=6.5,
        linewidth=2.2,
        zorder=3,
    )
    ax_top.plot(
        x, top10,
        color=C_TOP10,
        marker="^",
        markersize=7.0,
        linewidth=2.2,
        zorder=3,
    )

    # 只保留虚线，不再写 Selected T 文本，避免和折线重叠
    ax_top.axvline(t_sel, color=C_VLINE, linestyle="--", linewidth=1.6, zorder=2)

    ax_top.set_title(ds["name"], pad=12)
    ax_top.set_ylim(30, 104)
    ax_top.set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
    ax_top.tick_params(axis="x", labelbottom=False)

    if show_ylabels:
        ax_top.set_ylabel("Percentage (%)")
    else:
        ax_top.set_yticklabels([])

    ax_top.text(
        -0.13,
        1.16,
        ds["panel"],
        transform=ax_top.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # ---------- bottom: false negatives ----------
    style_axis(ax_bot)

    ax_bot.bar(
        x,
        fn,
        width=bar_width,
        color=C_FN_BAR,
        edgecolor=C_FN_EDGE,
        linewidth=1.2,
        alpha=0.78,
        zorder=3,
    )
    ax_bot.axvline(t_sel, color=C_VLINE, linestyle="--", linewidth=1.45, zorder=2)

    ax_bot.text(
        t_sel,
        fn_sel + 0.95,
        f"FN={fn_sel}",
        ha="center",
        va="bottom",
        fontsize=11.0,
        color=C_FN_TEXT,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.18),
        zorder=5,
    )

    ax_bot.set_ylim(0, 10.5)
    ax_bot.set_yticks([0, 3, 6, 9])
    ax_bot.set_xlim(x.min() - step * 0.45, x.max() + step * 0.45)
    ax_bot.set_xticks(x)
    ax_bot.set_xlabel("RT threshold T (s)")

    # 右侧那个 False negatives 轴不要了，只左图显示 FN 轴
    if show_ylabels:
        ax_bot.set_ylabel("False negatives")
    else:
        ax_bot.set_yticklabels([])

    return ax_top, ax_bot


def main():
    fig = plt.figure(figsize=(14.6, 6.9))

    outer = fig.add_gridspec(
        1,
        2,
        left=0.065,
        right=0.985,
        top=0.88,
        bottom=0.20,
        wspace=0.26,
    )

    draw_one_dataset(fig, outer[0, 0], DATASETS[0], show_ylabels=True)
    draw_one_dataset(fig, outer[0, 1], DATASETS[1], show_ylabels=False)

    handles = [
        Line2D(
            [0], [0],
            color=C_REDUCTION,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label="Candidate reduction (%)",
        ),
        Line2D(
            [0], [0],
            color=C_RETENTION,
            marker="s",
            linewidth=2.2,
            markersize=7,
            label="True-candidate retention (%)",
        ),
        Line2D(
            [0], [0],
            color=C_TOP10,
            marker="^",
            linewidth=2.2,
            markersize=7,
            label="Top-10 accuracy (%)",
        ),
        Patch(
            facecolor=C_FN_BAR,
            edgecolor=C_FN_EDGE,
            alpha=0.78,
            label="False negatives",
        ),
        Line2D(
            [0], [0],
            color=C_VLINE,
            linestyle="--",
            linewidth=1.6,
            label="Selected threshold",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
        columnspacing=1.4,
        handlelength=2.0,
        handletextpad=0.55,
    )

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")

    print("[SAVE]", OUT_PDF)
    print("[SAVE]", OUT_PNG)


if __name__ == "__main__":
    main()

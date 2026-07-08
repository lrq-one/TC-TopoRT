from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(".").resolve()
OUT_DIR = ROOT / "manuscript_figures_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "fig_threshold_sensitivity_candidate_filtering.pdf"
OUT_PNG = OUT_DIR / "fig_threshold_sensitivity_candidate_filtering.png"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DATA = {
    "MetaboBase": {
        "T": [30, 45, 60, 75, 90],
        "Reduction": [81.48, 74.86, 69.14, 64.37, 60.01],
        "True retained": [80.00, 88.89, 93.33, 93.33, 93.33],
        "Top-10": [77.78, 86.67, 88.89, 88.89, 86.67],
        "FN": [9, 5, 3, 3, 3],
        "Selected": 60,
    },
    "RIKEN-PlaSMA": {
        "T": [30, 40, 50, 60, 70],
        "Reduction": [63.16, 54.08, 46.23, 40.11, 35.05],
        "True retained": [89.41, 95.29, 97.65, 97.65, 98.82],
        "Top-10": [82.35, 87.06, 89.41, 88.24, 88.24],
        "FN": [9, 4, 2, 2, 1],
        "Selected": 50,
    },
}

C_RED = "#5F88C4"
C_RET = "#57B7A6"
C_TOP10 = "#6E63B6"
C_FN = "#E5964A"
C_VLINE = "#555555"

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.30)
    ax.set_axisbelow(True)

def draw_panel(ax, name, panel):
    d = DATA[name]
    T = np.array(d["T"])
    reduction = np.array(d["Reduction"])
    retained = np.array(d["True retained"])
    top10 = np.array(d["Top-10"])
    fn = np.array(d["FN"])
    selected = d["Selected"]

    style_ax(ax)
    ax2 = ax.twinx()

    bar_w = (T[1] - T[0]) * 0.34
    ax2.bar(
        T, fn,
        width=bar_w,
        color=C_FN,
        alpha=0.14,
        edgecolor=C_FN,
        linewidth=0.5,
        zorder=1,
    )

    ax.plot(T, reduction, marker="o", color=C_RED, linewidth=1.8, markersize=4.0,
            label="Candidate reduction (%)", zorder=3)
    ax.plot(T, retained, marker="s", color=C_RET, linewidth=1.8, markersize=3.8,
            label="True-candidate retention (%)", zorder=3)
    ax.plot(T, top10, marker="^", color=C_TOP10, linewidth=1.8, markersize=4.2,
            label="Top-10 accuracy (%)", zorder=3)

    ax.axvline(selected, linestyle="--", color=C_VLINE, linewidth=1.0, zorder=2)

    # selected T 标签放在顶部，不压曲线
    ax.text(
        selected,
        101.0,
        f"Selected T = {selected} s",
        fontsize=7.0,
        ha="center",
        va="top",
        color=C_VLINE,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
    )

    # 只标 selected threshold 的 FN，不标所有点
    idx = list(T).index(selected)
    ax2.text(
        selected,
        fn[idx] + 0.35,
        f"FN={int(fn[idx])}",
        fontsize=6.8,
        color=C_FN,
        ha="center",
        va="bottom",
    )

    ax.set_title(name, pad=5)
    ax.set_xlabel("RT threshold T (s)")
    ax.set_xlim(min(T) - 5, max(T) + 5)
    ax.set_ylim(30, 104)
    ax2.set_ylim(0, max(fn) + 2)
    ax.set_xticks(T)

    ax.text(-0.13, 1.07, panel, transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="bottom")

    return ax2

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.65), sharey=True)

ax2a = draw_panel(axes[0], "MetaboBase", "A")
ax2b = draw_panel(axes[1], "RIKEN-PlaSMA", "B")

axes[0].set_ylabel("Percentage (%)")
axes[1].set_ylabel("")
ax2a.set_ylabel("")
ax2a.tick_params(labelright=False)
ax2b.set_ylabel("False negatives")

legend_handles = [
    Line2D([0], [0], color=C_RED, marker="o", lw=1.8, label="Candidate reduction (%)"),
    Line2D([0], [0], color=C_RET, marker="s", lw=1.8, label="True-candidate retention (%)"),
    Line2D([0], [0], color=C_TOP10, marker="^", lw=1.8, label="Top-10 accuracy (%)"),
    Patch(facecolor=C_FN, edgecolor=C_FN, alpha=0.14, label="False negatives"),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.005),
    ncol=4,
    frameon=False,
)

fig.subplots_adjust(left=0.08, right=0.93, top=0.90, bottom=0.27, wspace=0.28)

fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.035)
fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.035)

print("[SAVE]", OUT_PDF)
print("[SAVE]", OUT_PNG)

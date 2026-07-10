from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================
# Output
# =========================
OUT_DIR = Path("manuscript_figures_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "fig5_ablation_summary.pdf"
OUT_PNG = OUT_DIR / "fig5_ablation_summary.png"

# =========================
# Data
# =========================
# Panel A: dual-view and fusion ablation, five-seed mean ± std
dual_labels = [
    "Original\nview only",
    "Strict tautomer\nview only",
    "Same-seed\npaired mean",
    "OOF Huber\nstack",
]
dual_mae = np.array([25.278, 25.217, 25.059, 25.055], dtype=float)
dual_std = np.array([0.054, 0.070, 0.038, 0.039], dtype=float)

# Panel B: structural ablation, corresponding structural-ablation runs
struct_labels = [
    "Full TC-TopoRT\n(seed 5)",
    "w/o explicit\nring 2-cells",
    "w/o CWN\nmessage passing",
]
struct_mae = np.array([25.012, 25.102, 39.645], dtype=float)
struct_delta = struct_mae - struct_mae[0]

# =========================
# Style
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_ORIG = "#8FA9C9"
C_TAUT = "#63B7A6"
C_MEAN = "#8D79C7"
C_STACK = "#5E4FB0"

C_FULL = "#5E4FB0"
C_RING = "#9B8BD0"
C_CWN = "#D18B4A"

EDGE = "#3F3F3F"
GRID = "#E6E6E6"
TEXT = "#222222"

# =========================
# Figure
# =========================
fig = plt.figure(figsize=(7.25, 3.25))
gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.12, 1.02],
    left=0.075,
    right=0.985,
    top=0.885,
    bottom=0.31,
    wspace=0.34,
)

# =========================
# Panel A
# =========================
ax1 = fig.add_subplot(gs[0, 0])

x = np.arange(len(dual_labels))
colors = [C_ORIG, C_TAUT, C_MEAN, C_STACK]

bars = ax1.bar(
    x,
    dual_mae,
    yerr=dual_std,
    capsize=3,
    width=0.62,
    color=colors,
    edgecolor=EDGE,
    linewidth=0.65,
    error_kw={
        "elinewidth": 0.8,
        "capthick": 0.8,
        "ecolor": EDGE,
    },
)

ax1.set_title("Dual-view and fusion ablation", fontsize=9, fontweight="bold", pad=8)
ax1.set_ylabel("MAE (s)")
ax1.set_xticks(x)
ax1.set_xticklabels(dual_labels, fontsize=7)
ax1.set_ylim(24.90, 25.40)
ax1.set_yticks([24.9, 25.0, 25.1, 25.2, 25.3, 25.4])
ax1.grid(axis="y", color=GRID, linewidth=0.6, linestyle="-", zorder=0)
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

for i, (b, m, s) in enumerate(zip(bars, dual_mae, dual_std)):
    ax1.text(
        b.get_x() + b.get_width() / 2,
        m + s + 0.018,
        f"{m:.3f}\n±{s:.3f}",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=TEXT,
        linespacing=0.9,
    )

ax1.text(
    -0.18, 1.08, "A",
    transform=ax1.transAxes,
    fontsize=10,
    fontweight="bold",
    va="top",
    ha="left",
)

# =========================
# Panel B
# =========================
ax2 = fig.add_subplot(gs[0, 1])

y = np.arange(len(struct_labels))
struct_colors = [C_FULL, C_RING, C_CWN]

bars2 = ax2.barh(
    y,
    struct_mae,
    height=0.48,
    color=struct_colors,
    edgecolor=EDGE,
    linewidth=0.65,
)

ax2.set_title("Structural ablation", fontsize=9, fontweight="bold", pad=8)
ax2.set_xlabel("MAE (s)")
ax2.set_yticks(y)
ax2.set_yticklabels(struct_labels, fontsize=7)
ax2.invert_yaxis()
ax2.set_xlim(0, 44.5)
ax2.set_xticks([0, 10, 20, 30, 40])
ax2.grid(axis="x", color=GRID, linewidth=0.6, linestyle="-", zorder=0)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

for i, (b, m, d) in enumerate(zip(bars2, struct_mae, struct_delta)):
    y0 = b.get_y() + b.get_height() / 2

    if i == 0:
        label = f"{m:.3f}"
    else:
        label = f"{m:.3f}  (Δ={d:+.3f})"

    ax2.text(
        m + 0.55,
        y0,
        label,
        ha="left",
        va="center",
        fontsize=7,
        color=TEXT,
    )

ax2.text(
    -0.18, 1.08, "B",
    transform=ax2.transAxes,
    fontsize=10,
    fontweight="bold",
    va="top",
    ha="left",
)

# =========================
# Small legend / note
# =========================
legend_handles = [
    Patch(facecolor=C_ORIG, edgecolor=EDGE, label="Original"),
    Patch(facecolor=C_TAUT, edgecolor=EDGE, label="Tautomer"),
    Patch(facecolor=C_MEAN, edgecolor=EDGE, label="Paired mean"),
    Patch(facecolor=C_STACK, edgecolor=EDGE, label="OOF Huber stack"),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.50, 0.02),
    ncol=2,
    frameon=False,
    fontsize=7.2,
    handlelength=1.35,
    columnspacing=1.6,
    handletextpad=0.55,
)


# =========================
# Save
# =========================
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
plt.close(fig)

print("[WROTE]", OUT_PDF)
print("[WROTE]", OUT_PNG)

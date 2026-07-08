from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(".").resolve()
OUT_DIR = ROOT / "manuscript_figures_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "fig3_candidate_filtering_summary.pdf"
OUT_PNG = OUT_DIR / "fig3_candidate_filtering_summary.png"

# ============================================================
# Data
# ============================================================
datasets = ["MetaboBase", "RIKEN-PlaSMA"]

# Panel A
initial = [3023, 5044]
retained = [933, 2712]
reduction_tc = [69.14, 46.23]

# Panel B
reduction_baselines = {
    "RT-Transformer-TL": [8.18, 11.14],
    "DeepGCN-RT-TL": [29.64, 35.37],
    "ABCoRT-TL": [38.35, 28.46],
    "TC-TopoRT": [69.14, 46.23],
}

# Panel C
topk_labels = ["Top-1", "Top-5", "Top-10"]
topk_methods = [
    "MS-FINDER only / No RT",
    "RT-Transformer-TL",
    "DeepGCN-RT-TL",
    "ABCoRT-TL",
    "TC-TopoRT",
]

topk_metabobase = {
    "MS-FINDER only / No RT": [44.44, 75.56, 84.44],
    "RT-Transformer-TL": [33.33, 64.44, 73.33],
    "DeepGCN-RT-TL": [42.22, 66.67, 75.56],
    "ABCoRT-TL": [51.11, 73.33, 82.22],
    "TC-TopoRT": [55.56, 82.22, 88.89],
}

topk_riken = {
    "MS-FINDER only / No RT": [47.06, 70.59, 82.35],
    "RT-Transformer-TL": [47.06, 77.65, 83.53],
    "DeepGCN-RT-TL": [57.65, 75.29, 83.53],
    "ABCoRT-TL": [52.94, 76.47, 83.53],
    "TC-TopoRT": [54.12, 77.65, 89.41],
}

# Panel D
true_retention = [93.33, 97.65]
fn_text = ["FN = 3 / 45", "FN = 2 / 85"]

# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.7,
    "ytick.labelsize": 6.7,
    "legend.fontsize": 5.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_INIT = "#C7CBD3"
C_RETAIN = "#7B68B8"

C_MSF = "#D7D7D7"
C_RTTR = "#B8C7D9"
C_DGCN = "#5F88C4"
C_ABC = "#4DB6AC"
C_TC = "#6E63B6"
C_FN = "#F39C34"

method_colors = {
    "MS-FINDER only / No RT": C_MSF,
    "RT-Transformer-TL": C_RTTR,
    "DeepGCN-RT-TL": C_DGCN,
    "ABCoRT-TL": C_ABC,
    "TC-TopoRT": C_TC,
}

def style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.35, alpha=0.24)
    ax.set_axisbelow(True)

def panel_title(ax, letter, title):
    ax.text(
        -0.13, 1.08, letter,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.text(
        -0.01, 1.08, title,
        transform=ax.transAxes,
        fontsize=7.9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

fig = plt.figure(figsize=(7.55, 5.25))
gs = fig.add_gridspec(
    2, 2,
    left=0.075,
    right=0.985,
    top=0.955,
    bottom=0.185,
    wspace=0.35,
    hspace=0.62,
)

# ============================================================
# A. Candidate-space reduction
# ============================================================
axA = fig.add_subplot(gs[0, 0])
style_ax(axA, "y")
panel_title(axA, "A", "Candidate-space reduction by TC-TopoRT")

x = np.arange(len(datasets))
w = 0.34

b_init = axA.bar(
    x - w / 2, initial, width=w,
    color=C_INIT, edgecolor="#7A7A7A", linewidth=0.65, label="Initial"
)
b_ret = axA.bar(
    x + w / 2, retained, width=w,
    color=C_RETAIN, edgecolor="#51468A", linewidth=0.65, label="Retained"
)

axA.set_xticks(x)
axA.set_xticklabels(datasets)
axA.set_ylabel("Candidates")
axA.set_ylim(0, 6500)

axA.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.00, 0.98),
    ncol=2,
    handlelength=1.1,
    columnspacing=0.7,
    handletextpad=0.35,
)

for bars in [b_init, b_ret]:
    for b in bars:
        h = b.get_height()
        axA.text(
            b.get_x() + b.get_width()/2,
            h + 85,
            f"{int(h)}",
            ha="center", va="bottom", fontsize=6.6
        )

for i, red in enumerate(reduction_tc):
    axA.text(
        x[i],
        max(initial[i], retained[i]) + 420,
        f"{red:.2f}% reduction",
        ha="center",
        va="bottom",
        fontsize=6.9,
        color=C_TC,
        fontweight="bold",
    )

# ============================================================
# B. Reduction comparison
# ============================================================
axB = fig.add_subplot(gs[0, 1])
style_ax(axB, "y")
panel_title(axB, "B", "Candidate reduction compared with RT-filtering baselines")

methods_B = ["RT-Transformer-TL", "DeepGCN-RT-TL", "ABCoRT-TL", "TC-TopoRT"]
xB = np.arange(len(datasets))
wB = 0.17
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * wB

for j, method in enumerate(methods_B):
    vals = reduction_baselines[method]
    bars = axB.bar(
        xB + offsets[j],
        vals,
        width=wB,
        color=method_colors[method],
        edgecolor="#666666",
        linewidth=0.55,
        label=method,
    )
    if method == "TC-TopoRT":
        for b, v in zip(bars, vals):
            axB.text(
                b.get_x() + b.get_width()/2, v + 1.0, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.3
            )

axB.set_xticks(xB)
axB.set_xticklabels(datasets)
axB.set_ylabel("Reduction (%)")
axB.set_ylim(0, 82)

axB.legend(
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(0.995, 1.01),
    ncol=2,
    handlelength=1.1,
    columnspacing=0.6,
    handletextpad=0.35,
)

# ============================================================
# C. Top-k prioritization comparison  -> grouped BAR
# ============================================================
outerC = fig.add_subplot(gs[1, 0])
outerC.axis("off")
outerC.text(
    -0.13, 1.08, "C",
    transform=outerC.transAxes,
    fontsize=9.0,
    fontweight="bold",
    ha="left",
    va="bottom",
)
outerC.text(
    -0.01, 1.08, "Top-k prioritization comparison",
    transform=outerC.transAxes,
    fontsize=7.9,
    fontweight="bold",
    ha="left",
    va="bottom",
)

legend_labels_short = {
    "MS-FINDER only / No RT": "MSF only / No RT",
    "RT-Transformer-TL": "RT-Transf.-TL",
    "DeepGCN-RT-TL": "DeepGCN-RT-TL",
    "ABCoRT-TL": "ABCoRT-TL",
    "TC-TopoRT": "TC-TopoRT",
}

legend_handles = [
    Patch(
        facecolor=method_colors[m],
        edgecolor="#666666",
        linewidth=0.45,
        label=legend_labels_short[m],
    )
    for m in topk_methods
]

# C panel legend: horizontal, placed below x-axis area
legC = outerC.legend(
    handles=legend_handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.50, -0.13),
    ncol=3,
    columnspacing=0.90,
    handlelength=1.10,
    handletextpad=0.35,
    borderaxespad=0.0,
    fontsize=5.3,
)

# 关键：legend 横向放在标题下面，不放到底部
legend_labels_short = {
    "MS-FINDER only / No RT": "MSF only / No RT",
    "RT-Transformer-TL": "RT-Transf.-TL",
    "DeepGCN-RT-TL": "DeepGCN-RT-TL",
    "ABCoRT-TL": "ABCoRT-TL",
    "TC-TopoRT": "TC-TopoRT",
}
legend_handles = [
    Patch(
        facecolor=method_colors[m],
        edgecolor="#666666",
        linewidth=0.45,
        label=legend_labels_short[m]
    )
    for m in topk_methods
]

# 给 C 面板上方留更多空位，避免 legend 压子图
subC = gs[1, 0].subgridspec(1, 2, wspace=0.22)
axC1 = fig.add_subplot(subC[0, 0])
axC2 = fig.add_subplot(subC[0, 1], sharey=axC1)

def draw_topk_grouped_bar(ax, title, data):
    style_ax(ax, "y")
    ax.set_title(title, fontsize=7.2, pad=2)

    x = np.arange(len(topk_labels))
    w = 0.14
    offsets = np.linspace(-2, 2, len(topk_methods)) * w

    for j, method in enumerate(topk_methods):
        vals = data[method]
        ax.bar(
            x + offsets[j],
            vals,
            width=w,
            color=method_colors[method],
            edgecolor="#666666",
            linewidth=0.45,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(topk_labels)
    ax.set_ylim(30, 95)

draw_topk_grouped_bar(axC1, "MetaboBase", topk_metabobase)
draw_topk_grouped_bar(axC2, "RIKEN-PlaSMA", topk_riken)

axC1.set_ylabel("Accuracy (%)")
axC2.tick_params(labelleft=False)

# 把子图整体往下压一点，避免和上面的 legend 打架
for ax in [axC1, axC2]:
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + 0.018, pos.width, pos.height * 0.82])

# ============================================================
# D. True-candidate retention and false negatives
# ============================================================
axD = fig.add_subplot(gs[1, 1])
style_ax(axD, "y")
panel_title(axD, "D", "True-candidate retention and false negatives")

xD = np.arange(len(datasets))
barsD = axD.bar(
    xD,
    true_retention,
    width=0.46,
    color=C_TC,
    edgecolor="#51468A",
    linewidth=0.65,
)

axD.set_xticks(xD)
axD.set_xticklabels(datasets)
axD.set_ylabel("True-candidate retention (%)")
axD.set_ylim(0, 106)

for b, rt, fn in zip(barsD, true_retention, fn_text):
    axD.text(
        b.get_x() + b.get_width()/2,
        rt + 1.1,
        f"{rt:.2f}%",
        ha="center", va="bottom",
        fontsize=6.9, fontweight="bold"
    )
    axD.text(
        b.get_x() + b.get_width()/2,
        rt - 7.0,
        fn,
        ha="center", va="top",
        fontsize=6.8, color=C_FN, fontweight="bold"
    )

# ============================================================
# Bottom note
# ============================================================
fig.text(
    0.52, 0.018,
    "RT-Transformer-TL and DeepGCN-RT-TL are literature-reported reduction-rate baselines from the ABCoRT study.",
    ha="center", va="center", fontsize=5.6
)

fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

print("[SAVE]", OUT_PDF.relative_to(ROOT))
print("[SAVE]", OUT_PNG.relative_to(ROOT))

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(".").resolve()
OUT_DIR = ROOT / "manuscript_figures_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "fig3_candidate_filtering_summary.pdf"
OUT_PNG = OUT_DIR / "fig3_candidate_filtering_summary.png"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.2,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.6,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

datasets = ["MetaboBase", "RIKEN-PlaSMA"]

initial = [3023, 5044]
retained = [933, 2712]
reduction_tc = [69.14, 46.23]

reduction_baselines = {
    "MetaboBase": {
        "RT-Transformer-TL": 8.18,
        "DeepGCN-RT-TL": 29.64,
        "ABCoRT-TL": 38.35,
        "TC-TopoRT": 69.14,
    },
    "RIKEN-PlaSMA": {
        "RT-Transformer-TL": 11.14,
        "DeepGCN-RT-TL": 35.37,
        "ABCoRT-TL": 28.46,
        "TC-TopoRT": 46.23,
    },
}

topk = {
    "MetaboBase": {
        "MS-FINDER only / No RT": [44.44, 75.56, 84.44],
        "ABCoRT-TL": [51.11, 73.33, 82.22],
        "TC-TopoRT": [55.56, 82.22, 88.89],
    },
    "RIKEN-PlaSMA": {
        "MS-FINDER only / No RT": [47.06, 70.59, 82.35],
        "ABCoRT-TL": [52.94, 76.47, 83.53],
        "TC-TopoRT": [54.12, 77.65, 89.41],
    },
}

true_retained = [93.33, 97.65]
fn_text = ["FN = 3 / 45", "FN = 2 / 85"]

C_MSF = "#D9D9D9"
C_RTTR = "#BCC7D6"
C_DGCN = "#648BC6"
C_ABC = "#40B6AA"
C_TC = "#7869BC"

C_INIT = "#BEC3CB"
C_RETAIN = "#7B69BB"

EDGE_SOFT = "#808080"
EDGE_TC = "#5A4A9C"
C_FN_TEXT = "#F39A2F"

def style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.45, alpha=0.22)
    ax.set_axisbelow(True)

def panel_title(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", pad=12)

fig = plt.figure(figsize=(9.2, 7.0))
gs = fig.add_gridspec(
    2, 2,
    left=0.07,
    right=0.985,
    top=0.92,
    bottom=0.145,
    wspace=0.36,
    hspace=0.72,
)

# ---------------- A ----------------
axA = fig.add_subplot(gs[0, 0])
style_ax(axA, "y")
panel_title(axA, "A  Candidate-space reduction by TC-TopoRT")

x = np.arange(len(datasets))
w = 0.34

b1 = axA.bar(x - w/2, initial, width=w, color=C_INIT, edgecolor=EDGE_SOFT, linewidth=0.8, label="Initial")
b2 = axA.bar(x + w/2, retained, width=w, color=C_RETAIN, edgecolor=EDGE_TC, linewidth=0.8, label="Retained")

axA.set_xticks(x)
axA.set_xticklabels(datasets)
axA.set_ylabel("Candidates")
axA.set_ylim(0, 6900)

axA.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.01, 0.97),
    ncol=2,
    columnspacing=0.8,
    handlelength=1.2,
    handletextpad=0.4,
)

for rect in list(b1) + list(b2):
    h = rect.get_height()
    axA.text(rect.get_x() + rect.get_width()/2, h + 90, f"{int(h)}",
             ha="center", va="bottom", fontsize=7.2)

for i, red in enumerate(reduction_tc):
    local_top = max(initial[i], retained[i])
    axA.text(x[i], local_top + 650, f"{red:.2f}% reduction",
             ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=C_TC)

# ---------------- B ----------------
axB = fig.add_subplot(gs[0, 1])
style_ax(axB, "y")
axB.set_title("B  Candidate reduction compared with RT-filtering baselines", loc="left", fontweight="bold", pad=18)

methods_b = ["RT-Transformer-TL", "DeepGCN-RT-TL", "ABCoRT-TL", "TC-TopoRT"]
colors_b = [C_RTTR, C_DGCN, C_ABC, C_TC]

xB = np.arange(len(datasets))
wB = 0.18
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * wB

for j, method in enumerate(methods_b):
    vals = [reduction_baselines[d][method] for d in datasets]
    axB.bar(
        xB + offsets[j], vals, width=wB,
        color=colors_b[j], edgecolor="#666666", linewidth=0.8, label=method
    )

axB.set_xticks(xB)
axB.set_xticklabels(datasets)
axB.set_ylabel("Reduction (%)")
axB.set_ylim(0, 82)

# Fig3-B legend: place inside the empty upper-right area to avoid overlapping the title.
axB.legend(
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    ncol=1,
    handlelength=1.35,
    handletextpad=0.45,
    borderaxespad=0.0,
    fontsize=6.6,
)


for i, v in enumerate([69.14, 46.23]):
    axB.text(xB[i] + offsets[-1], v + 1.5, f"{v:.2f}",
             ha="center", va="bottom", fontsize=7.2)

# ---------------- C ----------------
outerC = fig.add_subplot(gs[1, 0])
outerC.axis("off")
outerC.text(
    0.0, 1.08,
    "C  Top-k prioritization after reranking",
    transform=outerC.transAxes,
    ha="left", va="bottom",
    fontsize=9.0, fontweight="bold"
)

subgsC = gs[1, 0].subgridspec(1, 2, wspace=0.34)
axC1 = fig.add_subplot(subgsC[0, 0])
axC2 = fig.add_subplot(subgsC[0, 1], sharey=axC1)

topk_labels = ["Top-1", "Top-5", "Top-10"]
xx = np.arange(len(topk_labels))
ww = 0.22

methods_c = ["MS-FINDER only / No RT", "ABCoRT-TL", "TC-TopoRT"]
colors_c = [C_MSF, C_ABC, C_TC]

for ax, ds in [(axC1, "MetaboBase"), (axC2, "RIKEN-PlaSMA")]:
    style_ax(ax, "y")
    for j, method in enumerate(methods_c):
        vals = topk[ds][method]
        ax.bar(xx + (j - 1) * ww, vals, width=ww,
               color=colors_c[j], edgecolor="#666666", linewidth=0.8)
    ax.set_xticks(xx)
    ax.set_xticklabels(topk_labels)
    ax.set_title(ds, fontsize=8.4, pad=3)
    ax.set_ylim(35, 100)

axC1.set_ylabel("Accuracy (%)")
axC2.tick_params(labelleft=False)

handles_c = [
    Patch(facecolor=C_MSF, edgecolor="#666666", label="MS-FINDER only / No RT"),
    Patch(facecolor=C_ABC, edgecolor="#666666", label="ABCoRT-TL"),
    Patch(facecolor=C_TC, edgecolor="#666666", label="TC-TopoRT"),
]
outerC.legend(
    handles=handles_c,
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.60, -0.23),
    ncol=3,
    fontsize=6.6,
    columnspacing=0.9,
    handlelength=1.3,
)

# ---------------- D ----------------
axD = fig.add_subplot(gs[1, 1])
style_ax(axD, "y")
panel_title(axD, "D  True-candidate retention and false negatives")

xD = np.arange(len(datasets))
barsD = axD.bar(xD, true_retained, width=0.46, color=C_TC, edgecolor=EDGE_TC, linewidth=0.8)

axD.set_xticks(xD)
axD.set_xticklabels(datasets)
axD.set_ylabel("True-candidate retention (%)")
axD.set_ylim(0, 106)

for i, b in enumerate(barsD):
    h = b.get_height()
    axD.text(b.get_x() + b.get_width()/2, h + 1.0, f"{h:.2f}%",
             ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    axD.text(b.get_x() + b.get_width()/2, h - 8.5, fn_text[i],
             ha="center", va="top", fontsize=8.0, color=C_FN_TEXT, fontweight="bold")

fig.text(
    0.53, 0.055,
    "RT-Transformer-TL and DeepGCN-RT-TL are literature-reported reduction-rate baselines from the ABCoRT study.",
    ha="center", va="center", fontsize=7.0
)

fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.03)
print("[SAVE]", OUT_PDF)
print("[SAVE]", OUT_PNG)

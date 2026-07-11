from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(".").resolve()

OUT_DIRS = [
    ROOT / "manuscript_figures_final",
    ROOT / "manuscript_figures_jcim",
]

for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

OUT_FILES = []
for d in OUT_DIRS:
    OUT_FILES.extend([
        d / "fig5_ablation_summary.pdf",
        d / "fig5_ablation_summary.png",
    ])

# -----------------------------
# Data
# -----------------------------
dual_labels = [
    "Original\nview only",
    "Strict tautomer\nview only",
    "Same-seed\npaired mean\nfusion",
    "OOF Huber\nstack",
]
dual_means = np.array([25.278, 25.217, 25.059, 25.055])
dual_stds  = np.array([0.054, 0.070, 0.038, 0.039])

full_mae = 25.012
struct_labels = [
    "Full TC-TopoRT",
    "w/o explicit\nring 2-cells",
    "Atom-bond GNN\n(same protocol)",
    "w/o CWN\nmessage passing",
]
struct_mae = np.array([25.012, 25.102, 28.252, 39.645])
struct_delta = struct_mae - full_mae

# -----------------------------
# Style
# -----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

c_original = "#8FA7C5"
c_tautomer = "#59B3A9"
c_paired   = "#8F79C6"
c_huber    = "#6550AE"

dual_colors = [c_original, c_tautomer, c_paired, c_huber]

c_full   = "#6550AE"
c_ring   = "#9A87C8"
c_atom   = "#6E97C4"
c_nocwn  = "#D28C4C"

struct_colors = [c_full, c_ring, c_atom, c_nocwn]

# -----------------------------
# Figure
# -----------------------------
fig = plt.figure(figsize=(14.8, 7.6))
gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.25, 1.08],
    left=0.07, right=0.985, top=0.91, bottom=0.12, wspace=0.34
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])

# -----------------------------
# Panel A
# -----------------------------
x = np.arange(len(dual_labels))
barsA = axA.bar(
    x, dual_means, yerr=dual_stds,
    color=dual_colors, edgecolor="#4A4A4A", linewidth=1.2,
    capsize=6, width=0.62
)

axA.set_title("Dual-view and fusion ablation", fontweight="bold", pad=20)
axA.set_ylabel("MAE (s)")
axA.set_xticks(x)
axA.set_xticklabels(dual_labels)
axA.set_ylim(24.90, 25.40)
axA.set_yticks([24.9, 25.0, 25.1, 25.2, 25.3, 25.4])
axA.grid(axis="y", linestyle="--", alpha=0.35)
axA.set_axisbelow(True)

for b, mean, std in zip(barsA, dual_means, dual_stds):
    axA.text(
        b.get_x() + b.get_width()/2,
        mean + std + 0.016,
        f"{mean:.3f}\n±{std:.3f}",
        ha="center", va="bottom",
        fontsize=15, color="#303030", linespacing=0.9
    )

axA.text(-0.13, 1.04, "A", transform=axA.transAxes,
         fontsize=30, fontweight="bold")

# -----------------------------
# Panel B
# -----------------------------
y = np.arange(len(struct_labels))
barsB = axB.barh(
    y, struct_mae,
    color=struct_colors, edgecolor="#4A4A4A", linewidth=1.2,
    height=0.54
)

axB.set_title("Structural ablation", fontweight="bold", pad=20)
axB.set_xlabel("MAE (s)")
axB.set_yticks(y)
axB.set_yticklabels(struct_labels)
axB.invert_yaxis()
axB.grid(axis="x", linestyle="--", alpha=0.35)
axB.set_axisbelow(True)
axB.set_xlim(0, 44.5)

for idx, (yy, val, delta) in enumerate(zip(y, struct_mae, struct_delta)):
    if idx == 0:
        text = f"{val:.3f}"
    else:
        sign = "+" if delta >= 0 else ""
        text = f"{val:.3f}  (Δ={sign}{delta:.3f})"
    axB.text(
        val + 0.60, yy, text,
        va="center", ha="left",
        fontsize=15, color="#303030",
        clip_on=False
    )

axB.text(-0.20, 1.04, "B", transform=axB.transAxes,
         fontsize=30, fontweight="bold")

# -----------------------------
# Axes spine cleanup
# -----------------------------
for ax in [axA, axB]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.tick_params(width=1.6, length=7)

# -----------------------------
# Save
# -----------------------------
for out in OUT_FILES:
    if out.suffix.lower() == ".png":
        fig.savefig(out, dpi=600, bbox_inches="tight")
    else:
        fig.savefig(out, bbox_inches="tight")

plt.close(fig)

print("[DONE] Figure 5 saved to:")
for out in OUT_FILES:
    print(" ", out.relative_to(ROOT))
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path.cwd()
INFILE = ROOT / "manuscript_figures_final" / "source_formula_level_guarded_soft_final.csv"
OUTDIR = ROOT / "manuscript_figures_final"
OUTDIR.mkdir(parents=True, exist_ok=True)

if not INFILE.exists():
    raise FileNotFoundError(f"Missing input file: {INFILE}")

df = pd.read_csv(INFILE)

# 统一数据集名字
def normalize_dataset(x):
    s = str(x).strip().lower()
    if "riken" in s or "plasma" in s:
        return "RIKEN-PlaSMA"
    if "metabo" in s:
        return "MetaboBase"
    return x

df["dataset"] = df["dataset"].map(normalize_dataset)

# 公式标签列
label_col = "formula_label" if "formula_label" in df.columns else "formula"

# 只保留这两个数据集
df = df[df["dataset"].isin(["RIKEN-PlaSMA", "MetaboBase"])].copy()

# 分开排序：形成“上下拼起来”的视觉
riken = df[df["dataset"] == "RIKEN-PlaSMA"].copy()
metabo = df[df["dataset"] == "MetaboBase"].copy()

# 按 total_candidates 从小到大排序，更接近你师兄那种形状
riken = riken.sort_values(["total_candidates", "retained_candidates"], ascending=[True, True]).reset_index(drop=True)
metabo = metabo.sort_values(["total_candidates", "retained_candidates"], ascending=[True, True]).reset_index(drop=True)

# 上下拼接：上面 RIKEN，下面 MetaboBase
plot_df = pd.concat([riken, metabo], axis=0).reset_index(drop=True)

# 给中间留一个视觉小间隔
gap = 3
y_riken = np.arange(len(riken))
y_metabo = np.arange(len(metabo)) + len(riken) + gap

# 配色：不和你师兄一模一样
# RIKEN -> 暖橙系
riken_total = "#f3d9c9"
riken_keep  = "#d98b73"

# MetaboBase -> 紫蓝系
metabo_total = "#d8def3"
metabo_keep  = "#7f90c9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig_h = max(10, 0.12 * len(plot_df))
fig, ax = plt.subplots(figsize=(8.6, fig_h))

# 画 RIKEN
ax.barh(
    y_riken,
    riken["total_candidates"],
    color=riken_total,
    edgecolor="none",
    height=0.78,
    zorder=1,
)
ax.barh(
    y_riken,
    riken["retained_candidates"],
    color=riken_keep,
    edgecolor="none",
    height=0.78,
    zorder=2,
)

# 画 MetaboBase
ax.barh(
    y_metabo,
    metabo["total_candidates"],
    color=metabo_total,
    edgecolor="none",
    height=0.78,
    zorder=1,
)
ax.barh(
    y_metabo,
    metabo["retained_candidates"],
    color=metabo_keep,
    edgecolor="none",
    height=0.78,
    zorder=2,
)

# y 轴标签
all_y = np.concatenate([y_riken, y_metabo])
all_labels = list(riken[label_col].astype(str)) + list(metabo[label_col].astype(str))
ax.set_yticks(all_y)
ax.set_yticklabels(all_labels, fontsize=6)

# 坐标轴
ax.set_xlabel("Number of Candidates", fontsize=12)
ax.set_ylabel("Formula", fontsize=12)

# 不要网格太花
ax.grid(False)

# 中间分割线
split_y = len(riken) + gap / 2 - 0.5
ax.axhline(split_y, color="#777777", lw=0.8, alpha=0.5)

# 左侧写 dataset 名字
x_left = -0.06 * max(plot_df["total_candidates"].max(), 1)
ax.text(x_left, y_riken.mean(), "RIKEN-PlaSMA", rotation=90,
        va="center", ha="center", fontsize=12, fontweight="bold", color="#8a4f39")
ax.text(x_left, y_metabo.mean(), "MetaboBase", rotation=90,
        va="center", ha="center", fontsize=12, fontweight="bold", color="#4d5c9a")

# 图例
handles = [
    Patch(facecolor=riken_total, label="Total candidates in RIKEN-PlaSMA"),
    Patch(facecolor=riken_keep,  label="Retained candidates in RIKEN-PlaSMA"),
    Patch(facecolor=metabo_total, label="Total candidates in MetaboBase"),
    Patch(facecolor=metabo_keep,  label="Retained candidates in MetaboBase"),
]
leg = ax.legend(handles=handles, loc="upper right", fontsize=10, frameon=True)
leg.get_frame().set_alpha(0.95)

# 边距
ax.set_xlim(left=x_left * 0.2, right=plot_df["total_candidates"].max() * 1.07)

# 让 RIKEN 在上面，MetaboBase 在下面
ax.invert_yaxis()

plt.tight_layout()

png_path = OUTDIR / "fig_formula_level_guarded_soft_final_pretty.png"
pdf_path = OUTDIR / "fig_formula_level_guarded_soft_final_pretty.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print("[SAVE]", png_path)
print("[SAVE]", pdf_path)

# 顺便打印汇总，确认没动结果
print("\n=== Summary ===")
for name, sub in [("RIKEN-PlaSMA", riken), ("MetaboBase", metabo)]:
    total = sub["total_candidates"].sum()
    keep = sub["retained_candidates"].sum()
    red = (1 - keep / total) * 100 if total > 0 else np.nan
    print(f"{name:14s} total={int(total):4d} retained={int(keep):4d} reduction={red:.2f}%")

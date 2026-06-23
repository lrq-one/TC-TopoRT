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

# -----------------------------
# 1) 统一字段
# -----------------------------
def normalize_dataset(x):
    s = str(x).strip().lower()
    if "riken" in s or "plasma" in s:
        return "RIKEN-PlaSMA"
    if "metabo" in s:
        return "MetaboBase"
    return x

df["dataset"] = df["dataset"].map(normalize_dataset)

label_col = "formula_label" if "formula_label" in df.columns else "formula"
if label_col not in df.columns:
    raise ValueError("Cannot find formula label column.")

need_cols = ["dataset", label_col, "total_candidates", "retained_candidates"]
for c in need_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

df = df[df["dataset"].isin(["RIKEN-PlaSMA", "MetaboBase"])].copy()

# -----------------------------
# 2) 分 dataset 排序
#    形成你师兄那种上下拼接的 wedge 视觉
# -----------------------------
riken = df[df["dataset"] == "RIKEN-PlaSMA"].copy()
metabo = df[df["dataset"] == "MetaboBase"].copy()

# 从小到大排序：顶部细，往下逐渐变宽
riken = riken.sort_values(["total_candidates", "retained_candidates"], ascending=[True, True]).reset_index(drop=True)
metabo = metabo.sort_values(["total_candidates", "retained_candidates"], ascending=[False, False]).reset_index(drop=True)

# 注意：
# 为了让最终图看起来和你师兄那张更像：
# - RIKEN 上半部分：从上到下逐渐变宽
# - Metabo 下半部分：从上到下逐渐变窄
# 所以下半部分这里直接按 total_candidates 从大到小

# -----------------------------
# 3) 颜色：换成接近你师兄那套
# -----------------------------
# RIKEN 绿色
RIKEN_TOTAL = "#dcebd8"   # 浅绿
RIKEN_KEEP  = "#9fd19d"   # 深一点的绿

# MetaboBase 蓝色
METABO_TOTAL = "#d8e4f3"  # 浅蓝
METABO_KEEP  = "#9eacbf"  # 灰蓝/深蓝

# -----------------------------
# 4) 画图参数
# -----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

n1 = len(riken)
n2 = len(metabo)
gap = 0

y_riken = np.arange(n1)
y_metabo = np.arange(n2) + n1 + gap

fig_h = max(12, 0.11 * (n1 + n2))
fig, ax = plt.subplots(figsize=(8.2, fig_h))

# -----------------------------
# 5) 上半部分 RIKEN
# -----------------------------
ax.barh(
    y_riken,
    riken["total_candidates"],
    color=RIKEN_TOTAL,
    edgecolor="none",
    height=0.82,
    zorder=1,
)
ax.barh(
    y_riken,
    riken["retained_candidates"],
    color=RIKEN_KEEP,
    edgecolor="none",
    height=0.82,
    zorder=2,
)

# -----------------------------
# 6) 下半部分 MetaboBase
# -----------------------------
ax.barh(
    y_metabo,
    metabo["total_candidates"],
    color=METABO_TOTAL,
    edgecolor="none",
    height=0.82,
    zorder=1,
)
ax.barh(
    y_metabo,
    metabo["retained_candidates"],
    color=METABO_KEEP,
    edgecolor="none",
    height=0.82,
    zorder=2,
)

# -----------------------------
# 7) y 标签
# -----------------------------
all_y = np.concatenate([y_riken, y_metabo])
all_labels = list(riken[label_col].astype(str)) + list(metabo[label_col].astype(str))

ax.set_yticks(all_y)
ax.set_yticklabels(all_labels, fontsize=6)

# 让第一组在最上面
ax.invert_yaxis()

# -----------------------------
# 8) 坐标轴样式
# -----------------------------
ax.set_xlabel("Number of Candidates", fontsize=12)
ax.set_ylabel("Formula", fontsize=12)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", length=0)

# 去掉网格，更接近你师兄那张
ax.grid(False)

# 中间淡分割线
sep_y = n1 - 0.5
ax.axhline(sep_y, color="#cfcfcf", lw=0.9)

# 边框保留，但不要太重
for sp in ax.spines.values():
    sp.set_linewidth(0.8)
    sp.set_color("#888888")

# x 轴范围
xmax = max(
    riken["total_candidates"].max() if len(riken) else 0,
    metabo["total_candidates"].max() if len(metabo) else 0,
)
ax.set_xlim(0, xmax * 1.08)

# -----------------------------
# 9) 图例
# -----------------------------
handles = [
    Patch(facecolor=RIKEN_TOTAL,  edgecolor="none", label="Total Number of Candidates in RIKEN_PlaSMA"),
    Patch(facecolor=RIKEN_KEEP,   edgecolor="none", label="Retained Candidates in RIKEN_PlaSMA"),
    Patch(facecolor=METABO_TOTAL, edgecolor="none", label="Total Number of Candidates in MetaboBASE"),
    Patch(facecolor=METABO_KEEP,  edgecolor="none", label="Retained Candidates in MetaboBASE"),
]
leg = ax.legend(
    handles=handles,
    loc="upper right",
    fontsize=9,
    frameon=True,
    fancybox=True,
)
leg.get_frame().set_edgecolor("#dddddd")
leg.get_frame().set_linewidth(0.8)
leg.get_frame().set_alpha(0.95)

plt.tight_layout()

png_path = OUTDIR / "fig_formula_level_guarded_soft_brostyle.png"
pdf_path = OUTDIR / "fig_formula_level_guarded_soft_brostyle.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print("[SAVE]", png_path)
print("[SAVE]", pdf_path)

print("\n=== summary ===")
for name, sub in [("RIKEN-PlaSMA", riken), ("MetaboBase", metabo)]:
    total = sub["total_candidates"].sum()
    keep = sub["retained_candidates"].sum()
    red = (1 - keep / total) * 100 if total > 0 else np.nan
    print(f"{name:14s} total={int(total):4d} retained={int(keep):4d} reduction={red:.2f}%")

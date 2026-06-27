from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parent

DUAL_TABLE_CANDIDATES = [
    ROOT / "paper_results_TCDV_TopoRT/tables/03_dualview_and_fusion_ablation.csv",
    ROOT / "gwn/final_paper_tables/Table_3_dualview_ablation.csv",
]

STRUCT_TABLE_CANDIDATES = [
    ROOT / "paper_results_TCDV_TopoRT/tables/04_structural_ablation_seed5.csv",
]

OUT_DIR = ROOT / "manuscript_figures_final"
OUT_STEM = "fig2_main_ablation"

# 浅蓝单色系
BLUE_1 = "#DCEFFA"   # very light
BLUE_2 = "#BFDDF4"
BLUE_3 = "#91C4EA"
BLUE_4 = "#5FA8DD"   # medium light
BLUE_5 = "#2E7FBF"   # deeper accent

LEFT_COLORS  = [BLUE_2, BLUE_3, BLUE_4, BLUE_5]
RIGHT_COLORS = [BLUE_2, BLUE_3, BLUE_5]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.8,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("No candidate table found:\n" + "\n".join(str(p) for p in paths))


def norm_text(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def parse_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        raise ValueError(f"Cannot parse numeric value from: {x!r}")
    return float(m.group(0))


def find_method_col(df):
    for c in ["variant", "method", "Method", "setting", "Setting", "name", "Name"]:
        if c in df.columns:
            return c
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    return obj_cols[0] if obj_cols else df.columns[0]


def find_mae_col(df):
    for c in ["mae", "MAE", "test_mae", "Test MAE", "MAE (s)"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "mae" in c.lower() and "delta" not in c.lower():
            return c
    raise ValueError(f"Cannot find MAE column from columns: {df.columns.tolist()}")


def load_dual_values():
    p = first_existing(DUAL_TABLE_CANDIDATES)
    df = pd.read_csv(p)
    method_col = find_method_col(df)
    mae_col = find_mae_col(df)

    out = {}
    for _, row in df.iterrows():
        key = norm_text(row[method_col])
        val = parse_float(row[mae_col])

        if key in {"originonly", "originalonly", "originviewonly", "originalviewonly"}:
            out["Original view only"] = val
        elif key in {"tautomeronly", "tautonly", "stricttautomerviewonly", "tautomerviewonly"}:
            out["Strict tautomer view only"] = val
        elif key in {"meanfusion", "averagefusion", "avgfusion", "mean"}:
            out["Mean fusion"] = val
        elif key in {"huberstack", "oofhuberstack", "oofstack", "stack", "stacking"}:
            out["OOF Huber stack"] = val

    needed = ["Original view only", "Strict tautomer view only", "Mean fusion", "OOF Huber stack"]
    miss = [k for k in needed if k not in out]
    if miss:
        print("[DEBUG] dual table:", p)
        print(df.to_string(index=False))
        raise ValueError(f"Missing dual-view items: {miss}")
    return p, out


def load_struct_values():
    p = first_existing(STRUCT_TABLE_CANDIDATES)
    df = pd.read_csv(p)
    method_col = find_method_col(df)
    mae_col = find_mae_col(df)

    out = {}
    for _, row in df.iterrows():
        raw_name = str(row[method_col]).strip().lower()
        key = norm_text(row[method_col])
        val = parse_float(row[mae_col])

        if (
            ("full" in raw_name and "tcdv" in raw_name)
            or ("full" in raw_name and "toport" in raw_name)
            or key in {"full", "fullmodel", "fulltcdvtoportseed5", "fulltcdvtoport"}
        ):
            out["Full model"] = val
        elif (
            "ring 2-cell" in raw_name
            or "no2cell" in key
            or "ring2cells" in key
            or "explicitring2cells" in key
        ):
            out["w/o ring 2-cells"] = val
        elif (
            "cwn" in raw_name
            or "cwn0" in key
            or "wocwn" in key
            or "withoutcwn" in key
        ):
            out["w/o CWN message passing"] = val

    needed = ["Full model", "w/o ring 2-cells", "w/o CWN message passing"]
    miss = [k for k in needed if k not in out]
    if miss:
        print("[DEBUG] structural table:", p)
        print(df.to_string(index=False))
        print("[DEBUG] parsed:", out)
        raise ValueError(f"Structural ablation missing items: {miss}\nFrom file: {p}")
    return p, out


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.18, color="#5B84A5")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3, color="#444444")


def add_group_span(ax, x0, x1, text, y=-0.10):
    trans = ax.get_xaxis_transform()
    ax.plot([x0, x1], [y, y], transform=trans, color="#8FB8D8", linewidth=0.9, clip_on=False)
    ax.text((x0 + x1) / 2, y - 0.055, text, transform=trans,
            ha="center", va="top", fontsize=7.4, color="#4E6E8E")


def main():
    dual_path, dual = load_dual_values()
    struct_path, struct = load_struct_values()

    dual_order = [
        "Original view only",
        "Strict tautomer view only",
        "Mean fusion",
        "OOF Huber stack",
    ]
    dual_vals = [dual[k] for k in dual_order]

    struct_order = [
        "Full model",
        "w/o ring 2-cells",
        "w/o CWN message passing",
    ]
    struct_vals = [struct[k] for k in struct_order]
    full = struct["Full model"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(7.8, 3.45),
        dpi=300,
        gridspec_kw={"wspace": 0.30}
    )
    fig.patch.set_facecolor("white")

    # ---------- Panel A ----------
    x = list(range(len(dual_order)))
    bars1 = ax1.bar(
        x, dual_vals,
        width=0.56,
        color=LEFT_COLORS,
        edgecolor="#EAF4FB",
        linewidth=0.8
    )

    ax1.set_title("A  Dual-view and fusion ablation", loc="left", fontweight="bold", pad=3)
    ax1.set_ylabel("Test MAE (s)")
    ax1.set_xlabel("Ablation setting", labelpad=26)
    ax1.set_xticks(x)
    ax1.set_xticklabels([
        "Original\nview only",
        "Strict tautomer\nview only",
        "Mean\nfusion",
        "OOF Huber\nstack",
    ])

    ax1.set_ylim(24.995, 25.335)
    ax1.set_yticks([25.00, 25.05, 25.10, 25.15, 25.20, 25.25, 25.30])
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    style_axes(ax1)

    for b, v in zip(bars1, dual_vals):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            v + 0.010,
            f"{v:.3f}",
            ha="center", va="bottom",
            fontsize=7.8, color="#2A2A2A"
        )

    add_group_span(ax1, -0.15, 1.15, "Single-view ablations", y=-0.10)
    add_group_span(ax1, 1.85, 3.15, "Fusion ablations", y=-0.10)

    # ---------- Panel B ----------
    x2 = list(range(len(struct_order)))
    bars2 = ax2.bar(
        x2, struct_vals,
        width=0.56,
        color=RIGHT_COLORS,
        edgecolor="#EAF4FB",
        linewidth=0.8
    )

    ax2.set_title("B  Structural ablation", loc="left", fontweight="bold", pad=3)
    ax2.set_ylabel("Test MAE (s)")
    ax2.set_xlabel("Structural variant", labelpad=26)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([
        "Full\nmodel",
        "w/o ring\n2-cells",
        "w/o CWN\nmessage passing",
    ])

    ax2.set_ylim(24.0, 41.5)
    ax2.set_yticks([24, 26, 28, 30, 32, 34, 36, 38, 40])
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    style_axes(ax2)

    labels2 = [
        f"{struct['Full model']:.3f}",
        f"{struct['w/o ring 2-cells']:.3f}\n(+{struct['w/o ring 2-cells'] - full:.3f})",
        f"{struct['w/o CWN message passing']:.3f}\n(+{struct['w/o CWN message passing'] - full:.3f})",
    ]

    for b, v, lab in zip(bars2, struct_vals, labels2):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + 0.35,
            lab,
            ha="center", va="bottom",
            fontsize=7.7, color="#2A2A2A",
            linespacing=0.95
        )

    src = pd.DataFrame({
        "panel": ["A"] * 4 + ["B"] * 3,
        "item": dual_order + struct_order,
        "mae_s": dual_vals + struct_vals,
        "delta_vs_full_s": ["", "", "", "", 0.0,
                            struct["w/o ring 2-cells"] - full,
                            struct["w/o CWN message passing"] - full],
    })

    src_path = OUT_DIR / f"{OUT_STEM}_source_data.csv"
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"

    src.to_csv(src_path, index=False)

    fig.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=0.31, wspace=0.30)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", dpi=600, facecolor="white")
    plt.close(fig)

    print("[OK] dual table   :", dual_path)
    print("[OK] struct table :", struct_path)
    print("[OK] pdf          :", pdf_path)
    print("[OK] png          :", png_path)
    print("[OK] source data  :", src_path)


if __name__ == "__main__":
    main()

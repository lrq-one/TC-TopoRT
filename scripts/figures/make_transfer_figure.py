from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    ROOT
    / "artifacts"
    / "results"
    / "external_transfer"
    / "Table_8_transfer_learning_effectiveness.csv"
)

OUT_DIR = ROOT / "artifacts" / "figures" / "transfer"
OUT_PNG = OUT_DIR / "fig_tl_vs_scratch_bar.png"
OUT_PDF = OUT_DIR / "fig_tl_vs_scratch_bar.pdf"



LABEL_MAP = {
    "FEM_short_73": "FEM-short",
    "FEM_long_412": "FEM-long",
    "UniToyama_Atlantis_143": "UniToyama-Atlantis",
    "Eawag_XBridgeC18_364": "Eawag-XBridgeC18",
    "FEM_lipids_72": "FEM-lipids",
    "LIFE_old_194": "LIFE-old",
    "LIFE_new_184": "LIFE-new",
    "MTBLS87_147": "MTBLS87",
    "IPB_Halle_82": "IPB-Halle",
    "Cao_HILIC_116": "Cao-HILIC",
}

def clean_dataset_label(name: str) -> str:
    s = str(name).strip()
    if s in LABEL_MAP:
        return LABEL_MAP[s]

    s2 = re.sub(r"_\d+$", "", s)
    if s2 in LABEL_MAP:
        return LABEL_MAP[s2]

    return s2.replace("_", "-")




def pick_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for c in df.columns:
        lc = c.lower()
        if any(cand.lower() in lc for cand in candidates):
            return c
    return None


def expand_xlim_to_fit_text(ax, texts, pad_px=10, max_iter=8):
    """
    Expand xlim until all value labels have at least pad_px inside the plot frame.
    This avoids manual guessing and guarantees visible gaps near both spines.
    """
    fig = ax.figure

    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        ax_box = ax.get_window_extent(renderer=renderer)
        x0, x1 = ax.get_xlim()
        data_per_px = (x1 - x0) / ax_box.width

        left_need_px = 0.0
        right_need_px = 0.0

        for t in texts:
            b = t.get_window_extent(renderer=renderer)

            if b.x0 < ax_box.x0 + pad_px:
                left_need_px = max(left_need_px, (ax_box.x0 + pad_px) - b.x0)

            if b.x1 > ax_box.x1 - pad_px:
                right_need_px = max(right_need_px, b.x1 - (ax_box.x1 - pad_px))

        if left_need_px <= 0 and right_need_px <= 0:
            break

        # Add a tiny extra cushion so it does not sit exactly on the boundary.
        new_x0 = x0 - left_need_px * data_per_px * 1.15
        new_x1 = x1 + right_need_px * data_per_px * 1.15
        ax.set_xlim(new_x0, new_x1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    in_path = INPUT_CSV
    if not in_path.is_file():
        raise FileNotFoundError(
            "Transfer Table 8 was not found. Run both public transfer "
            f"workflows first: {in_path}"
        )

    df = pd.read_csv(in_path)

    dataset_col = pick_col(df, ["dataset_name", "dataset", "Dataset"])
    scratch_col = pick_col(df, ["scratch_mae", "Scratch MAE"])
    transfer_col = pick_col(df, ["transfer_mae", "Transfer MAE"])
    improvement_col = pick_col(df, ["mae_improvement_s", "mae_improvement", "MAE improvement"])

    if dataset_col is None:
        raise RuntimeError(f"Cannot find dataset column in {in_path}")

    if improvement_col is None:
        if scratch_col is None or transfer_col is None:
            raise RuntimeError("Cannot find improvement column or scratch/transfer MAE columns.")
        df["mae_improvement_s"] = pd.to_numeric(df[scratch_col]) - pd.to_numeric(df[transfer_col])
        improvement_col = "mae_improvement_s"

    plot_df = df[[dataset_col, improvement_col]].copy()
    plot_df.columns = ["dataset", "improvement"]
    plot_df["improvement"] = pd.to_numeric(plot_df["improvement"], errors="coerce")
    plot_df = plot_df.dropna(subset=["improvement"])
    plot_df["label"] = plot_df["dataset"].map(clean_dataset_label)
    plot_df = plot_df.sort_values("improvement", ascending=False).reset_index(drop=True)

    labels = plot_df["label"].tolist()
    values = plot_df["improvement"].to_numpy()
    y = np.arange(len(values))

    positive_color = "#5B84B1"
    negative_color = "#B58AA5"
    colors = [positive_color if v >= 0 else negative_color for v in values]

    fig, ax = plt.subplots(figsize=(7.25, 4.55))

    ax.barh(
        y,
        values,
        color=colors,
        edgecolor="none",
        linewidth=0.0,
        height=0.62,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.invert_yaxis()

    ax.axvline(0, color="#2B2B2B", linewidth=0.9, zorder=4)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.30, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlabel("MAE improvement from transfer learning (s)", fontsize=9.2)

    
    ax.set_xlim(-5.8, 35.6)
    ax.set_xticks([-5, 0, 5, 10, 15, 20, 25, 30, 35])

    
    
    texts = []
    for yi, v in zip(y, values):
        txt = f"{v:+.3f}"

        if v >= 0:
            t = ax.annotate(
                txt,
                xy=(v, yi),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=7.6,
                color="black",
                annotation_clip=False,
                zorder=10,
            )
        else:
            t = ax.annotate(
                txt,
                xy=(v, yi),
                xytext=(-7, 0),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=7.6,
                color="black",
                annotation_clip=False,
                zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.10),
            )

        texts.append(t)

    
    expand_xlim_to_fit_text(ax, texts, pad_px=12, max_iter=8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#444444")

    ax.tick_params(axis="x", labelsize=8.0)
    ax.tick_params(axis="y", length=0, pad=8)

    fig.subplots_adjust(left=0.28, right=0.965, bottom=0.165, top=0.965)

    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print("[INPUT]", in_path.relative_to(ROOT))
    print("[SAVE]", OUT_PDF.relative_to(ROOT))
    print("[SAVE]", OUT_PNG.relative_to(ROOT))
    print()
    print("final xlim =", ax.get_xlim())
    print(plot_df.to_string(index=False))


if __name__ == "__main__":
    main()

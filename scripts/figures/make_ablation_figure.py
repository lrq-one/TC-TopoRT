#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DUALVIEW_DEFAULT = (
    ROOT
    / "artifacts"
    / "results"
    / "paper_tables"
    / "ablation"
    / "dualview_fusion_ablation_summary.csv"
)
STRUCTURAL_DEFAULT = (
    ROOT
    / "artifacts"
    / "results"
    / "paper_tables"
    / "ablation"
    / "structural_ablation_seed5.csv"
)
OUT_DIR_DEFAULT = ROOT / "artifacts" / "figures" / "ablation"


PANEL_A_ORDER = [
    "Original view only",
    "Tautomer view only",
    "Same-seed paired mean fusion",
    "OOF Huber stack",
]

PANEL_A_LABELS = {
    "Original view only": "Original\nview only",
    "Tautomer view only": "Strict tautomer\nview only",
    "Same-seed paired mean fusion": "Same-seed\npaired mean\nfusion",
    "OOF Huber stack": "OOF Huber\nstack",
}

PANEL_B_ORDER = [
    "Full TC-TopoRT",
    "w/o explicit ring 2-cells",
    "Conventional atom-bond GNN",
    "w/o CWN message passing",
]

PANEL_B_LABELS = {
    "Full TC-TopoRT": "Full TC-TopoRT",
    "w/o explicit ring 2-cells": "w/o explicit\nring 2-cells",
    "Conventional atom-bond GNN": "Atom-bond GNN\n(same protocol)",
    "w/o CWN message passing": "w/o CWN\nmessage passing",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def require_unique_row(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    rows = frame[frame[column].astype(str).eq(value)]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one row where {column}={value!r}; found {len(rows)}"
        )
    return rows.iloc[0]


def load_panel_a(path: Path) -> tuple[list[float], list[float]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dual-view ablation summary not found: {path}\n"
            "Run: python scripts/analysis/build_dualview_ablation.py"
        )

    frame = pd.read_csv(path)
    required = {"method", "mae_mean", "mae_sd"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")

    values: list[float] = []
    errors: list[float] = []
    for method in PANEL_A_ORDER:
        row = require_unique_row(frame, "method", method)
        values.append(float(row["mae_mean"]))
        errors.append(float(row["mae_sd"]))

    return values, errors


def load_panel_b(path: Path) -> tuple[list[float], list[float]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Structural ablation summary not found: {path}\n"
            "Run: python scripts/analysis/collect_structural_ablation.py"
        )

    frame = pd.read_csv(path)
    required = {"variant", "mae", "delta_mae_vs_full"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")

    values: list[float] = []
    deltas: list[float] = []
    for variant in PANEL_B_ORDER:
        row = require_unique_row(frame, "variant", variant)
        values.append(float(row["mae"]))
        deltas.append(float(row["delta_mae_vs_full"]))

    return values, deltas


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the manuscript Figure 5 ablation plot from the public "
            "dual-view and structural-ablation summary tables."
        )
    )
    parser.add_argument(
        "--dualview_csv",
        default=str(DUALVIEW_DEFAULT.relative_to(ROOT)),
    )
    parser.add_argument(
        "--structural_csv",
        default=str(STRUCTURAL_DEFAULT.relative_to(ROOT)),
    )
    parser.add_argument(
        "--out_dir",
        default=str(OUT_DIR_DEFAULT.relative_to(ROOT)),
    )
    args = parser.parse_args()

    dualview_csv = resolve_path(args.dualview_csv)
    structural_csv = resolve_path(args.structural_csv)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_a_values, panel_a_errors = load_panel_a(dualview_csv)
    panel_b_values, panel_b_deltas = load_panel_b(structural_csv)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors_a = ["#8EA6C6", "#5FB3AC", "#8A73BE", "#6253AE"]
    colors_b = ["#6A55B5", "#9A87CB", "#6F9CC4", "#D99552"]

    fig = plt.figure(figsize=(7.35, 3.45))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.08, 1.0],
        left=0.085,
        right=0.985,
        bottom=0.23,
        top=0.88,
        wspace=0.31,
    )

    # Panel A: dual-view and fusion ablation.
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(PANEL_A_ORDER))
    bars_a = ax_a.bar(
        x,
        panel_a_values,
        yerr=panel_a_errors,
        capsize=3,
        width=0.62,
        color=colors_a,
        edgecolor="#555555",
        linewidth=0.6,
        zorder=3,
    )

    ax_a.set_title("A   Dual-view and fusion ablation", loc="left", fontweight="bold")
    ax_a.set_ylabel("MAE (s)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([PANEL_A_LABELS[name] for name in PANEL_A_ORDER])
    ax_a.set_ylim(24.90, 25.40)
    ax_a.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35, zorder=0)
    ax_a.set_axisbelow(True)

    for bar, value, error in zip(bars_a, panel_a_values, panel_a_errors):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + 0.012,
            f"{value:.3f}\n±{error:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            linespacing=0.9,
        )

    # Panel B: structural ablation and atom-bond control.
    ax_b = fig.add_subplot(gs[0, 1])
    y = np.arange(len(PANEL_B_ORDER))
    bars_b = ax_b.barh(
        y,
        panel_b_values,
        height=0.62,
        color=colors_b,
        edgecolor="#555555",
        linewidth=0.6,
        zorder=3,
    )

    ax_b.set_title("B   Structural ablation", loc="left", fontweight="bold")
    ax_b.set_xlabel("MAE (s)")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([PANEL_B_LABELS[name] for name in PANEL_B_ORDER])
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 43.5)
    ax_b.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.35, zorder=0)
    ax_b.set_axisbelow(True)

    for index, (bar, value, delta) in enumerate(
        zip(bars_b, panel_b_values, panel_b_deltas)
    ):
        label = f"{value:.3f}"
        if index > 0:
            label += f"  (Δ=+{delta:.3f})"
        ax_b.text(
            value + 0.35,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="left",
            va="center",
            fontsize=6.7,
        )

    for axis in (ax_a, ax_b):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(width=0.7, length=3)

    out_pdf = out_dir / "fig5_ablation_analysis.pdf"
    out_png = out_dir / "fig5_ablation_analysis.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print("[INPUT]", dualview_csv.relative_to(ROOT))
    print("[INPUT]", structural_csv.relative_to(ROOT))
    print("[SAVE]", out_pdf.relative_to(ROOT))
    print("[SAVE]", out_png.relative_to(ROOT))


if __name__ == "__main__":
    main()

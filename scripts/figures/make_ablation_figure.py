#!/usr/bin/env python3
"""Plot the final five-run structural controls from a computed summary table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = [
    "Full TC-TopoRT",
    "w/o explicit ring 2-cells",
    "parameter-matched atom-bond GINE",
]
LABELS = ["Full", "w/o explicit\nring 2-cells", "matched GINE"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/analysis/structural_ablation.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/figures/structural_ablation")
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input).set_index("variant")
    missing = [variant for variant in ORDER if variant not in frame.index]
    if missing:
        raise ValueError(f"Structural summary is missing variants: {missing}")
    means = [float(frame.loc[name, "mae_mean_seconds"]) for name in ORDER]
    errors = [float(frame.loc[name, "mae_sample_sd_seconds"]) for name in ORDER]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.4, 3.5))
    x = np.arange(len(ORDER))
    bars = axis.bar(
        x, means, yerr=errors, capsize=3, width=0.62,
        color=["#4C78A8", "#72A0C1", "#9B8ABF"], edgecolor="#3A4654",
    )
    axis.set_xticks(x, LABELS)
    axis.set_ylabel("MAE (s)")
    axis.set_ylim(min(means) - 0.15, max(means) + 0.18)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, means):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.3f}", ha="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(args.output_dir / "structural_ablation.pdf")
    figure.savefig(args.output_dir / "structural_ablation.png", dpi=400)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


SCRATCH_ROWS = [
    ["FEM_short_73", 73, 99.4390],
    ["UniToyama_Atlantis_143", 143, 72.1139],
    ["FEM_long_412", 412, 117.1540],
    ["Eawag_XBridgeC18_364", 364, 58.5117],
    ["LIFE_old_194", 194, 11.6040],
    ["MTBLS87_147", 147, 69.4439],
    ["LIFE_new_184", 184, 14.7011],
    ["Cao_HILIC_116", 116, 67.7614],
    ["IPB_Halle_82", 82, 13.0508],
    ["FEM_lipids_72", 72, 55.7618],
]

# 当前主文 Table 2 的 6 个 fixed raw AutoSelect 新 TL 结果
FIXED6_TL = {
    "Eawag_XBridgeC18_364": 47.218,
    "FEM_lipids_72": 51.907,
    "FEM_long_412": 88.493,
    "IPB_Halle_82": 13.340,
    "LIFE_new_184": 13.341,
    "LIFE_old_194": 8.105,
}

MISSING4 = {
    "FEM_short_73",
    "UniToyama_Atlantis_143",
    "MTBLS87_147",
    "Cao_HILIC_116",
}


def read_missing4_tl(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing missing4 transfer summary: {path}\n"
            "请先运行 134_external_tcdv_transfer_only.py 补跑 4 个 TL 数据集。"
        )

    df = pd.read_csv(path)

    if "dataset_name" not in df.columns:
        raise KeyError(f"dataset_name not found in {path}")

    if "mae_mean" in df.columns:
        mae_col = "mae_mean"
    elif "mae" in df.columns:
        mae_col = "mae"
    else:
        raise KeyError(f"Cannot find MAE column in {path}; expected mae_mean or mae")

    out = {}
    for _, r in df.iterrows():
        ds = str(r["dataset_name"])
        if ds in MISSING4:
            out[ds] = float(r[mae_col])

    missing = sorted(MISSING4 - set(out))
    if missing:
        raise RuntimeError(f"Missing TL rows for: {missing}")

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--missing4_csv",
        default="experiments_transfer_effectiveness/fixed_raw_autoselect_missing4_cvseed1/external_table2_fixed_raw_autoselect_summary.csv",
    )
    ap.add_argument(
        "--out_dir",
        default="experiments_transfer_effectiveness/tl_vs_scratch_final",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tl_map = dict(FIXED6_TL)
    tl_map.update(read_missing4_tl(Path(args.missing4_csv)))

    rows = []
    for ds, n, scratch_mae in SCRATCH_ROWS:
        if ds not in tl_map:
            raise RuntimeError(f"No TL result for {ds}")

        tl_mae = float(tl_map[ds])
        rows.append({
            "dataset_name": ds,
            "n": int(n),
            "mae_Scratch_random_init": float(scratch_mae),
            "mae_TL_pretrained": tl_mae,
            "MAE_improvement_s": float(scratch_mae - tl_mae),
            "TL_better_MAE": bool(scratch_mae > tl_mae),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("MAE_improvement_s", ascending=False).reset_index(drop=True)

    csv_path = out_dir / "tl_vs_scratch_summary.csv"
    md_path = out_dir / "tl_vs_scratch_summary.md"
    png_path = out_dir / "fig_tl_vs_scratch_bar.png"
    pdf_path = out_dir / "fig_tl_vs_scratch_bar.pdf"

    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False, floatfmt=".3f"))

    plot_df = df.sort_values("MAE_improvement_s", ascending=True)

    plt.figure(figsize=(9.0, 5.6))
    plt.barh(plot_df["dataset_name"], plot_df["MAE_improvement_s"])
    plt.axvline(0, linewidth=1.0)
    plt.xlabel("MAE improvement from transfer learning, scratch - TL (s)")
    plt.title("Transfer learning versus scratch training")

    for i, v in enumerate(plot_df["MAE_improvement_s"]):
        x = v + 0.35 if v >= 0 else v - 0.35
        ha = "left" if v >= 0 else "right"
        plt.text(x, i, f"{v:+.3f}", va="center", ha=ha, fontsize=8)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print("\n=== TL vs scratch final summary ===")
    print(df.to_string(index=False))

    print("\n=== Overall ===")
    print("num_datasets:", len(df))
    print("TL better:", int(df["TL_better_MAE"].sum()), "/", len(df))
    print("Scratch better:", int((~df["TL_better_MAE"]).sum()), "/", len(df))

    print("\n[SAVE]", csv_path)
    print("[SAVE]", md_path)
    print("[SAVE]", png_path)
    print("[SAVE]", pdf_path)


if __name__ == "__main__":
    main()
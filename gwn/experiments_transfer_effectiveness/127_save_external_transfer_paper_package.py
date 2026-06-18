#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Save external transfer experiment outputs into a paper-ready archive package.

This script preserves:
1. 119 base TL prediction outputs.
2. 122c fixed raw-only no-leak AutoSelect outputs.
3. Paper-ready Table-2-style comparison against ABCoRT-TL.
4. Manifest + README + checksums for later manuscript/reviewer use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


ABCORT_TL = {
    "Eawag_XBridgeC18_364": 45.30,
    "FEM_lipids_72": 85.46,
    "FEM_long_412": 87.16,
    "IPB_Halle_82": 13.81,
    "LIFE_new_184": 15.62,
    "LIFE_old_194": 9.97,
}

DISPLAY_NAME = {
    "Eawag_XBridgeC18_364": "Eawag_XBridgeC18",
    "FEM_lipids_72": "FEM_lipids",
    "FEM_long_412": "FEM_long",
    "IPB_Halle_82": "IPB_Halle",
    "LIFE_new_184": "LIFE_new",
    "LIFE_old_194": "LIFE_old",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_if_exists(src: Path, dst: Path):
    if not src.exists():
        print("[MISS]", src)
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print("[COPY]", src, "->", dst)
    return dst


def metric_table(summary_csv: Path, out_dir: Path):
    df = pd.read_csv(summary_csv)
    df = df[df["method"] == "tcdv_fixed_noleak_autocal"].copy()

    rows = []
    for _, r in df.iterrows():
        ds = r["dataset_name"]
        abc = ABCORT_TL.get(ds)

        row = {
            "dataset_name": ds,
            "display_name": DISPLAY_NAME.get(ds, ds),
            "tcdv_fixed_raw_autoselect_mae": float(r["mae"]),
            "tcdv_fixed_raw_autoselect_medae": float(r["medae"]),
            "tcdv_fixed_raw_autoselect_rmse": float(r["rmse"]),
            "tcdv_fixed_raw_autoselect_r2": float(r["r2"]),
            "tcdv_fixed_raw_autoselect_spearman": float(r["spearman"]),
            "abcort_tl_mae": abc,
        }

        if abc is not None:
            row["delta_mae_tcdv_minus_abcort"] = float(r["mae"]) - float(abc)
            row["better_than_abcort"] = bool(float(r["mae"]) < float(abc))
        else:
            row["delta_mae_tcdv_minus_abcort"] = None
            row["better_than_abcort"] = None

        rows.append(row)

    table = pd.DataFrame(rows)
    table = table.sort_values("dataset_name").reset_index(drop=True)

    table_path = out_dir / "paper_table2_external_transfer_fixed_raw_autoselect.csv"
    table.to_csv(table_path, index=False)

    md_path = out_dir / "paper_table2_external_transfer_fixed_raw_autoselect.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Dataset | TCDV fixed raw AutoSelect MAE | ABCoRT-TL MAE | ΔMAE | Result |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for _, r in table.iterrows():
            delta = r["delta_mae_tcdv_minus_abcort"]
            result = "better" if r["better_than_abcort"] else "competitive"
            f.write(
                f"| {r['display_name']} | "
                f"{r['tcdv_fixed_raw_autoselect_mae']:.3f} | "
                f"{r['abcort_tl_mae']:.2f} | "
                f"{delta:+.3f} | "
                f"{result} |\n"
            )

    return table, table_path, md_path


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--base_dir",
        default="experiments_transfer_effectiveness/results_122_base_tl_all6_seed5_src0to4_cvseed_1",
    )
    ap.add_argument(
        "--rawonly_dir",
        default="experiments_transfer_effectiveness/results_122c_autocal_rawonly_all6_seed5_src0to4_cvseed_1",
    )
    ap.add_argument(
        "--out_dir",
        default="experiments_transfer_effectiveness/paper_saved_external_transfer_fixed_raw_autoselect_cvseed1",
    )
    ap.add_argument("--make_zip", type=int, default=1)

    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    rawonly_dir = Path(args.rawonly_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_out = out_dir / "raw_outputs"
    paper_out = out_dir / "paper_tables"

    files_to_copy = {
        # 119 base outputs
        base_dir / "external_tl_predictions.csv": raw_out / "119_external_tl_predictions.csv",
        base_dir / "external_tl_fold_metrics.csv": raw_out / "119_external_tl_fold_metrics.csv",
        base_dir / "external_tl_metrics_by_run.csv": raw_out / "119_external_tl_metrics_by_run.csv",
        base_dir / "external_tl_summary.csv": raw_out / "119_external_tl_summary.csv",

        # 122c raw-only outputs
        rawonly_dir / "tcdv_fixed_noleak_autocal_summary.csv": raw_out / "122c_rawonly_autoselect_summary.csv",
        rawonly_dir / "tcdv_fixed_noleak_autocal_predictions.csv": raw_out / "122c_rawonly_autoselect_predictions.csv",
        rawonly_dir / "tcdv_fixed_noleak_autocal_selection_counts.csv": raw_out / "122c_rawonly_autoselect_selection_counts.csv",
        rawonly_dir / "tcdv_fixed_noleak_autocal_meta.json": raw_out / "122c_rawonly_autoselect_meta.json",
        rawonly_dir / "tcdv_autocal_prediction_bank.csv": raw_out / "122c_rawonly_prediction_bank.csv",
    }

    copied = []
    for src, dst in files_to_copy.items():
        p = copy_if_exists(src, dst)
        if p is not None:
            copied.append(p)

    summary_csv = rawonly_dir / "tcdv_fixed_noleak_autocal_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    paper_out.mkdir(parents=True, exist_ok=True)
    table, table_path, md_path = metric_table(summary_csv, paper_out)
    copied.extend([table_path, md_path])

    better_n = int(table["better_than_abcort"].sum())
    total_n = int(len(table))

    readme = f"""# External transfer paper result package

Generated: {datetime.now().isoformat(timespec="seconds")}

## Protocol

Fixed no-leak raw AutoSelect external transfer protocol.

- Base predictions are generated by `119_external_tcdv_scratch_vs_tl.py`.
- Final aggregation is generated by `122c_external_tcdv_fixed_autocal_from_wide.py`.
- Final protocol uses `--calib_modes raw` and `--selection_metric mae`.
- For each external held-out fold, the aggregation candidate is selected only using the remaining training folds.
- No additional calibration model is fitted in the final fixed protocol.

## Source experiment directories

- Base TL outputs: `{base_dir}`
- Raw-only AutoSelect outputs: `{rawonly_dir}`

## Main result

TCDV-TopoRT fixed raw AutoSelect is better than ABCoRT-TL on {better_n}/{total_n} external datasets.

Paper-ready table:

- `paper_tables/paper_table2_external_transfer_fixed_raw_autoselect.csv`
- `paper_tables/paper_table2_external_transfer_fixed_raw_autoselect.md`

## Raw preserved files

- `raw_outputs/119_external_tl_predictions.csv`
- `raw_outputs/119_external_tl_fold_metrics.csv`
- `raw_outputs/119_external_tl_metrics_by_run.csv`
- `raw_outputs/119_external_tl_summary.csv`
- `raw_outputs/122c_rawonly_autoselect_summary.csv`
- `raw_outputs/122c_rawonly_autoselect_predictions.csv`
- `raw_outputs/122c_rawonly_autoselect_selection_counts.csv`
- `raw_outputs/122c_rawonly_autoselect_meta.json`
- `raw_outputs/122c_rawonly_prediction_bank.csv`
"""

    readme_path = out_dir / "README_external_transfer_results.md"
    readme_path.write_text(readme, encoding="utf-8")
    copied.append(readme_path)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base_dir),
        "rawonly_dir": str(rawonly_dir),
        "out_dir": str(out_dir),
        "protocol": "fixed no-leak raw AutoSelect",
        "calib_modes": ["raw"],
        "selection_metric": "mae",
        "abcort_tl_reference_mae": ABCORT_TL,
        "better_than_abcort_count": better_n,
        "num_datasets": total_n,
        "files": [],
    }

    for p in sorted(copied):
        manifest["files"].append({
            "path": str(p.relative_to(out_dir)),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        })

    manifest_path = out_dir / "manifest_external_transfer_results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    checksum_path = out_dir / "SHA256SUMS.txt"
    with open(checksum_path, "w", encoding="utf-8") as f:
        for item in manifest["files"]:
            f.write(f"{item['sha256']}  {item['path']}\n")

    if int(args.make_zip) == 1:
        zip_base = out_dir
        zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=out_dir)
        print("[ZIP]", zip_path)

    print("\n=== Paper table ===")
    print(table.to_string(index=False))

    print("\n[SAVE DIR]", out_dir)
    print("[SAVE]", manifest_path)
    print("[SAVE]", checksum_path)


if __name__ == "__main__":
    main()

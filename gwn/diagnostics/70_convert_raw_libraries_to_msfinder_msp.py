#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import math
import re
from pathlib import Path

import pandas as pd


def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip().replace("\r", " ").replace("\n", " ")


def is_valid(x):
    return x is not None and (not pd.isna(x)) and str(x).strip() and str(x).strip().lower() != "nan"


def build_colmap(columns):
    return {str(c).strip().lower(): c for c in columns}


def get_col(row, colmap, candidates, default=""):
    for c in candidates:
        key = c.strip().lower()
        if key in colmap:
            v = row.get(colmap[key], default)
            if is_valid(v):
                return v
    return default


def to_float(x, default=None):
    try:
        if not is_valid(x):
            return default
        v = float(str(x).strip())
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def ion_mode_norm(x):
    s = clean(x).upper()
    if s in ["N", "NEG", "NEGATIVE", "-"]:
        return "Negative"
    if s in ["P", "POS", "POSITIVE", "+"]:
        return "Positive"
    if "NEG" in s:
        return "Negative"
    if "POS" in s:
        return "Positive"
    return clean(x)


def spectrum_type_norm(x):
    s = clean(x)
    if not s:
        return "MS2"
    return s


def parse_peaks(x):
    """
    Supports common formats:
    1) "['107.01 10', '108.02 50']"
    2) "107.01 10;108.02 50"
    3) "107.01:10 108.02:50"
    4) newline separated
    5) a flat sequence of numbers: mz inten mz inten ...
    """
    if not is_valid(x):
        return []

    raw = str(x).strip()
    items = None

    # Python list string
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, (list, tuple)):
            items = [str(v) for v in obj]
    except Exception:
        items = None

    peaks = []

    if items is not None:
        for item in items:
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", str(item))
            if len(nums) >= 2:
                mz = to_float(nums[0])
                inten = to_float(nums[1])
                if mz is not None and inten is not None and mz > 0:
                    peaks.append((mz, inten))
    else:
        raw2 = raw.replace("\\n", "\n")

        # colon pairs: 100.1:23.4
        colon_pairs = re.findall(
            r"([-+]?\d*\.\d+|[-+]?\d+)\s*[:]\s*([-+]?\d*\.\d+|[-+]?\d+)",
            raw2,
        )
        if colon_pairs:
            for a, b in colon_pairs:
                mz = to_float(a)
                inten = to_float(b)
                if mz is not None and inten is not None and mz > 0:
                    peaks.append((mz, inten))
        else:
            # split by semicolon / newline / vertical bar
            segs = re.split(r"[;\n|]+", raw2)
            if len(segs) > 1:
                for seg in segs:
                    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", seg)
                    if len(nums) >= 2:
                        mz = to_float(nums[0])
                        inten = to_float(nums[1])
                        if mz is not None and inten is not None and mz > 0:
                            peaks.append((mz, inten))
            else:
                # final fallback: all numbers paired sequentially
                nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", raw2)
                if len(nums) >= 2:
                    for i in range(0, len(nums) - 1, 2):
                        mz = to_float(nums[i])
                        inten = to_float(nums[i + 1])
                        if mz is not None and inten is not None and mz > 0:
                            peaks.append((mz, inten))

    # remove invalid and sort by m/z
    clean_peaks = []
    for mz, inten in peaks:
        if mz is not None and inten is not None and math.isfinite(mz) and math.isfinite(inten) and mz > 0:
            clean_peaks.append((float(mz), float(inten)))

    clean_peaks = sorted(clean_peaks, key=lambda z: z[0])
    return clean_peaks


def read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        xl = pd.ExcelFile(path)
        sheet = "mona_filter_result" if "mona_filter_result" in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet)
        return df, sheet

    # CSV
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        df = pd.read_csv(path, engine="python")
    return df, ""


def infer_rt_seconds(row, colmap, dataset_name):
    # Prefer RT column because previous audit treated RT as seconds.
    rt = get_col(row, colmap, ["RT", "rt"], default="")
    if is_valid(rt):
        return to_float(rt)

    # RIKEN sometimes has retention time in minutes.
    rt_min = get_col(row, colmap, ["retention time", "Retention time", "retention_time"], default="")
    if is_valid(rt_min):
        v = to_float(rt_min)
        if v is not None:
            return v * 60.0

    return None


def write_record(f, rec):
    f.write(f"NAME: {rec['name']}\n")
    f.write(f"SCANNUMBER: {rec['query_id']}\n")
    f.write(f"PRECURSORMZ: {rec['precursor_mz']:.6f}\n")

    if rec["precursor_type"]:
        f.write(f"PRECURSORTYPE: {rec['precursor_type']}\n")
    if rec["ion_mode"]:
        f.write(f"IONMODE: {rec['ion_mode']}\n")
    if rec["spectrum_type"]:
        f.write(f"SPECTRUMTYPE: {rec['spectrum_type']}\n")
    if rec["collision_energy"]:
        f.write(f"COLLISIONENERGY: {rec['collision_energy']}\n")
    if rec["formula"]:
        f.write(f"FORMULA: {rec['formula']}\n")
    if rec["smiles"]:
        f.write(f"SMILES: {rec['smiles']}\n")
    if rec["inchi"]:
        f.write(f"INCHI: {rec['inchi']}\n")
    if rec["inchikey"]:
        f.write(f"INCHIKEY: {rec['inchikey']}\n")

    # MS-FINDER GUI shows retention time in minutes.
    if rec["rt_sec"] is not None:
        f.write(f"RETENTIONTIME: {rec['rt_sec'] / 60.0:.6f}\n")

    f.write(
        "COMMENT: "
        f"dataset={rec['dataset']}; "
        f"source_file={rec['source_file']}; "
        f"source_row={rec['source_row']}; "
        f"RT_sec={rec['rt_sec'] if rec['rt_sec'] is not None else ''}\n"
    )

    f.write(f"Num Peaks: {len(rec['peaks'])}\n")
    for mz, inten in rec["peaks"]:
        f.write(f"{mz:.6f} {inten:.6f}\n")
    f.write("\n")


def convert_one_file(path, out_dir):
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = path.stem
    df, sheet = read_table(path)
    colmap = build_colmap(df.columns)

    written = []
    skipped = []

    msp_path = out_dir / f"{dataset}.msp"

    with msp_path.open("w", encoding="utf-8", newline="\n") as f:
        for i, row in df.iterrows():
            name = clean(get_col(row, colmap, ["NAME", "name", "compound_name", "compound name"], default=""))
            if not name:
                name = f"{dataset}_{i}"

            precursor_mz = to_float(get_col(row, colmap, [
                "PrecursorMZ", "precursor mz", "precursor m/z", "precursor_mz", "PRECURSORMZ"
            ], default=""))

            precursor_type = clean(get_col(row, colmap, [
                "Precursor_type", "precursor type", "PRECURSORTYPE", "adduct"
            ], default=""))

            ion_mode = ion_mode_norm(get_col(row, colmap, [
                "Ion_mode", "ion mode", "ionization mode", "IONMODE"
            ], default=""))

            spectrum_type = spectrum_type_norm(get_col(row, colmap, [
                "Spectrum_type", "spectrum type", "SPECTRUMTYPE"
            ], default="MS2"))

            collision_energy = clean(get_col(row, colmap, [
                "Collision_energy", "collision energy", "collision_energy"
            ], default=""))

            formula = clean(get_col(row, colmap, [
                "Formula", "formula", "molecular formula"
            ], default=""))

            smiles = clean(get_col(row, colmap, [
                "SMILES", "smiles", "computedSMILES", "computed_smiles"
            ], default=""))

            inchi = clean(get_col(row, colmap, [
                "InChI", "inchi"
            ], default=""))

            inchikey = clean(get_col(row, colmap, [
                "InChIKey", "inchikey", "INCHIKEY"
            ], default=""))

            rt_sec = infer_rt_seconds(row, colmap, dataset)

            peaks_raw = get_col(row, colmap, [
                "Peaks", "peaks", "spectrum", "Spectrum", "MSMS", "msms", "MS2", "ms2"
            ], default="")
            peaks = parse_peaks(peaks_raw)

            query_id = f"{dataset}_{i}"

            reason = []
            if precursor_mz is None:
                reason.append("missing_precursor_mz")
            if len(peaks) == 0:
                reason.append("missing_or_invalid_peaks")

            if reason:
                skipped.append({
                    "dataset": dataset,
                    "source_file": str(path),
                    "source_row": i,
                    "name": name,
                    "reason": "|".join(reason),
                    "has_precursor_mz": int(precursor_mz is not None),
                    "n_peaks": len(peaks),
                })
                continue

            rec = {
                "dataset": dataset,
                "source_file": str(path),
                "source_row": i,
                "query_id": query_id,
                "name": name,
                "precursor_mz": precursor_mz,
                "precursor_type": precursor_type,
                "ion_mode": ion_mode,
                "spectrum_type": spectrum_type,
                "collision_energy": collision_energy,
                "formula": formula,
                "smiles": smiles,
                "inchi": inchi,
                "inchikey": inchikey,
                "rt_sec": rt_sec,
                "peaks": peaks,
            }

            write_record(f, rec)

            written.append({
                "dataset": dataset,
                "query_id": query_id,
                "source_file": str(path),
                "source_row": i,
                "name": name,
                "precursor_mz": precursor_mz,
                "precursor_type": precursor_type,
                "ion_mode": ion_mode,
                "formula": formula,
                "smiles": smiles,
                "inchi": inchi,
                "inchikey": inchikey,
                "rt_sec": rt_sec,
                "rt_min_written": rt_sec / 60.0 if rt_sec is not None else None,
                "n_peaks": len(peaks),
                "msp_file": str(msp_path),
            })

    written_df = pd.DataFrame(written)
    skipped_df = pd.DataFrame(skipped)

    print(f"[{dataset}] input={df.shape}, written={len(written_df)}, skipped={len(skipped_df)}, msp={msp_path}")

    return msp_path, written_df, skipped_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="external_data/predret10/raw_libraries",
        help="Directory containing MassBank1.csv, MassBank2.csv, MetaboBase.csv, RIKEN_MONA.xlsx",
    )
    parser.add_argument(
        "--out_dir",
        default="experiments_candidate_filtering/msfinder_queries",
        help="Output directory for MSP and metadata",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        input_dir / "MetaboBase.csv",
        input_dir / "RIKEN_MONA.xlsx",
        input_dir / "MassBank1.csv",
        input_dir / "MassBank2.csv",
    ]

    all_meta = []
    all_skip = []
    msp_files = []

    for path in files:
        if not path.exists():
            print(f"[WARN] missing: {path}")
            continue
        msp_path, meta_df, skip_df = convert_one_file(path, out_dir)
        msp_files.append(msp_path)
        all_meta.append(meta_df)
        all_skip.append(skip_df)

    meta = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
    skipped = pd.concat(all_skip, ignore_index=True) if all_skip else pd.DataFrame()

    meta_path = out_dir / "msfinder_query_metadata.csv"
    skip_path = out_dir / "msfinder_query_skipped.csv"
    combined_path = out_dir / "ALL_raw_libraries_msfinder_queries.msp"

    meta.to_csv(meta_path, index=False)
    skipped.to_csv(skip_path, index=False)

    with combined_path.open("w", encoding="utf-8", newline="\n") as out:
        for msp in msp_files:
            if not msp.exists():
                continue
            text = msp.read_text(encoding="utf-8")
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")

    print("=" * 80)
    print(f"Saved metadata: {meta_path} shape={meta.shape}")
    print(f"Saved skipped:  {skip_path} shape={skipped.shape}")
    print(f"Saved combined MSP: {combined_path}")
    print("=" * 80)

    if len(meta) > 0:
        print("\n[Written by dataset]")
        print(meta.groupby("dataset")["query_id"].count().to_string())

    if len(skipped) > 0:
        print("\n[Skipped by dataset/reason]")
        print(skipped.groupby(["dataset", "reason"]).size().to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import csv
import json
import pandas as pd

ROOT = Path(".").resolve()
OUT = Path("experiments_candidate_filtering/metabobase_full45_recovery")
OUT.mkdir(parents=True, exist_ok=True)

MISSING = [
    "BXPBSBBFPNFTFT-UHFFFAOYSA-N",
    "FDRBRRRBUPLHXEM-PUVROTEMSA-N",
    "GAMYVSCDDILXAQW-MIUGBVLSSA-N",
    "GYLUFQJZYAJODI-UHFFFAOYSA-N",
    "KXPQYWKYYDYOCCQ-UHFFFAOYSA-N",
    "UKMSUNONTOPQIO-UHFFFAOYSA-N",
]

MISSING14 = [x.split("-")[0] for x in MISSING]
KEYS = sorted(set(MISSING + MISSING14))

SCAN_EXT = {
    ".csv", ".tsv", ".txt", ".msp", ".sdf", ".json", ".md", ".mgf", ".ms", ".mat"
}

SKIP_DIR_PARTS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "processed_r6_Full46D_Embedded_E",
    "checkpoints",
}

MAX_FILE_MB = 300


def should_skip(path: Path):
    parts = set(path.parts)
    if parts & SKIP_DIR_PARTS:
        return True
    if path.suffix.lower() not in SCAN_EXT:
        return True
    try:
        mb = path.stat().st_size / 1024 / 1024
    except Exception:
        return True
    if mb > MAX_FILE_MB:
        return True
    return False


def scan_text_file(path: Path):
    hits = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                for k in KEYS:
                    if k in line:
                        hits.append({
                            "file": str(path),
                            "line_no": i,
                            "key": k,
                            "line": line.strip()[:2000],
                        })
    except Exception as e:
        hits.append({
            "file": str(path),
            "line_no": -1,
            "key": "READ_ERROR",
            "line": repr(e),
        })
    return hits


def try_extract_table_rows(path: Path, keys):
    rows = []
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return rows

    try:
        df_iter = pd.read_csv(path, sep=sep, dtype=str, chunksize=50000, low_memory=False)
        for ci, chunk in enumerate(df_iter):
            chunk = chunk.fillna("")
            joined = chunk.astype(str).agg(" | ".join, axis=1)
            mask = joined.apply(lambda x: any(k in x for k in keys))
            sub = chunk[mask].copy()
            for local_idx, r in sub.iterrows():
                val = " | ".join(str(x) for x in r.values)
                matched = [k for k in keys if k in val]
                rows.append({
                    "file": str(path),
                    "chunk": ci,
                    "row_index": int(local_idx),
                    "matched_keys": ";".join(matched),
                    "columns": json.dumps(list(chunk.columns), ensure_ascii=False),
                    "row_json": json.dumps(r.to_dict(), ensure_ascii=False),
                })
    except Exception:
        pass
    return rows


def main():
    files = []
    for p in ROOT.rglob("*"):
        if p.is_file() and not should_skip(p):
            files.append(p)

    print("=" * 100)
    print("root:", ROOT)
    print("files to scan:", len(files))
    print("missing full InChIKeys:", len(MISSING))
    print("=" * 100)

    line_hits = []
    table_hits = []

    for n, p in enumerate(files, start=1):
        if n % 500 == 0:
            print("scanned", n, "/", len(files))

        hs = scan_text_file(p)
        if hs:
            line_hits.extend(hs)

        th = try_extract_table_rows(p, KEYS)
        if th:
            table_hits.extend(th)

    line_df = pd.DataFrame(line_hits)
    table_df = pd.DataFrame(table_hits)

    line_out = OUT / "missing6_line_hits.tsv"
    table_out = OUT / "missing6_table_row_hits.tsv"
    summary_out = OUT / "missing6_hit_summary.csv"

    if len(line_df):
        line_df.to_csv(line_out, sep="\t", index=False)
    else:
        pd.DataFrame(columns=["file", "line_no", "key", "line"]).to_csv(line_out, sep="\t", index=False)

    if len(table_df):
        table_df.to_csv(table_out, sep="\t", index=False)
    else:
        pd.DataFrame(columns=["file", "chunk", "row_index", "matched_keys", "columns", "row_json"]).to_csv(table_out, sep="\t", index=False)

    # summary by missing full key and first block
    rows = []
    for full, k14 in zip(MISSING, MISSING14):
        hit_line_full = line_df[line_df["key"].eq(full)] if len(line_df) else pd.DataFrame()
        hit_line_14 = line_df[line_df["key"].eq(k14)] if len(line_df) else pd.DataFrame()

        rows.append({
            "missing_inchikey": full,
            "inchikey14": k14,
            "line_hits_full": len(hit_line_full),
            "line_hits_14": len(hit_line_14),
            "files_full": ";".join(sorted(hit_line_full["file"].unique().tolist())) if len(hit_line_full) else "",
            "files_14": ";".join(sorted(hit_line_14["file"].unique().tolist())) if len(hit_line_14) else "",
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_out, index=False)

    print("\n[SUMMARY]")
    print(summary.to_string(index=False))

    print("\nTop line hits:")
    if len(line_df):
        print(line_df.head(80).to_string(index=False))
    else:
        print("NO LINE HITS")

    print("\nTop table hits:")
    if len(table_df):
        cols = ["file", "matched_keys", "row_json"]
        print(table_df[cols].head(40).to_string(index=False))
    else:
        print("NO TABLE HITS")

    print("\nSaved:")
    print(line_out)
    print(table_out)
    print(summary_out)
    print("=" * 100)


if __name__ == "__main__":
    main()

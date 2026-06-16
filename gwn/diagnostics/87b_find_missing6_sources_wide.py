#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import json
import os

SEARCH_ROOTS = [
    Path("/home/lwh/projects/lrq_q/ABCoRT-main"),
    Path("/home/lwh/projects/lrq_q"),
]

OUT = Path("experiments_candidate_filtering/metabobase_full45_recovery_wide")
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

EXTS = {
    ".csv", ".tsv", ".txt", ".msp", ".sdf", ".json", ".md", ".mgf", ".ms",
    ".mat", ".xlsx", ".xls", ".xml", ".html", ".htm"
}

SKIP_PARTS = {
    ".git", "__pycache__", ".ipynb_checkpoints",
    "processed_r6_Full46D_Embedded_E",
    "checkpoints",
    "node_modules",
}

MAX_MB = 600


def should_skip(p: Path):
    if any(x in p.parts for x in SKIP_PARTS):
        return True
    if p.suffix.lower() not in EXTS:
        return True
    try:
        if p.stat().st_size / 1024 / 1024 > MAX_MB:
            return True
    except Exception:
        return True
    return False


def scan_text(path):
    hits = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                for k in KEYS:
                    if k in line:
                        hits.append({
                            "file": str(path),
                            "line_no": i,
                            "key": k,
                            "line": line.strip()[:3000],
                        })
    except Exception:
        pass
    return hits


def scan_csv_tsv(path):
    rows = []
    if path.suffix.lower() not in [".csv", ".tsv"]:
        return rows
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        for ci, chunk in enumerate(pd.read_csv(path, sep=sep, dtype=str, chunksize=50000, low_memory=False)):
            chunk = chunk.fillna("")
            joined = chunk.astype(str).agg(" | ".join, axis=1)
            mask = joined.apply(lambda x: any(k in x for k in KEYS))
            sub = chunk[mask]
            for idx, r in sub.iterrows():
                val = " | ".join(str(x) for x in r.values)
                rows.append({
                    "file": str(path),
                    "chunk": ci,
                    "row_index": int(idx),
                    "matched_keys": ";".join([k for k in KEYS if k in val]),
                    "columns": json.dumps(list(chunk.columns), ensure_ascii=False),
                    "row_json": json.dumps(r.to_dict(), ensure_ascii=False),
                })
    except Exception:
        pass
    return rows


def scan_excel(path):
    rows = []
    if path.suffix.lower() not in [".xlsx", ".xls"]:
        return rows
    try:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet, dtype=str).fillna("")
            joined = df.astype(str).agg(" | ".join, axis=1)
            mask = joined.apply(lambda x: any(k in x for k in KEYS))
            sub = df[mask]
            for idx, r in sub.iterrows():
                val = " | ".join(str(x) for x in r.values)
                rows.append({
                    "file": str(path),
                    "sheet": sheet,
                    "row_index": int(idx),
                    "matched_keys": ";".join([k for k in KEYS if k in val]),
                    "columns": json.dumps(list(df.columns), ensure_ascii=False),
                    "row_json": json.dumps(r.to_dict(), ensure_ascii=False),
                })
    except Exception:
        pass
    return rows


def main():
    files = []
    seen = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and not should_skip(p):
                sp = str(p.resolve())
                if sp not in seen:
                    seen.add(sp)
                    files.append(p)

    print("=" * 100)
    print("search roots:", [str(x) for x in SEARCH_ROOTS])
    print("files to scan:", len(files))
    print("=" * 100)

    line_hits = []
    table_hits = []
    excel_hits = []

    for i, p in enumerate(files, 1):
        if i % 1000 == 0:
            print("scanned", i, "/", len(files))

        line_hits.extend(scan_text(p))
        table_hits.extend(scan_csv_tsv(p))
        excel_hits.extend(scan_excel(p))

    line_df = pd.DataFrame(line_hits)
    table_df = pd.DataFrame(table_hits)
    excel_df = pd.DataFrame(excel_hits)

    line_df.to_csv(OUT / "missing6_wide_line_hits.tsv", sep="\t", index=False)
    table_df.to_csv(OUT / "missing6_wide_table_hits.tsv", sep="\t", index=False)
    excel_df.to_csv(OUT / "missing6_wide_excel_hits.tsv", sep="\t", index=False)

    summary_rows = []
    for full, k14 in zip(MISSING, MISSING14):
        files_full = []
        files14 = []

        for df in [line_df, table_df, excel_df]:
            if len(df) == 0:
                continue
            if "key" in df.columns:
                files_full += df[df["key"].eq(full)]["file"].unique().tolist()
                files14 += df[df["key"].eq(k14)]["file"].unique().tolist()
            elif "matched_keys" in df.columns:
                files_full += df[df["matched_keys"].astype(str).str.contains(full, regex=False)]["file"].unique().tolist()
                files14 += df[df["matched_keys"].astype(str).str.contains(k14, regex=False)]["file"].unique().tolist()

        summary_rows.append({
            "missing_inchikey": full,
            "inchikey14": k14,
            "n_files_full": len(set(files_full)),
            "n_files_14": len(set(files14)),
            "files_full": ";".join(sorted(set(files_full))),
            "files_14": ";".join(sorted(set(files14))),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "missing6_wide_summary.csv", index=False)

    print("\n[SUMMARY]")
    print(summary.to_string(index=False))

    print("\n[Excel hits]")
    if len(excel_df):
        print(excel_df.head(50).to_string(index=False))
    else:
        print("NO EXCEL HITS")

    print("\n[Table hits]")
    if len(table_df):
        print(table_df.head(80).to_string(index=False))
    else:
        print("NO TABLE HITS")

    print("\nSaved dir:", OUT)
    print("=" * 100)


if __name__ == "__main__":
    main()

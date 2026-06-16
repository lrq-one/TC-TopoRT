#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import json
import pandas as pd

REPO = Path("external_repos/RiverCCC_ABCoRT").resolve()
OUT = Path("experiments_candidate_filtering/river_abcort_split_inspect")
OUT.mkdir(parents=True, exist_ok=True)

KEYWORDS = [
    "MetaboBase", "MetaboBASE", "metabobase",
    "RIKEN", "PlaSM",
    "train_test_split", "random_state", "test_size",
    "shuffle", "seed", "45", "181", "226",
]

MISSING6 = [
    "BXPBSBBFPNFTFT-UHFFFAOYSA-N",
    "FDRBRRRBUPLHXEM-PUVROTEMSA-N",
    "GAMYVSCDDILXAQW-MIUGBVLSSA-N",
    "GYLUFQJZYAJODI-UHFFFAOYSA-N",
    "KXPQYWKYYDYOCCQ-UHFFFAOYSA-N",
    "UKMSUNONTOPQIO-UHFFFAOYSA-N",
]

EXT_TEXT = {".py", ".ipynb", ".md", ".txt", ".csv", ".tsv", ".json", ".yaml", ".yml", ".R", ".m"}
EXT_TABLE = {".csv", ".tsv", ".xlsx", ".xls"}

SKIP = {".git", "__pycache__", ".ipynb_checkpoints"}


def skip(p):
    return any(x in p.parts for x in SKIP)


def read_text(p):
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def scan_code_and_text():
    rows = []
    for p in REPO.rglob("*"):
        if not p.is_file() or skip(p) or p.suffix.lower() not in EXT_TEXT:
            continue
        text = read_text(p)
        if not text:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            hit_keys = [k for k in KEYWORDS if k.lower() in line.lower()]
            if hit_keys:
                rows.append({
                    "file": str(p.relative_to(REPO)),
                    "line_no": i,
                    "hit_keys": ";".join(hit_keys),
                    "line": line[:1000],
                })
    return pd.DataFrame(rows)


def table_summary():
    rows = []
    missing_hits = []

    for p in REPO.rglob("*"):
        if not p.is_file() or skip(p) or p.suffix.lower() not in EXT_TABLE:
            continue

        try:
            if p.suffix.lower() == ".tsv":
                dfs = {"__tsv__": pd.read_csv(p, sep="\t", dtype=str, low_memory=False)}
            elif p.suffix.lower() == ".csv":
                dfs = {"__csv__": pd.read_csv(p, dtype=str, low_memory=False)}
            else:
                xls = pd.ExcelFile(p)
                dfs = {s: pd.read_excel(p, sheet_name=s, dtype=str) for s in xls.sheet_names}
        except Exception as e:
            continue

        for sheet, df in dfs.items():
            df = df.fillna("")
            cols = list(df.columns)
            n = len(df)
            joined_all = " ".join(cols).lower() + " " + " ".join(df.astype(str).head(20).agg(" ".join, axis=1).tolist()).lower()

            looks_related = any(k.lower() in joined_all for k in ["metabobase", "metabobase", "riken", "smiles", "inchi", "retention", "rt"])
            interesting_n = n in [45, 181, 226, 85, 341, 426]

            if looks_related or interesting_n:
                rows.append({
                    "file": str(p.relative_to(REPO)),
                    "sheet": sheet,
                    "n_rows": n,
                    "n_cols": len(cols),
                    "columns": json.dumps(cols, ensure_ascii=False),
                    "head_json": json.dumps(df.head(3).to_dict(orient="records"), ensure_ascii=False)[:3000],
                })

            # search missing6
            joined = df.astype(str).agg(" | ".join, axis=1)
            mask = joined.apply(lambda x: any(k in x for k in MISSING6))
            sub = df[mask]
            for idx, r in sub.iterrows():
                val = " | ".join(str(x) for x in r.values)
                missing_hits.append({
                    "file": str(p.relative_to(REPO)),
                    "sheet": sheet,
                    "row_index": int(idx),
                    "matched": ";".join([k for k in MISSING6 if k in val]),
                    "columns": json.dumps(cols, ensure_ascii=False),
                    "row_json": json.dumps(r.to_dict(), ensure_ascii=False),
                })

    return pd.DataFrame(rows), pd.DataFrame(missing_hits)


def main():
    print("=" * 100)
    print("repo:", REPO)
    print("exists:", REPO.exists())
    print("=" * 100)

    if not REPO.exists():
        raise SystemExit("RiverCCC_ABCoRT repo not found. git clone may have failed.")

    text_hits = scan_code_and_text()
    tables, missing_hits = table_summary()

    text_hits.to_csv(OUT / "code_text_keyword_hits.tsv", sep="\t", index=False)
    tables.to_csv(OUT / "table_file_summary.tsv", sep="\t", index=False)
    missing_hits.to_csv(OUT / "missing6_hits_in_river_repo.tsv", sep="\t", index=False)

    print("\n[Possible split/random code hits]")
    if len(text_hits):
        pat = r"train_test_split|random_state|test_size|shuffle|seed|MetaboBase|MetaboBASE|metabobase|45|181|226"
        show = text_hits[text_hits["line"].str.contains(pat, case=False, regex=True, na=False)].copy()
        print(show.head(120).to_string(index=False))
    else:
        print("NO TEXT HITS")

    print("\n[Possible dataset/split tables]")
    if len(tables):
        print(tables.sort_values(["n_rows", "file"]).head(120).to_string(index=False))
    else:
        print("NO TABLE SUMMARY")

    print("\n[Missing6 hits in River repo]")
    if len(missing_hits):
        print(missing_hits.to_string(index=False))
    else:
        print("NO MISSING6 HITS")

    print("\nSaved:")
    print(OUT / "code_text_keyword_hits.tsv")
    print(OUT / "table_file_summary.tsv")
    print(OUT / "missing6_hits_in_river_repo.tsv")
    print("=" * 100)


if __name__ == "__main__":
    main()

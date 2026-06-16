#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

OUT = Path("experiments_candidate_filtering/riken_raw_inspect")
OUT.mkdir(parents=True, exist_ok=True)

ROOT = Path(".")

patterns = [
    "*Structure result-2080.txt",
    "*Structure*2080*.txt",
    "*structure*2080*.txt",
    "*Formula result-2080.txt",
    "*Formula*2080*.txt",
    "*formula*2080*.txt",
]

files = set()
for pat in patterns:
    for p in ROOT.rglob(pat):
        if p.is_file():
            files.add(p)

files = sorted(files)

print("=" * 100)
print("[FOUND FILES]")
for p in files:
    print(p)
print("=" * 100)

if not files:
    print("NO RIKEN MS-FINDER raw files found.")
    print("Run this to locate them:")
    print("find . -iname '*Structure*2080*.txt' -o -iname '*Formula*2080*.txt'")
    raise SystemExit(0)


encodings = ["utf-8-sig", "utf-16", "utf-16le", "utf-16be", "gb18030", "latin1"]
seps = [
    ("tab", "\t"),
    ("comma", ","),
    ("semicolon", ";"),
]


def read_lines_any_encoding(path):
    for enc in encodings:
        try:
            text = path.read_text(encoding=enc, errors="replace")
            lines = text.splitlines()
            if len(lines) > 0:
                return enc, lines
        except Exception:
            pass
    text = path.read_text(errors="replace")
    return "default", text.splitlines()


for p in files:
    print("\n" + "=" * 100)
    print("[FILE]", p)
    print("size:", p.stat().st_size)
    print("=" * 100)

    enc, lines = read_lines_any_encoding(p)

    print("\n[RAW TEXT ENCODING USED]", enc)
    print("[FIRST 80 RAW LINES]")
    for i, line in enumerate(lines[:80], start=1):
        print(f"{i:04d}: {line}")

    for enc_try in encodings:
        for sep_name, sep in seps:
            print("\n" + "-" * 100)
            print(f"[try read_csv encoding={enc_try}, sep={sep_name}]")
            try:
                df = pd.read_csv(p, sep=sep, dtype=str, low_memory=False, encoding=enc_try)
                print("shape:", df.shape)
                print("columns:")
                for j, c in enumerate(df.columns):
                    print(f"  {j:02d}: {repr(c)}")
                print("\nhead:")
                print(df.head(8).to_string())

                safe_stem = p.stem.replace(" ", "_").replace("/", "_")
                out_csv = OUT / f"{safe_stem}_{enc_try.replace('-', '')}_{sep_name}_head50.csv"
                df.head(50).to_csv(out_csv, index=False)
                print("saved head:", out_csv)
            except Exception as e:
                print("FAILED:", repr(e))

print("\nSaved head csv files to:", OUT)
print("=" * 100)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import string

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

files = []
for pat in patterns:
    files.extend([p for p in ROOT.rglob(pat) if p.is_file()])
files = sorted(set(files))

print("=" * 100)
print("[FOUND FILES]")
for p in files:
    print(p)
print("=" * 100)

if not files:
    print("NO FILE FOUND. Run:")
    print("find . -iname '*Structure*2080*.txt' -o -iname '*Formula*2080*.txt'")
    raise SystemExit(0)

encodings = [
    "utf-16",
    "utf-16le",
    "utf-16be",
    "utf-8-sig",
    "gb18030",
    "latin1",
]

keywords = [
    "Title", "Precursor", "SMILES", "InChIKey", "Formula",
    "Structure", "Name", "Total score", "RT", "Msp", "MS-FINDER",
    "Ontology", "Comment"
]


def score_text(txt):
    if not txt:
        return -1
    head = txt[:20000]
    printable = sum(ch in string.printable for ch in head)
    keyword_score = sum(head.count(k) for k in keywords) * 1000
    tab_score = head.count("\t") * 5
    newline_score = head.count("\n")
    weird_score = sum("\u4e00" <= ch <= "\u9fff" for ch in head)
    return printable + keyword_score + tab_score + newline_score - weird_score * 3


def best_decode(path):
    raw = path.read_bytes()
    results = []
    for enc in encodings:
        try:
            txt = raw.decode(enc, errors="strict")
            results.append((score_text(txt), enc, txt))
        except Exception:
            pass

    # fallback: 不严格解码，但只作为兜底
    for enc in encodings:
        try:
            txt = raw.decode(enc, errors="replace")
            results.append((score_text(txt) - 100000, enc + "_replace", txt))
        except Exception:
            pass

    results.sort(reverse=True, key=lambda x: x[0])
    return results[0]


for p in files:
    print("\n" + "=" * 100)
    print("[FILE]", p)
    print("size:", p.stat().st_size)

    score, enc, txt = best_decode(p)
    print("best_encoding:", enc)
    print("decode_score:", score)

    lines = txt.splitlines()
    print("\n[FIRST 40 LINES, correctly decoded if possible]")
    for i, line in enumerate(lines[:40], start=1):
        print(f"{i:04d}: {line}")

    decoded_txt = OUT / (p.stem.replace(" ", "_") + "_decoded_preview.txt")
    decoded_txt.write_text("\n".join(lines[:300]), encoding="utf-8")
    print("saved decoded preview:", decoded_txt)

    # 用最佳编码读 tab 表
    enc_clean = enc.replace("_replace", "")
    print("\n[TRY TAB TABLE]")
    try:
        df = pd.read_csv(p, sep="\t", dtype=str, low_memory=False, encoding=enc_clean)
        print("shape:", df.shape)
        print("columns:")
        for j, c in enumerate(df.columns):
            print(f"  {j:02d}: {repr(c)}")
        print("\nhead:")
        print(df.head(10).to_string())

        out_csv = OUT / (p.stem.replace(" ", "_") + "_tab_head100.csv")
        df.head(100).to_csv(out_csv, index=False)
        print("saved:", out_csv)
    except Exception as e:
        print("tab read failed:", repr(e))

print("=" * 100)

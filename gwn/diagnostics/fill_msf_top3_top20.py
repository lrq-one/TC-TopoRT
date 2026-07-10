from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
OUT = ROOT / "gwn/final_paper_tables/Table_S26_MSF_topk_fill_audit.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

DATASETS = ["MetaboBase", "RIKEN-PlaSMA"]

SKIP_DIRS = {
    ".git", "__pycache__", "cleanup_backups",
    "TCDV-TopoRT_cleanup_backups",
}

def skip_path(p: Path) -> bool:
    return bool(set(p.parts) & SKIP_DIRS)

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_col(cols, patterns):
    for c in cols:
        nc = norm(c)
        for p in patterns:
            if re.search(p, nc):
                return c
    return None

def dataset_from_path_or_df(path, df):
    txt = str(path).lower()
    found = []
    for ds in DATASETS:
        if ds.lower().replace("-", "") in txt.replace("-", ""):
            found.append(ds)

    if found:
        return found[0]

    for c in df.columns:
        vals = df[c].astype(str).head(200).str.lower().tolist()
        joined = " ".join(vals)
        if "metabobase" in joined:
            return "MetaboBase"
        if "riken" in joined or "plasma" in joined:
            return "RIKEN-PlaSMA"

    return None

def coerce_bool_true(s):
    x = str(s).strip().lower()
    return x in {"1", "true", "yes", "y", "target", "correct", "real", "matched"}

def compute_topk(ranks, n=None):
    ranks = pd.to_numeric(pd.Series(ranks), errors="coerce").dropna().astype(float)
    if n is None:
        n = len(ranks)
    out = {}
    for k in [1, 3, 5, 10, 20]:
        out[f"Top-{k} (%)"] = 100.0 * float((ranks <= k).sum()) / float(n)
    out["N"] = int(n)
    return out

rows = []

for p in ROOT.rglob("*.csv"):
    if skip_path(p):
        continue

    # final summary tables may not contain query-level rank; include them in scan but they usually fail detection.
    try:
        df = pd.read_csv(p)
    except Exception:
        continue

    if df.empty:
        continue

    cols = list(df.columns)
    ds = dataset_from_path_or_df(p, df)

    # Find query id column.
    query_col = find_col(cols, [
        r"queryid", r"query", r"recordid", r"sampleid", r"caseid", r"qid"
    ])

    # Find original MS-FINDER rank column.
    rank_col = None
    rank_candidates = []
    for c in cols:
        nc = norm(c)
        if ("rank" in nc) and (
            "msf" in nc or "msfinder" in nc or "original" in nc or "initial" in nc or "before" in nc
        ):
            rank_candidates.append(c)

    # Prefer explicit MSF / MS-FINDER rank.
    for c in rank_candidates:
        nc = norm(c)
        if "msf" in nc or "msfinder" in nc:
            rank_col = c
            break
    if rank_col is None and rank_candidates:
        rank_col = rank_candidates[0]

    if rank_col is None:
        continue

    # Case 1: query-level table already has one true MSF rank per query.
    # Usually columns look like Dataset, Query, MSF_rank / MSF_Rank / MS-FINDER rank.
    if query_col is not None:
        tmp = df.copy()
        tmp[rank_col] = pd.to_numeric(tmp[rank_col], errors="coerce")
        tmp = tmp.dropna(subset=[rank_col])

        # If candidate-level rows exist, identify true candidate rows.
        true_col = find_col(cols, [
            r"istrue", r"truecandidate", r"target", r"iscorrect", r"matched"
        ])

        if true_col is not None:
            mask = tmp[true_col].map(coerce_bool_true)
            tmp = tmp[mask]

        if len(tmp) == 0:
            continue

        # If dataset column exists, split by dataset.
        ds_col = find_col(cols, [r"dataset", r"source"])
        if ds_col is not None:
            for ds_name in DATASETS:
                sub = tmp[tmp[ds_col].astype(str).str.lower().str.contains(ds_name.lower().split("-")[0], na=False)]
                if len(sub) == 0:
                    continue
                ranks = sub.groupby(query_col)[rank_col].min()
                if len(ranks) in {45, 85}:
                    res = compute_topk(ranks)
                    rows.append({
                        "dataset": ds_name,
                        "file": str(p.relative_to(ROOT)),
                        "rank_col": rank_col,
                        "query_col": query_col,
                        "true_col": true_col if true_col else "",
                        **res,
                    })
        else:
            ranks = tmp.groupby(query_col)[rank_col].min()
            if len(ranks) in {45, 85}:
                ds_name = ds
                if ds_name is None:
                    ds_name = "MetaboBase" if len(ranks) == 45 else "RIKEN-PlaSMA"
                res = compute_topk(ranks)
                rows.append({
                    "dataset": ds_name,
                    "file": str(p.relative_to(ROOT)),
                    "rank_col": rank_col,
                    "query_col": query_col,
                    "true_col": true_col if true_col else "",
                    **res,
                })

    # Case 2: table itself only has one rank per row but no query column.
    else:
        tmp = df.copy()
        tmp[rank_col] = pd.to_numeric(tmp[rank_col], errors="coerce")
        tmp = tmp.dropna(subset=[rank_col])
        if len(tmp) in {45, 85}:
            ds_name = ds or ("MetaboBase" if len(tmp) == 45 else "RIKEN-PlaSMA")
            res = compute_topk(tmp[rank_col].values)
            rows.append({
                "dataset": ds_name,
                "file": str(p.relative_to(ROOT)),
                "rank_col": rank_col,
                "query_col": "",
                "true_col": "",
                **res,
            })

out = pd.DataFrame(rows)

if out.empty:
    print("[NO QUERY-LEVEL MSF RANK TABLE FOUND]")
    print("请用下面命令人工找 rank 文件：")
    print("find . -type f -name '*.csv' | grep -Ei 'candidate|filter|rank|msf|msfinder|topk|metabobase|riken' | sort")
else:
    out = out.sort_values(["dataset", "N", "file"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)

    print("===== candidate MSF Top-k audit =====")
    show = out[[
        "dataset", "N", "Top-1 (%)", "Top-3 (%)", "Top-5 (%)", "Top-10 (%)", "Top-20 (%)",
        "rank_col", "query_col", "true_col", "file"
    ]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n===== likely final MSF rows =====")
    for ds_name, n_expected in [("MetaboBase", 45), ("RIKEN-PlaSMA", 85)]:
        sub = out[(out["dataset"].eq(ds_name)) & (out["N"].eq(n_expected))]
        if len(sub) == 0:
            print(f"{ds_name}: NOT FOUND")
            continue

        # Prefer rows matching the known Top-1/Top-5/Top-10 from your table.
        if ds_name == "MetaboBase":
            known = (44.44, 75.56, 84.44)
        else:
            known = (47.06, 70.59, 82.35)

        sub = sub.copy()
        sub["score"] = (
            (sub["Top-1 (%)"] - known[0]).abs()
            + (sub["Top-5 (%)"] - known[1]).abs()
            + (sub["Top-10 (%)"] - known[2]).abs()
        )
        row = sub.sort_values("score").iloc[0]

        print(
            f"{ds_name} MSF: "
            f"Top-1={row['Top-1 (%)']:.2f}, "
            f"Top-3={row['Top-3 (%)']:.2f}, "
            f"Top-5={row['Top-5 (%)']:.2f}, "
            f"Top-10={row['Top-10 (%)']:.2f}, "
            f"Top-20={row['Top-20 (%)']:.2f} "
            f" | file={row['file']} | rank_col={row['rank_col']}"
        )

    print("\n[WROTE]", OUT.relative_to(ROOT))

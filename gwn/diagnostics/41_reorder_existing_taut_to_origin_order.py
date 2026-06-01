import argparse
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canon(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def normalize_cols(df):
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    if "smiles" in df.columns and "smile" not in df.columns:
        df.rename(columns={"smiles": "smile"}, inplace=True)
    return df


def rt_key(x):
    return f"{float(x):.4f}"


def load_origin(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = normalize_cols(pd.read_csv(path, engine="python"))

    if "smile" not in df.columns or "rt" not in df.columns:
        raise ValueError(f"{path} must have smile/rt columns, got {df.columns.tolist()}")

    df["rt"] = df["rt"].astype(float)

    rows = []
    for source_row, row in df.iterrows():
        smi = str(row["smile"])
        rt = float(row["rt"])

        if rt <= 300.0:
            continue

        c = canon(smi)
        if c is None:
            continue

        item = row.to_dict()
        item["_source_row"] = int(source_row)
        item["_origin_smile"] = smi
        item["_origin_canon"] = c
        item["_rt_key"] = rt_key(rt)
        item["_key"] = c + "||" + item["_rt_key"]
        rows.append(item)

    out = pd.DataFrame(rows)
    out["_dup_id"] = out.groupby("_key").cumcount()

    print("\n[origin]")
    print("path:", path)
    print("valid rows:", len(out))
    print("head keys:", out["_key"].head().tolist())

    return out


def load_existing_taut(path, audit_path=None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = normalize_cols(pd.read_csv(path, engine="python"))

    if "smile" not in df.columns or "rt" not in df.columns:
        raise ValueError(f"{path} must have smile/rt columns, got {df.columns.tolist()}")

    if "orig_smile" not in df.columns:
        raise ValueError(
            f"{path} has no orig_smile column. "
            f"必须使用 build_tautomer_strict_csv.py 生成的 strict tautomer CSV。"
        )

    # 可选：把 audit 里的 real_changed 等字段带过来
    if audit_path and os.path.exists(audit_path):
        audit = pd.read_csv(audit_path)
        for c in ["raw_changed", "real_changed", "formula_same", "heavy_same", "fallback", "reason"]:
            if c in audit.columns and c not in df.columns and len(audit) == len(df):
                df[c] = audit[c].values

    df["rt"] = df["rt"].astype(float)

    rows = []
    for source_row, row in df.iterrows():
        orig_smi = str(row["orig_smile"])
        taut_smi = str(row["smile"])
        rt = float(row["rt"])

        if rt <= 300.0:
            continue

        c = canon(orig_smi)
        taut_c = canon(taut_smi)

        if c is None or taut_c is None:
            continue

        item = row.to_dict()
        item["_taut_source_row"] = int(source_row)
        item["_orig_canon"] = c
        item["_taut_canon"] = taut_c
        item["_rt_key"] = rt_key(rt)
        item["_key"] = c + "||" + item["_rt_key"]
        rows.append(item)

    out = pd.DataFrame(rows)
    out["_dup_id"] = out.groupby("_key").cumcount()

    print("\n[existing taut]")
    print("path:", path)
    print("valid rows:", len(out))
    print("head keys:", out["_key"].head().tolist())

    return out


def reorder_one(origin_csv, existing_taut_csv, output_csv, audit_csv=None, output_audit_csv=None):
    origin = load_origin(origin_csv)
    taut = load_existing_taut(existing_taut_csv, audit_csv)

    keep_cols = [
        "_key",
        "_dup_id",
        "smile",
        "orig_smile",
        "_orig_canon",
        "_taut_canon",
        "_taut_source_row",
    ]

    extra_cols = []
    for c in ["raw_changed", "real_changed", "formula_same", "heavy_same", "fallback", "reason"]:
        if c in taut.columns:
            extra_cols.append(c)

    taut_small = taut[keep_cols + extra_cols].copy()
    taut_small = taut_small.rename(columns={
        "smile": "_taut_smile",
        "orig_smile": "_taut_orig_smile",
    })

    merged = origin.merge(
        taut_small,
        on=["_key", "_dup_id"],
        how="left",
        suffixes=("", "_taut"),
    )

    print("\n[merge]")
    print("origin rows:", len(origin))
    print("matched taut:", int(merged["_taut_smile"].notna().sum()))

    if merged["_taut_smile"].isna().any():
        bad = merged[merged["_taut_smile"].isna()].head(20)
        print("❌ missing examples:")
        print(bad[["_origin_smile", "rt", "_key", "_dup_id"]].to_string(index=False))
        raise SystemExit(1)

    # canonical identity check
    same = merged["_origin_canon"].astype(str).values == merged["_orig_canon_taut"].astype(str).values
    print("canonical identity matched:", int(same.sum()), "/", len(same))

    if same.sum() != len(same):
        bad_idx = np.where(~same)[0][:20]
        print("❌ identity mismatch examples:")
        print(merged.iloc[bad_idx][[
            "_origin_smile",
            "_taut_orig_smile",
            "_taut_smile",
            "rt",
            "_origin_canon",
            "_orig_canon_taut",
        ]].to_string(index=False))
        raise SystemExit(1)

    # 输出保持 origin CSV 的原始列和顺序，只替换 smile 为 taut smile
    original_cols = [c for c in origin.columns if not c.startswith("_")]
    out = origin[original_cols].copy()

    out["orig_smile"] = merged["_origin_smile"].astype(str).values
    out["smile"] = merged["_taut_smile"].astype(str).values

    # 保留 audit 信息，方便后面分组分析
    for c in extra_cols:
        out[c] = merged[c].values

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("\n✅ saved:", output_csv)
    print("rows:", len(out))
    changed = (merged["_origin_canon"].astype(str).values != merged["_taut_canon"].astype(str).values)
    print("canonical taut changed:", int(changed.sum()), "/", len(changed), "ratio:", float(changed.mean()))

    if output_audit_csv:
        audit_out = pd.DataFrame({
            "valid_i": np.arange(len(merged)),
            "origin_smile": merged["_origin_smile"].astype(str).values,
            "taut_smile": merged["_taut_smile"].astype(str).values,
            "rt": merged["rt"].astype(float).values,
            "origin_canon": merged["_origin_canon"].astype(str).values,
            "taut_canon": merged["_taut_canon"].astype(str).values,
            "taut_changed": changed.astype(int),
            "taut_source_row": merged["_taut_source_row"].astype(int).values,
        })
        for c in extra_cols:
            audit_out[c] = merged[c].values

        audit_out.to_csv(output_audit_csv, index=False)
        print("✅ saved audit:", output_audit_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin_train", required=True)
    ap.add_argument("--origin_test", required=True)
    ap.add_argument("--taut_train_existing", required=True)
    ap.add_argument("--taut_test_existing", required=True)
    ap.add_argument("--taut_train_audit", default="")
    ap.add_argument("--taut_test_audit", default="")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    reorder_one(
        args.origin_train,
        args.taut_train_existing,
        os.path.join(args.out_dir, "SMRT_train_tautomer_strict.csv"),
        audit_csv=args.taut_train_audit if args.taut_train_audit else None,
        output_audit_csv=os.path.join(args.out_dir, "SMRT_train_tautomer_strict_reorder_audit.csv"),
    )

    reorder_one(
        args.origin_test,
        args.taut_test_existing,
        os.path.join(args.out_dir, "SMRT_test_tautomer_strict.csv"),
        audit_csv=args.taut_test_audit if args.taut_test_audit else None,
        output_audit_csv=os.path.join(args.out_dir, "SMRT_test_tautomer_strict_reorder_audit.csv"),
    )

    print("\n✅ ALL DONE. Existing tautomer CSV has been reordered to origin CSV order.")


if __name__ == "__main__":
    main()

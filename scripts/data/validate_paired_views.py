import argparse
import os
import pandas as pd
import numpy as np
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


def load_rt_smiles(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, engine="python")
    raw_cols = df.columns.tolist()
    df.columns = [str(c).lower().strip() for c in df.columns]

    if "smile" in df.columns and "smiles" not in df.columns:
        df.rename(columns={"smile": "smiles"}, inplace=True)

    if "smiles" not in df.columns or "rt" not in df.columns:
        raise ValueError(f"{name}: missing smiles/rt columns. raw columns={raw_cols}")

    df = df.copy()
    df["rt"] = df["rt"].astype(float)

    
    df = df[df["rt"] > 300.0].reset_index(drop=False).rename(columns={"index": "source_row"})

    rows = []
    bad = 0

    for i, row in df.iterrows():
        smi = str(row["smiles"])
        rt = float(row["rt"])
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad += 1
            continue

        item = {
            "valid_i": len(rows),
            "source_row": int(row["source_row"]),
            "smiles": smi,
            "canon": Chem.MolToSmiles(m, canonical=True, isomericSmiles=True),
            "rt": rt,
        }

        if "orig_smile" in df.columns:
            item["orig_smile"] = str(row["orig_smile"])
            item["orig_canon"] = canon(row["orig_smile"])

        rows.append(item)

    out = pd.DataFrame(rows)
    print(f"\n[{name}]")
    print("path:", path)
    print("raw rows after rt>300:", len(df))
    print("valid rows:", len(out))
    print("rdkit invalid skipped:", bad)
    print("columns:", raw_cols)

    return out


def compare_pair(origin_csv, taut_csv, split_name):
    origin = load_rt_smiles(origin_csv, f"{split_name}-origin")
    taut = load_rt_smiles(taut_csv, f"{split_name}-taut")

    print(f"\n========== {split_name} PAIR CHECK ==========")
    print("origin valid rows:", len(origin))
    print("taut valid rows  :", len(taut))

    if len(origin) != len(taut):
        print("❌ row count mismatch")
        print("origin head rt:", origin["rt"].head().tolist())
        print("taut head rt  :", taut["rt"].head().tolist())
        raise SystemExit(1)

    rt_diff = np.abs(origin["rt"].values - taut["rt"].values)
    print("max rt diff :", float(rt_diff.max()))
    print("mean rt diff:", float(rt_diff.mean()))

    if rt_diff.max() > 1e-8:
        bad = np.where(rt_diff > 1e-8)[0][:10]
        print("❌ RT order mismatch examples:")
        for i in bad:
            print(
                "i=", int(i),
                "origin:", origin.iloc[i]["smiles"], origin.iloc[i]["rt"],
                "taut:", taut.iloc[i]["smiles"], taut.iloc[i]["rt"],
            )
        raise SystemExit(1)

    
    if "orig_smile" in taut.columns:
        raw_same = (origin["smiles"].astype(str).values == taut["orig_smile"].astype(str).values)
        canon_same = (origin["canon"].astype(str).values == taut["orig_canon"].astype(str).values)

        print("raw origin == taut.orig_smile:", int(raw_same.sum()), "/", len(raw_same))
        print("canon origin == taut.orig_smile:", int(canon_same.sum()), "/", len(canon_same))

        if canon_same.sum() != len(canon_same):
            bad = np.where(~canon_same)[0][:10]
            print("❌ canonical identity mismatch examples:")
            for i in bad:
                print(
                    "i=", int(i),
                    "origin:", origin.iloc[i]["smiles"],
                    "taut.orig:", taut.iloc[i]["orig_smile"],
                    "taut.new:", taut.iloc[i]["smiles"],
                )
            raise SystemExit(1)
    else:
        print("⚠️ taut CSV has no orig_smile column. 不建议用于正式 dual-view。")

    changed = (origin["canon"].astype(str).values != taut["canon"].astype(str).values)
    print("taut view changed canonical:", int(changed.sum()), "/", len(changed), "ratio:", float(changed.mean()))

    print(f"✅ {split_name} origin/taut pair check passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin_train", required=True)
    ap.add_argument("--origin_test", required=True)
    ap.add_argument("--taut_train", required=True)
    ap.add_argument("--taut_test", required=True)
    args = ap.parse_args()

    compare_pair(args.origin_train, args.taut_train, "TRAIN")
    compare_pair(args.origin_test, args.taut_test, "TEST")

    print("\n✅ ALL CHECKS PASSED. 可以进入 dual-view 训练脚本阶段。")


if __name__ == "__main__":
    main()

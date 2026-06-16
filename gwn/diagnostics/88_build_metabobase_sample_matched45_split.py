#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

IN_DIR = Path("experiments_candidate_filtering/metabobase_tl_exact39")
OUT_BASE = Path("experiments_candidate_filtering/metabobase_tl_sample45")
OUT_BASE.mkdir(parents=True, exist_ok=True)


def normalize_key(s):
    return str(s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_extra", type=int, default=6)
    args = ap.parse_args()

    out_dir = OUT_BASE / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    test39_meta = pd.read_csv(IN_DIR / "metabobase_test_exact39_metadata.csv")
    train187_meta = pd.read_csv(IN_DIR / "metabobase_train_exact39_metadata.csv")

    test39_meta["inchikey"] = test39_meta["inchikey"].map(normalize_key)
    train187_meta["inchikey"] = train187_meta["inchikey"].map(normalize_key)

    # 防止已有 test 泄漏到 train
    test39_keys = set(test39_meta["inchikey"])
    pool = train187_meta[~train187_meta["inchikey"].isin(test39_keys)].copy()

    if len(test39_meta) != 39:
        raise RuntimeError(f"test39_meta should be 39, got {len(test39_meta)}")
    if len(pool) < args.n_extra:
        raise RuntimeError(f"not enough pool rows: {len(pool)}")

    rng = np.random.default_rng(args.seed)
    sampled_idx = rng.choice(pool.index.to_numpy(), size=args.n_extra, replace=False)

    extra6 = pool.loc[sampled_idx].copy()
    extra6["sample45_role"] = "random_extra_test"
    extra6["sample45_seed"] = args.seed

    test39_meta = test39_meta.copy()
    test39_meta["sample45_role"] = "exact39_original_test"
    test39_meta["sample45_seed"] = args.seed

    test45_meta = pd.concat([test39_meta, extra6], axis=0, ignore_index=True)

    extra6_keys = set(extra6["inchikey"])
    train181_meta = train187_meta[~train187_meta["inchikey"].isin(extra6_keys)].copy()
    train181_meta["sample45_role"] = "train181"
    train181_meta["sample45_seed"] = args.seed

    assert len(test45_meta) == 45, len(test45_meta)
    assert len(train181_meta) == 181, len(train181_meta)
    assert len(set(test45_meta["inchikey"]) & set(train181_meta["inchikey"])) == 0

    def make_for_model(df):
        out = pd.DataFrame()
        out["name"] = df["name"] if "name" in df.columns else df.get("true_name", "")
        out["smiles"] = df["smiles"]
        if "rt" in df.columns:
            out["rt"] = df["rt"].astype(float)
        elif "rt_sec" in df.columns:
            out["rt"] = df["rt_sec"].astype(float)
        else:
            raise RuntimeError("cannot find rt or rt_sec column")
        out["inchikey"] = df["inchikey"]
        return out

    train181_for_model = make_for_model(train181_meta)
    test45_for_model = make_for_model(test45_meta)

    train181_meta.to_csv(out_dir / "metabobase_train181_sample45_metadata.csv", index=False)
    test45_meta.to_csv(out_dir / "metabobase_test45_sample45_metadata.csv", index=False)
    extra6.to_csv(out_dir / "metabobase_random_extra6_test_metadata.csv", index=False)

    train181_for_model.to_csv(out_dir / "metabobase_train181_sample45_for_model.csv", index=False)
    test45_for_model.to_csv(out_dir / "metabobase_test45_sample45_for_model.csv", index=False)

    with open(out_dir / "split_note.txt", "w") as f:
        f.write("sample-size matched MetaboBase-45 split\n")
        f.write("test45 = previous exact39 + 6 random compounds sampled from previous train187\n")
        f.write("train181 = previous train187 minus sampled extra6\n")
        f.write(f"seed = {args.seed}\n")
        f.write("This is NOT guaranteed to be the same 45 compounds as ABCoRT reported split.\n")

    print("=" * 100)
    print("[sample-size matched MetaboBase-45 split]")
    print("seed:", args.seed)
    print("exact39:", len(test39_meta))
    print("extra6:", len(extra6))
    print("test45:", len(test45_meta))
    print("train181:", len(train181_meta))
    print("overlap train/test:", len(set(test45_meta["inchikey"]) & set(train181_meta["inchikey"])))
    print("=" * 100)

    show_cols = ["name", "inchikey", "smiles"]
    show_cols = [c for c in show_cols if c in extra6.columns]
    print("\n[random extra6]")
    print(extra6[show_cols].to_string(index=False))

    print("\nSaved:")
    print(out_dir / "metabobase_train181_sample45_for_model.csv")
    print(out_dir / "metabobase_test45_sample45_for_model.csv")
    print(out_dir / "metabobase_train181_sample45_metadata.csv")
    print(out_dir / "metabobase_test45_sample45_metadata.csv")
    print(out_dir / "metabobase_random_extra6_test_metadata.csv")
    print(out_dir / "split_note.txt")
    print("=" * 100)


if __name__ == "__main__":
    main()

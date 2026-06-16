#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

IN_DIR = Path("experiments_candidate_filtering/metabobase_tl_exact39")
CAND_VALID = Path("experiments_candidate_filtering/parsed_candidates/msfinder_candidates_valid.csv")
OUT_BASE = Path("experiments_candidate_filtering/metabobase_tl_evaluable45")
OUT_BASE.mkdir(parents=True, exist_ok=True)


def to_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def make_for_model(df):
    out = pd.DataFrame()
    out["name"] = df["name"]
    out["smiles"] = df["smiles"]
    if "rt" in df.columns:
        out["rt"] = df["rt"].astype(float)
    elif "rt_sec" in df.columns:
        out["rt"] = df["rt_sec"].astype(float)
    else:
        raise RuntimeError("cannot find rt or rt_sec")
    out["inchikey"] = df["inchikey"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_extra", type=int, default=6)
    args = ap.parse_args()

    out_dir = OUT_BASE / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    test39 = pd.read_csv(IN_DIR / "metabobase_test_exact39_metadata.csv")
    train187 = pd.read_csv(IN_DIR / "metabobase_train_exact39_metadata.csv")
    cand = pd.read_csv(CAND_VALID, dtype=str, low_memory=False).fillna("")
    cand["is_true"] = to_bool_series(cand["is_true"])

    qstats = []
    for qid, sub in cand.groupby("query_id"):
        qstats.append({
            "query_id": str(qid),
            "n_valid_candidates": len(sub),
            "true_in_valid": bool(sub["is_true"].any()),
            "true_rank": int(pd.to_numeric(sub[sub["is_true"]]["candidate_rank"], errors="coerce").min()) if sub["is_true"].any() else np.nan,
        })
    qstats = pd.DataFrame(qstats)

    pool = train187.merge(qstats, on="query_id", how="left")
    pool["n_valid_candidates"] = pool["n_valid_candidates"].fillna(0).astype(int)
    pool["true_in_valid"] = pool["true_in_valid"].fillna(False).astype(bool)

    eligible = pool[pool["true_in_valid"]].copy()

    if len(eligible) < args.n_extra:
        raise RuntimeError(f"eligible rows not enough: {len(eligible)}")

    rng = np.random.default_rng(args.seed)
    picked_idx = rng.choice(eligible.index.to_numpy(), size=args.n_extra, replace=False)

    extra6 = eligible.loc[picked_idx].copy()
    extra6["sample45_role"] = "candidate_evaluable_extra_test"
    extra6["sample45_seed"] = args.seed

    test39 = test39.copy()
    test39["sample45_role"] = "exact39_original_test"
    test39["sample45_seed"] = args.seed

    test45 = pd.concat([test39, extra6[test39.columns.intersection(extra6.columns)]], ignore_index=True, sort=False)

    extra_keys = set(extra6["inchikey"].astype(str))
    train181 = train187[~train187["inchikey"].astype(str).isin(extra_keys)].copy()
    train181["sample45_role"] = "train181"
    train181["sample45_seed"] = args.seed

    assert len(test45) == 45, len(test45)
    assert len(train181) == 181, len(train181)
    assert len(set(test45["inchikey"].astype(str)) & set(train181["inchikey"].astype(str))) == 0

    train181.to_csv(out_dir / "metabobase_train181_evaluable45_metadata.csv", index=False)
    test45.to_csv(out_dir / "metabobase_test45_evaluable45_metadata.csv", index=False)
    extra6.to_csv(out_dir / "metabobase_extra6_evaluable_test_metadata.csv", index=False)

    make_for_model(train181).to_csv(out_dir / "metabobase_train181_evaluable45_for_model.csv", index=False)
    make_for_model(test45).to_csv(out_dir / "metabobase_test45_evaluable45_for_model.csv", index=False)

    eligible.sort_values(["true_rank", "n_valid_candidates"]).to_csv(out_dir / "eligible_extra_pool_true_in_valid.csv", index=False)

    with open(out_dir / "split_note.txt", "w") as f:
        f.write("candidate-evaluable MetaboBase-45 split\n")
        f.write("test45 = exact39 + 6 compounds sampled from train187 with true_in_valid_candidates=True\n")
        f.write("train181 = remaining train187 after removing selected extra6\n")
        f.write(f"seed = {args.seed}\n")
        f.write("This is sample-size matched, not guaranteed same 45 as ABCoRT reported split.\n")

    print("=" * 100)
    print("[candidate-evaluable MetaboBase-45 split]")
    print("seed:", args.seed)
    print("exact39:", len(test39))
    print("eligible pool:", len(eligible))
    print("extra6:", len(extra6))
    print("test45:", len(test45))
    print("train181:", len(train181))
    print("overlap:", len(set(test45["inchikey"].astype(str)) & set(train181["inchikey"].astype(str))))
    print("=" * 100)

    show_cols = ["query_id", "name", "inchikey", "rt_sec", "n_valid_candidates", "true_rank"]
    show_cols = [c for c in show_cols if c in extra6.columns]
    print("\n[extra6 candidate-evaluable]")
    print(extra6[show_cols].to_string(index=False))

    print("\nSaved:")
    print(out_dir / "metabobase_train181_evaluable45_for_model.csv")
    print(out_dir / "metabobase_test45_evaluable45_for_model.csv")
    print(out_dir / "metabobase_extra6_evaluable_test_metadata.csv")
    print(out_dir / "eligible_extra_pool_true_in_valid.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

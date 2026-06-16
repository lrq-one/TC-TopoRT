#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
from difflib import get_close_matches

IN_META = Path("experiments_candidate_filtering/riken_parsed/riken_metadata_with_candidate_evaluable_flag.csv")
IN_VALID = Path("experiments_candidate_filtering/riken_parsed/riken_candidates_valid.csv")

OUT = Path("experiments_candidate_filtering/riken_tl_exact85")
OUT.mkdir(parents=True, exist_ok=True)

# Table S11, RIKEN_PlaSMA test set, extracted from the paper.
# Use first 14 characters of InChIKey for robust matching.
TABLE_S11_FULL_OVERRIDES = {
    # These three InChIKey14 values are duplicated in RIKEN metadata.
    # Use full InChIKeys from Table S11 to select the exact test molecule.
    "KYWSCMDFVARMPN": "KYWSCMDFVARMPN-MSSMMRRTSA-N",
    "RKUNBYITZUJHSG": "RKUNBYITZUJHSG-VFSICIBPSA-N",
    "UIDGLYUNOUKLBM": "UIDGLYUNOUKLBM-GEBJFKNCSA-N",
}

TABLE_S11_IK14 = [
    "AIFRHYZBTHREPW",
    "ATLJNLYIJOCWJE",
    "BCJMNZRQJAVDLD",
    "BQGJXFQCMYJENQ",
    "BXNJHAXVSOCGBA",
    "CDICDSOGTRCHMG",
    "CJDRUOGAGYHKKD",
    "CJXMVKYNVIGQBS",
    "CLDCTFPNFRITPI",
    "CRQDWQWZCNKKAC",
    "CTSPAMFJBXKSOY",
    "CZLWGXKWXLVFJU",
    "DEMKZLAVQYISIA",
    "DFNXNCCYQRPZMD",
    "DOUMFZQKYFQNTF",
    "DTGZHCFJNDAHEN",
    "DVGGLGXQSFURLP",
    "FAZIYUIDUNHZRG",
    "FBMORZZOJSDNRQ",

    "FIAAVMJLAGNUKW",
    "GRTOGORTSDXSFK",
    "GUAFOGOEJLSQBT",
    "HGNHIFJNOKGSKI",
    "HITJFUSPLYBJPE",
    "HJRVLGWTJSLQIG",
    "INYYVPJSBIVGPH",
    "IYLRRIUNGGQRTN",
    "IZQSVPBOUDKVDZ",
    "JVHNBFFHWQQPLL",
    "JVIKUDVTJCANPX",
    "KUBCEEMXQZUPDQ",
    "KYWSCMDFVARMPN",
    "LDIRGNDMTOGVRB",
    "LNTHITQWFMADLM",
    "LTOOPESWVADEAE",
    "LZPNXAULYJPXEH",
    "MGJLSBDCWOSMHL",
    "MOJZMWJRUKIQGL",

    "NLAWPKPYBMEWIR",
    "NLGUKXQDDTZCDG",
    "NMLUOJBSAYAYEM",
    "OIUBYZLTFSLSBY",
    "OMQADRGFMLGFJF",
    "PCMORTLOPMLEFB",
    "PIWJSAMCEMZIDO",
    "PLAPMLGJVGLZOV",
    "PSFDQSOCUJVVGF",
    "PTNLHDGQWUGONS",
    "QEEBRPGZBVVINN",
    "QUQPHWDTPGMPEX",
    "RBALEJFQJCAPLN",
    "RKBDCPZCGRWNMP",
    "RKUNBYITZUJHSG",
    "RMCRQBAILCLJGU",
    "RODXRVNMMDRFIK",
    "RYENLSMHLCNXJT",
    "RYHDIBJJJRNDSX",

    "SLCKJKWFULXZBD",
    "SQFSKOYWJBQGKQ",
    "SWIROVJVGRGSPO",
    "SXYMMDGPXYVCER",
    "TUIJPUWSXVFWSH",
    "TZJALUIVHRYQQB",
    "UIDGLYUNOUKLBM",
    "USNPULRDBDVJAO",
    "VHBFFQKBGNRLFZ",
    "VLEUZFDZJKSGMX",
    "VLSMHEGGTFMBBZ",
    "VLSRUFWCGBMYDJ",
    "VNBUMBNLPGLBML",
    "VRSRXLJTYQVOHC",
    "VVOAZFWZEDHOOU",
    "WMBWREPUVVBILR",
    "WUFQLZTXIWKION",
    "XFSBVAOIAHNAPC",
    "XLTFNNCXVBYBSX",

    "XQYZDYMELSJDRZ",
    "XRBIHOLQAKITPP",
    "YCIMNLLNPGFGHC",
    "YDDUMTOHNYZQPO",
    "YFPYXTNSQOUHPS",
    "YNMFDPCLPIMRFD",
    "ZONYXWQDUYMKFB",
    "ZQSIJRDFPHDXIC",
    "ZTVIKZXZYLEVOL",
]


def ik14(x):
    s = str(x).strip()
    return s[:14] if len(s) >= 14 else ""


def make_for_model(df):
    out = pd.DataFrame()
    out["name"] = df["name"].astype(str)
    out["smiles"] = df["smiles"].astype(str)
    out["rt"] = pd.to_numeric(df["rt_sec"], errors="coerce")
    out["inchikey"] = df["inchikey"].astype(str)
    return out


def bool_series(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def main():
    meta = pd.read_csv(IN_META, dtype=str, low_memory=False).fillna("")
    valid = pd.read_csv(IN_VALID, dtype=str, low_memory=False).fillna("")

    table = pd.DataFrame({
        "table_s11_order": range(1, len(TABLE_S11_IK14) + 1),
        "table_ik14": TABLE_S11_IK14,
    })

    print("=" * 100)
    print("[Table S11 extracted keys]")
    print("n_table_rows:", len(table))
    print("n_unique_table_ik14:", table["table_ik14"].nunique())
    if len(table) != 85 or table["table_ik14"].nunique() != 85:
        print("[ERROR] Table S11 list is not 85 unique InChIKey14.")
        dup = table[table["table_ik14"].duplicated(keep=False)]
        print(dup.to_string(index=False))
        raise SystemExit(1)

    meta["meta_ik14"] = meta["inchikey"].map(ik14)
    valid["query_ik14"] = valid["query_inchikey"].map(ik14)
    valid["is_true"] = bool_series(valid["is_true"])

    dup_meta = meta[meta["meta_ik14"].duplicated(keep=False) & meta["meta_ik14"].ne("")]
    print("metadata rows:", len(meta))
    print("metadata unique query_id:", meta["query_id"].nunique())
    print("metadata unique ik14:", meta["meta_ik14"].nunique())
    print("duplicated ik14 rows in metadata:", len(dup_meta))

    if len(dup_meta):
        print("\n[WARNING duplicated ik14 in metadata]")
        print(dup_meta[["query_id", "name", "inchikey", "meta_ik14"]].head(80).to_string(index=False))

    # Match Table S11 keys one by one.
    # Most rows are unique by InChIKey14. A few duplicated InChIKey14 rows are resolved by full InChIKey.
    matched_rows = []
    missing_rows = []
    ambiguous_rows = []

    for _, r in table.iterrows():
        key = str(r["table_ik14"])
        cand = meta[meta["meta_ik14"].eq(key)].copy()

        if key in TABLE_S11_FULL_OVERRIDES:
            full_key = TABLE_S11_FULL_OVERRIDES[key]
            cand = cand[cand["inchikey"].astype(str).eq(full_key)].copy()

        if len(cand) == 0:
            missing_rows.append(r.to_dict())
            continue

        if len(cand) > 1:
            tmp = cand[["query_id", "name", "inchikey", "meta_ik14"]].copy()
            tmp.insert(0, "table_s11_order", r["table_s11_order"])
            tmp.insert(1, "table_ik14", key)
            ambiguous_rows.append(tmp)
            continue

        row = r.to_dict()
        row.update(cand.iloc[0].to_dict())
        matched_rows.append(row)

    test = pd.DataFrame(matched_rows)
    missing = pd.DataFrame(missing_rows)

    print("\n[matching Table S11 -> RIKEN metadata]")
    print("matched test rows:", len(test), "/", len(table))
    print("missing rows:", len(missing))
    print("ambiguous rows:", sum(len(x) for x in ambiguous_rows))

    if len(missing):
        all_ik14 = meta["meta_ik14"].dropna().astype(str).unique().tolist()
        miss_rows = []
        for _, r in missing.iterrows():
            key = r["table_ik14"]
            suggestions = get_close_matches(key, all_ik14, n=5, cutoff=0.75)
            miss_rows.append({
                "table_s11_order": r["table_s11_order"],
                "table_ik14": key,
                "suggestions": ";".join(suggestions),
            })
        miss_df = pd.DataFrame(miss_rows)
        miss_df.to_csv(OUT / "table_s11_missing_ik14_suggestions.csv", index=False)
        print(miss_df.to_string(index=False))
        print("saved:", OUT / "table_s11_missing_ik14_suggestions.csv")
        raise SystemExit(1)

    if ambiguous_rows:
        amb = pd.concat(ambiguous_rows, axis=0, ignore_index=True)
        amb.to_csv(OUT / "table_s11_ambiguous_ik14_matches.csv", index=False)
        print("[ERROR] still ambiguous Table S11 matches:")
        print(amb.to_string(index=False))
        print("saved:", OUT / "table_s11_ambiguous_ik14_matches.csv")
        raise SystemExit(1)

    if test["query_id"].duplicated().any():
        print("[ERROR] duplicated query_id matched from Table S11")
        print(test[test["query_id"].duplicated(keep=False)][["table_s11_order", "table_ik14", "query_id", "name", "inchikey"]].to_string(index=False))
        raise SystemExit(1)

    test = test.sort_values("table_s11_order").reset_index(drop=True)
    test_keys = set(test["query_id"].astype(str))

    train = meta[~meta["query_id"].astype(str).isin(test_keys)].copy().reset_index(drop=True)

    print("\n[split sizes]")
    print("test85:", len(test))
    print("train341:", len(train))
    print("overlap:", len(set(test["query_id"]) & set(train["query_id"])))

    if len(test) != 85 or len(train) != 341:
        print("[ERROR] split size is not 85/341.")
        raise SystemExit(1)

    # Candidate coverage on strict exact85
    cand = valid[valid["query_id"].astype(str).isin(test_keys)].copy()
    qrows = []
    for qid, sub in cand.groupby("query_id"):
        sub = sub.copy()
        sub["candidate_rank"] = pd.to_numeric(sub["candidate_rank"], errors="coerce")
        true_sub = sub[sub["is_true"]].sort_values("candidate_rank")
        qrows.append({
            "query_id": qid,
            "n_valid_candidates": len(sub),
            "true_in_valid": bool(len(true_sub)),
            "true_rank": int(true_sub["candidate_rank"].iloc[0]) if len(true_sub) else np.nan,
            "top1_before": bool(len(true_sub) and true_sub["candidate_rank"].iloc[0] <= 1),
            "top5_before": bool(len(true_sub) and true_sub["candidate_rank"].iloc[0] <= 5),
            "top10_before": bool(len(true_sub) and true_sub["candidate_rank"].iloc[0] <= 10),
        })

    qstat = pd.DataFrame(qrows)
    test = test.merge(qstat, on="query_id", how="left")

    for c in ["n_valid_candidates"]:
        test[c] = pd.to_numeric(test[c], errors="coerce").fillna(0).astype(int)
    for c in ["true_in_valid", "top1_before", "top5_before", "top10_before"]:
        test[c] = test[c].fillna(False).astype(bool)

    n_true = int(test["true_in_valid"].sum())
    print("\n[candidate coverage on strict exact85]")
    print("candidate rows:", len(cand))
    print("queries with candidate rows:", cand["query_id"].nunique())
    print("queries with true candidate:", n_true, "/", len(test))

    eval_q = test[test["true_in_valid"]].copy()
    if len(eval_q):
        print("\n[MS-FINDER original on strict Table-S11 exact85]")
        print("N:", len(eval_q))
        print("Top1:", round(100 * eval_q["top1_before"].mean(), 4))
        print("Top5:", round(100 * eval_q["top5_before"].mean(), 4))
        print("Top10:", round(100 * eval_q["top10_before"].mean(), 4))
        print("median true rank:", eval_q["true_rank"].median())
        print("mean candidates/query:", round(eval_q["n_valid_candidates"].mean(), 4))

    problem = test[~test["true_in_valid"]].copy()
    if len(problem):
        print("\n[WARNING exact85 queries without true candidate]")
        print(problem[["table_s11_order", "query_id", "name", "inchikey", "n_valid_candidates", "true_in_valid"]].to_string(index=False))

    # Save
    test.to_csv(OUT / "riken_test85_exactS11_metadata.csv", index=False)
    train.to_csv(OUT / "riken_train341_exactS11_metadata.csv", index=False)
    cand.to_csv(OUT / "riken_test85_exactS11_candidates_valid.csv", index=False)
    table.to_csv(OUT / "table_s11_ik14_extracted.csv", index=False)

    make_for_model(test).to_csv(OUT / "riken_test85_exactS11_for_model.csv", index=False)
    make_for_model(train).to_csv(OUT / "riken_train341_exactS11_for_model.csv", index=False)

    print("\n[head exact85]")
    show = ["table_s11_order", "query_id", "name", "inchikey", "rt_sec", "n_valid_candidates", "true_rank"]
    show = [c for c in show if c in test.columns]
    print(test[show].head(100).to_string(index=False))

    print("\nSaved:")
    print(OUT / "riken_test85_exactS11_metadata.csv")
    print(OUT / "riken_train341_exactS11_metadata.csv")
    print(OUT / "riken_test85_exactS11_candidates_valid.csv")
    print(OUT / "riken_test85_exactS11_for_model.csv")
    print(OUT / "riken_train341_exactS11_for_model.csv")
    print(OUT / "table_s11_ik14_extracted.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

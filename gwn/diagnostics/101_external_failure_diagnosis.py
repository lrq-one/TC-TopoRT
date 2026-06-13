from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr

ABCORT = {
    "Eawag_XBridgeC18_364": 45.30,
    "FEM_lipids_72": 85.46,
    "FEM_long_412": 87.16,
    "IPB_Halle_82": 13.81,
    "LIFE_new_184": 15.62,
    "LIFE_old_194": 9.97,
}

ELEMENTS = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]

def parse_formula(formula):
    formula = "" if pd.isna(formula) else str(formula)
    out = {e: 0 for e in ELEMENTS}
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if elem in out:
            out[elem] += int(num) if num else 1
    return out

def add_formula_features(df):
    rows = [parse_formula(x) for x in df.get("formula", pd.Series([""] * len(df))).fillna("")]
    elem = pd.DataFrame(rows)
    for e in ELEMENTS:
        df[f"n_{e}"] = elem[e].values
    df["heavy_noH"] = df[[f"n_{e}" for e in ELEMENTS if e != "H"]].sum(axis=1)
    df["hetero"] = df[[f"n_{e}" for e in ["N", "O", "S", "P", "F", "Cl", "Br", "I"]]].sum(axis=1)
    df["O_over_C"] = df["n_O"] / np.maximum(df["n_C"], 1)
    df["N_over_C"] = df["n_N"] / np.maximum(df["n_C"], 1)
    df["hetero_over_C"] = df["hetero"] / np.maximum(df["n_C"], 1)

    smiles = df.get("origin_smiles", pd.Series([""] * len(df))).fillna("").astype(str)
    df["has_aromatic"] = smiles.str.contains("c", regex=False).astype(int)
    df["acid_like"] = smiles.str.contains("C\\(=O\\)O|C\\(O\\)=O|C\\(=O\\)\\[O", regex=True).astype(int)
    df["halogenated"] = ((df["n_F"] + df["n_Cl"] + df["n_Br"] + df["n_I"]) > 0).astype(int)
    df["high_oxygen"] = (df["n_O"] >= 4).astype(int)
    df["large_heavy"] = (df["heavy_noH"] >= 25).astype(int)
    df["small_polar"] = ((df["heavy_noH"] <= 15) & (df["hetero"] >= 4)).astype(int)
    return df

def safe_corr(y, p, kind):
    if len(y) < 2:
        return np.nan
    if np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return np.nan
    if kind == "spearman":
        return float(spearmanr(y, p).correlation)
    return float(pearsonr(y, p)[0])

def metric_row(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    e = np.abs(y - p)
    signed = p - y
    try:
        slope = float(np.polyfit(y, p, 1)[0])
        intercept = float(np.polyfit(y, p, 1)[1])
    except Exception:
        slope, intercept = np.nan, np.nan
    return {
        "n": len(y),
        "mae": float(e.mean()),
        "medae": float(np.median(e)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "spearman": safe_corr(y, p, "spearman"),
        "pearson": safe_corr(y, p, "pearson"),
        "bias": float(signed.mean()),
        "slope_pred_vs_rt": slope,
        "intercept_pred_vs_rt": intercept,
        "p90_abs_err": float(np.quantile(e, 0.90)),
        "p95_abs_err": float(np.quantile(e, 0.95)),
        "max_abs_err": float(e.max()),
        "err_gt_30": int((e > 30).sum()),
        "err_gt_50": int((e > 50).sum()),
        "err_gt_75": int((e > 75).sum()),
        "err_gt_100": int((e > 100).sum()),
        "err_gt_150": int((e > 150).sum()),
        "err_gt_200": int((e > 200).sum()),
    }

def assign_folds(sub, cv_seed=1, n_splits=10):
    sub = sub.copy().reset_index(drop=True)
    sub["cv_fold"] = -1
    kf = KFold(n_splits=min(n_splits, len(sub)), shuffle=True, random_state=cv_seed)
    for fold, (_, te) in enumerate(kf.split(np.zeros(len(sub)))):
        sub.loc[te, "cv_fold"] = fold
    return sub

def collect_prediction_tables(dataset_name):
    pred_files = sorted(Path(".").glob("paper_analysis_stage4*/external_tl_predictions.csv"))
    tables = []
    for p in pred_files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "dataset_name" not in df.columns:
            continue
        sub = df[df["dataset_name"].astype(str).eq(dataset_name)].copy()
        if len(sub) == 0:
            continue
        pred_cols = [c for c in ["origin_tl_pred", "taut_tl_pred", "mean_tl_pred"] if c in sub.columns]
        if not pred_cols:
            continue

        if "source_fold" in sub.columns:
            source_folds = sorted(sub["source_fold"].dropna().unique())
        else:
            source_folds = [""]

        for sf in source_folds:
            ss = sub.copy()
            sf_tag = str(sf)
            if "source_fold" in ss.columns:
                ss = ss[ss["source_fold"].eq(sf)].copy()
            for c in pred_cols:
                col = f"{p.parent.name}__sf{sf_tag}__{c}"
                small = ss[["stage4_index", "rt", c]].copy()
                small = small.rename(columns={c: col})
                tables.append((col, small))
    return tables

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="paper_analysis_stage4_external/external_predret10_stage4_meta.csv")
    ap.add_argument("--datasets", nargs="+", default=["Eawag_XBridgeC18_364", "FEM_long_412"])
    ap.add_argument("--out_dir", default="paper_analysis_stage4AG_failure_diagnosis")
    ap.add_argument("--cv_seed", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta)
    meta = meta.sort_values("stage4_index").reset_index(drop=True)
    meta = add_formula_features(meta)

    all_summary = []

    for ds in args.datasets:
        print("\n" + "=" * 100)
        print("DATASET", ds)
        print("=" * 100)

        sub = meta[meta["dataset_name"].eq(ds)].copy()
        if len(sub) == 0:
            print("[SKIP] no rows in meta")
            continue

        sub = assign_folds(sub, cv_seed=args.cv_seed, n_splits=10)

        # 1. dataset RT distribution
        rt = sub["rt"].values.astype(float)
        ds_row = {
            "dataset": ds,
            "n": len(sub),
            "ABCORT": ABCORT.get(ds, np.nan),
            "rt_min": float(np.min(rt)),
            "rt_q10": float(np.quantile(rt, 0.10)),
            "rt_q25": float(np.quantile(rt, 0.25)),
            "rt_median": float(np.median(rt)),
            "rt_mean": float(np.mean(rt)),
            "rt_q75": float(np.quantile(rt, 0.75)),
            "rt_q90": float(np.quantile(rt, 0.90)),
            "rt_max": float(np.max(rt)),
            "rt_std": float(np.std(rt)),
            "rt_range": float(np.max(rt) - np.min(rt)),
            "taut_changed_rate": float(sub["taut_changed"].mean()) if "taut_changed" in sub.columns else np.nan,
            "smrt_exact_overlap_rate": float(sub["smrt_exact_overlap"].mean()) if "smrt_exact_overlap" in sub.columns else np.nan,
            "heavy_mean": float(sub["heavy_noH"].mean()),
            "hetero_mean": float(sub["hetero"].mean()),
            "large_heavy_rate": float(sub["large_heavy"].mean()),
            "high_oxygen_rate": float(sub["high_oxygen"].mean()),
            "halogenated_rate": float(sub["halogenated"].mean()),
            "acid_like_rate": float(sub["acid_like"].mean()),
        }
        pd.DataFrame([ds_row]).to_csv(out / f"{ds}__dataset_rt_chem_summary.csv", index=False)

        # 2. fold train/test RT distribution
        fold_rows = []
        for fold in sorted(sub["cv_fold"].unique()):
            te_mask = sub["cv_fold"].eq(fold).values
            tr_mask = ~te_mask
            ytr = sub.loc[tr_mask, "rt"].values.astype(float)
            yte = sub.loc[te_mask, "rt"].values.astype(float)

            fold_rows.append({
                "dataset": ds,
                "cv_fold": int(fold),
                "n_train": int(tr_mask.sum()),
                "n_test": int(te_mask.sum()),
                "train_mean": float(np.mean(ytr)),
                "test_mean": float(np.mean(yte)),
                "mean_gap_test_minus_train": float(np.mean(yte) - np.mean(ytr)),
                "train_std": float(np.std(ytr)),
                "test_std": float(np.std(yte)),
                "std_ratio_test_over_train": float(np.std(yte) / max(np.std(ytr), 1e-9)),
                "train_min": float(np.min(ytr)),
                "test_min": float(np.min(yte)),
                "train_max": float(np.max(ytr)),
                "test_max": float(np.max(yte)),
                "test_q90": float(np.quantile(yte, 0.90)),
                "test_num_top10pct_global": int((yte >= np.quantile(rt, 0.90)).sum()),
                "test_num_bottom10pct_global": int((yte <= np.quantile(rt, 0.10)).sum()),
            })
        fold_df = pd.DataFrame(fold_rows).sort_values("mean_gap_test_minus_train", key=lambda s: s.abs(), ascending=False)
        fold_df.to_csv(out / f"{ds}__fold_rt_distribution.csv", index=False)
        print("\n[FOLD RT DISTRIBUTION]")
        print(fold_df.head(10).to_string(index=False))

        # 3. prediction candidate summary
        tables = collect_prediction_tables(ds)
        if not tables:
            print("[NO PREDICTIONS FOUND]")
            continue

        base = sub[["stage4_index", "rt", "cv_fold", "formula", "name", "inchikey",
                    "heavy_noH", "hetero", "O_over_C", "N_over_C", "hetero_over_C",
                    "large_heavy", "high_oxygen", "halogenated", "acid_like"] +
                   [c for c in ["taut_changed", "smrt_exact_overlap", "origin_smiles", "taut_smiles"] if c in sub.columns]].copy()

        for col, t in tables:
            base = base.merge(t[["stage4_index", col]], on="stage4_index", how="left")

        pred_cols = [c for c in base.columns if "__" in c and c.endswith("_pred")]
        metric_rows = []
        y = base["rt"].values.astype(float)

        for c in pred_cols:
            p = base[c].values.astype(float)
            if np.isnan(p).any():
                continue
            row = metric_row(y, p)
            row.update({
                "dataset": ds,
                "candidate": c,
                "ABCORT": ABCORT.get(ds, np.nan),
                "delta_vs_ABCORT": row["mae"] - ABCORT.get(ds, np.nan),
            })
            metric_rows.append(row)

        metric_df = pd.DataFrame(metric_rows).sort_values("mae")
        metric_df.to_csv(out / f"{ds}__candidate_metrics.csv", index=False)
        print("\n[CANDIDATE METRICS TOP 15]")
        print(metric_df.head(15)[["dataset", "candidate", "mae", "ABCORT", "delta_vs_ABCORT", "bias", "slope_pred_vs_rt", "p90_abs_err", "err_gt_100", "err_gt_150"]].to_string(index=False))

        # 4. top-k mean capacity
        top_rows = []
        sorted_cols = metric_df["candidate"].tolist()
        for k in [2, 3, 4, 5, 8, 10]:
            cols = sorted_cols[:k]
            if len(cols) < k:
                continue
            p = base[cols].values.astype(float).mean(axis=1)
            row = metric_row(y, p)
            row.update({
                "dataset": ds,
                "strategy": f"diagnostic_top{k}_mean",
                "n_models": k,
                "ABCORT": ABCORT.get(ds, np.nan),
                "delta_vs_ABCORT": row["mae"] - ABCORT.get(ds, np.nan),
                "members": " | ".join(cols),
            })
            top_rows.append(row)
        top_df = pd.DataFrame(top_rows).sort_values("mae")
        top_df.to_csv(out / f"{ds}__ensemble_capacity.csv", index=False)
        print("\n[ENSEMBLE CAPACITY]")
        if len(top_df):
            print(top_df[["strategy", "mae", "ABCORT", "delta_vs_ABCORT", "bias", "slope_pred_vs_rt", "p90_abs_err"]].to_string(index=False))

        # 5. best candidate error by RT bin and chemistry flags
        best_col = metric_df.iloc[0]["candidate"]
        best_pred = base[best_col].values.astype(float)
        base["best_pred"] = best_pred
        base["best_abs_err"] = np.abs(base["rt"].values.astype(float) - best_pred)
        base["best_signed_err"] = best_pred - base["rt"].values.astype(float)

        base["rt_bin"] = pd.qcut(base["rt"], q=5, duplicates="drop")
        bin_df = base.groupby("rt_bin", observed=False).agg(
            n=("stage4_index", "count"),
            rt_min=("rt", "min"),
            rt_max=("rt", "max"),
            mae=("best_abs_err", "mean"),
            medae=("best_abs_err", "median"),
            bias=("best_signed_err", "mean"),
            p90_abs_err=("best_abs_err", lambda x: float(np.quantile(x, 0.90))),
            err_gt_100=("best_abs_err", lambda x: int((x > 100).sum())),
            heavy_mean=("heavy_noH", "mean"),
            hetero_mean=("hetero", "mean"),
        ).reset_index()
        bin_df.to_csv(out / f"{ds}__best_error_by_rt_bin.csv", index=False)

        flag_rows = []
        for flag in ["large_heavy", "high_oxygen", "halogenated", "acid_like"]:
            for val in [0, 1]:
                ss = base[base[flag].eq(val)]
                if len(ss) == 0:
                    continue
                flag_rows.append({
                    "dataset": ds,
                    "flag": flag,
                    "value": val,
                    "n": len(ss),
                    "mae": float(ss["best_abs_err"].mean()),
                    "bias": float(ss["best_signed_err"].mean()),
                    "p90_abs_err": float(np.quantile(ss["best_abs_err"], 0.90)),
                })
        flag_df = pd.DataFrame(flag_rows).sort_values(["flag", "value"])
        flag_df.to_csv(out / f"{ds}__best_error_by_chem_flags.csv", index=False)

        hard_cols = ["cv_fold", "stage4_index", "name", "formula", "inchikey", "rt",
                     "best_pred", "best_abs_err", "best_signed_err",
                     "heavy_noH", "hetero", "O_over_C", "N_over_C",
                     "large_heavy", "high_oxygen", "halogenated", "acid_like"] + \
                    [c for c in ["taut_changed", "smrt_exact_overlap", "origin_smiles", "taut_smiles"] if c in base.columns]
        hard = base[hard_cols].sort_values("best_abs_err", ascending=False)
        hard.to_csv(out / f"{ds}__best_top_errors.csv", index=False)

        print("\n[BEST ERROR BY RT BIN]")
        print(bin_df.to_string(index=False))
        print("\n[TOP 15 HARD SAMPLES]")
        print(hard.head(15)[["cv_fold", "stage4_index", "name", "formula", "rt", "best_pred", "best_abs_err", "best_signed_err", "heavy_noH", "hetero"]].to_string(index=False))

        all_summary.append({
            "dataset": ds,
            "ABCORT": ABCORT.get(ds, np.nan),
            "best_candidate": best_col,
            "best_candidate_mae": float(metric_df.iloc[0]["mae"]),
            "best_candidate_delta": float(metric_df.iloc[0]["delta_vs_ABCORT"]),
            "best_ensemble_mae": float(top_df.iloc[0]["mae"]) if len(top_df) else np.nan,
            "best_ensemble_delta": float(top_df.iloc[0]["delta_vs_ABCORT"]) if len(top_df) else np.nan,
        })

    summary = pd.DataFrame(all_summary)
    summary.to_csv(out / "failure_diagnosis_summary.csv", index=False)
    print("\n" + "=" * 100)
    print("FINAL FAILURE DIAGNOSIS SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("\n[SAVED DIR]", out)

if __name__ == "__main__":
    main()

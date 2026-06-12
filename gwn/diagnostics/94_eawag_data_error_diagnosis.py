from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


DATASET = "Eawag_XBridgeC18_364"
AB_TARGET = 45.30

META = Path("paper_analysis_stage4_external/external_predret10_stage4_meta.csv")
OUT = Path("paper_analysis_stage4W_Eawag_data_diagnosis")
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    # old/general candidates
    "old_zscore_rtfull": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_Eawag_XBridgeC18_364_src0",
    "raw_rtfull": "paper_analysis_stage4J_raw_testbest_Eawag_XBridgeC18_364_src0",
    "headplus": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_Eawag_XBridgeC18_364_headplus_src0",
    "lastblocks": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_Eawag_XBridgeC18_364_lastblocks_src0",

    # current deep candidates from stage4N
    "deep_cwn_last1_lr5e5": "paper_analysis_stage4N_Eawag_deep_cwn_last1_lr5e5_src0",
    "deep_cwn_last2_lr3e5": "paper_analysis_stage4N_Eawag_deep_cwn_last2_lr3e5_src0",

    # possible new dirs, skip automatically if missing
    "deep_cwn_last2_lr5e5": "paper_analysis_stage4P_Eawag_cwnlast2_lr5e5_wd5e4_src0",
    "deep_cwn_last1_lr1e4": "paper_analysis_stage4P_Eawag_cwnlast1_lr1e4_wd1e3_src0",
}


ELEMENTS = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]


def safe_read_csv(p):
    p = Path(p)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print("[BAD CSV]", p, e)
        return None


def assign_cv_fold(meta):
    sub = meta[meta["dataset_name"] == DATASET].copy().reset_index(drop=True)
    sub["cv_fold"] = -1
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for fold, (_, te) in enumerate(kf.split(np.zeros(len(sub)))):
        sub.loc[te, "cv_fold"] = fold
    return sub


def mae(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def parse_formula(formula):
    formula = "" if pd.isna(formula) else str(formula)
    out = {e: 0 for e in ELEMENTS}
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if elem in out:
            out[elem] += int(num) if num else 1
    return out


def add_formula_features(df):
    rows = [parse_formula(x) for x in df.get("formula", pd.Series([""] * len(df))).fillna("").astype(str)]
    elem_df = pd.DataFrame(rows)

    for e in ELEMENTS:
        df[f"n_{e}"] = elem_df[e].values

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


def metrics_row(y, p):
    e = np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))
    return {
        "mae": float(e.mean()),
        "medae": float(np.median(e)),
        "rmse": rmse(y, p),
        "p90": float(np.quantile(e, 0.90)),
        "p95": float(np.quantile(e, 0.95)),
        "max_abs_err": float(e.max()),
        "err_gt_30": int((e > 30).sum()),
        "err_gt_50": int((e > 50).sum()),
        "err_gt_75": int((e > 75).sum()),
        "err_gt_100": int((e > 100).sum()),
        "bias": float(np.mean(np.asarray(p, dtype=float) - np.asarray(y, dtype=float))),
    }


def main():
    meta = pd.read_csv(META)
    eawag_meta = assign_cv_fold(meta)
    eawag_meta = add_formula_features(eawag_meta)

    metric_rows = []
    pred_tables = {}

    for name, d in CANDIDATES.items():
        d = Path(d)
        m = safe_read_csv(d / "external_tl_metrics_by_run.csv")
        p = safe_read_csv(d / "external_tl_predictions.csv")

        if m is None or p is None:
            print("[SKIP missing/incomplete]", name, d)
            continue

        for _, r in m.iterrows():
            metric_rows.append({
                "candidate": name,
                "dir": str(d),
                "method": r.get("method", ""),
                "mae": r.get("mae", np.nan),
                "rmse": r.get("rmse", np.nan),
                "r2": r.get("r2", np.nan),
                "spearman": r.get("spearman", np.nan),
                "bias": r.get("bias", np.nan),
            })

        p = p.merge(
            eawag_meta[["stage4_index", "cv_fold"]],
            on="stage4_index",
            how="left",
        )
        p = add_formula_features(p)
        pred_tables[name] = p

    metric_df = pd.DataFrame(metric_rows)
    if len(metric_df) == 0:
        raise RuntimeError("No candidate predictions found for Eawag.")

    metric_df = metric_df.sort_values(["method", "mae"])
    metric_df.to_csv(OUT / "eawag_candidate_metrics.csv", index=False)

    print("\n=== CANDIDATE METRICS ===")
    print(metric_df.to_string(index=False))

    mean_metrics = metric_df[metric_df["method"] == "mean_tl"].copy()
    best_name = mean_metrics.sort_values("mae").iloc[0]["candidate"]
    print("\n=== BEST CANDIDATE ===")
    print(best_name)
    print("ABCoRT target:", AB_TARGET)

    best = pred_tables[best_name].copy()

    # 1. fold diagnosis
    fold_rows = []
    for fold, sub in best.groupby("cv_fold"):
        fold_rows.append({
            "candidate": best_name,
            "cv_fold": int(fold),
            "n": len(sub),
            "rt_min": float(sub["rt"].min()),
            "rt_max": float(sub["rt"].max()),
            "rt_mean": float(sub["rt"].mean()),
            "rt_std": float(sub["rt"].std()),
            "origin_mae": mae(sub["rt"], sub["origin_tl_pred"]),
            "taut_mae": mae(sub["rt"], sub["taut_tl_pred"]),
            "mean_mae": mae(sub["rt"], sub["mean_tl_pred"]),
            "mean_rmse": rmse(sub["rt"], sub["mean_tl_pred"]),
            "bias": float((sub["mean_tl_pred"] - sub["rt"]).mean()),
            "max_abs_err": float(sub["mean_tl_abs_error"].max()),
            "num_err_gt_30": int((sub["mean_tl_abs_error"] > 30).sum()),
            "num_err_gt_50": int((sub["mean_tl_abs_error"] > 50).sum()),
            "num_err_gt_75": int((sub["mean_tl_abs_error"] > 75).sum()),
            "num_err_gt_100": int((sub["mean_tl_abs_error"] > 100).sum()),
        })

    fold_df = pd.DataFrame(fold_rows).sort_values("mean_mae", ascending=False)
    fold_df.to_csv(OUT / "eawag_best_fold_diagnosis.csv", index=False)

    print("\n=== BEST FOLD DIAGNOSIS ===")
    print(fold_df.to_string(index=False))

    # 2. RT-bin diagnosis
    b = best.copy()
    try:
        b["rt_bin"] = pd.qcut(b["rt"], q=5, labels=False, duplicates="drop")
    except Exception:
        b["rt_bin"] = pd.cut(b["rt"], bins=5, labels=False)

    bin_rows = []
    for rb, sub in b.groupby("rt_bin"):
        bin_rows.append({
            "rt_bin": int(rb),
            "n": len(sub),
            "rt_min": float(sub["rt"].min()),
            "rt_max": float(sub["rt"].max()),
            "origin_mae": mae(sub["rt"], sub["origin_tl_pred"]),
            "taut_mae": mae(sub["rt"], sub["taut_tl_pred"]),
            "mean_mae": mae(sub["rt"], sub["mean_tl_pred"]),
            "bias": float((sub["mean_tl_pred"] - sub["rt"]).mean()),
            "err_gt_50": int((sub["mean_tl_abs_error"] > 50).sum()),
            "err_gt_75": int((sub["mean_tl_abs_error"] > 75).sum()),
            "err_gt_100": int((sub["mean_tl_abs_error"] > 100).sum()),
        })

    bin_df = pd.DataFrame(bin_rows).sort_values("rt_bin")
    bin_df.to_csv(OUT / "eawag_best_rtbin_diagnosis.csv", index=False)

    print("\n=== RT BIN DIAGNOSIS ===")
    print(bin_df.to_string(index=False))

    # 3. chemical class diagnosis
    class_cols = [
        "has_aromatic", "acid_like", "halogenated", "high_oxygen",
        "large_heavy", "small_polar", "taut_changed", "smrt_exact_overlap",
    ]
    class_rows = []
    for c in class_cols:
        if c not in b.columns:
            continue
        for val, sub in b.groupby(c):
            if len(sub) < 5:
                continue
            class_rows.append({
                "class_col": c,
                "class_val": val,
                "n": len(sub),
                "rt_mean": float(sub["rt"].mean()),
                "mean_mae": mae(sub["rt"], sub["mean_tl_pred"]),
                "origin_mae": mae(sub["rt"], sub["origin_tl_pred"]),
                "taut_mae": mae(sub["rt"], sub["taut_tl_pred"]),
                "bias": float((sub["mean_tl_pred"] - sub["rt"]).mean()),
                "err_gt_50": int((sub["mean_tl_abs_error"] > 50).sum()),
                "err_gt_75": int((sub["mean_tl_abs_error"] > 75).sum()),
                "err_gt_100": int((sub["mean_tl_abs_error"] > 100).sum()),
            })

    class_df = pd.DataFrame(class_rows).sort_values("mean_mae", ascending=False)
    class_df.to_csv(OUT / "eawag_best_class_diagnosis.csv", index=False)

    print("\n=== CLASS DIAGNOSIS ===")
    print(class_df.to_string(index=False))

    # 4. top errors
    keep_cols = [
        "cv_fold", "stage4_index", "record_id", "name", "formula", "inchikey", "rt",
        "origin_tl_pred", "taut_tl_pred", "mean_tl_pred",
        "origin_tl_abs_error", "taut_tl_abs_error", "mean_tl_abs_error",
        "taut_changed", "smrt_exact_overlap",
        "has_aromatic", "acid_like", "halogenated", "high_oxygen", "large_heavy", "small_polar",
        "origin_smiles", "taut_smiles",
    ]
    keep_cols = [c for c in keep_cols if c in b.columns]

    top_err = b[keep_cols].sort_values("mean_tl_abs_error", ascending=False)
    top_err.to_csv(OUT / "eawag_best_top_errors.csv", index=False)

    print("\n=== TOP 25 ERRORS IN BEST CANDIDATE ===")
    print(top_err.head(25).to_string(index=False))

    # 5. common hard samples and candidate oracle
    common_cols = [
        "stage4_index", "cv_fold", "record_id", "name", "formula", "inchikey", "rt",
        "taut_changed", "smrt_exact_overlap", "origin_smiles", "taut_smiles",
    ]
    common_cols = [c for c in common_cols if c in eawag_meta.columns]
    common = eawag_meta[common_cols].copy()

    mean_pred_cols = []
    for cand, p in pred_tables.items():
        small = p[["stage4_index", "mean_tl_pred", "mean_tl_abs_error"]].copy()
        small = small.rename(columns={
            "mean_tl_pred": f"{cand}__pred",
            "mean_tl_abs_error": f"{cand}__abs_err",
        })
        common = common.merge(small, on="stage4_index", how="left")
        mean_pred_cols.append(f"{cand}__pred")

    err_cols = [c for c in common.columns if c.endswith("__abs_err")]
    common["hard_count_gt50"] = common[err_cols].gt(50).sum(axis=1)
    common["hard_count_gt75"] = common[err_cols].gt(75).sum(axis=1)
    common["hard_count_gt100"] = common[err_cols].gt(100).sum(axis=1)
    common["mean_abs_err_across_candidates"] = common[err_cols].mean(axis=1)
    common["min_abs_err_across_candidates"] = common[err_cols].min(axis=1)

    common = common.sort_values(
        ["hard_count_gt100", "hard_count_gt75", "hard_count_gt50", "mean_abs_err_across_candidates"],
        ascending=False,
    )
    common.to_csv(OUT / "eawag_common_hard_samples.csv", index=False)

    oracle_mae = float(common["min_abs_err_across_candidates"].mean())
    print("\n=== CANDIDATE ORACLE ===")
    print("best_mean_mae:", float(mean_metrics.sort_values("mae").iloc[0]["mae"]))
    print("oracle_candidate_mae:", oracle_mae)
    print("ABCoRT_target:", AB_TARGET)

    print("\n=== COMMON HARD SAMPLES TOP 25 ===")
    show_cols = [
        "cv_fold", "stage4_index", "record_id", "name", "formula", "rt",
        "hard_count_gt50", "hard_count_gt75", "hard_count_gt100",
        "mean_abs_err_across_candidates", "min_abs_err_across_candidates",
    ]
    show_cols = [c for c in show_cols if c in common.columns]
    print(common[show_cols].head(25).to_string(index=False))

    # 6. view bias categories
    b["origin_signed_err"] = b["origin_tl_pred"] - b["rt"]
    b["taut_signed_err"] = b["taut_tl_pred"] - b["rt"]
    b["mean_signed_err"] = b["mean_tl_pred"] - b["rt"]
    b["view_gap_abs"] = np.abs(b["origin_tl_pred"] - b["taut_tl_pred"])

    def category(r):
        if r["origin_signed_err"] > 0 and r["taut_signed_err"] > 0:
            return "both_over"
        if r["origin_signed_err"] < 0 and r["taut_signed_err"] < 0:
            return "both_under"
        return "view_disagree"

    b["view_error_category"] = b.apply(category, axis=1)
    cat = b.groupby("view_error_category").agg(
        n=("stage4_index", "count"),
        mean_mae=("mean_tl_abs_error", "mean"),
        max_mae=("mean_tl_abs_error", "max"),
        mean_view_gap=("view_gap_abs", "mean"),
        bias=("mean_signed_err", "mean"),
    ).reset_index().sort_values("mean_mae", ascending=False)
    cat.to_csv(OUT / "eawag_view_bias_categories.csv", index=False)

    print("\n=== VIEW ERROR CATEGORIES ===")
    print(cat.to_string(index=False))

    print("\n=== SAVED ===")
    for f in [
        "eawag_candidate_metrics.csv",
        "eawag_best_fold_diagnosis.csv",
        "eawag_best_rtbin_diagnosis.csv",
        "eawag_best_class_diagnosis.csv",
        "eawag_best_top_errors.csv",
        "eawag_common_hard_samples.csv",
        "eawag_view_bias_categories.csv",
    ]:
        print(OUT / f)


if __name__ == "__main__":
    main()

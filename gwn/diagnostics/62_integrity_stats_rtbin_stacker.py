import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.exceptions import ConvergenceWarning

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


DEFAULT_RUNS = [
    ("seed1", "results_OOF_DualView_Stack_v1"),
    ("seed79", "results_OOF_DualView_Stack_seed79"),
    ("seed123", "results_OOF_DualView_Stack_seed123"),
    ("seed256", "results_OOF_DualView_Stack_seed256"),
    ("seed5", "results_OOF_DualView_Stack_seed5"),
]


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_csv(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path} shape={df.shape}")


def metric_dict(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = np.abs(y - p)
    rel = e / (np.abs(y) + 1e-8) * 100.0
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(len(y)),
        "mae": float(np.mean(e)),
        "mre": float(np.mean(rel)),
        "medae": float(np.median(e)),
        "medre": float(np.median(rel)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "gt50": int((e > 50).sum()),
        "gt80": int((e > 80).sum()),
        "gt100": int((e > 100).sum()),
        "gt150": int((e > 150).sum()),
        "gt200": int((e > 200).sum()),
        "bias": float(np.mean(p - y)),
    }


def summarize_mean_std(df, group_cols, metric_cols):
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        if "run" in sub.columns:
            row["num_runs"] = int(sub["run"].nunique())
        else:
            row["num_rows"] = int(len(sub))
        for c in metric_cols:
            if c in sub.columns:
                vals = pd.to_numeric(sub[c], errors="coerce")
                row[f"{c}_mean"] = float(vals.mean())
                row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def norm_columns(df):
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    if "smile" in df.columns and "smiles" not in df.columns:
        df = df.rename(columns={"smile": "smiles"})
    return df


def parse_boolish(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (bool, np.bool_)):
        return float(x)
    s = str(x).strip().lower()
    if s in ["1", "true", "yes", "y", "t"]:
        return 1.0
    if s in ["0", "false", "no", "n", "f", "none", ""]:
        return 0.0
    try:
        return float(s)
    except Exception:
        return np.nan


def safe_mol(smi):
    try:
        return Chem.MolFromSmiles(str(smi))
    except Exception:
        return None


def safe_canon(smi):
    m = safe_mol(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def safe_formula(smi):
    m = safe_mol(smi)
    if m is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(m)
    except Exception:
        return None


def safe_inchikey(smi):
    m = safe_mol(smi)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)
    except Exception:
        return None


def load_valid_table(csv_path):
    raw = pd.read_csv(csv_path, engine="python")
    df = norm_columns(raw)

    if "smiles" not in df.columns or "rt" not in df.columns:
        raise ValueError(f"{csv_path} must contain smiles/smile and rt columns. Existing columns={df.columns.tolist()}")

    df["rt"] = df["rt"].astype(float)
    after_rt = df[df["rt"] > 300.0].copy()

    rows = []
    for source_idx, row in after_rt.iterrows():
        smi = str(row["smiles"])
        mol = safe_mol(smi)
        if mol is None:
            continue
        item = row.to_dict()
        item["source_index"] = int(source_idx)
        item["smiles"] = smi
        item["rt"] = float(row["rt"])
        item["canonical_smiles"] = safe_canon(smi)
        ik = safe_inchikey(smi)
        item["inchikey"] = ik
        item["inchikey14"] = ik[:14] if isinstance(ik, str) and len(ik) >= 14 else None
        item["formula"] = safe_formula(smi)
        rows.append(item)

    valid = pd.DataFrame(rows).reset_index(drop=True)
    return raw, df, after_rt, valid


def count_bool_col(df, col):
    if col not in df.columns:
        return np.nan
    vals = df[col].apply(parse_boolish)
    return int(np.nansum(vals.values))


def collect_integrity(args):
    paths = {
        "origin_train": args.origin_train_csv,
        "origin_test": args.origin_test_csv,
        "taut_train": args.taut_train_csv,
        "taut_test": args.taut_test_csv,
    }

    loaded = {}
    audit_rows = []

    for name, path in paths.items():
        raw, norm, after_rt, valid = load_valid_table(path)
        loaded[name] = {
            "raw": raw,
            "norm": norm,
            "after_rt": after_rt,
            "valid": valid,
        }

        row = {
            "table": name,
            "path": path,
            "raw_rows": int(len(raw)),
            "rows_after_rt_gt_300": int(len(after_rt)),
            "valid_rdkit_rows": int(len(valid)),
            "invalid_or_filtered_by_rdkit_after_rt": int(len(after_rt) - len(valid)),
            "unique_exact_smiles_valid": int(valid["smiles"].nunique()) if "smiles" in valid.columns else np.nan,
            "unique_canonical_smiles_valid": int(valid["canonical_smiles"].nunique()) if "canonical_smiles" in valid.columns else np.nan,
            "unique_inchikey14_valid": int(valid["inchikey14"].nunique()) if "inchikey14" in valid.columns else np.nan,
        }

        for c in ["raw_changed", "real_changed", "fallback", "formula_same", "heavy_same"]:
            if c in valid.columns:
                row[f"{c}_true_count"] = count_bool_col(valid, c)
                row[f"{c}_false_count"] = int(len(valid) - count_bool_col(valid, c))
            else:
                row[f"{c}_true_count"] = np.nan
                row[f"{c}_false_count"] = np.nan

        audit_rows.append(row)

    audit = pd.DataFrame(audit_rows)

    # Pairing audit.
    pair_rows = []
    for split in ["train", "test"]:
        ori = loaded[f"origin_{split}"]["valid"]
        tau = loaded[f"taut_{split}"]["valid"]

        row = {
            "split": split,
            "origin_rows": int(len(ori)),
            "taut_rows": int(len(tau)),
            "same_row_count": int(len(ori) == len(tau)),
            "max_rt_diff": np.nan,
            "mean_rt_diff": np.nan,
            "exact_origin_matches_taut_orig_smile_count": np.nan,
            "canonical_origin_matches_taut_orig_smile_count": np.nan,
            "taut_changed_count": np.nan,
            "taut_changed_ratio": np.nan,
            "fallback_count": np.nan,
            "formula_same_count": np.nan,
            "heavy_same_count": np.nan,
        }

        if len(ori) == len(tau):
            rt_diff = np.abs(ori["rt"].values.astype(float) - tau["rt"].values.astype(float))
            row["max_rt_diff"] = float(np.max(rt_diff))
            row["mean_rt_diff"] = float(np.mean(rt_diff))

            if "orig_smile" in tau.columns:
                row["exact_origin_matches_taut_orig_smile_count"] = int((ori["smiles"].astype(str).values == tau["orig_smile"].astype(str).values).sum())

                ori_can = ori["smiles"].map(safe_canon).astype(str).values
                tau_orig_can = tau["orig_smile"].map(safe_canon).astype(str).values
                row["canonical_origin_matches_taut_orig_smile_count"] = int((ori_can == tau_orig_can).sum())

            changed_col = "real_changed" if "real_changed" in tau.columns else ("raw_changed" if "raw_changed" in tau.columns else None)
            if changed_col:
                vals = tau[changed_col].apply(parse_boolish)
                row["taut_changed_count"] = int(np.nansum(vals.values))
                row["taut_changed_ratio"] = float(np.nanmean(vals.values))

            row["fallback_count"] = count_bool_col(tau, "fallback") if "fallback" in tau.columns else np.nan
            row["formula_same_count"] = count_bool_col(tau, "formula_same") if "formula_same" in tau.columns else np.nan
            row["heavy_same_count"] = count_bool_col(tau, "heavy_same") if "heavy_same" in tau.columns else np.nan

        pair_rows.append(row)

    pairing = pd.DataFrame(pair_rows)

    # Train/test overlap audit.
    train = loaded["origin_train"]["valid"]
    test = loaded["origin_test"]["valid"]

    def set_no_none(series):
        return set([x for x in series.tolist() if isinstance(x, str) and x and x != "None" and x != "nan"])

    train_exact = set_no_none(train["smiles"].astype(str))
    test_exact = set_no_none(test["smiles"].astype(str))
    train_can = set_no_none(train["canonical_smiles"].astype(str))
    test_can = set_no_none(test["canonical_smiles"].astype(str))
    train_ik = set_no_none(train["inchikey"].astype(str))
    test_ik = set_no_none(test["inchikey"].astype(str))
    train_ik14 = set_no_none(train["inchikey14"].astype(str))
    test_ik14 = set_no_none(test["inchikey14"].astype(str))

    overlap_rows = [
        {
            "key_type": "exact_smiles",
            "train_unique": len(train_exact),
            "test_unique": len(test_exact),
            "overlap_unique": len(train_exact & test_exact),
            "overlap_ratio_vs_test_unique": len(train_exact & test_exact) / max(len(test_exact), 1),
        },
        {
            "key_type": "canonical_smiles",
            "train_unique": len(train_can),
            "test_unique": len(test_can),
            "overlap_unique": len(train_can & test_can),
            "overlap_ratio_vs_test_unique": len(train_can & test_can) / max(len(test_can), 1),
        },
        {
            "key_type": "full_inchikey",
            "train_unique": len(train_ik),
            "test_unique": len(test_ik),
            "overlap_unique": len(train_ik & test_ik),
            "overlap_ratio_vs_test_unique": len(train_ik & test_ik) / max(len(test_ik), 1),
        },
        {
            "key_type": "inchikey14_connectivity",
            "train_unique": len(train_ik14),
            "test_unique": len(test_ik14),
            "overlap_unique": len(train_ik14 & test_ik14),
            "overlap_ratio_vs_test_unique": len(train_ik14 & test_ik14) / max(len(test_ik14), 1),
        },
    ]

    overlap = pd.DataFrame(overlap_rows)

    return audit, pairing, overlap, loaded


def build_stack_features_by_variant(o, t, changed, variant):
    o = np.asarray(o, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    changed = np.asarray(changed, dtype=np.float64)

    diff = np.abs(o - t)
    mean = 0.5 * (o + t)
    minp = np.minimum(o, t)
    maxp = np.maximum(o, t)

    if variant == "origin_taut":
        cols = [o, t]
    elif variant == "origin_taut_diff":
        cols = [o, t, diff]
    elif variant == "origin_taut_mean_min_max":
        cols = [o, t, diff, mean, minp, maxp]
    elif variant == "origin_taut_changed":
        cols = [o, t, changed]
    elif variant == "origin_taut_diff_changed":
        cols = [o, t, diff, changed, diff * changed]
    elif variant == "all_features":
        cols = [
            o,
            t,
            diff,
            mean,
            minp,
            maxp,
            changed,
            diff * changed,
            o * changed / 1000.0,
            t * changed / 1000.0,
        ]
    else:
        raise ValueError(f"Unknown feature variant: {variant}")

    return np.vstack(cols).T


def fit_huber_feature_variant(oof, test, variant, huber_alpha):
    y_oof = oof["Actual_RT"].values.astype(np.float64)

    x_oof = build_stack_features_by_variant(
        oof["Origin_OOF_Pred"].values,
        oof["Taut_OOF_Pred"].values,
        oof["Taut_Changed"].values,
        variant,
    )
    x_test = build_stack_features_by_variant(
        test["Origin_Test_Pred"].values,
        test["Taut_Test_Pred"].values,
        test["Taut_Changed"].values,
        variant,
    )

    model = make_pipeline(
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=huber_alpha, max_iter=1000),
    )
    model.fit(x_oof, y_oof)
    return model.predict(x_oof), model.predict(x_test)


def load_run(run_name, run_dir):
    run_dir = Path(run_dir)
    oof_path = run_dir / "oof_predictions.csv"
    test_path = run_dir / "test_predictions.csv"

    if not oof_path.exists():
        raise FileNotFoundError(oof_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    oof = pd.read_csv(oof_path)
    test = pd.read_csv(test_path)

    needed_oof = ["Actual_RT", "Origin_OOF_Pred", "Taut_OOF_Pred", "Taut_Changed"]
    needed_test = ["Actual_RT", "Origin_Test_Pred", "Taut_Test_Pred", "Final_Pred", "Taut_Changed"]

    for c in needed_oof:
        if c not in oof.columns:
            raise ValueError(f"{oof_path} missing column {c}")
    for c in needed_test:
        if c not in test.columns:
            raise ValueError(f"{test_path} missing column {c}")

    return oof, test


def collect_fold_averaging(run_specs):
    rows = []

    for run_name, run_dir in run_specs:
        run_dir = Path(run_dir)
        test_df = pd.read_csv(run_dir / "test_predictions.csv")
        y = test_df["Actual_RT"].values.astype(np.float64)

        for view, final_col in [
            ("origin", "Origin_Test_Pred"),
            ("taut", "Taut_Test_Pred"),
        ]:
            fold_preds = []
            for fold in range(5):
                p = run_dir / "folds" / f"fold_{fold}" / view / "test_pred.npy"
                if not p.exists():
                    print(f"[WARN] missing fold prediction: {p}")
                    continue

                pred = np.load(p).astype(np.float64)
                fold_preds.append(pred)

                rows.append({
                    "run": run_name,
                    "view": view,
                    "level": "single_fold",
                    "fold": fold,
                    **metric_dict(y, pred),
                })

            if fold_preds:
                mean_pred = np.mean(np.vstack(fold_preds), axis=0)
                rows.append({
                    "run": run_name,
                    "view": view,
                    "level": "five_fold_mean_from_npy",
                    "fold": -1,
                    **metric_dict(y, mean_pred),
                })

            if final_col in test_df.columns:
                rows.append({
                    "run": run_name,
                    "view": view,
                    "level": "five_fold_mean_from_csv",
                    "fold": -1,
                    **metric_dict(y, test_df[final_col].values.astype(np.float64)),
                })

    fold_df = pd.DataFrame(rows)
    metric_cols = ["mae", "mre", "medae", "rmse", "r2", "p95", "p99", "gt100", "gt200", "bias"]
    summary = summarize_mean_std(fold_df, ["view", "level"], metric_cols)
    return fold_df, summary


def collect_seed_stability(run_specs):
    rows = []
    for run_name, run_dir in run_specs:
        _, test = load_run(run_name, run_dir)
        y = test["Actual_RT"].values.astype(np.float64)

        preds = {
            "origin_only": test["Origin_Test_Pred"].values.astype(np.float64),
            "taut_only": test["Taut_Test_Pred"].values.astype(np.float64),
            "mean_origin_taut": 0.5 * (
                test["Origin_Test_Pred"].values.astype(np.float64)
                + test["Taut_Test_Pred"].values.astype(np.float64)
            ),
            "huber_stack": test["Final_Pred"].values.astype(np.float64),
        }

        for method, pred in preds.items():
            rows.append({"run": run_name, "method": method, **metric_dict(y, pred)})

    df = pd.DataFrame(rows)
    metric_cols = ["mae", "mre", "medae", "rmse", "r2", "p95", "p99", "gt100", "gt200", "bias"]
    summary = summarize_mean_std(df, ["method"], metric_cols)
    return df, summary


def paired_bootstrap(run_specs, repeats, seed):
    rng = np.random.default_rng(seed)
    rows = []

    comparisons = [
        ("huber_stack", "origin_only"),
        ("huber_stack", "taut_only"),
        ("huber_stack", "mean_origin_taut"),
        ("mean_origin_taut", "origin_only"),
        ("mean_origin_taut", "taut_only"),
        ("taut_only", "origin_only"),
    ]

    for run_name, run_dir in run_specs:
        _, test = load_run(run_name, run_dir)

        y = test["Actual_RT"].values.astype(np.float64)
        pred = {
            "origin_only": test["Origin_Test_Pred"].values.astype(np.float64),
            "taut_only": test["Taut_Test_Pred"].values.astype(np.float64),
            "mean_origin_taut": 0.5 * (
                test["Origin_Test_Pred"].values.astype(np.float64)
                + test["Taut_Test_Pred"].values.astype(np.float64)
            ),
            "huber_stack": test["Final_Pred"].values.astype(np.float64),
        }

        n = len(y)

        for method_a, method_b in comparisons:
            diff_samples = []
            ea = np.abs(y - pred[method_a])
            eb = np.abs(y - pred[method_b])

            observed_delta = float(np.mean(ea) - np.mean(eb))

            for _ in range(repeats):
                idx = rng.integers(0, n, size=n)
                diff_samples.append(float(np.mean(ea[idx]) - np.mean(eb[idx])))

            diff_samples = np.asarray(diff_samples, dtype=np.float64)

            # delta < 0 means method_a has lower MAE than method_b.
            p_a_better = float(np.mean(diff_samples < 0))
            p_two_sided = float(2.0 * min(np.mean(diff_samples <= 0), np.mean(diff_samples >= 0)))
            p_two_sided = min(p_two_sided, 1.0)

            rows.append({
                "run": run_name,
                "method_a": method_a,
                "method_b": method_b,
                "observed_delta_mae_a_minus_b": observed_delta,
                "bootstrap_delta_mean": float(np.mean(diff_samples)),
                "ci95_low": float(np.percentile(diff_samples, 2.5)),
                "ci95_high": float(np.percentile(diff_samples, 97.5)),
                "prob_method_a_better": p_a_better,
                "two_sided_p_approx": p_two_sided,
                "bootstrap_repeats": int(repeats),
            })

    df = pd.DataFrame(rows)

    summary_rows = []
    for (a, b), sub in df.groupby(["method_a", "method_b"]):
        row = {
            "method_a": a,
            "method_b": b,
            "num_runs": int(sub["run"].nunique()),
            "observed_delta_mae_mean": float(sub["observed_delta_mae_a_minus_b"].mean()),
            "observed_delta_mae_std": float(sub["observed_delta_mae_a_minus_b"].std(ddof=1)),
            "ci95_low_mean": float(sub["ci95_low"].mean()),
            "ci95_high_mean": float(sub["ci95_high"].mean()),
            "prob_method_a_better_mean": float(sub["prob_method_a_better"].mean()),
            "two_sided_p_approx_mean": float(sub["two_sided_p_approx"].mean()),
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    return df, summary


def rt_bins_fixed(y):
    y = np.asarray(y, dtype=np.float64)
    bins = [
        (300, 600, "300-600"),
        (600, 800, "600-800"),
        (800, 1000, "800-1000"),
        (1000, 1200, "1000-1200"),
        (1200, np.inf, ">1200"),
    ]
    labels = np.empty(len(y), dtype=object)
    for lo, hi, lab in bins:
        labels[(y >= lo) & (y < hi)] = lab
    labels[pd.isna(labels)] = "outside"
    return labels


def rt_bins_quantile(y, q=5):
    y = np.asarray(y, dtype=np.float64)
    try:
        codes = pd.qcut(y, q=q, labels=False, duplicates="drop")
        codes = np.asarray(codes)
        labels = np.array([f"Q{int(c)+1}" for c in codes], dtype=object)
        return labels
    except Exception:
        return np.array(["QNA"] * len(y), dtype=object)


def collect_rtbin(run_specs):
    rows = []

    for run_name, run_dir in run_specs:
        _, test = load_run(run_name, run_dir)
        y = test["Actual_RT"].values.astype(np.float64)

        methods = {
            "origin_only": test["Origin_Test_Pred"].values.astype(np.float64),
            "taut_only": test["Taut_Test_Pred"].values.astype(np.float64),
            "mean_origin_taut": 0.5 * (
                test["Origin_Test_Pred"].values.astype(np.float64)
                + test["Taut_Test_Pred"].values.astype(np.float64)
            ),
            "huber_stack": test["Final_Pred"].values.astype(np.float64),
        }

        bin_sets = {
            "fixed_rt": rt_bins_fixed(y),
            "quantile_rt": rt_bins_quantile(y, q=5),
        }

        for bin_type, labels in bin_sets.items():
            for lab in sorted(pd.unique(labels), key=lambda x: str(x)):
                mask = labels == lab
                if mask.sum() == 0:
                    continue
                for method, p in methods.items():
                    rows.append({
                        "run": run_name,
                        "bin_type": bin_type,
                        "bin_label": str(lab),
                        "method": method,
                        **metric_dict(y[mask], p[mask]),
                    })

    df = pd.DataFrame(rows)
    metric_cols = ["mae", "mre", "medae", "rmse", "r2", "p95", "p99", "gt100", "gt200", "bias"]
    summary = summarize_mean_std(df, ["bin_type", "bin_label", "method"], metric_cols)
    return df, summary


def collect_stacker_feature_ablation(run_specs, huber_alpha):
    variants = [
        "origin_taut",
        "origin_taut_diff",
        "origin_taut_mean_min_max",
        "origin_taut_changed",
        "origin_taut_diff_changed",
        "all_features",
    ]

    rows = []

    for run_name, run_dir in run_specs:
        oof, test = load_run(run_name, run_dir)
        y_test = test["Actual_RT"].values.astype(np.float64)

        # Baselines.
        baseline_preds = {
            "origin_only": test["Origin_Test_Pred"].values.astype(np.float64),
            "taut_only": test["Taut_Test_Pred"].values.astype(np.float64),
            "mean_origin_taut": 0.5 * (
                test["Origin_Test_Pred"].values.astype(np.float64)
                + test["Taut_Test_Pred"].values.astype(np.float64)
            ),
            "reported_final_huber_stack": test["Final_Pred"].values.astype(np.float64),
        }

        for name, pred in baseline_preds.items():
            rows.append({
                "run": run_name,
                "feature_set": name,
                "model": "baseline_or_existing",
                **metric_dict(y_test, pred),
            })

        for variant in variants:
            _, pred_test = fit_huber_feature_variant(oof, test, variant, huber_alpha)
            rows.append({
                "run": run_name,
                "feature_set": variant,
                "model": "StandardScaler+HuberRegressor",
                **metric_dict(y_test, pred_test),
            })

    df = pd.DataFrame(rows)
    metric_cols = ["mae", "mre", "medae", "rmse", "r2", "p95", "p99", "gt100", "gt200", "bias"]
    summary = summarize_mean_std(df, ["feature_set", "model"], metric_cols)
    return df, summary


def disagreement_labels_quantile(d):
    d = np.asarray(d, dtype=np.float64)
    q1, q2 = np.quantile(d, [1/3, 2/3])
    labels = np.empty(len(d), dtype=object)
    labels[d <= q1] = "low_disagreement"
    labels[(d > q1) & (d <= q2)] = "mid_disagreement"
    labels[d > q2] = "high_disagreement"
    return labels, {"q33": float(q1), "q67": float(q2)}


def disagreement_labels_fixed(d):
    d = np.asarray(d, dtype=np.float64)
    labels = np.empty(len(d), dtype=object)
    labels[d < 5] = "<5s"
    labels[(d >= 5) & (d < 20)] = "5-20s"
    labels[(d >= 20) & (d < 50)] = "20-50s"
    labels[d >= 50] = ">=50s"
    return labels


def collect_view_disagreement(run_specs):
    rows = []
    thresholds_rows = []

    for run_name, run_dir in run_specs:
        _, test = load_run(run_name, run_dir)
        y = test["Actual_RT"].values.astype(np.float64)
        o = test["Origin_Test_Pred"].values.astype(np.float64)
        t = test["Taut_Test_Pred"].values.astype(np.float64)
        final = test["Final_Pred"].values.astype(np.float64)
        mean = 0.5 * (o + t)

        d = np.abs(o - t)

        q_labels, qinfo = disagreement_labels_quantile(d)
        thresholds_rows.append({
            "run": run_name,
            "q33_abs_origin_taut_diff": qinfo["q33"],
            "q67_abs_origin_taut_diff": qinfo["q67"],
        })

        label_sets = {
            "quantile_disagreement": q_labels,
            "fixed_disagreement": disagreement_labels_fixed(d),
        }

        methods = {
            "origin_only": o,
            "taut_only": t,
            "mean_origin_taut": mean,
            "huber_stack": final,
        }

        for group_type, labels in label_sets.items():
            for lab in sorted(pd.unique(labels), key=lambda x: str(x)):
                mask = labels == lab
                if mask.sum() == 0:
                    continue
                for method, pred in methods.items():
                    rows.append({
                        "run": run_name,
                        "group_type": group_type,
                        "group": str(lab),
                        "method": method,
                        "mean_view_disagreement": float(np.mean(d[mask])),
                        "median_view_disagreement": float(np.median(d[mask])),
                        **metric_dict(y[mask], pred[mask]),
                    })

    df = pd.DataFrame(rows)
    thresholds = pd.DataFrame(thresholds_rows)

    metric_cols = [
        "mae", "mre", "medae", "rmse", "r2", "p95", "p99",
        "gt100", "gt200", "bias", "mean_view_disagreement", "median_view_disagreement",
    ]
    summary = summarize_mean_std(df, ["group_type", "group", "method"], metric_cols)
    return df, summary, thresholds


def collect_plot_data(run_specs, out_dir):
    # 输出可画图数据，不直接画，避免后续改图麻烦。
    seed_rows = []
    scatter_rows = []
    hist_rows = []
    ecdf_rows = []

    for run_name, run_dir in run_specs:
        _, test = load_run(run_name, run_dir)
        y = test["Actual_RT"].values.astype(np.float64)
        pred = test["Final_Pred"].values.astype(np.float64)
        err = pred - y
        abs_err = np.abs(err)

        m = metric_dict(y, pred)
        seed_rows.append({"run": run_name, **m})

        tmp = pd.DataFrame({
            "run": run_name,
            "Actual_RT": y,
            "Final_Pred": pred,
            "Residual": err,
            "Abs_Error": abs_err,
            "Taut_Changed": test["Taut_Changed"].values,
            "Origin_Test_Pred": test["Origin_Test_Pred"].values,
            "Taut_Test_Pred": test["Taut_Test_Pred"].values,
        })
        scatter_rows.append(tmp)

        counts, edges = np.histogram(err, bins=80)
        for i in range(len(counts)):
            hist_rows.append({
                "run": run_name,
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "count": int(counts[i]),
            })

        xs = np.sort(abs_err)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        # downsample for compact figure data
        idx = np.linspace(0, len(xs) - 1, min(1000, len(xs))).astype(int)
        for j in idx:
            ecdf_rows.append({
                "run": run_name,
                "abs_error": float(xs[j]),
                "ecdf": float(ys[j]),
            })

    seed_df = pd.DataFrame(seed_rows)
    scatter_df = pd.concat(scatter_rows, ignore_index=True)
    hist_df = pd.DataFrame(hist_rows)
    ecdf_df = pd.DataFrame(ecdf_rows)

    save_csv(seed_df, Path(out_dir) / "plot_seed_stability.csv")
    save_csv(scatter_df, Path(out_dir) / "plot_actual_vs_predicted_data.csv")
    save_csv(hist_df, Path(out_dir) / "plot_residual_histogram_data.csv")
    save_csv(ecdf_df, Path(out_dir) / "plot_abs_error_ecdf_data.csv")

    return seed_df, scatter_df, hist_df, ecdf_df


def parse_runs(s):
    if s is None or s.strip() == "":
        return DEFAULT_RUNS

    runs = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Custom runs must be formatted as name:path,name:path")
        name, path = item.split(":", 1)
        runs.append((name.strip(), path.strip()))
    return runs


def write_readme(out_dir):
    text = """TCDV-TopoRT stage-1 paper analysis outputs

This folder contains no-retraining analyses:
1. dataset_integrity_audit.csv
2. tautomer_pairing_audit_summary.csv
3. train_test_overlap_audit.csv
4. fold_averaging_protocol_audit.csv
5. fold_averaging_protocol_summary.csv
6. seed_stability_metrics.csv
7. seed_stability_summary.csv
8. paired_bootstrap_significance.csv
9. paired_bootstrap_significance_summary.csv
10. rtbin_error_summary.csv
11. rtbin_error_summary_mean_std.csv
12. stacker_feature_ablation.csv
13. stacker_feature_ablation_summary.csv
14. view_disagreement_subgroup.csv
15. view_disagreement_subgroup_summary.csv
16. view_disagreement_thresholds.csv
17. plot_* CSV files for figures

Interpretation notes:
- Fold-averaged origin/taut results are not single-checkpoint results.
- Paired bootstrap uses test-set resampling only for statistical comparison; no model fitting or test tuning is performed.
- Stacker feature ablation refits Huber stackers from OOF predictions only.
- RT-bin and view-disagreement analyses are stratified test-set diagnostics.
"""
    path = Path(out_dir) / "README_outputs.txt"
    path.write_text(text, encoding="utf-8")
    print(f"[SAVE] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="paper_analysis_62")
    ap.add_argument("--origin_train_csv", default="data/SMRT_train.csv")
    ap.add_argument("--origin_test_csv", default="data/SMRT_test.csv")
    ap.add_argument("--taut_train_csv", default="data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv")
    ap.add_argument("--taut_test_csv", default="data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv")
    ap.add_argument("--runs", default="")
    ap.add_argument("--bootstrap_repeats", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=20260609)
    ap.add_argument("--huber_alpha", type=float, default=1e-4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    run_specs = parse_runs(args.runs)

    print("=== Stage-1 TCDV-TopoRT no-retraining analysis ===")
    print("out_dir:", out_dir)
    print("bootstrap_repeats:", args.bootstrap_repeats)
    print("runs:")
    for name, path in run_specs:
        print(" ", name, "->", path)

    print("\n=== 1. Dataset integrity / leakage audit ===")
    integrity, pairing, overlap, loaded = collect_integrity(args)
    save_csv(integrity, out_dir / "dataset_integrity_audit.csv")
    save_csv(pairing, out_dir / "tautomer_pairing_audit_summary.csv")
    save_csv(overlap, out_dir / "train_test_overlap_audit.csv")

    print("\n=== 2. Fold averaging protocol audit ===")
    fold_df, fold_summary = collect_fold_averaging(run_specs)
    save_csv(fold_df, out_dir / "fold_averaging_protocol_audit.csv")
    save_csv(fold_summary, out_dir / "fold_averaging_protocol_summary.csv")

    print("\n=== 3. Seed stability metrics ===")
    seed_df, seed_summary = collect_seed_stability(run_specs)
    save_csv(seed_df, out_dir / "seed_stability_metrics.csv")
    save_csv(seed_summary, out_dir / "seed_stability_summary.csv")

    print("\n=== 4. Paired bootstrap significance ===")
    boot, boot_summary = paired_bootstrap(
        run_specs,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    save_csv(boot, out_dir / "paired_bootstrap_significance.csv")
    save_csv(boot_summary, out_dir / "paired_bootstrap_significance_summary.csv")

    print("\n=== 5. RT-bin stratified error analysis ===")
    rtbin, rtbin_summary = collect_rtbin(run_specs)
    save_csv(rtbin, out_dir / "rtbin_error_summary.csv")
    save_csv(rtbin_summary, out_dir / "rtbin_error_summary_mean_std.csv")

    print("\n=== 6. Stacker feature ablation ===")
    stacker, stacker_summary = collect_stacker_feature_ablation(run_specs, huber_alpha=args.huber_alpha)
    save_csv(stacker, out_dir / "stacker_feature_ablation.csv")
    save_csv(stacker_summary, out_dir / "stacker_feature_ablation_summary.csv")

    print("\n=== 7. View-disagreement subgroup analysis ===")
    vd, vd_summary, vd_thresholds = collect_view_disagreement(run_specs)
    save_csv(vd, out_dir / "view_disagreement_subgroup.csv")
    save_csv(vd_summary, out_dir / "view_disagreement_subgroup_summary.csv")
    save_csv(vd_thresholds, out_dir / "view_disagreement_thresholds.csv")

    print("\n=== 8. Plot data ===")
    collect_plot_data(run_specs, out_dir)

    write_readme(out_dir)

    print("\n=== Quick check: dataset integrity ===")
    print(integrity.to_string(index=False))

    print("\n=== Quick check: pairing ===")
    print(pairing.to_string(index=False))

    print("\n=== Quick check: train/test overlap ===")
    print(overlap.to_string(index=False))

    print("\n=== Quick check: fold averaging summary ===")
    show_cols = ["view", "level", "mae_mean", "mae_std", "rmse_mean", "r2_mean", "gt100_mean", "gt200_mean"]
    show_cols = [c for c in show_cols if c in fold_summary.columns]
    print(fold_summary[show_cols].to_string(index=False))

    print("\n=== Quick check: bootstrap significance ===")
    print(boot_summary.to_string(index=False))

    print("\n=== Quick check: stacker feature ablation ===")
    show_cols = ["feature_set", "model", "mae_mean", "mae_std", "medae_mean", "rmse_mean", "r2_mean", "gt100_mean", "gt200_mean"]
    show_cols = [c for c in show_cols if c in stacker_summary.columns]
    print(stacker_summary[show_cols].sort_values("mae_mean").to_string(index=False))

    print("\n✅ Done. All files are in:", out_dir)


if __name__ == "__main__":
    main()

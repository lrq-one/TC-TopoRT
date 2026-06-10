import argparse
import json
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


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_csv(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path} shape={df.shape}")


def norm_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "smile" in df.columns and "smiles" not in df.columns:
        df = df.rename(columns={"smile": "smiles"})
    return df


def safe_mol(smiles):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def safe_formula(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


def safe_canon(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def safe_inchikey(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


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
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "gt100": int((e > 100).sum()),
        "gt200": int((e > 200).sum()),
        "bias": float(np.mean(p - y)),
    }


def load_smrt_valid(csv_path, split):
    raw = pd.read_csv(csv_path, engine="python")
    df = norm_columns(raw)

    if "smiles" not in df.columns or "rt" not in df.columns:
        raise ValueError(f"{csv_path} must contain smile/smiles and rt columns. columns={df.columns.tolist()}")

    df["rt"] = df["rt"].astype(float)
    df = df[df["rt"] > 300.0].copy()

    rows = []
    for source_idx, row in df.iterrows():
        smi = str(row["smiles"])
        mol = safe_mol(smi)
        if mol is None:
            continue

        ik = safe_inchikey(smi)
        rows.append({
            "split": split,
            "source_index": int(source_idx),
            "local_index": len(rows),
            "candidate_id": f"{split}_{len(rows)}",
            "smiles": smi,
            "rt": float(row["rt"]),
            "formula": safe_formula(smi),
            "canonical_smiles": safe_canon(smi),
            "inchikey": ik,
            "inchikey14": ik[:14] if isinstance(ik, str) and len(ik) >= 14 else None,
        })

    out = pd.DataFrame(rows)
    return out


def build_stack_features(origin_pred, taut_pred, changed):
    origin_pred = np.asarray(origin_pred, dtype=np.float64)
    taut_pred = np.asarray(taut_pred, dtype=np.float64)
    changed = np.asarray(changed, dtype=np.float64)

    diff = np.abs(origin_pred - taut_pred)
    mean_pred = 0.5 * (origin_pred + taut_pred)
    min_pred = np.minimum(origin_pred, taut_pred)
    max_pred = np.maximum(origin_pred, taut_pred)

    return np.vstack([
        origin_pred,
        taut_pred,
        diff,
        mean_pred,
        min_pred,
        max_pred,
        changed,
        diff * changed,
        origin_pred * changed / 1000.0,
        taut_pred * changed / 1000.0,
    ]).T


def fit_huber_oof_test(oof, test, huber_alpha):
    y_oof = oof["Actual_RT"].values.astype(np.float64)

    x_oof = build_stack_features(
        oof["Origin_OOF_Pred"].values,
        oof["Taut_OOF_Pred"].values,
        oof["Taut_Changed"].values,
    )
    x_test = build_stack_features(
        test["Origin_Test_Pred"].values,
        test["Taut_Test_Pred"].values,
        test["Taut_Changed"].values,
    )

    model = make_pipeline(
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=huber_alpha, max_iter=1000),
    )
    model.fit(x_oof, y_oof)

    return model.predict(x_oof), model.predict(x_test)


def load_run_predictions(run_name, run_dir, train_meta, test_meta, huber_alpha):
    run_dir = Path(run_dir)
    oof_path = run_dir / "oof_predictions.csv"
    test_path = run_dir / "test_predictions.csv"

    if not oof_path.exists():
        raise FileNotFoundError(oof_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    oof = pd.read_csv(oof_path)
    test = pd.read_csv(test_path)

    need_oof = ["Actual_RT", "Origin_OOF_Pred", "Taut_OOF_Pred", "Taut_Changed"]
    need_test = ["Actual_RT", "Origin_Test_Pred", "Taut_Test_Pred", "Final_Pred", "Taut_Changed"]

    for c in need_oof:
        if c not in oof.columns:
            raise ValueError(f"{oof_path} missing {c}")
    for c in need_test:
        if c not in test.columns:
            raise ValueError(f"{test_path} missing {c}")

    if len(oof) != len(train_meta):
        raise RuntimeError(f"{run_name}: oof rows {len(oof)} != train_meta rows {len(train_meta)}")
    if len(test) != len(test_meta):
        raise RuntimeError(f"{run_name}: test rows {len(test)} != test_meta rows {len(test_meta)}")

    # sanity check: order and RT alignment
    train_rt_diff = np.max(np.abs(oof["Actual_RT"].values.astype(float) - train_meta["rt"].values.astype(float)))
    test_rt_diff = np.max(np.abs(test["Actual_RT"].values.astype(float) - test_meta["rt"].values.astype(float)))
    if train_rt_diff > 1e-6:
        raise RuntimeError(f"{run_name}: train RT order mismatch, max diff={train_rt_diff}")
    if test_rt_diff > 1e-6:
        raise RuntimeError(f"{run_name}: test RT order mismatch, max diff={test_rt_diff}")

    huber_oof, huber_test = fit_huber_oof_test(oof, test, huber_alpha=huber_alpha)

    train_pred = pd.DataFrame({
        "candidate_id": train_meta["candidate_id"].values,
        "split": "train",
        "formula": train_meta["formula"].values,
        "actual_rt": train_meta["rt"].values.astype(float),
        "origin_pred": oof["Origin_OOF_Pred"].values.astype(float),
        "taut_pred": oof["Taut_OOF_Pred"].values.astype(float),
        "mean_pred": 0.5 * (
            oof["Origin_OOF_Pred"].values.astype(float)
            + oof["Taut_OOF_Pred"].values.astype(float)
        ),
        "huber_pred": huber_oof.astype(float),
    })

    test_pred = pd.DataFrame({
        "candidate_id": test_meta["candidate_id"].values,
        "split": "test",
        "formula": test_meta["formula"].values,
        "actual_rt": test_meta["rt"].values.astype(float),
        "origin_pred": test["Origin_Test_Pred"].values.astype(float),
        "taut_pred": test["Taut_Test_Pred"].values.astype(float),
        "mean_pred": 0.5 * (
            test["Origin_Test_Pred"].values.astype(float)
            + test["Taut_Test_Pred"].values.astype(float)
        ),
        "huber_pred": test["Final_Pred"].values.astype(float),
    })

    # Also keep refit-huber test version for auditing consistency.
    huber_refit_mae = metric_dict(test_meta["rt"].values.astype(float), huber_test)["mae"]
    final_csv_mae = metric_dict(test_meta["rt"].values.astype(float), test["Final_Pred"].values.astype(float))["mae"]

    if abs(huber_refit_mae - final_csv_mae) > 0.05:
        print(f"[WARN] {run_name}: refit huber MAE {huber_refit_mae:.6f} differs from csv Final_Pred MAE {final_csv_mae:.6f}")

    all_pred = pd.concat([train_pred, test_pred], ignore_index=True)

    method_mae_oof = {
        "origin": metric_dict(train_meta["rt"].values, train_pred["origin_pred"].values)["mae"],
        "taut": metric_dict(train_meta["rt"].values, train_pred["taut_pred"].values)["mae"],
        "mean": metric_dict(train_meta["rt"].values, train_pred["mean_pred"].values)["mae"],
        "huber": metric_dict(train_meta["rt"].values, train_pred["huber_pred"].values)["mae"],
    }

    method_mae_test = {
        "origin": metric_dict(test_meta["rt"].values, test_pred["origin_pred"].values)["mae"],
        "taut": metric_dict(test_meta["rt"].values, test_pred["taut_pred"].values)["mae"],
        "mean": metric_dict(test_meta["rt"].values, test_pred["mean_pred"].values)["mae"],
        "huber": metric_dict(test_meta["rt"].values, test_pred["huber_pred"].values)["mae"],
    }

    return all_pred, method_mae_oof, method_mae_test


def make_thresholds(method, oof_mae):
    return [
        ("fixed_25s", 25.0),
        ("fixed_50s", 50.0),
        ("fixed_75s", 75.0),
        ("fixed_100s", 100.0),
        ("fixed_150s", 150.0),
        ("1x_oof_mae", float(oof_mae)),
        ("2x_oof_mae", float(2.0 * oof_mae)),
        ("3x_oof_mae", float(3.0 * oof_mae)),
    ]


def rank_of_true(pool, query_id, query_rt, pred_col):
    tmp = pool.copy()
    tmp["rt_distance"] = np.abs(tmp[pred_col].values.astype(float) - float(query_rt))
    tmp = tmp.sort_values(["rt_distance", "candidate_id"], ascending=[True, True]).reset_index(drop=True)
    match = np.where(tmp["candidate_id"].values == query_id)[0]
    if len(match) == 0:
        return np.nan, np.nan
    rank = int(match[0] + 1)
    true_dist = float(tmp.loc[match[0], "rt_distance"])
    return rank, true_dist


def evaluate_formula_filtering_for_run(run_name, all_pred, test_meta, method_mae_oof, min_pool_size):
    methods = [
        ("origin", "origin_pred"),
        ("taut", "taut_pred"),
        ("mean", "mean_pred"),
        ("huber", "huber_pred"),
    ]

    all_pred = all_pred[all_pred["formula"].notna()].copy()
    formula_groups = {f: sub.reset_index(drop=True) for f, sub in all_pred.groupby("formula")}

    per_query_rows = []
    rank_rows = []

    eligible_count = 0

    for _, q in test_meta.iterrows():
        formula = q["formula"]
        if not isinstance(formula, str) or formula not in formula_groups:
            continue

        pool = formula_groups[formula]
        pool_n = int(len(pool))
        if pool_n < min_pool_size:
            continue

        eligible_count += 1

        query_id = q["candidate_id"]
        query_rt = float(q["rt"])

        for method, pred_col in methods:
            full_rank, true_rt_distance = rank_of_true(pool, query_id, query_rt, pred_col)

            rank_rows.append({
                "run": run_name,
                "query_id": query_id,
                "formula": formula,
                "method": method,
                "pool_n": pool_n,
                "true_rank_full_pool": full_rank,
                "true_top1_full_pool": int(full_rank <= 1) if not pd.isna(full_rank) else 0,
                "true_top5_full_pool": int(full_rank <= 5) if not pd.isna(full_rank) else 0,
                "true_top10_full_pool": int(full_rank <= 10) if not pd.isna(full_rank) else 0,
                "true_rt_distance": true_rt_distance,
                "query_rt": query_rt,
            })

            for threshold_label, threshold_value in make_thresholds(method, method_mae_oof[method]):
                dist = np.abs(pool[pred_col].values.astype(float) - query_rt)
                keep_mask = dist <= threshold_value
                kept = pool.loc[keep_mask].copy()
                after_n = int(len(kept))

                true_retained = int((kept["candidate_id"].values == query_id).any())
                reduction_rate = float((pool_n - after_n) / max(pool_n, 1))

                if true_retained and after_n > 0:
                    kept_rank, _ = rank_of_true(kept, query_id, query_rt, pred_col)
                else:
                    kept_rank = np.nan

                per_query_rows.append({
                    "run": run_name,
                    "query_id": query_id,
                    "formula": formula,
                    "method": method,
                    "threshold_label": threshold_label,
                    "threshold_value": float(threshold_value),
                    "pool_n_before": pool_n,
                    "pool_n_after": after_n,
                    "reduction_rate": reduction_rate,
                    "true_retained": true_retained,
                    "true_rank_full_pool": full_rank,
                    "true_rank_after_filter": kept_rank,
                    "true_top1_full_pool": int(full_rank <= 1) if not pd.isna(full_rank) else 0,
                    "true_top5_full_pool": int(full_rank <= 5) if not pd.isna(full_rank) else 0,
                    "true_top10_full_pool": int(full_rank <= 10) if not pd.isna(full_rank) else 0,
                    "true_top1_after_filter": int(kept_rank <= 1) if not pd.isna(kept_rank) else 0,
                    "true_top5_after_filter": int(kept_rank <= 5) if not pd.isna(kept_rank) else 0,
                    "true_top10_after_filter": int(kept_rank <= 10) if not pd.isna(kept_rank) else 0,
                    "query_rt": query_rt,
                    "true_rt_distance": true_rt_distance,
                })

    per_query = pd.DataFrame(per_query_rows)
    rank_df = pd.DataFrame(rank_rows)

    print(f"[{run_name}] eligible test queries with formula pool >= {min_pool_size}: {eligible_count} / {len(test_meta)}")

    return per_query, rank_df


def summarize_filtering(per_query):
    if per_query.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["run", "method", "threshold_label"]

    for keys, sub in per_query.groupby(group_cols):
        run, method, threshold_label = keys
        total_before = float(sub["pool_n_before"].sum())
        total_after = float(sub["pool_n_after"].sum())

        rows.append({
            "run": run,
            "method": method,
            "threshold_label": threshold_label,
            "n_queries": int(sub["query_id"].nunique()),
            "threshold_value_mean": float(sub["threshold_value"].mean()),
            "pool_n_before_mean": float(sub["pool_n_before"].mean()),
            "pool_n_before_median": float(sub["pool_n_before"].median()),
            "pool_n_after_mean": float(sub["pool_n_after"].mean()),
            "pool_n_after_median": float(sub["pool_n_after"].median()),
            "per_query_reduction_rate_mean": float(sub["reduction_rate"].mean()),
            "global_reduction_rate": float((total_before - total_after) / max(total_before, 1.0)),
            "true_retained_rate": float(sub["true_retained"].mean()),
            "true_top1_full_pool_rate": float(sub["true_top1_full_pool"].mean()),
            "true_top5_full_pool_rate": float(sub["true_top5_full_pool"].mean()),
            "true_top10_full_pool_rate": float(sub["true_top10_full_pool"].mean()),
            "true_top1_after_filter_rate": float(sub["true_top1_after_filter"].mean()),
            "true_top5_after_filter_rate": float(sub["true_top5_after_filter"].mean()),
            "true_top10_after_filter_rate": float(sub["true_top10_after_filter"].mean()),
        })

    return pd.DataFrame(rows)


def summarize_rank(rank_df):
    if rank_df.empty:
        return pd.DataFrame()

    rows = []
    for (run, method), sub in rank_df.groupby(["run", "method"]):
        rows.append({
            "run": run,
            "method": method,
            "n_queries": int(sub["query_id"].nunique()),
            "pool_n_mean": float(sub["pool_n"].mean()),
            "pool_n_median": float(sub["pool_n"].median()),
            "true_rank_mean": float(sub["true_rank_full_pool"].mean()),
            "true_rank_median": float(sub["true_rank_full_pool"].median()),
            "true_top1_full_pool_rate": float(sub["true_top1_full_pool"].mean()),
            "true_top5_full_pool_rate": float(sub["true_top5_full_pool"].mean()),
            "true_top10_full_pool_rate": float(sub["true_top10_full_pool"].mean()),
            "true_rt_distance_mean": float(sub["true_rt_distance"].mean()),
            "true_rt_distance_median": float(sub["true_rt_distance"].median()),
        })
    return pd.DataFrame(rows)


def mean_std_summary(df, group_cols, value_cols):
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        if "run" in sub.columns:
            row["num_runs"] = int(sub["run"].nunique())
        else:
            row["num_rows"] = int(len(sub))
        for c in value_cols:
            if c in sub.columns:
                vals = pd.to_numeric(sub[c], errors="coerce")
                row[f"{c}_mean"] = float(vals.mean())
                row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def formula_pool_overview(train_meta, test_meta, min_pool_size):
    all_meta = pd.concat([train_meta, test_meta], ignore_index=True)
    all_meta = all_meta[all_meta["formula"].notna()].copy()

    formula_counts = all_meta.groupby("formula").agg(
        total_pool_n=("candidate_id", "count"),
        train_pool_n=("split", lambda x: int((x == "train").sum())),
        test_pool_n=("split", lambda x: int((x == "test").sum())),
    ).reset_index()

    test_with_pool = test_meta.merge(formula_counts, on="formula", how="left")
    eligible = test_with_pool[test_with_pool["total_pool_n"] >= min_pool_size].copy()

    overview = pd.DataFrame([{
        "train_rows": int(len(train_meta)),
        "test_rows": int(len(test_meta)),
        "unique_formulas_all": int(all_meta["formula"].nunique()),
        "unique_formulas_train": int(train_meta["formula"].nunique()),
        "unique_formulas_test": int(test_meta["formula"].nunique()),
        "test_queries_total": int(len(test_meta)),
        "test_queries_with_formula_pool_ge_min": int(len(eligible)),
        "min_pool_size": int(min_pool_size),
        "eligible_query_ratio": float(len(eligible) / max(len(test_meta), 1)),
        "eligible_pool_n_mean": float(eligible["total_pool_n"].mean()) if len(eligible) else np.nan,
        "eligible_pool_n_median": float(eligible["total_pool_n"].median()) if len(eligible) else np.nan,
        "eligible_pool_n_max": int(eligible["total_pool_n"].max()) if len(eligible) else 0,
    }])

    pool_dist = test_with_pool.groupby("total_pool_n").agg(
        n_test_queries=("candidate_id", "count")
    ).reset_index().sort_values("total_pool_n")

    return overview, pool_dist, formula_counts, eligible


def parse_runs(s):
    if s is None or s.strip() == "":
        return DEFAULT_RUNS
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Custom runs must be formatted as name:path,name:path")
        name, path = item.split(":", 1)
        out.append((name.strip(), path.strip()))
    return out


def write_readme(out_dir):
    text = """SMRT formula-level candidate filtering simulation

This analysis constructs formula-level candidate pools from retained SMRT train+test molecules.

Important design choices:
- Test molecules are used as queries.
- Candidate pool = all retained SMRT train+test molecules sharing the same molecular formula.
- Train candidates use OOF predictions, avoiding in-sample training predictions.
- Test candidates use fold-averaged test predictions.
- Query experimental RT is used as the practical RT filtering target.
- Thresholds include fixed windows and OOF-MAE-based windows.
- This is a formula-level candidate filtering simulation, not an MS-FINDER candidate-list experiment.

Main outputs:
- smrt_formula_pool_overview.csv
- smrt_formula_pool_size_distribution.csv
- smrt_formula_counts.csv
- smrt_formula_candidate_filtering_per_query.csv
- smrt_formula_candidate_filtering_summary.csv
- smrt_formula_candidate_filtering_summary_5seed.csv
- smrt_formula_candidate_rank_summary.csv
- smrt_formula_candidate_rank_summary_5seed.csv
"""
    path = Path(out_dir) / "README_outputs.txt"
    path.write_text(text, encoding="utf-8")
    print(f"[SAVE] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="paper_analysis_63")
    ap.add_argument("--origin_train_csv", default="data/SMRT_train.csv")
    ap.add_argument("--origin_test_csv", default="data/SMRT_test.csv")
    ap.add_argument("--runs", default="")
    ap.add_argument("--min_pool_size", type=int, default=2)
    ap.add_argument("--huber_alpha", type=float, default=1e-4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    run_specs = parse_runs(args.runs)

    print("=== Stage 2A: SMRT formula-level candidate filtering simulation ===")
    print("out_dir:", out_dir)
    print("min_pool_size:", args.min_pool_size)
    print("runs:")
    for name, path in run_specs:
        print(" ", name, "->", path)

    print("\n=== Load SMRT retained data ===")
    train_meta = load_smrt_valid(args.origin_train_csv, split="train")
    test_meta = load_smrt_valid(args.origin_test_csv, split="test")

    print("train_meta:", train_meta.shape)
    print("test_meta:", test_meta.shape)

    overview, pool_dist, formula_counts, eligible = formula_pool_overview(
        train_meta, test_meta, min_pool_size=args.min_pool_size
    )

    save_csv(overview, out_dir / "smrt_formula_pool_overview.csv")
    save_csv(pool_dist, out_dir / "smrt_formula_pool_size_distribution.csv")
    save_csv(formula_counts, out_dir / "smrt_formula_counts.csv")
    save_csv(eligible, out_dir / "smrt_formula_eligible_test_queries.csv")

    all_per_query = []
    all_rank = []
    run_metric_rows = []

    for run_name, run_dir in run_specs:
        print(f"\n=== Run {run_name}: {run_dir} ===")
        all_pred, method_mae_oof, method_mae_test = load_run_predictions(
            run_name,
            run_dir,
            train_meta=train_meta,
            test_meta=test_meta,
            huber_alpha=args.huber_alpha,
        )

        for method in ["origin", "taut", "mean", "huber"]:
            run_metric_rows.append({
                "run": run_name,
                "method": method,
                "oof_mae_for_thresholds": method_mae_oof[method],
                "test_mae": method_mae_test[method],
            })

        per_query, rank_df = evaluate_formula_filtering_for_run(
            run_name,
            all_pred=all_pred,
            test_meta=test_meta,
            method_mae_oof=method_mae_oof,
            min_pool_size=args.min_pool_size,
        )

        all_per_query.append(per_query)
        all_rank.append(rank_df)

    run_metric_df = pd.DataFrame(run_metric_rows)
    save_csv(run_metric_df, out_dir / "smrt_formula_filtering_method_mae_by_run.csv")

    per_query_all = pd.concat(all_per_query, ignore_index=True)
    rank_all = pd.concat(all_rank, ignore_index=True)

    save_csv(per_query_all, out_dir / "smrt_formula_candidate_filtering_per_query.csv")
    save_csv(rank_all, out_dir / "smrt_formula_candidate_rank_per_query.csv")

    filtering_summary = summarize_filtering(per_query_all)
    save_csv(filtering_summary, out_dir / "smrt_formula_candidate_filtering_summary.csv")

    filtering_summary_5seed = mean_std_summary(
        filtering_summary,
        group_cols=["method", "threshold_label"],
        value_cols=[
            "threshold_value_mean",
            "n_queries",
            "pool_n_before_mean",
            "pool_n_after_mean",
            "per_query_reduction_rate_mean",
            "global_reduction_rate",
            "true_retained_rate",
            "true_top1_full_pool_rate",
            "true_top5_full_pool_rate",
            "true_top10_full_pool_rate",
            "true_top1_after_filter_rate",
            "true_top5_after_filter_rate",
            "true_top10_after_filter_rate",
        ],
    )
    save_csv(filtering_summary_5seed, out_dir / "smrt_formula_candidate_filtering_summary_5seed.csv")

    rank_summary = summarize_rank(rank_all)
    save_csv(rank_summary, out_dir / "smrt_formula_candidate_rank_summary.csv")

    rank_summary_5seed = mean_std_summary(
        rank_summary,
        group_cols=["method"],
        value_cols=[
            "n_queries",
            "pool_n_mean",
            "true_rank_mean",
            "true_rank_median",
            "true_top1_full_pool_rate",
            "true_top5_full_pool_rate",
            "true_top10_full_pool_rate",
            "true_rt_distance_mean",
            "true_rt_distance_median",
        ],
    )
    save_csv(rank_summary_5seed, out_dir / "smrt_formula_candidate_rank_summary_5seed.csv")

    write_readme(out_dir)

    print("\n=== Quick check: formula pool overview ===")
    print(overview.to_string(index=False))

    print("\n=== Quick check: method MAE used for thresholds ===")
    print(run_metric_df.to_string(index=False))

    print("\n=== Quick check: rank summary 5seed ===")
    print(rank_summary_5seed.to_string(index=False))

    print("\n=== Quick check: filtering summary 5seed, selected thresholds ===")
    sel = filtering_summary_5seed[
        filtering_summary_5seed["threshold_label"].isin(["fixed_50s", "fixed_100s", "2x_oof_mae", "3x_oof_mae"])
    ].copy()
    cols = [
        "method",
        "threshold_label",
        "threshold_value_mean_mean",
        "pool_n_before_mean_mean",
        "pool_n_after_mean_mean",
        "per_query_reduction_rate_mean_mean",
        "global_reduction_rate_mean",
        "true_retained_rate_mean",
        "true_top1_full_pool_rate_mean",
        "true_top5_full_pool_rate_mean",
        "true_top10_full_pool_rate_mean",
    ]
    cols = [c for c in cols if c in sel.columns]
    print(sel[cols].sort_values(["threshold_label", "method"]).to_string(index=False))

    print("\n✅ Done. All files are in:", out_dir)


if __name__ == "__main__":
    main()

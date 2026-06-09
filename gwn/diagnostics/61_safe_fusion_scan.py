import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings

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


def metrics(y, p):
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
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "gt80": int((e > 80).sum()),
        "gt100": int((e > 100).sum()),
        "gt150": int((e > 150).sum()),
        "gt200": int((e > 200).sum()),
        "bias": float(np.mean(p - y)),
    }


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


def fit_huber(oof, test, huber_alpha):
    y_oof = oof["Actual_RT"].values.astype(np.float64)

    o_oof = oof["Origin_OOF_Pred"].values.astype(np.float64)
    t_oof = oof["Taut_OOF_Pred"].values.astype(np.float64)
    c_oof = oof["Taut_Changed"].values.astype(np.float64)

    o_test = test["Origin_Test_Pred"].values.astype(np.float64)
    t_test = test["Taut_Test_Pred"].values.astype(np.float64)
    c_test = test["Taut_Changed"].values.astype(np.float64)

    x_oof = build_stack_features(o_oof, t_oof, c_oof)
    x_test = build_stack_features(o_test, t_test, c_test)

    model = make_pipeline(
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=huber_alpha, max_iter=1000),
    )
    model.fit(x_oof, y_oof)

    return model.predict(x_oof), model.predict(x_test)


def candidate_preds(oof, test, huber_alpha):
    y_oof = oof["Actual_RT"].values.astype(np.float64)
    y_test = test["Actual_RT"].values.astype(np.float64)

    o_oof = oof["Origin_OOF_Pred"].values.astype(np.float64)
    t_oof = oof["Taut_OOF_Pred"].values.astype(np.float64)

    o_test = test["Origin_Test_Pred"].values.astype(np.float64)
    t_test = test["Taut_Test_Pred"].values.astype(np.float64)

    mean_oof = 0.5 * (o_oof + t_oof)
    mean_test = 0.5 * (o_test + t_test)

    huber_oof, huber_test = fit_huber(oof, test, huber_alpha)

    diff_oof = np.abs(o_oof - t_oof)
    diff_test = np.abs(o_test - t_test)

    dev_oof = np.abs(huber_oof - mean_oof)
    dev_test = np.abs(huber_test - mean_test)

    lo_oof = np.minimum(o_oof, t_oof)
    hi_oof = np.maximum(o_oof, t_oof)
    lo_test = np.minimum(o_test, t_test)
    hi_test = np.maximum(o_test, t_test)

    candidates = []

    def add(name, p_oof, p_test, params):
        candidates.append({
            "method": name,
            "oof_pred": p_oof,
            "test_pred": p_test,
            "params": params,
            "oof_mae": metrics(y_oof, p_oof)["mae"],
        })

    add("origin_only", o_oof, o_test, {})
    add("taut_only", t_oof, t_test, {})
    add("mean_origin_taut", mean_oof, mean_test, {})
    add("huber_stack", huber_oof, huber_test, {})

    # 1. Huber 向 mean 收缩：lambda=1 是 huber，lambda=0 是 mean。
    for lam in np.linspace(0.0, 1.0, 51):
        p_oof = mean_oof + lam * (huber_oof - mean_oof)
        p_test = mean_test + lam * (huber_test - mean_test)
        add("shrink_huber_to_mean", p_oof, p_test, {"lambda": float(lam)})

    # 2. 固定 cap：限制 huber 偏离 mean 的最大幅度。
    for cap in [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 120, 200]:
        p_oof = mean_oof + np.clip(huber_oof - mean_oof, -cap, cap)
        p_test = mean_test + np.clip(huber_test - mean_test, -cap, cap)
        add("cap_huber_from_mean", p_oof, p_test, {"cap": float(cap)})

    # 3. 自适应 cap：允许分歧越大，cap 越大。
    for c0 in [0, 1, 2, 5, 10, 20]:
        for c1 in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            cap_oof = c0 + c1 * diff_oof
            cap_test = c0 + c1 * diff_test
            p_oof = mean_oof + np.clip(huber_oof - mean_oof, -cap_oof, cap_oof)
            p_test = mean_test + np.clip(huber_test - mean_test, -cap_test, cap_test)
            add("adaptive_cap_huber_from_mean", p_oof, p_test, {"c0": float(c0), "c1": float(c1)})

    # 4. origin/taut 分歧过大时，不信任 huber，回退到 mean。
    for tau in [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 120, 200]:
        p_oof = np.where(diff_oof > tau, mean_oof, huber_oof)
        p_test = np.where(diff_test > tau, mean_test, huber_test)
        add("fallback_mean_if_view_disagree", p_oof, p_test, {"tau": float(tau)})

    # 5. huber 偏离 mean 太大时，回退到 mean。
    for tau in [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 120, 200]:
        p_oof = np.where(dev_oof > tau, mean_oof, huber_oof)
        p_test = np.where(dev_test > tau, mean_test, huber_test)
        add("fallback_mean_if_huber_deviates", p_oof, p_test, {"tau": float(tau)})

    # 6. 把 huber 限制在 origin/taut 区间附近。
    for margin in [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 120]:
        p_oof = np.clip(huber_oof, lo_oof - margin, hi_oof + margin)
        p_test = np.clip(huber_test, lo_test - margin, hi_test + margin)
        add("clip_huber_to_view_range", p_oof, p_test, {"margin": float(margin)})

    # 只用 OOF MAE 选择最终 safe fusion。
    best = min(candidates, key=lambda x: x["oof_mae"])

    return candidates, best


def summarize(df, group_cols):
    metric_cols = [
        "mae", "mre", "medae", "medre", "rmse", "r2",
        "p95", "p99", "gt80", "gt100", "gt150", "gt200", "bias",
        "delta_mae_vs_huber",
        "delta_mae_vs_mean",
    ]

    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["num_runs"] = int(sub["run"].nunique()) if "run" in sub.columns else len(sub)
        for c in metric_cols:
            if c in sub.columns:
                vals = pd.to_numeric(sub[c], errors="coerce")
                row[f"{c}_mean"] = float(vals.mean())
                row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def parse_runs(s):
    if not s:
        return DEFAULT_RUNS
    out = []
    for item in s.split(","):
        name, path = item.split(":", 1)
        out.append((name.strip(), path.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="paper_analysis_safe_fusion")
    ap.add_argument("--huber_alpha", type=float, default=1e-4)
    ap.add_argument("--runs", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    run_specs = parse_runs(args.runs)

    rows_all = []
    rows_best = []

    print("=== Safe fusion scan ===")
    for run, path in run_specs:
        print(f"\n[RUN] {run} {path}")

        oof_path = Path(path) / "oof_predictions.csv"
        test_path = Path(path) / "test_predictions.csv"

        if not oof_path.exists():
            raise FileNotFoundError(oof_path)
        if not test_path.exists():
            raise FileNotFoundError(test_path)

        oof = pd.read_csv(oof_path)
        test = pd.read_csv(test_path)

        y_test = test["Actual_RT"].values.astype(np.float64)

        candidates, best = candidate_preds(oof, test, args.huber_alpha)

        # baseline references
        ref = {}
        for c in candidates:
            if c["method"] in ["huber_stack", "mean_origin_taut"]:
                ref[c["method"]] = metrics(y_test, c["test_pred"])["mae"]

        # 每类方法只保留 OOF 最优参数，避免表太大。
        best_by_method = {}
        for c in candidates:
            m = c["method"]
            if m not in best_by_method or c["oof_mae"] < best_by_method[m]["oof_mae"]:
                best_by_method[m] = c

        for method, c in best_by_method.items():
            mt = metrics(y_test, c["test_pred"])
            row = {
                "run": run,
                "method": method,
                "selected_by_oof": int(c is best),
                "oof_mae": float(c["oof_mae"]),
                "params_json": json.dumps(c["params"], ensure_ascii=False),
                **mt,
            }
            row["delta_mae_vs_huber"] = row["mae"] - ref["huber_stack"]
            row["delta_mae_vs_mean"] = row["mae"] - ref["mean_origin_taut"]
            rows_all.append(row)

        mt_best = metrics(y_test, best["test_pred"])
        best_row = {
            "run": run,
            "method": "safe_selected_by_oof",
            "base_method": best["method"],
            "oof_mae": float(best["oof_mae"]),
            "params_json": json.dumps(best["params"], ensure_ascii=False),
            **mt_best,
        }
        best_row["delta_mae_vs_huber"] = best_row["mae"] - ref["huber_stack"]
        best_row["delta_mae_vs_mean"] = best_row["mae"] - ref["mean_origin_taut"]
        rows_best.append(best_row)

        print(
            f"best={best['method']} params={best['params']} "
            f"test_mae={best_row['mae']:.6f} "
            f"delta_vs_huber={best_row['delta_mae_vs_huber']:.6f}"
        )

    df_all = pd.DataFrame(rows_all)
    df_best = pd.DataFrame(rows_best)

    df_all.to_csv(out_dir / "safe_fusion_scan_by_method_5seed.csv", index=False)
    df_best.to_csv(out_dir / "safe_fusion_selected_by_oof_5seed.csv", index=False)

    summary_all = summarize(df_all, ["method"])
    summary_best = summarize(df_best, ["method", "base_method"])

    summary_all.to_csv(out_dir / "safe_fusion_scan_by_method_5seed_summary.csv", index=False)
    summary_best.to_csv(out_dir / "safe_fusion_selected_by_oof_5seed_summary.csv", index=False)

    print("\n=== Method summary ===")
    show = [
        "method", "num_runs",
        "mae_mean", "mae_std",
        "medae_mean",
        "rmse_mean",
        "r2_mean",
        "p95_mean",
        "p99_mean",
        "gt100_mean",
        "gt200_mean",
        "delta_mae_vs_huber_mean",
        "delta_mae_vs_huber_std",
    ]
    show = [c for c in show if c in summary_all.columns]
    print(summary_all[show].sort_values("mae_mean").to_string(index=False))

    print("\n=== OOF-selected safe fusion summary ===")
    show = [
        "method", "base_method", "num_runs",
        "mae_mean", "mae_std",
        "medae_mean",
        "rmse_mean",
        "r2_mean",
        "gt100_mean",
        "gt200_mean",
        "delta_mae_vs_huber_mean",
        "delta_mae_vs_huber_std",
    ]
    show = [c for c in show if c in summary_best.columns]
    print(summary_best[show].to_string(index=False))

    print("\n[SAVE]", out_dir)


if __name__ == "__main__":
    main()

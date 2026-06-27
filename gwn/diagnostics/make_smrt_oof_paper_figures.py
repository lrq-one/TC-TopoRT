import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def full_metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = np.abs(y - p)

    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rel = e / (np.abs(y) + 1e-8) * 100.0

    return {
        "MAE": float(e.mean()),
        "MRE": float(rel.mean()),
        "MedAE": float(np.median(e)),
        "MedRE": float(np.median(rel)),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "R2": float(r2),
        "P95": float(np.percentile(e, 95)),
        "P99": float(np.percentile(e, 99)),
        ">100": int((e > 100).sum()),
        ">200": int((e > 200).sum()),
        "Bias": float(np.mean(p - y)),
    }


def save_table(df, path):
    df.to_csv(path, index=False)
    print("saved table:", path)


def plot_rt_distribution(train_csv, test_csv, out_path):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    train.columns = [str(c).lower().strip() for c in train.columns]
    test.columns = [str(c).lower().strip() for c in test.columns]

    train = train[train["rt"] > 300.0]
    test = test[test["rt"] > 300.0]

    plt.figure(figsize=(7.5, 5.0))
    plt.hist(train["rt"], bins=60, alpha=0.55, label=f"Train, n={len(train)}")
    plt.hist(test["rt"], bins=60, alpha=0.55, label=f"Test, n={len(test)}")
    plt.xlabel("Retention time (s)")
    plt.ylabel("Count")
    plt.title("RT distribution of SMRT train/test sets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("saved:", out_path)


def plot_actual_vs_pred(test, out_path):
    y = test["Actual_RT"].values
    p = test["Final_Pred"].values
    e = np.abs(y - p)

    m = full_metrics(y, p)

    plt.figure(figsize=(6.2, 6.0))
    sc = plt.scatter(y, p, c=e, s=12, alpha=0.65)
    lo = min(y.min(), p.min())
    hi = max(y.max(), p.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
    plt.xlabel("Experimental RT (s)")
    plt.ylabel("Predicted RT (s)")
    plt.title(f"OOF dual-view stack on SMRT test\nMAE={m['MAE']:.2f}s, R²={m['R2']:.3f}")
    cb = plt.colorbar(sc)
    cb.set_label("Absolute error (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("saved:", out_path)


def plot_error_hist_violin(test, out_prefix):
    y = test["Actual_RT"].values
    p = test["Final_Pred"].values
    err = p - y
    abs_err = np.abs(err)

    plt.figure(figsize=(7.0, 4.8))
    plt.hist(err, bins=80, alpha=0.8)
    plt.axvline(err.mean(), linestyle="--", linewidth=1.2, label=f"Bias={err.mean():.2f}s")
    plt.xlabel("Prediction error, Predicted - Experimental (s)")
    plt.ylabel("Count")
    plt.title("Error distribution on SMRT test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_prefix) + "_error_hist.png", dpi=300)
    plt.close()
    print("saved:", str(out_prefix) + "_error_hist.png")

    plt.figure(figsize=(4.2, 5.2))
    plt.violinplot(abs_err, showmeans=True, showmedians=True)
    plt.ylabel("Absolute error (s)")
    plt.xticks([1], ["OOF dual-view stack"])
    plt.title("Absolute error distribution")
    plt.tight_layout()
    plt.savefig(str(out_prefix) + "_abs_error_violin.png", dpi=300)
    plt.close()
    print("saved:", str(out_prefix) + "_abs_error_violin.png")


def plot_ablation_bar(test, out_path):
    y = test["Actual_RT"].values
    origin = test["Origin_Test_Pred"].values
    taut = test["Taut_Test_Pred"].values
    mean = 0.5 * (origin + taut)
    final = test["Final_Pred"].values

    methods = [
        ("Origin 5-fold", origin),
        ("Taut 5-fold", taut),
        ("Mean fusion", mean),
        ("OOF Huber stack", final),
    ]

    rows = []
    for name, pred in methods:
        m = full_metrics(y, pred)
        rows.append({"Method": name, **m})

    df = pd.DataFrame(rows)

    plt.figure(figsize=(7.2, 4.8))
    plt.bar(df["Method"], df["MAE"])
    plt.ylabel("MAE (s)")
    plt.title("Dual-view ablation on SMRT test")
    plt.xticks(rotation=25, ha="right")
    for i, v in enumerate(df["MAE"].values):
        plt.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("saved:", out_path)

    return df


def plot_group_by_taut_changed(test, out_path):
    if "Taut_Changed" not in test.columns:
        print("skip taut_changed plot: no Taut_Changed column")
        return None

    rows = []
    for g in [0, 1]:
        sub = test[test["Taut_Changed"] == g]
        if len(sub) == 0:
            continue

        y = sub["Actual_RT"].values
        candidates = {
            "Origin": sub["Origin_Test_Pred"].values,
            "Taut": sub["Taut_Test_Pred"].values,
            "Final": sub["Final_Pred"].values,
        }

        for name, p in candidates.items():
            m = full_metrics(y, p)
            rows.append({
                "Taut_Changed": int(g),
                "Method": name,
                "N": len(sub),
                **m,
            })

    df = pd.DataFrame(rows)

    plt.figure(figsize=(7.4, 4.8))
    labels = []
    values = []
    for g in [0, 1]:
        for method in ["Origin", "Taut", "Final"]:
            row = df[(df["Taut_Changed"] == g) & (df["Method"] == method)]
            if len(row):
                labels.append(f"Changed={g}\n{method}")
                values.append(float(row["MAE"].iloc[0]))

    plt.bar(labels, values)
    plt.ylabel("MAE (s)")
    plt.title("Performance grouped by tautomer change")
    plt.xticks(rotation=0)
    for i, v in enumerate(values):
        plt.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("saved:", out_path)

    return df


def plot_learning_curves(result_dir, out_path):
    logs = []
    fold_root = Path(result_dir) / "folds"

    for fold_dir in sorted(fold_root.glob("fold_*")):
        for view in ["origin", "taut"]:
            log_path = fold_dir / view / "train_log.jsonl"
            if not log_path.exists():
                continue
            df = pd.read_json(log_path, lines=True)
            df["fold_view"] = f"{fold_dir.name}_{view}"
            logs.append(df)

    if not logs:
        print("skip learning curve: no train_log.jsonl found")
        return None

    log = pd.concat(logs, ignore_index=True)

    plt.figure(figsize=(8.0, 5.0))
    for name, sub in log.groupby("fold_view"):
        plt.plot(sub["epoch"], sub["val_mae"], alpha=0.35, linewidth=1.0)

    mean_val = log.groupby("epoch")["val_mae"].mean().reset_index()
    plt.plot(mean_val["epoch"], mean_val["val_mae"], linewidth=2.5, label="Mean validation MAE")

    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE (s)")
    plt.title("Learning curves across 5-fold origin/taut models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("saved:", out_path)

    return log


def make_main_comparison_table(test):
    y = test["Actual_RT"].values

    rows = [
        {
            "Method": "GNN-RT",
            "MAE": 39.87,
            "MRE": 5.00,
            "MedAE": 25.24,
            "MedRE": np.nan,
            "RMSE": np.nan,
            "R2": 0.850,
            "Source": "ABCoRT paper Table 1",
        },
        {
            "Method": "1D-CNN",
            "MAE": 34.70,
            "MRE": 4.30,
            "MedAE": 18.70,
            "MedRE": 2.40,
            "RMSE": np.nan,
            "R2": np.nan,
            "Source": "ABCoRT paper Table 1",
        },
        {
            "Method": "MPNN",
            "MAE": 31.50,
            "MRE": 4.00,
            "MedAE": 16.00,
            "MedRE": np.nan,
            "RMSE": np.nan,
            "R2": 0.879,
            "Source": "ABCoRT paper Table 1",
        },
        {
            "Method": "GIN",
            "MAE": 32.71,
            "MRE": np.nan,
            "MedAE": 19.06,
            "MedRE": np.nan,
            "RMSE": np.nan,
            "R2": 0.880,
            "Source": "ABCoRT paper Table 1",
        },
        {
            "Method": "RT-Transformer",
            "MAE": 27.30,
            "MRE": 3.42,
            "MedAE": 12.46,
            "MedRE": 1.58,
            "RMSE": np.nan,
            "R2": 0.880,
            "Source": "ABCoRT paper Table 1",
        },
        {
            "Method": "DeepGCN-RT",
            "MAE": 26.55,
            "MRE": np.nan,
            "MedAE": 12.38,
            "MedRE": np.nan,
            "RMSE": np.nan,
            "R2": 0.890,
            "Source": "ABCoRT paper Table 1",
        },
        {
            "Method": "ABCoRT",
            "MAE": 25.75,
            "MRE": 3.24,
            "MedAE": 11.78,
            "MedRE": 1.50,
            "RMSE": np.nan,
            "R2": 0.895,
            "Source": "ABCoRT paper Table 1",
        },
    ]

    ours = {
        "Origin 5-fold": test["Origin_Test_Pred"].values,
        "Taut 5-fold": test["Taut_Test_Pred"].values,
        "Mean fusion": 0.5 * (test["Origin_Test_Pred"].values + test["Taut_Test_Pred"].values),
        "OOF Huber stack": test["Final_Pred"].values,
    }

    for name, pred in ours.items():
        m = full_metrics(y, pred)
        rows.append({
            "Method": name,
            "MAE": m["MAE"],
            "MRE": m["MRE"],
            "MedAE": m["MedAE"],
            "MedRE": m["MedRE"],
            "RMSE": m["RMSE"],
            "R2": m["R2"],
            "Source": "Ours seed=1",
        })

    return pd.DataFrame(rows)


def plot_main_comparison_bar(df, out_path):
    plot_df = df.dropna(subset=["MAE"]).copy()

    plt.figure(figsize=(9.5, 5.0))
    plt.bar(plot_df["Method"], plot_df["MAE"])
    plt.ylabel("MAE (s)")
    plt.title("SMRT test MAE comparison")
    plt.xticks(rotation=35, ha="right")
    for i, v in enumerate(plot_df["MAE"].values):
        plt.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default="results_OOF_DualView_Stack_v1")
    ap.add_argument("--train_csv", default="data/SMRT_train.csv")
    ap.add_argument("--test_csv", default="data/SMRT_test.csv")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir) if args.out_dir else result_dir / "paper_figures"
    ensure_dir(out_dir)

    test_path = result_dir / "test_predictions.csv"
    oof_path = result_dir / "oof_predictions.csv"

    if not test_path.exists():
        raise FileNotFoundError(test_path)
    if not oof_path.exists():
        print("warning: oof_predictions.csv not found, some plots may be skipped")

    test = pd.read_csv(test_path)

    # tables
    main_table = make_main_comparison_table(test)
    save_table(main_table, out_dir / "table_main_comparison.csv")

    ablation_df = plot_ablation_bar(test, out_dir / "fig_dualview_ablation_mae.png")
    save_table(ablation_df, out_dir / "table_dualview_ablation.csv")

    group_df = plot_group_by_taut_changed(test, out_dir / "fig_taut_changed_group_mae.png")
    if group_df is not None:
        save_table(group_df, out_dir / "table_taut_changed_group.csv")

    # figures
    plot_main_comparison_bar(main_table, out_dir / "fig_main_comparison_mae.png")
    plot_rt_distribution(args.train_csv, args.test_csv, out_dir / "fig_rt_distribution_train_test.png")
    plot_actual_vs_pred(test, out_dir / "fig_actual_vs_pred_final.png")
    plot_error_hist_violin(test, out_dir / "fig_final")
    log_df = plot_learning_curves(result_dir, out_dir / "fig_learning_curves_val_mae.png")
    if log_df is not None:
        log_df.to_csv(out_dir / "table_learning_curves_raw.csv", index=False)

    print("\n✅ all figures saved to:", out_dir)


if __name__ == "__main__":
    main()

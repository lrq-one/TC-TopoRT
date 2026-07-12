import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


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


def resolve_repo_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the core SMRT evaluation figures from a public "
            "TC-TopoRT training output."
        )
    )
    parser.add_argument(
        "--result_dir",
        default="artifacts/results/smrt/seed5",
        help=(
            "Directory containing test_predictions.csv. "
            "Default: artifacts/results/smrt/seed5"
        ),
    )
    parser.add_argument(
        "--train_csv",
        default="gwn/data/SMRT_train.csv",
    )
    parser.add_argument(
        "--test_csv",
        default="gwn/data/SMRT_test.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/figures/smrt",
    )
    args = parser.parse_args()

    result_dir = resolve_repo_path(args.result_dir)
    train_csv = resolve_repo_path(args.train_csv)
    test_csv = resolve_repo_path(args.test_csv)
    out_dir = resolve_repo_path(args.out_dir)
    ensure_dir(out_dir)

    test_path = result_dir / "test_predictions.csv"
    if not test_path.is_file():
        raise FileNotFoundError(
            "Public SMRT test predictions were not found. Run the "
            "single-seed training entry first or pass --result_dir: "
            f"{test_path}"
        )

    test = pd.read_csv(test_path)

    required = {
        "Actual_RT",
        "Origin_Test_Pred",
        "Taut_Test_Pred",
        "Final_Pred",
    }
    missing = sorted(required - set(test.columns))
    if missing:
        raise RuntimeError(
            f"Missing required columns in {test_path}: {missing}"
        )

    metrics = full_metrics(
        test["Actual_RT"].to_numpy(),
        test["Final_Pred"].to_numpy(),
    )
    metrics_table = pd.DataFrame(
        [
            {
                "result_dir": str(result_dir.relative_to(ROOT)),
                "n": len(test),
                **metrics,
            }
        ]
    )
    save_table(
        metrics_table,
        out_dir / "table_smrt_metrics.csv",
    )

    plot_actual_vs_pred(
        test,
        out_dir / "fig_actual_vs_pred_final.png",
    )
    plot_error_hist_violin(
        test,
        out_dir / "fig_final",
    )

    if train_csv.is_file() and test_csv.is_file():
        plot_rt_distribution(
            train_csv,
            test_csv,
            out_dir / "fig_rt_distribution_train_test.png",
        )
    else:
        print(
            "warning: SMRT train/test CSV files were not found; "
            "RT-distribution figure skipped."
        )

    print()
    print("[INPUT]", test_path.relative_to(ROOT))
    print("[OUTPUT]", out_dir.relative_to(ROOT))
    print("SMRT core figure generation completed.")


if __name__ == "__main__":
    main()

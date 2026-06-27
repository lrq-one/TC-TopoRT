from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("experiments_transfer_effectiveness")
OUT = ROOT / "external_transfer_vs_scratch_effectiveness"
OUT.mkdir(parents=True, exist_ok=True)

PAPER = Path("final_paper_tables")
PAPER.mkdir(exist_ok=True)

TL_DIR = ROOT / "results_figure4_tl_seed1_src0"
SCRATCH_DIR = ROOT / "results_figure4_scratch_seed1"

tl_csv = TL_DIR / "external_tl_metrics_by_run.csv"
scratch_csv = SCRATCH_DIR / "external_tl_metrics_by_run.csv"

print("[READ TL]", tl_csv, "exists=", tl_csv.exists())
print("[READ Scratch]", scratch_csv, "exists=", scratch_csv.exists())

if not tl_csv.exists():
    raise FileNotFoundError(tl_csv)
if not scratch_csv.exists():
    raise FileNotFoundError(scratch_csv)

def load_mean_result(path, mode):
    df = pd.read_csv(path)
    df = df[df["method"].astype(str).eq("mean_tl")].copy()
    if len(df) == 0:
        raise RuntimeError(f"No mean_tl rows in {path}")

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "mode": mode,
            "dataset_name": r["dataset_name"],
            "n": int(r["n"]),
            "mae": float(r["mae"]),
            "medae": float(r["medae"]),
            "rmse": float(r["rmse"]),
            "r2": float(r["r2"]),
            "pearson": float(r.get("pearson", np.nan)),
            "spearman": float(r.get("spearman", np.nan)),
            "bias": float(r["bias"]),
        })
    return pd.DataFrame(rows)

tl = load_mean_result(tl_csv, "TL_pretrained")
scratch = load_mean_result(scratch_csv, "Scratch_random_init")

all_rows = pd.concat([tl, scratch], ignore_index=True)
all_rows.to_csv(OUT / "figure4_tl_vs_scratch_by_mode.csv", index=False)

wide = all_rows.pivot_table(
    index=["dataset_name", "n"],
    columns="mode",
    values=["mae", "medae", "rmse", "r2", "pearson", "spearman", "bias"],
    aggfunc="mean",
)
wide.columns = [f"{metric}_{mode}" for metric, mode in wide.columns]
wide = wide.reset_index()

required = [
    "mae_TL_pretrained",
    "mae_Scratch_random_init",
    "medae_TL_pretrained",
    "medae_Scratch_random_init",
    "r2_TL_pretrained",
    "r2_Scratch_random_init",
]

missing = [c for c in required if c not in wide.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

wide["MAE_improvement_s"] = wide["mae_Scratch_random_init"] - wide["mae_TL_pretrained"]
wide["MedAE_improvement_s"] = wide["medae_Scratch_random_init"] - wide["medae_TL_pretrained"]
wide["RMSE_improvement_s"] = wide["rmse_Scratch_random_init"] - wide["rmse_TL_pretrained"]
wide["R2_improvement_abs"] = wide["r2_TL_pretrained"] - wide["r2_Scratch_random_init"]

wide["TL_better_MAE"] = wide["MAE_improvement_s"] > 0
wide["TL_better_MedAE"] = wide["MedAE_improvement_s"] > 0
wide["TL_better_R2"] = wide["R2_improvement_abs"] > 0

wide = wide.sort_values("MAE_improvement_s", ascending=False)

wide.to_csv(OUT / "external_transfer_vs_scratch_effectiveness_summary.csv", index=False)
wide.to_csv(PAPER / "Table_8_transfer_learning_effectiveness.csv", index=False)

def bar_plot(df, col, ylabel, filename):
    plot_df = df.sort_values(col)
    plt.figure(figsize=(12, 4.5))
    plt.bar(plot_df["dataset_name"], plot_df[col])
    plt.axhline(0, linewidth=1)
    plt.axhline(plot_df[col].mean(), linestyle="--", linewidth=1)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=300)
    plt.close()

bar_plot(wide, "MAE_improvement_s", "MAE improvement: scratch - TL (s)", "Figure4A_MAE_improvement.png")
bar_plot(wide, "MedAE_improvement_s", "MedAE improvement: scratch - TL (s)", "Figure4B_MedAE_improvement.png")
bar_plot(wide, "R2_improvement_abs", "R² improvement: TL - scratch", "Figure4C_R2_improvement.png")

summary = {
    "num_datasets": int(len(wide)),
    "TL_better_MAE_count": int(wide["TL_better_MAE"].sum()),
    "TL_better_MedAE_count": int(wide["TL_better_MedAE"].sum()),
    "TL_better_R2_count": int(wide["TL_better_R2"].sum()),
    "MAE_improvement_mean": float(wide["MAE_improvement_s"].mean()),
    "MAE_improvement_median": float(wide["MAE_improvement_s"].median()),
    "MedAE_improvement_mean": float(wide["MedAE_improvement_s"].mean()),
    "R2_improvement_abs_mean": float(wide["R2_improvement_abs"].mean()),
}

pd.DataFrame([summary]).to_csv(OUT / "external_transfer_vs_scratch_effectiveness_overall_summary.csv", index=False)

md = []
md.append("# Transfer-learning effectiveness summary\n\n")
md.append(f"- Number of external datasets: {summary['num_datasets']}\n")
md.append(f"- TL better than scratch by MAE: {summary['TL_better_MAE_count']} / {summary['num_datasets']}\n")
md.append(f"- TL better than scratch by MedAE: {summary['TL_better_MedAE_count']} / {summary['num_datasets']}\n")
md.append(f"- TL better than scratch by R2: {summary['TL_better_R2_count']} / {summary['num_datasets']}\n")
md.append(f"- Mean MAE improvement, scratch - TL: {summary['MAE_improvement_mean']:.6f} s\n")
md.append(f"- Median MAE improvement, scratch - TL: {summary['MAE_improvement_median']:.6f} s\n")
md.append(f"- Mean MedAE improvement, scratch - TL: {summary['MedAE_improvement_mean']:.6f} s\n")
md.append(f"- Mean R2 improvement, TL - scratch: {summary['R2_improvement_abs_mean']:.6f}\n\n")

if summary["MAE_improvement_mean"] > 0:
    md.append("Interpretation: under this controlled protocol, SMRT-pretrained transfer learning improves the average MAE over random initialization.\n")
else:
    md.append("Interpretation: under this controlled protocol, random initialization is competitive or better on average; the transfer-learning effect is dataset-dependent and should not be overclaimed.\n")

(OUT / "external_transfer_vs_scratch_effectiveness_interpretation.md").write_text("".join(md), encoding="utf-8")

print("\n=== TL vs Scratch summary ===")
print(wide[[
    "dataset_name", "n",
    "mae_TL_pretrained", "mae_Scratch_random_init", "MAE_improvement_s",
    "medae_TL_pretrained", "medae_Scratch_random_init", "MedAE_improvement_s",
    "r2_TL_pretrained", "r2_Scratch_random_init", "R2_improvement_abs",
    "TL_better_MAE"
]].to_string(index=False))

print("\n=== Overall summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

print("\n[SAVE]", OUT / "figure4_tl_vs_scratch_by_mode.csv")
print("[SAVE]", OUT / "external_transfer_vs_scratch_effectiveness_summary.csv")
print("[SAVE]", OUT / "external_transfer_vs_scratch_effectiveness_overall_summary.csv")
print("[SAVE]", OUT / "external_transfer_vs_scratch_effectiveness_interpretation.md")
print("[SAVE]", PAPER / "Table_8_transfer_learning_effectiveness.csv")
print("[SAVE]", OUT / "Figure4A_MAE_improvement.png")
print("[SAVE]", OUT / "Figure4B_MedAE_improvement.png")
print("[SAVE]", OUT / "Figure4C_R2_improvement.png")

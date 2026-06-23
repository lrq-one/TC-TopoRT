#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path.cwd()
OUT = ROOT / "manuscript_figures_abcort_style"
OUT.mkdir(parents=True, exist_ok=True)

# Detect paths after you renamed the project root.
GWN = ROOT / "gwn"
if not GWN.exists():
    GWN = ROOT

ABL = ROOT / "ablations" / "gwn_cwn_structural_ablation"

SEED_DIRS = {
    "seed1-v1": GWN / "results_OOF_DualView_Stack_v1",
    "seed5": GWN / "results_OOF_DualView_Stack_seed5",
    "seed79": GWN / "results_OOF_DualView_Stack_seed79",
    "seed123": GWN / "results_OOF_DualView_Stack_seed123",
    "seed256": GWN / "results_OOF_DualView_Stack_seed256",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def savefig(name):
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    plt.tight_layout()
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()
    print("[SAVE]", pdf)
    print("[SAVE]", png)


def calc_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    abs_err = np.abs(err)
    mae = abs_err.mean()
    medae = np.median(abs_err)
    rmse = np.sqrt(np.mean(err ** 2))
    mre = np.mean(abs_err / np.maximum(np.abs(y), 1e-12)) * 100.0
    medre = np.median(abs_err / np.maximum(np.abs(y), 1e-12)) * 100.0
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return {
        "MAE": mae,
        "MRE": mre,
        "MedAE": medae,
        "MedRE": medre,
        "R2": r2,
    }


def read_seed_predictions():
    rows = []
    seed_frames = {}

    for seed, d in SEED_DIRS.items():
        p = d / "test_predictions.csv"
        if not p.exists():
            print("[MISS seed pred]", p)
            continue

        df = pd.read_csv(p)
        required = ["Actual_RT", "Origin_Test_Pred", "Taut_Test_Pred", "Final_Pred"]
        miss = [c for c in required if c not in df.columns]
        if miss:
            print("[SKIP]", p, "missing", miss)
            continue

        df = df.copy()
        df["Mean_Fusion_Pred"] = (df["Origin_Test_Pred"] + df["Taut_Test_Pred"]) / 2.0
        seed_frames[seed] = df

        variants = {
            "Original": "Origin_Test_Pred",
            "Tautomer": "Taut_Test_Pred",
            "Mean fusion": "Mean_Fusion_Pred",
            "OOF stack": "Final_Pred",
        }

        for v, col in variants.items():
            m = calc_metrics(df["Actual_RT"], df[col])
            rows.append({"seed": seed, "variant": v, **m})

    met = pd.DataFrame(rows)
    met.to_csv(OUT / "source_per_seed_variant_metrics.csv", index=False)

    return seed_frames, met


def aggregate_variant_metrics(met):
    out = []
    for v, sub in met.groupby("variant"):
        row = {"variant": v}
        for m in ["MAE", "MRE", "MedAE", "MedRE", "R2"]:
            row[m] = sub[m].mean()
            row[m + "_std"] = sub[m].std(ddof=1)
        out.append(row)
    agg = pd.DataFrame(out)

    order = ["Original", "Tautomer", "Mean fusion", "OOF stack"]
    agg["order"] = agg["variant"].map({v: i for i, v in enumerate(order)})
    agg = agg.sort_values("order").drop(columns=["order"])
    agg.to_csv(OUT / "source_dualview_fusion_ablation_mean_std.csv", index=False)
    return agg


def make_ensemble_metrics(seed_frames):
    # Make sure molecule order is aligned by Source_Index if available.
    frames = []
    for seed, df in seed_frames.items():
        tmp = df[["Actual_RT", "Final_Pred"]].copy()
        if "Source_Index" in df.columns:
            tmp["Source_Index"] = df["Source_Index"]
            tmp = tmp.sort_values("Source_Index")
        tmp = tmp.reset_index(drop=True)
        tmp = tmp.rename(columns={"Final_Pred": f"Final_Pred_{seed}"})
        frames.append(tmp)

    if not frames:
        return None

    base = frames[0][["Actual_RT"]].copy()
    pred_cols = []
    for tmp in frames:
        col = [c for c in tmp.columns if c.startswith("Final_Pred_")][0]
        base[col] = tmp[col].values
        pred_cols.append(col)

    base["TCDV-E"] = base[pred_cols].mean(axis=1)
    m = calc_metrics(base["Actual_RT"], base["TCDV-E"])
    pd.DataFrame([m]).to_csv(OUT / "source_tcdv_ensemble_metrics.csv", index=False)
    return m


def read_structural_metrics():
    rows = []

    # Full seed5
    full_json = GWN / "results_OOF_DualView_Stack_seed5" / "final_metrics.json"
    if full_json.exists():
        m = json.loads(full_json.read_text())
        tm = m["test_final"]
        rows.append({
            "variant": "Full",
            "MAE": tm["mae"],
            "MRE": tm.get("mre", np.nan),
            "MedAE": tm["medae"],
            "MedRE": tm.get("medre", np.nan),
            "R2": tm["r2"],
        })
    else:
        print("[MISS]", full_json)

    # No explicit ring 2-cells
    no2_json = ABL / "results_Ablation_No2Cell_DualView_Stack_seed5" / "final_metrics.json"
    if no2_json.exists():
        m = json.loads(no2_json.read_text())
        tm = m["test_final"]
        rows.append({
            "variant": "w/o ring 2-cells",
            "MAE": tm["mae"],
            "MRE": tm.get("mre", np.nan),
            "MedAE": tm["medae"],
            "MedRE": tm.get("medre", np.nan),
            "R2": tm["r2"],
        })
    else:
        print("[MISS]", no2_json)

    # No CWN message passing
    cwn0_json = ABL / "results_Ablation_CWN0_DualView_Stack_seed5" / "final_metrics.json"
    if cwn0_json.exists():
        m = json.loads(cwn0_json.read_text())
        tm = m["test_final"]
        rows.append({
            "variant": "w/o CWN",
            "MAE": tm["mae"],
            "MRE": tm.get("mre", np.nan),
            "MedAE": tm["medae"],
            "MedRE": tm.get("medre", np.nan),
            "R2": tm["r2"],
        })
    else:
        print("[MISS]", cwn0_json)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "source_structural_ablation_metrics.csv", index=False)
    return df


def normalize_for_radar(df, metrics, lower_better):
    """
    Normalize each metric to [0, 1], higher is better.
    This is how we can put MAE/MRE/MedAE/MedRE/R2 on one radar chart.
    """
    score = df.copy()
    for m in metrics:
        vals = df[m].astype(float).values
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)

        if abs(vmax - vmin) < 1e-12:
            score[m] = 1.0
            continue

        if m in lower_better:
            # lower raw value -> higher score
            score[m] = (vmax - vals) / (vmax - vmin)
        else:
            # higher raw value -> higher score
            score[m] = (vals - vmin) / (vmax - vmin)

        # keep a visible lower bound so the smallest polygon does not collapse completely
        score[m] = 0.20 + 0.80 * score[m]

    return score


def plot_radar(df, name_col, metrics, labels, title, out_name, legend_anchor=(1.35, 1.10)):
    lower_better = {"MAE", "MRE", "MedAE", "MedRE"}
    score = normalize_for_radar(df, metrics, lower_better)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6.2, 5.3))
    ax = plt.subplot(111, polar=True)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.0, 1.03)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(True, linewidth=0.7, alpha=0.6)

    for _, row in score.iterrows():
        vals = [row[m] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.8, label=row[name_col])
        ax.fill(angles, vals, alpha=0.16)

    ax.set_title(title, y=1.08, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=legend_anchor, frameon=False, fontsize=9)

    # Save normalized scores for transparency
    score.to_csv(OUT / f"source_{out_name}_normalized_scores.csv", index=False)

    savefig(out_name)


def plot_radar_on_axis(ax, df, name_col, metrics, labels, title, legend=True):
    lower_better = {"MAE", "MRE", "MedAE", "MedRE"}
    score = normalize_for_radar(df, metrics, lower_better)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.03)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.grid(True, linewidth=0.7, alpha=0.6)
    ax.set_title(title, y=1.08, fontsize=11)

    for _, row in score.iterrows():
        vals = [row[m] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.6, label=row[name_col])
        ax.fill(angles, vals, alpha=0.13)

    if legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), frameon=False, fontsize=8)


def make_two_panel_ablation_radar(dual_df, struct_df):
    metrics = ["MAE", "MRE", "MedAE", "MedRE", "R2"]
    labels = ["MAE", "MRE", "MedAE", "MedRE", r"$R^2$"]

    fig = plt.figure(figsize=(11.5, 5.4))
    ax1 = fig.add_subplot(121, polar=True)
    ax2 = fig.add_subplot(122, polar=True)

    plot_radar_on_axis(
        ax1, dual_df, "variant", metrics, labels,
        "(A) Dual-view/fusion ablation",
        legend=True
    )
    plot_radar_on_axis(
        ax2, struct_df, "variant", metrics, labels,
        "(B) Structural ablation",
        legend=True
    )

    fig.suptitle("Ablation radar analysis of TCDV-TopoRT", y=1.04, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT / "fig_radar_ablation_two_panel.pdf", bbox_inches="tight")
    plt.savefig(OUT / "fig_radar_ablation_two_panel.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[SAVE]", OUT / "fig_radar_ablation_two_panel.pdf")
    print("[SAVE]", OUT / "fig_radar_ablation_two_panel.png")


def make_smrt_tcdv_vs_abcort_radar(dual_agg, ensemble_metrics):
    # ABCoRT values from the benchmark table; TCDV-S from OOF stack mean; TCDV-E from ensemble.
    stack = dual_agg[dual_agg["variant"] == "OOF stack"].iloc[0]

    rows = [
        {
            "variant": "ABCoRT",
            "MAE": 25.75,
            "MRE": 3.24,
            "MedAE": 11.78,
            "MedRE": 1.50,
            "R2": 0.895,
        },
        {
            "variant": "TCDV-S",
            "MAE": stack["MAE"],
            "MRE": stack["MRE"],
            "MedAE": stack["MedAE"],
            "MedRE": stack["MedRE"],
            "R2": stack["R2"],
        },
    ]

    if ensemble_metrics is not None:
        rows.append({
            "variant": "TCDV-E",
            "MAE": ensemble_metrics["MAE"],
            "MRE": ensemble_metrics["MRE"],
            "MedAE": ensemble_metrics["MedAE"],
            "MedRE": ensemble_metrics["MedRE"],
            "R2": ensemble_metrics["R2"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "source_smrt_tcdv_vs_abcort_radar_metrics.csv", index=False)

    plot_radar(
        df,
        "variant",
        ["MAE", "MRE", "MedAE", "MedRE", "R2"],
        ["MAE", "MRE", "MedAE", "MedRE", r"$R^2$"],
        "SMRT performance radar",
        "fig_radar_smrt_tcdv_vs_abcort",
        legend_anchor=(1.28, 1.10)
    )


# ============================================================
# Candidate filtering summary plot
# ============================================================

def make_candidate_summary_plot():
    rows = [
        ["MetaboBase", "ABCoRT-TL", 38.35, 51.11, 73.33, 82.22],
        ["MetaboBase", "TCDV-TopoRT", 69.14, 55.56, 82.22, 88.89],
        ["RIKEN-PlaSMA", "ABCoRT-TL", 28.46, 52.94, 76.47, 83.53],
        ["RIKEN-PlaSMA", "TCDV-TopoRT", 46.23, 54.12, 77.65, 89.41],
    ]
    df = pd.DataFrame(rows, columns=["dataset", "method", "Reduction", "Top-1", "Top-5", "Top-10"])
    df.to_csv(OUT / "source_candidate_filtering_summary.csv", index=False)

    metrics = ["Reduction", "Top-1", "Top-5", "Top-10"]
    datasets = ["MetaboBase", "RIKEN-PlaSMA"]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        abc = sub[sub["method"] == "ABCoRT-TL"].iloc[0]
        ours = sub[sub["method"] == "TCDV-TopoRT"].iloc[0]

        x = np.arange(len(metrics))
        w = 0.35

        ax.bar(x - w/2, [abc[m] for m in metrics], width=w,
               label="ABCoRT-TL", edgecolor="black", linewidth=0.5, alpha=0.65)
        ax.bar(x + w/2, [ours[m] for m in metrics], width=w,
               label="TCDV-TopoRT", edgecolor="black", linewidth=0.5, alpha=0.85)

        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)")

        for i, m in enumerate(metrics):
            ax.text(i - w/2, abc[m] + 1.2, f"{abc[m]:.1f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + w/2, ours[m] + 1.2, f"{ours[m]:.1f}", ha="center", va="bottom", fontsize=7)

    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("RT-aware candidate filtering and reranking", y=1.03)
    plt.tight_layout()
    savefig("fig_candidate_filtering_summary")


# ============================================================
# Formula-level candidate long horizontal bar
# Requires per-formula data: formula, total_candidates, retained_candidates
# The script tries to find it automatically.
# ============================================================

def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())


def try_read_csv_any_encoding(path):
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin1", "utf-16", "utf-16-le", "utf-16-be"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return None


def detect_formula_count_table(path):
    df = try_read_csv_any_encoding(path)
    if df is None or df.shape[0] < 5:
        return None

    cols = list(df.columns)
    nmap = {c: norm_col(c) for c in cols}

    formula_cols = []
    total_cols = []
    retained_cols = []
    dataset_cols = []

    for c, n in nmap.items():
        if any(x in n for x in ["formula", "molecularformula", "queryformula"]):
            formula_cols.append(c)

        if (
            ("total" in n and "candidate" in n)
            or ("num" in n and "candidate" in n and "retained" not in n and "remain" not in n and "after" not in n)
            or n in ["total", "before", "beforefilter", "totalnum"]
        ):
            total_cols.append(c)

        if (
            ("retained" in n and "candidate" in n)
            or ("remain" in n and "candidate" in n)
            or ("after" in n and "candidate" in n)
            or n in ["retained", "remaining", "after", "afterfilter"]
        ):
            retained_cols.append(c)

        if n in ["dataset", "datasetname", "source", "db", "database"]:
            dataset_cols.append(c)

    if not formula_cols or not total_cols or not retained_cols:
        return None

    formula_col = formula_cols[0]
    total_col = total_cols[0]
    retained_col = retained_cols[0]
    dataset_col = dataset_cols[0] if dataset_cols else None

    use_cols = [formula_col, total_col, retained_col]
    if dataset_col:
        use_cols.append(dataset_col)

    out = df[use_cols].copy()
    out = out.rename(columns={
        formula_col: "formula",
        total_col: "total_candidates",
        retained_col: "retained_candidates",
    })

    if dataset_col:
        out = out.rename(columns={dataset_col: "dataset"})
    else:
        out["dataset"] = "CandidateSet"

    out["formula"] = out["formula"].astype(str)
    out["dataset"] = out["dataset"].astype(str)
    out["total_candidates"] = pd.to_numeric(out["total_candidates"], errors="coerce")
    out["retained_candidates"] = pd.to_numeric(out["retained_candidates"], errors="coerce")

    out = out.dropna(subset=["formula", "total_candidates", "retained_candidates"])
    out = out[(out["total_candidates"] > 0) & (out["retained_candidates"] >= 0)]
    out = out[out["total_candidates"] >= out["retained_candidates"]]

    if len(out) < 5:
        return None

    out["source_file"] = str(path)
    return out


def make_formula_level_candidate_plot():
    preferred = [
        ROOT / "candidate_formula_counts.csv",
        ROOT / "paper_results_TCDV_TopoRT" / "candidate_filtering" / "candidate_formula_counts.csv",
        ROOT / "candidate_filtering" / "candidate_formula_counts.csv",
    ]

    found = []
    for p in preferred:
        if p.exists():
            t = detect_formula_count_table(p)
            if t is not None:
                print("[USE formula table]", p)
                found.append(t)

    if not found:
        csvs = []
        for p in ROOT.rglob("*.csv"):
            s = str(p).lower()
            if any(k in s for k in ["candidate", "filter", "rerank", "metabobase", "riken", "plasma"]):
                if p.stat().st_size < 200 * 1024 * 1024:
                    csvs.append(p)

        for p in sorted(csvs):
            t = detect_formula_count_table(p)
            if t is not None:
                print("[FOUND formula table]", p)
                found.append(t)

    if not found:
        print("")
        print("[SKIP] Formula-level candidate plot was not generated.")
        print("Reason: no CSV with formula,total_candidates,retained_candidates was detected.")
        print("Create this file if needed:")
        print("  candidate_formula_counts.csv")
        print("with columns:")
        print("  dataset,formula,total_candidates,retained_candidates")
        print("")
        return

    df = pd.concat(found, ignore_index=True)

    def group_name(x):
        x = str(x).lower()
        if "riken" in x or "plasma" in x:
            return "RIKEN-PlaSMA"
        if "metabo" in x:
            return "MetaboBase"
        return "CandidateSet"

    df["dataset_group"] = df["dataset"].map(group_name)

    df = (
        df.groupby(["dataset_group", "formula"], as_index=False)
          .agg(total_candidates=("total_candidates", "max"),
               retained_candidates=("retained_candidates", "min"))
    )

    order_map = {"RIKEN-PlaSMA": 0, "MetaboBase": 1, "CandidateSet": 2}
    df["order"] = df["dataset_group"].map(order_map).fillna(9)
    df = df.sort_values(["order", "total_candidates"], ascending=[True, True]).reset_index(drop=True)

    # Very long plot like ABCoRT Fig.5.
    max_rows = 180
    if len(df) > max_rows:
        df = df.tail(max_rows).copy().reset_index(drop=True)

    y = np.arange(len(df))
    height = max(8.0, 0.075 * len(df))

    plt.figure(figsize=(8.3, height))

    is_riken = df["dataset_group"].eq("RIKEN-PlaSMA")
    total_colors = np.where(is_riken, "#cdecc8", "#d8e6f7")
    retained_colors = np.where(is_riken, "#84cc82", "#8ea2bb")

    plt.barh(y, df["total_candidates"], color=total_colors, edgecolor="none")
    plt.barh(y, df["retained_candidates"], color=retained_colors, edgecolor="none")

    plt.yticks(y, df["formula"], fontsize=5)
    plt.xlabel("Number of candidates")
    plt.ylabel("Formula")

    handles = [
        Patch(facecolor="#cdecc8", label="Total candidates in RIKEN-PlaSMA"),
        Patch(facecolor="#84cc82", label="Retained candidates in RIKEN-PlaSMA"),
        Patch(facecolor="#d8e6f7", label="Total candidates in MetaboBase"),
        Patch(facecolor="#8ea2bb", label="Retained candidates in MetaboBase"),
    ]
    plt.legend(handles=handles, frameon=True, fontsize=8, loc="upper right")

    df.to_csv(OUT / "source_candidate_formula_level_retained.csv", index=False)

    savefig("fig_candidate_formula_level_retained")


def main():
    seed_frames, per_seed_metrics = read_seed_predictions()
    dual_agg = aggregate_variant_metrics(per_seed_metrics)
    ensemble_metrics = make_ensemble_metrics(seed_frames)
    struct_df = read_structural_metrics()

    metrics = ["MAE", "MRE", "MedAE", "MedRE", "R2"]
    labels = ["MAE", "MRE", "MedAE", "MedRE", r"$R^2$"]

    # 1. Similar to ABCoRT radar, but for main SMRT performance.
    make_smrt_tcdv_vs_abcort_radar(dual_agg, ensemble_metrics)

    # 2. Full dual-view/fusion ablation radar, 4 variants.
    plot_radar(
        dual_agg,
        "variant",
        metrics,
        labels,
        "Dual-view/fusion ablation radar",
        "fig_radar_dualview_fusion_ablation",
        legend_anchor=(1.38, 1.12)
    )

    # 3. Structural ablation radar, 3 structural variants.
    if len(struct_df) > 0:
        plot_radar(
            struct_df,
            "variant",
            metrics,
            labels,
            "Structural ablation radar",
            "fig_radar_structural_ablation",
            legend_anchor=(1.40, 1.12)
        )

        # 4. Two-panel ablation radar, this is the best one for main or SI.
        make_two_panel_ablation_radar(dual_agg, struct_df)

    # 5. Candidate filtering summary.
    make_candidate_summary_plot()

    # 6. Formula-level long candidate plot if formula table exists.
    make_formula_level_candidate_plot()

    print("")
    print("DONE. Outputs saved to:")
    print(OUT)
    print("")
    print("Recommended radar figures:")
    print("  fig_radar_dualview_fusion_ablation.pdf")
    print("  fig_radar_structural_ablation.pdf")
    print("  fig_radar_ablation_two_panel.pdf")
    print("")
    print("Formula-level candidate plot:")
    print("  fig_candidate_formula_level_retained.pdf  (only if formula-level data was found)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_result_base():
    candidates = [
        Path.cwd(),
        Path.cwd() / "gwn",
        Path.cwd().parent / "ABCoRT-main" / "gwn",
        Path.home() / "projects" / "lrq_q" / "ABCoRT-main" / "gwn",
    ]
    rel = Path("experiments_candidate_filtering")
    for b in candidates:
        if (b / rel).exists():
            return b
    raise FileNotFoundError("Cannot find experiments_candidate_filtering directory.")


BASE = find_result_base()
OUT = BASE / "experiments_candidate_filtering" / "filtering_parameter_sensitivity_audit"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "MetaboBase_evaluable45": {
        "summary": BASE / "experiments_candidate_filtering/metabobase_evaluable45_rank_guard_soft_eval_seed42/rank_guard_soft_rerank_summary.csv",
        "abcort": {"reduction": 38.35, "top1": 51.11, "top5": 73.33, "top10": 82.22},
        "selected": {
            "balanced": "rank_guard_filter_soft_th60.0_g3_tau75.17_alpha1.5",
            "high_reduction": "rank_guard_filter_soft_th50.0_g2_tau75.17_alpha1.5",
        },
    },
    "RIKEN_exact85": {
        "summary": BASE / "experiments_candidate_filtering/riken_exact85_rank_guard_soft_eval_seed42/rank_guard_soft_rerank_summary.csv",
        "abcort": {"reduction": 28.46, "top1": 52.94, "top5": 76.47, "top10": 83.53},
        "selected": {
            "balanced": "rank_guard_filter_soft_th50.0_g2_tau25.66_alpha2.0",
            "high_reduction": "rank_guard_filter_soft_th40.0_g2_tau25.66_alpha2.0",
        },
    },
}

METRIC_COLS = [
    "candidate_reduction_pct",
    "true_retention_pct",
    "top1_after_pct",
    "top5_after_pct",
    "top10_after_pct",
]


def safe_numeric(df):
    out = df.copy()
    for c in out.columns:
        if c != "method":
            out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def get_original_row(df):
    sub = df[df["method"].eq("original_msfinder_rank")]
    if len(sub) != 1:
        raise RuntimeError("original_msfinder_rank not found or duplicated.")
    return sub.iloc[0]


def add_flags(df, original, abcort):
    out = df.copy()

    out["delta_reduction_vs_original"] = out["candidate_reduction_pct"] - original["candidate_reduction_pct"]
    out["delta_top1_vs_original"] = out["top1_after_pct"] - original["top1_after_pct"]
    out["delta_top5_vs_original"] = out["top5_after_pct"] - original["top5_after_pct"]
    out["delta_top10_vs_original"] = out["top10_after_pct"] - original["top10_after_pct"]

    out["delta_reduction_vs_abcort_tl"] = out["candidate_reduction_pct"] - abcort["reduction"]
    out["delta_top1_vs_abcort_tl"] = out["top1_after_pct"] - abcort["top1"]
    out["delta_top5_vs_abcort_tl"] = out["top5_after_pct"] - abcort["top5"]
    out["delta_top10_vs_abcort_tl"] = out["top10_after_pct"] - abcort["top10"]

    out["beats_original_top1"] = out["top1_after_pct"] > original["top1_after_pct"]
    out["beats_original_top5"] = out["top5_after_pct"] > original["top5_after_pct"]
    out["beats_original_top10"] = out["top10_after_pct"] > original["top10_after_pct"]
    out["beats_original_all_topk"] = (
        out["beats_original_top1"] &
        out["beats_original_top5"] &
        out["beats_original_top10"]
    )

    out["beats_abcort_reduction"] = out["candidate_reduction_pct"] > abcort["reduction"]
    out["beats_abcort_top1"] = out["top1_after_pct"] >= abcort["top1"]
    out["beats_abcort_top5"] = out["top5_after_pct"] >= abcort["top5"]
    out["beats_abcort_top10"] = out["top10_after_pct"] >= abcort["top10"]
    out["beats_abcort_all_reported_metrics"] = (
        out["beats_abcort_reduction"] &
        out["beats_abcort_top1"] &
        out["beats_abcort_top5"] &
        out["beats_abcort_top10"]
    )
    return out


def mark_pareto_front(df):
    metrics = METRIC_COLS
    work = df.copy()
    unique = work[metrics].drop_duplicates().reset_index(drop=True)
    vals = unique[metrics].to_numpy(dtype=float)

    dominated = np.zeros(len(unique), dtype=bool)
    for i in range(len(unique)):
        vi = vals[i]
        ge = vals >= vi
        gt = vals > vi
        dom = np.all(ge, axis=1) & np.any(gt, axis=1)
        dom[i] = False
        dominated[i] = bool(np.any(dom))

    unique["is_pareto_5metric"] = ~dominated
    work = work.merge(unique[metrics + ["is_pareto_5metric"]], on=metrics, how="left")
    return work


def round_df(df, digits=3):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(digits)
    return out


def make_grid_values_table(dataset_name, grid):
    rows = []
    for label, col in [
        ("T / threshold_sec", "threshold_sec"),
        ("g / guard_k", "guard_k"),
        ("tau", "tau"),
        ("alpha", "alpha"),
    ]:
        vals = sorted([v for v in grid[col].dropna().unique().tolist()])
        rows.append({
            "Dataset": dataset_name,
            "Parameter": label,
            "Values explored": ", ".join(str(v) for v in vals),
            "Number of values": len(vals),
        })
    return pd.DataFrame(rows)


def selected_rows_table(dataset_name, grid, selected):
    rows = []
    for point_name, method_id in selected.items():
        sub = grid[grid["method"].eq(method_id)]
        if len(sub) != 1:
            print("[ERROR] selected method not found:", dataset_name, point_name, method_id)
            print(grid["method"].head(30).to_string(index=False))
            raise SystemExit(1)

        r = sub.iloc[0]
        rows.append({
            "Dataset": dataset_name,
            "Operating point": point_name,
            "Method id": method_id,
            "T / threshold": r["threshold_sec"],
            "g / guard_k": r["guard_k"],
            "tau": r["tau"],
            "alpha": r["alpha"],
            "Candidates before": r["n_candidate_rows_before"],
            "Candidates after": r["n_candidate_rows_after"],
            "Reduction (%)": r["candidate_reduction_pct"],
            "True retention (%)": r["true_retention_pct"],
            "Top1 (%)": r["top1_after_pct"],
            "Top5 (%)": r["top5_after_pct"],
            "Top10 (%)": r["top10_after_pct"],
            "ΔReduction vs ABCoRT-TL": r["delta_reduction_vs_abcort_tl"],
            "ΔTop1 vs ABCoRT-TL": r["delta_top1_vs_abcort_tl"],
            "ΔTop5 vs ABCoRT-TL": r["delta_top5_vs_abcort_tl"],
            "ΔTop10 vs ABCoRT-TL": r["delta_top10_vs_abcort_tl"],
            "Beats ABCoRT-TL all reported metrics": bool(r["beats_abcort_all_reported_metrics"]),
            "On 5-metric Pareto front": bool(r["is_pareto_5metric"]),
            "Reason for reporting": (
                "balanced operating point"
                if point_name == "balanced"
                else "higher-reduction operating point"
            ),
            "Universal optimum?": "No; representative point from sensitivity grid",
        })
    return pd.DataFrame(rows)


def make_tradeoff_rows(dataset_name, grid):
    sub = grid[grid["beats_abcort_all_reported_metrics"]].copy()
    if len(sub) == 0:
        return pd.DataFrame()

    sub = sub.sort_values(
        [
            "true_retention_pct",
            "top10_after_pct",
            "top5_after_pct",
            "top1_after_pct",
            "candidate_reduction_pct",
        ],
        ascending=[False, False, False, False, False],
    ).head(100)

    keep = [
        "method", "threshold_sec", "guard_k", "tau", "alpha",
        "n_candidate_rows_before", "n_candidate_rows_after",
        "candidate_reduction_pct", "true_retention_pct",
        "top1_after_pct", "top5_after_pct", "top10_after_pct",
        "delta_reduction_vs_abcort_tl", "delta_top1_vs_abcort_tl",
        "delta_top5_vs_abcort_tl", "delta_top10_vs_abcort_tl",
        "is_pareto_5metric",
    ]
    keep = [c for c in keep if c in sub.columns]
    out = sub[keep].copy()
    out.insert(0, "Dataset", dataset_name)
    return out

def plot_tradeoff(dataset_name, grid, selected, abcort, out_dir):
    """
    Publication-ready Figure S7 panel.

    The x-axis is extended to create a dedicated annotation region inside
    the black axes frame. Selected-point labels therefore do not cover the
    stars or sensitivity-grid points and do not extend beyond the axes.
    """

    title_map = {
        "MetaboBase_evaluable45": "MetaboBase",
        "RIKEN_exact85": "RIKEN-PlaSMA",
        "MetaboBase": "MetaboBase",
        "RIKEN-PlaSMA": "RIKEN-PlaSMA",
    }

    point_meta = {
        "balanced": {
            "legend": "Main operating point",
            "label": "Main operating point",
            "color": "#2CA02C",
        },
        "high_reduction": {
            "legend": "Higher-reduction point",
            "label": "Higher-reduction point",
            "color": "#D62728",
        },
    }

    # Vertical positions for annotation boxes.
    label_y_map = {
        "MetaboBase_evaluable45": {
            "balanced": 90.0,
            "high_reduction": 84.9,
        },
        "MetaboBase": {
            "balanced": 90.0,
            "high_reduction": 84.9,
        },
        "RIKEN_exact85": {
            "balanced": 89.7,
            "high_reduction": 87.3,
        },
        "RIKEN-PlaSMA": {
            "balanced": 89.7,
            "high_reduction": 87.3,
        },
    }

    # Wider canvas so the enlarged black axes frame remains readable.
    fig, ax = plt.subplots(figsize=(10.8, 5.9))

    # ------------------------------------------------------------
    # Sensitivity grid
    # ------------------------------------------------------------
    ax.scatter(
        grid["candidate_reduction_pct"],
        grid["top10_after_pct"],
        s=22,
        alpha=0.24,
        color="#5DA5DA",
        label="Sensitivity grid",
        zorder=2,
    )

    # Five-metric non-dominated set
    pareto = grid[grid["is_pareto_5metric"]].copy()
    ax.scatter(
        pareto["candidate_reduction_pct"],
        pareto["top10_after_pct"],
        s=40,
        alpha=0.90,
        color="#F28E2B",
        label="Five-metric non-dominated points",
        zorder=3,
    )

    # ------------------------------------------------------------
    # Enlarge the black axes frame itself
    # ------------------------------------------------------------
    x_min = float(grid["candidate_reduction_pct"].min())
    x_max = float(grid["candidate_reduction_pct"].max())
    y_min = float(grid["top10_after_pct"].min())
    y_max = float(grid["top10_after_pct"].max())

    x_span = max(x_max - x_min, 1.0)

    left_pad = max(2.0, 0.03 * x_span)

    # Large dedicated annotation band inside the axes.
    annotation_band = max(28.0, 0.38 * x_span)

    ax.set_xlim(
        x_min - left_pad,
        x_max + annotation_band,
    )

    ax.set_ylim(
        y_min - 0.8,
        y_max + 1.6,
    )

    # Annotation boxes begin several units to the right of all data points.
    label_x = x_max + max(4.0, 0.06 * x_span)

    # ------------------------------------------------------------
    # Selected operating points
    # ------------------------------------------------------------
    for point_name, method_id in selected.items():
        sub = grid[grid["method"].eq(method_id)]

        if len(sub) != 1:
            raise RuntimeError(
                f"Selected point not uniquely found: "
                f"dataset={dataset_name}, point={point_name}, method={method_id}"
            )

        r = sub.iloc[0]
        meta = point_meta[point_name]

        x0 = float(r["candidate_reduction_pct"])
        y0 = float(r["top10_after_pct"])

        # Keep the stars clearly visible above all other elements.
        ax.scatter(
            [x0],
            [y0],
            s=380,
            marker="*",
            color=meta["color"],
            edgecolor="white",
            linewidth=1.0,
            label=meta["legend"],
            zorder=15,
        )

        threshold = int(round(float(r["threshold_sec"])))
        guard = int(round(float(r["guard_k"])))
        tau = float(r["tau"])
        alpha = float(r["alpha"])

        label_text = (
            f"{meta['label']}\n"
            f"T = {threshold} s, g = {guard}\n"
            f"τ = {tau:.2f} s, α = {alpha:.1f}"
        )

        text_y = label_y_map.get(dataset_name, {}).get(
            point_name,
            y0,
        )

        text_y = min(text_y, ax.get_ylim()[1] - 0.55)
        text_y = max(text_y, ax.get_ylim()[0] + 0.55)

        ax.annotate(
            label_text,
            xy=(x0, y0),
            xycoords="data",
            xytext=(label_x, text_y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=9.0,
            linespacing=1.08,
            color="#303030",
            bbox=dict(
                boxstyle="round,pad=0.34",
                facecolor="white",
                edgecolor=meta["color"],
                linewidth=0.9,
                alpha=0.98,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=meta["color"],
                linewidth=1.0,
                shrinkA=7,
                shrinkB=8,
                connectionstyle="arc3,rad=0.02",
            ),
            annotation_clip=False,
            zorder=10,
        )

    # ------------------------------------------------------------
    # Formal labels
    # ------------------------------------------------------------
    ax.set_xlabel("Candidate-space reduction (%)", fontsize=11)
    ax.set_ylabel("Top-10 accuracy (%)", fontsize=11)
    ax.set_title(
        title_map.get(dataset_name, dataset_name),
        fontsize=14,
        pad=12,
    )

    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)

    # Make the enlarged black frame visually clear.
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    # Remove duplicate legend entries.
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    seen = set()

    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)

    ax.legend(
        unique_handles,
        unique_labels,
        fontsize=9,
        loc="lower left",
        framealpha=0.95,
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.975,
        bottom=0.14,
        top=0.90,
    )

    png = out_dir / f"{dataset_name}_sensitivity_tradeoff_reduction_vs_top10.png"
    pdf = out_dir / f"{dataset_name}_sensitivity_tradeoff_reduction_vs_top10.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )

    plt.close(fig)
    return png, pdf

def main():
    print("=" * 120)
    print("Result base:", BASE)
    print("Output dir :", OUT)
    print("=" * 120)

    all_grid_values = []
    all_selected = []
    all_tradeoff = []
    all_pareto = []

    for dataset_name, cfg in DATASETS.items():
        summary_path = cfg["summary"]
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)

        df = pd.read_csv(summary_path)
        df = safe_numeric(df)

        original = get_original_row(df)
        grid = df[df["method"].astype(str).str.startswith("rank_guard_filter_soft_")].copy()
        grid = add_flags(grid, original, cfg["abcort"])
        grid = mark_pareto_front(grid)

        dataset_out = OUT / dataset_name
        dataset_out.mkdir(parents=True, exist_ok=True)

        full_grid = round_df(grid)
        full_grid.to_csv(dataset_out / "full_rank_guard_soft_sensitivity_grid_with_flags.csv", index=False)

        grid_values = make_grid_values_table(dataset_name, grid)
        grid_values.to_csv(dataset_out / "parameter_grid_values.csv", index=False)
        all_grid_values.append(grid_values)

        selected = round_df(selected_rows_table(dataset_name, grid, cfg["selected"]))
        selected.to_csv(dataset_out / "selected_operating_points_with_rationale.csv", index=False)
        all_selected.append(selected)

        tradeoff = round_df(make_tradeoff_rows(dataset_name, grid))
        tradeoff.to_csv(dataset_out / "top_tradeoff_rows_beating_abcort_all_metrics.csv", index=False)
        if len(tradeoff):
            all_tradeoff.append(tradeoff)

        pareto = grid[grid["is_pareto_5metric"]].copy()
        pareto = pareto.sort_values(
            ["candidate_reduction_pct", "top10_after_pct", "top5_after_pct", "top1_after_pct"],
            ascending=[False, False, False, False],
        )
        pareto = round_df(pareto)
        pareto.to_csv(dataset_out / "non_dominated_pareto_rows_5metric.csv", index=False)
        tmp = pareto.copy()
        tmp.insert(0, "Dataset", dataset_name)
        all_pareto.append(tmp)

        png, pdf = plot_tradeoff(dataset_name, grid, cfg["selected"], cfg["abcort"], dataset_out)

        print("\n" + "=" * 120)
        print(dataset_name)
        print("summary:", summary_path)
        print("grid rows:", len(grid))
        print("parameter grid values:")
        print(grid_values.to_string(index=False))
        print("\nselected operating points:")
        print(selected.to_string(index=False))
        print("figure png:", png)
        print("figure pdf:", pdf)

    combined_grid_values = pd.concat(all_grid_values, ignore_index=True)
    combined_selected = pd.concat(all_selected, ignore_index=True)
    combined_pareto = pd.concat(all_pareto, ignore_index=True)

    combined_grid_values.to_csv(OUT / "combined_parameter_grid_values.csv", index=False)
    combined_selected.to_csv(OUT / "combined_selected_operating_points_with_rationale.csv", index=False)
    combined_pareto.to_csv(OUT / "combined_non_dominated_pareto_rows_5metric.csv", index=False)

    if all_tradeoff:
        combined_tradeoff = pd.concat(all_tradeoff, ignore_index=True)
        combined_tradeoff.to_csv(OUT / "combined_top_tradeoff_rows_beating_abcort_all_metrics.csv", index=False)

    si_cols = [
        "Dataset", "Operating point", "T / threshold", "g / guard_k", "tau", "alpha",
        "Reduction (%)", "True retention (%)", "Top1 (%)", "Top5 (%)", "Top10 (%)",
        "Beats ABCoRT-TL all reported metrics", "On 5-metric Pareto front",
        "Reason for reporting", "Universal optimum?",
    ]
    si_table = combined_selected[si_cols].copy()
    si_table.to_csv(OUT / "SI_selected_operating_points_parameter_source_table.csv", index=False)

    print("\n" + "=" * 120)
    print("[SAVED COMBINED FILES]")
    print(OUT / "combined_parameter_grid_values.csv")
    print(OUT / "combined_selected_operating_points_with_rationale.csv")
    print(OUT / "combined_non_dominated_pareto_rows_5metric.csv")
    print(OUT / "combined_top_tradeoff_rows_beating_abcort_all_metrics.csv")
    print(OUT / "SI_selected_operating_points_parameter_source_table.csv")
    print("\n[SI selected operating-point table]")
    print(si_table.to_string(index=False))
    print("=" * 120)


if __name__ == "__main__":
    main()

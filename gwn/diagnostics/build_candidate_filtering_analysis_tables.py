from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "gwn/final_paper_tables"
FIG = ROOT / "manuscript_figures_final"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "MetaboBase": {
        "candidate_file": ROOT / "gwn/experiments_candidate_filtering/metabobase_evaluable45_predictions_tl_seed42/evaluable45_candidate_predictions_tl_seed42.csv",
        "expected_n": 45,
        "T": 60.0,
        "g": 3,
        "tau": 75.17,
        "alpha": 1.5,
        "threshold_grid": [30, 45, 60, 75, 90],
        "abcort": {
            "reduction": 38.35,
            "top1": 51.11,
            "top5": 73.33,
            "top10": 82.22,
        },
    },
    "RIKEN-PlaSMA": {
        "candidate_file": ROOT / "gwn/experiments_candidate_filtering/riken_exact85_predictions_tl_seed42/riken_exact85_candidate_predictions_tl_seed42_for_eval.csv",
        "expected_n": 85,
        "T": 50.0,
        "g": 2,
        "tau": 25.66,
        "alpha": 2.0,
        "threshold_grid": [30, 40, 50, 60, 70],
        "abcort": {
            "reduction": 28.46,
            "top1": 52.94,
            "top5": 76.47,
            "top10": 83.53,
        },
    },
}

TABLE3_CSV = OUT / "Table_3_candidate_filtering_main.csv"
TABLE3_TEX = OUT / "Table_3_candidate_filtering_main.tex"

S24_CSV = OUT / "Table_S24_threshold_sensitivity.csv"
S24_TEX = OUT / "Table_S24_threshold_sensitivity.tex"
S25_CSV = OUT / "Table_S25_representative_candidate_filtering_cases.csv"
S25_TEX = OUT / "Table_S25_representative_candidate_filtering_cases.tex"

FIG_S6_PNG = FIG / "fig_s6_threshold_sensitivity.png"
FIG_S6_PDF = FIG / "fig_s6_threshold_sensitivity.pdf"


def norm(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def find_col(df, candidates=None, contains_any=None, contains_all=None, exclude_any=None):
    candidates = candidates or []
    contains_any = contains_any or []
    contains_all = contains_all or []
    exclude_any = exclude_any or []

    nmap = {norm(c): c for c in df.columns}

    for cand in candidates:
        k = norm(cand)
        if k in nmap:
            return nmap[k]

    for c in df.columns:
        k = norm(c)
        if contains_any and not any(x in k for x in contains_any):
            continue
        if contains_all and not all(x in k for x in contains_all):
            continue
        if exclude_any and any(x in k for x in exclude_any):
            continue
        return c

    return None


def bool_value(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "retained", "kept", "pass", "passed"}:
        return True
    if s in {"0", "false", "f", "no", "n", "removed", "miss", "false negative"}:
        return False
    try:
        return float(s) > 0
    except Exception:
        return False


def get_columns(df):
    qid = find_col(df, ["query_id", "s10_row", "query", "spectrum_id"], contains_any=["query", "s10", "spectrum"])
    formula = find_col(df, ["query_formula", "true_formula", "Formula", "formula"], contains_any=["formula"], exclude_any=["candidate", "cand"])
    inchikey = find_col(df, ["true_inchikey", "query_inchikey", "InChIKey", "inchikey"], contains_any=["inchikey"], exclude_any=["candidate", "cand"])
    rank = find_col(df, ["candidate_rank", "candidate_rank_raw"], contains_all=["rank"], exclude_any=["after", "rerank"])
    score = find_col(df, ["candidate_score", "score"], contains_any=["score"], exclude_any=["rt"])
    is_true = find_col(df, ["is_true", "is_true_full_inchikey", "is_true_inchikey14", "true_candidate"], contains_any=["true"], exclude_any=["rank"])

    abs_delta = find_col(df, ["abs_rt_delta", "rt_abs_delta", "abs_delta"], contains_all=["abs", "delta"])
    qrt = find_col(df, ["rt_sec", "query_rt_sec", "query_rt"], contains_any=["rt"])
    pred_rt = find_col(df, ["candidate_pred_rt", "pred_rt", "predicted_rt"], contains_all=["pred", "rt"])

    cols = {
        "query_id": qid,
        "formula": formula,
        "inchikey": inchikey,
        "candidate_rank": rank,
        "candidate_score": score,
        "is_true": is_true,
        "abs_rt_delta": abs_delta,
        "rt_sec": qrt,
        "candidate_pred_rt": pred_rt,
    }

    required = ["query_id", "formula", "inchikey", "candidate_rank", "candidate_score", "is_true"]
    missing = [k for k in required if cols[k] is None]
    if missing:
        print("[DEBUG] columns:", list(df.columns))
        raise RuntimeError(f"Missing required columns: {missing}")

    if cols["abs_rt_delta"] is None and (cols["rt_sec"] is None or cols["candidate_pred_rt"] is None):
        print("[DEBUG] columns:", list(df.columns))
        raise RuntimeError("Need either abs_rt_delta or both rt_sec and candidate_pred_rt.")

    return cols


def prepare_df(path):
    df = pd.read_csv(path)
    cols = get_columns(df)

    df = df.copy()
    df["_rank"] = pd.to_numeric(df[cols["candidate_rank"]], errors="coerce")
    df["_score"] = pd.to_numeric(df[cols["candidate_score"]], errors="coerce").fillna(0.0)
    df["_is_true"] = df[cols["is_true"]].map(bool_value)

    if cols["abs_rt_delta"] is not None:
        df["_abs_rt_delta"] = pd.to_numeric(df[cols["abs_rt_delta"]], errors="coerce")
    else:
        qrt = pd.to_numeric(df[cols["rt_sec"]], errors="coerce")
        pred = pd.to_numeric(df[cols["candidate_pred_rt"]], errors="coerce")
        df["_abs_rt_delta"] = (pred - qrt).abs()

    return df, cols


def evaluate_no_rt(df, cols):
    rows = []
    before_total = 0
    top1 = top5 = top10 = 0

    for qid, sub in df.groupby(cols["query_id"], sort=True):
        true_rows = sub[sub["_is_true"]]
        if len(true_rows) != 1:
            raise RuntimeError(f"query={qid}: expected one true row, got {len(true_rows)}")

        tr = true_rows.iloc[0]
        rank = int(round(float(tr["_rank"])))
        before_n = len(sub)

        before_total += before_n
        top1 += int(rank <= 1)
        top5 += int(rank <= 5)
        top10 += int(rank <= 10)

        rows.append({
            "query_id": qid,
            "InChIKey": str(tr[cols["inchikey"]]),
            "Formula": str(tr[cols["formula"]]),
            "Initial candidates": before_n,
            "Retained candidates": before_n,
            "MS-FINDER score rank": rank,
            "Rank after TC-TopoRT guarded reranking": rank,
            "Status": "Retained",
        })

    n = len(rows)
    return pd.DataFrame(rows), {
        "Queries": n,
        "Initial candidates": before_total,
        "Retained candidates": before_total,
        "Reduction (%)": 0.0,
        "True retained (%)": 100.0,
        "Top-1 (%)": 100.0 * top1 / n,
        "Top-5 (%)": 100.0 * top5 / n,
        "Top-10 (%)": 100.0 * top10 / n,
        "False negatives": 0,
    }


def evaluate_tc_toport(df, cols, T, g, tau, alpha):
    rows = []
    before_total = after_total = 0
    retained_count = 0
    top1 = top5 = top10 = 0
    false_neg = 0

    for qid, sub0 in df.groupby(cols["query_id"], sort=True):
        sub = sub0.copy()
        true_rows = sub[sub["_is_true"]]
        if len(true_rows) != 1:
            raise RuntimeError(f"query={qid}: expected one true row, got {len(true_rows)}")

        true_idx = true_rows.index[0]
        tr = true_rows.iloc[0]

        keep = (sub["_abs_rt_delta"] <= T) | (sub["_rank"] <= g)
        kept = sub[keep].copy()

        before_n = len(sub)
        after_n = len(kept)
        before_total += before_n
        after_total += after_n

        true_rank_before = int(round(float(tr["_rank"])))
        true_retained = bool(true_idx in kept.index)

        rank_after = ""

        if true_retained:
            kept["hybrid_score"] = kept["_rank"] + alpha * kept["_abs_rt_delta"] / tau
            kept = kept.sort_values(
                ["hybrid_score", "_rank", "_score"],
                ascending=[True, True, False],
                kind="mergesort",
            )
            kept["_rank_after"] = np.arange(1, len(kept) + 1)

            rank_after = int(kept.loc[true_idx, "_rank_after"])

            retained_count += 1
            top1 += int(rank_after <= 1)
            top5 += int(rank_after <= 5)
            top10 += int(rank_after <= 10)
            status = "Retained"
        else:
            false_neg += 1
            status = "False negative"

        rows.append({
            "query_id": qid,
            "InChIKey": str(tr[cols["inchikey"]]),
            "Formula": str(tr[cols["formula"]]),
            "Initial candidates": before_n,
            "Retained candidates": after_n,
            "MS-FINDER score rank": true_rank_before,
            "Rank after TC-TopoRT guarded reranking": rank_after,
            "Status": status,
        })

    n = len(rows)
    return pd.DataFrame(rows), {
        "Queries": n,
        "Initial candidates": before_total,
        "Retained candidates": after_total,
        "Reduction (%)": 100.0 * (before_total - after_total) / before_total,
        "True retained (%)": 100.0 * retained_count / n,
        "Top-1 (%)": 100.0 * top1 / n,
        "Top-5 (%)": 100.0 * top5 / n,
        "Top-10 (%)": 100.0 * top10 / n,
        "False negatives": false_neg,
    }


def fmt_num(x, nd=2):
    if pd.isna(x) or x == "":
        return ""
    try:
        v = float(x)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.{nd}f}"
    except Exception:
        return str(x)


def tex_escape(x):
    s = "" if pd.isna(x) else str(x)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def write_table3_tex(df):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{RT-guided candidate filtering and reranking performance.}")
    lines.append(r"\label{tab:main-candidate-filtering}")
    lines.append(r"\setlength{\tabcolsep}{3.0pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\begin{tabular}{llrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Method & Queries & Initial & Retained & Reduction (\%) & Top-1 (\%) & Top-5 (\%) & Top-10 (\%) \\"
    )
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        vals = [
            tex_escape(r["Dataset"]),
            tex_escape(r["Method"]),
            fmt_num(r["Queries"], 0),
            fmt_num(r["Initial candidates"], 0),
            fmt_num(r["Retained candidates"], 0),
            fmt_num(r["Reduction (%)"], 2),
            fmt_num(r["Top-1 (%)"], 2),
            fmt_num(r["Top-5 (%)"], 2),
            fmt_num(r["Top-10 (%)"], 2),
        ]
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    TABLE3_TEX.write_text("\n".join(lines) + "\n")


def write_simple_tex_table(df, path, caption, label, columns):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(rf"\caption{{{tex_escape(caption)}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\setlength{\tabcolsep}{3.0pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\begin{tabular}{" + "l" * len(columns) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join([tex_escape(c) for c in columns]) + r" \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        vals = [tex_escape(r.get(c, "")) for c in columns]
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")


def make_threshold_figure(th_df):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=False)

    metrics = [
        ("Reduction (%)", "Candidate reduction (%)"),
        ("Top-10 (%)", "Top-10 accuracy (%)"),
    ]

    for ax, (col, ylabel) in zip(axes, metrics):
        for dataset, sub in th_df.groupby("Dataset", sort=False):
            ax.plot(
                sub["T (s)"],
                sub[col],
                marker="o",
                linewidth=1.6,
                markersize=4.5,
                label=dataset,
            )

            # mark selected final T
            final_T = DATASETS[dataset]["T"]
            final_y = sub.loc[sub["T (s)"].eq(final_T), col]
            if len(final_y):
                ax.scatter([final_T], [float(final_y.iloc[0])], s=42, zorder=5)

        ax.set_xlabel("RT threshold T (s)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.tick_params(labelsize=8)

    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_S6_PDF, bbox_inches="tight")
    fig.savefig(FIG_S6_PNG, dpi=600, bbox_inches="tight")


def select_representative_cases(query_tables):
    all_rows = []
    for dataset, qdf in query_tables.items():
        qdf = qdf.copy()
        qdf["Dataset"] = dataset
        qdf["Reduction within query (%)"] = 100.0 * (
            qdf["Initial candidates"] - qdf["Retained candidates"]
        ) / qdf["Initial candidates"]

        rank_after_num = pd.to_numeric(qdf["Rank after TC-TopoRT guarded reranking"], errors="coerce")
        qdf["_rank_after_num"] = rank_after_num
        qdf["_rank_gain"] = qdf["MS-FINDER score rank"] - qdf["_rank_after_num"]

        retained = qdf[qdf["Status"].eq("Retained")].copy()
        false_neg = qdf[qdf["Status"].eq("False negative")].copy()

        chosen = []

        # Case 1: strong filtering and Top-1 success
        c1 = retained[retained["_rank_after_num"].eq(1)]
        if len(c1):
            row = c1.sort_values("Reduction within query (%)", ascending=False).iloc[0].copy()
            row["Case type"] = "Large candidate reduction with Top-1 retention"
            row["Interpretation"] = "TC-TopoRT substantially reduced the candidate set while keeping the true candidate ranked first."
            chosen.append(row)

        # Case 2: improved rank from low MS-FINDER rank to useful top-k
        c2 = retained[(retained["MS-FINDER score rank"] >= 10) & (retained["_rank_after_num"] <= 10)]
        if len(c2):
            row = c2.sort_values("_rank_gain", ascending=False).iloc[0].copy()
            row["Case type"] = "Reranking improvement from lower MS-FINDER rank"
            row["Interpretation"] = "The RT-guided reranking moved a low-ranked MS-FINDER true candidate into the top-k range."
            chosen.append(row)

        # Case 3: many initial candidates, still retained
        c3 = retained.sort_values("Initial candidates", ascending=False)
        if len(c3):
            row = c3.iloc[0].copy()
            row["Case type"] = "Large candidate pool"
            row["Interpretation"] = "The query had many initial candidates, and TC-TopoRT still retained the true candidate after filtering."
            chosen.append(row)

        # Case 4: false negative
        if len(false_neg):
            row = false_neg.sort_values("Initial candidates", ascending=False).iloc[0].copy()
            row["Case type"] = "False negative after RT filtering"
            row["Interpretation"] = "The true candidate was removed, illustrating the remaining risk of RT-guided filtering and the need for a guarded rule."
            chosen.append(row)

        for r in chosen:
            all_rows.append(r)

    out = pd.DataFrame(all_rows)

    keep_cols = [
        "Dataset",
        "Case type",
        "InChIKey",
        "Formula",
        "Initial candidates",
        "Retained candidates",
        "Reduction within query (%)",
        "MS-FINDER score rank",
        "Rank after TC-TopoRT guarded reranking",
        "Status",
        "Interpretation",
    ]
    out = out[keep_cols].copy()

    # Pretty formatting for CSV/TEX
    out["Reduction within query (%)"] = out["Reduction within query (%)"].map(lambda x: f"{x:.2f}")
    out["Rank after TC-TopoRT guarded reranking"] = out["Rank after TC-TopoRT guarded reranking"].replace("", "False negative")

    return out


def main():
    table3_rows = []
    th_rows = []
    query_tables = {}

    for dataset, cfg in DATASETS.items():
        print(f"\n===== {dataset} =====")
        df, cols = prepare_df(cfg["candidate_file"])

        # No-RT baseline
        no_rt_table, no_rt_metrics = evaluate_no_rt(df, cols)

        # TC-TopoRT final
        tc_table, tc_metrics = evaluate_tc_toport(
            df,
            cols,
            T=cfg["T"],
            g=cfg["g"],
            tau=cfg["tau"],
            alpha=cfg["alpha"],
        )

        query_tables[dataset] = tc_table

        # ABCoRT-TL existing benchmark row
        initial = no_rt_metrics["Initial candidates"]
        retained_abcort = int(round(initial * (1 - cfg["abcort"]["reduction"] / 100.0)))

        table3_rows.append({
            "Dataset": dataset,
            "Method": "MS-FINDER only / No RT",
            **no_rt_metrics,
        })
        table3_rows.append({
            "Dataset": dataset,
            "Method": "ABCoRT-TL",
            "Queries": cfg["expected_n"],
            "Initial candidates": initial,
            "Retained candidates": retained_abcort,
            "Reduction (%)": cfg["abcort"]["reduction"],
            "Top-1 (%)": cfg["abcort"]["top1"],
            "Top-5 (%)": cfg["abcort"]["top5"],
            "Top-10 (%)": cfg["abcort"]["top10"],
            "True retained (%)": "",
            "False negatives": "",
        })
        table3_rows.append({
            "Dataset": dataset,
            "Method": "TC-TopoRT",
            **tc_metrics,
        })

        # threshold sensitivity
        for T in cfg["threshold_grid"]:
            _, m = evaluate_tc_toport(
                df,
                cols,
                T=T,
                g=cfg["g"],
                tau=cfg["tau"],
                alpha=cfg["alpha"],
            )
            th_rows.append({
                "Dataset": dataset,
                "T (s)": T,
                "g": cfg["g"],
                "tau": cfg["tau"],
                "alpha": cfg["alpha"],
                **m,
            })

        print("[No RT]", no_rt_metrics)
        print("[TC-TopoRT]", tc_metrics)

    table3 = pd.DataFrame(table3_rows)
    table3 = table3[
        [
            "Dataset", "Method", "Queries", "Initial candidates", "Retained candidates",
            "Reduction (%)", "Top-1 (%)", "Top-5 (%)", "Top-10 (%)",
            "True retained (%)", "False negatives"
        ]
    ]
    table3.to_csv(TABLE3_CSV, index=False)
    write_table3_tex(table3)

    th_df = pd.DataFrame(th_rows)
    th_df.to_csv(S24_CSV, index=False)

    s24_pretty = th_df.copy()
    for c in ["Reduction (%)", "True retained (%)", "Top-1 (%)", "Top-5 (%)", "Top-10 (%)"]:
        s24_pretty[c] = s24_pretty[c].map(lambda x: f"{x:.2f}")
    for c in ["T (s)", "g", "tau", "alpha", "Queries", "Initial candidates", "Retained candidates", "False negatives"]:
        s24_pretty[c] = s24_pretty[c].map(lambda x: fmt_num(x, 2))

    write_simple_tex_table(
        s24_pretty,
        S24_TEX,
        "Threshold sensitivity analysis for RT-guided candidate filtering.",
        "tab:s24-threshold-sensitivity",
        [
            "Dataset", "T (s)", "g", "Initial candidates", "Retained candidates",
            "Reduction (%)", "True retained (%)", "Top-1 (%)", "Top-5 (%)", "Top-10 (%)",
            "False negatives",
        ],
    )

    make_threshold_figure(th_df)

    cases = select_representative_cases(query_tables)
    cases.to_csv(S25_CSV, index=False)

    write_simple_tex_table(
        cases,
        S25_TEX,
        "Representative query-level candidate filtering and reranking cases.",
        "tab:s25-representative-candidate-filtering-cases",
        [
            "Dataset", "Case type", "InChIKey", "Formula", "Initial candidates",
            "Retained candidates", "Reduction within query (%)", "MS-FINDER score rank",
            "Rank after TC-TopoRT guarded reranking", "Status",
        ],
    )

    print("\n===== WROTE =====")
    for p in [TABLE3_CSV, TABLE3_TEX, S24_CSV, S24_TEX, S25_CSV, S25_TEX, FIG_S6_PDF, FIG_S6_PNG]:
        print(p.relative_to(ROOT))

    print("\n===== Table 3 preview =====")
    print(table3.to_string(index=False))

    print("\n===== Threshold sensitivity preview =====")
    print(th_df.to_string(index=False))

    print("\n===== Representative cases preview =====")
    print(cases.to_string(index=False))


if __name__ == "__main__":
    main()

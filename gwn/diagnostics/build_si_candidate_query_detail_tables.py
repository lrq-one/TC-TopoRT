from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "gwn/final_paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_COLUMNS = [
    "InChIKey",
    "Formula",
    "Num. candidates from MS-FINDER",
    "Num. candidates retained by TC-TopoRT",
    "MS-FINDER score rank",
    "Rank after TC-TopoRT guarded reranking",
    "Status",
]

DATASETS = {
    "metabobase": {
        "label": "MetaboBase",
        "expected_n": 45,
        "candidate_file": ROOT / "gwn/experiments_candidate_filtering/metabobase_evaluable45_predictions_tl_seed42/evaluable45_candidate_predictions_tl_seed42.csv",
        "threshold": 60.0,
        "guard_k": 3,
        "tau": 75.17,
        "alpha": 1.5,
        "target_before": 3023,
        "target_after": 933,
        "target_reduction": 69.14,
        "target_retention": 93.33,
        "target_top1": 55.56,
        "target_top5": 82.22,
        "target_top10": 88.89,
        "out_csv": OUT_DIR / "Table_S20_metabobase_query_filtering_details.csv",
        "out_tex": OUT_DIR / "Table_S20_metabobase_query_filtering_details.tex",
        "caption": "Query-level candidate filtering and guarded soft reranking details on MetaboBase.",
        "tex_label": "tab:s20-metabobase-query-detail",
    },
    "riken": {
        "label": "RIKEN-PlaSMA",
        "expected_n": 85,
        "candidate_file": ROOT / "gwn/experiments_candidate_filtering/riken_exact85_predictions_tl_seed42/riken_exact85_candidate_predictions_tl_seed42_for_eval.csv",
        "threshold": 50.0,
        "guard_k": 2,
        "tau": 25.66,
        "alpha": 2.0,
        "target_before": 5044,
        "target_after": 2712,
        "target_reduction": 46.23,
        "target_retention": 97.65,
        "target_top1": 54.12,
        "target_top5": 77.65,
        "target_top10": 89.41,
        "out_csv": OUT_DIR / "Table_S21_riken_plasma_query_filtering_details.csv",
        "out_tex": OUT_DIR / "Table_S21_riken_plasma_query_filtering_details.tex",
        "caption": "Query-level candidate filtering and guarded soft reranking details on RIKEN-PlaSMA.",
        "tex_label": "tab:s21-riken-query-detail",
    },
}


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
    formula = find_col(df, ["Formula", "query_formula", "true_formula", "formula"], contains_any=["formula"], exclude_any=["candidate", "cand"])
    inchikey = find_col(df, ["true_inchikey", "query_inchikey", "inchikey", "InChIKey"], contains_any=["inchikey"], exclude_any=["candidate", "cand"])
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
        raise ValueError(f"Missing required columns: {missing}")

    if cols["abs_rt_delta"] is None and (cols["rt_sec"] is None or cols["candidate_pred_rt"] is None):
        print("[DEBUG] columns:", list(df.columns))
        raise ValueError("Need either abs_rt_delta or both rt_sec and candidate_pred_rt.")

    return cols


def add_abs_delta(df, cols):
    df = df.copy()

    df["_candidate_rank"] = pd.to_numeric(df[cols["candidate_rank"]], errors="coerce")
    df["_candidate_score"] = pd.to_numeric(df[cols["candidate_score"]], errors="coerce").fillna(0.0)
    df["_is_true"] = df[cols["is_true"]].map(bool_value)

    if cols["abs_rt_delta"] is not None:
        df["_abs_rt_delta"] = pd.to_numeric(df[cols["abs_rt_delta"]], errors="coerce")
    else:
        qrt = pd.to_numeric(df[cols["rt_sec"]], errors="coerce")
        pred = pd.to_numeric(df[cols["candidate_pred_rt"]], errors="coerce")
        df["_abs_rt_delta"] = (pred - qrt).abs()

    return df


def evaluate_guarded_soft(df, cols, cfg):
    rows = []
    before_total = 0
    after_total = 0
    retained_count = 0
    top1 = 0
    top5 = 0
    top10 = 0

    for qid, sub0 in df.groupby(cols["query_id"], sort=True):
        sub = sub0.copy()

        true_rows = sub[sub["_is_true"]]
        if len(true_rows) != 1:
            raise ValueError(f"query_id={qid}: expected exactly one true candidate, got {len(true_rows)}")

        true_idx = true_rows.index[0]
        tr = true_rows.iloc[0]

        keep = (sub["_abs_rt_delta"] <= cfg["threshold"]) | (sub["_candidate_rank"] <= cfg["guard_k"])
        kept = sub[keep].copy()

        before_n = len(sub)
        after_n = len(kept)

        before_total += before_n
        after_total += after_n

        true_rank_before = int(round(float(tr["_candidate_rank"])))
        true_retained = bool(true_idx in kept.index)

        rank_after = ""
        if true_retained:
            kept["hybrid_score"] = kept["_candidate_rank"] + cfg["alpha"] * kept["_abs_rt_delta"] / cfg["tau"]

            kept = kept.sort_values(
                ["hybrid_score", "_candidate_rank", "_candidate_score"],
                ascending=[True, True, False],
                kind="mergesort",
            )
            kept["_rank_after"] = np.arange(1, len(kept) + 1)

            rank_after = int(kept.loc[true_idx, "_rank_after"])

            retained_count += 1
            if rank_after <= 1:
                top1 += 1
            if rank_after <= 5:
                top5 += 1
            if rank_after <= 10:
                top10 += 1

        rows.append({
            "InChIKey": str(tr[cols["inchikey"]]),
            "Formula": str(tr[cols["formula"]]),
            "Num. candidates from MS-FINDER": before_n,
            "Num. candidates retained by TC-TopoRT": after_n,
            "MS-FINDER score rank": true_rank_before,
            "Rank after TC-TopoRT guarded reranking": rank_after,
            "Status": "Retained" if true_retained else "False negative",
        })

    n = len(rows)
    metrics = {
        "n": n,
        "before": before_total,
        "after": after_total,
        "reduction": 100.0 * (before_total - after_total) / max(before_total, 1),
        "retention": 100.0 * retained_count / n,
        "top1": 100.0 * top1 / n,
        "top5": 100.0 * top5 / n,
        "top10": 100.0 * top10 / n,
    }

    table = pd.DataFrame(rows)
    return table, metrics


def check_metrics(metrics, cfg):
    checks = [
        ("n", metrics["n"], cfg["expected_n"], 0.01),
        ("before", metrics["before"], cfg["target_before"], 0.01),
        ("after", metrics["after"], cfg["target_after"], 0.01),
        ("reduction", metrics["reduction"], cfg["target_reduction"], 0.05),
        ("retention", metrics["retention"], cfg["target_retention"], 0.05),
        ("top1", metrics["top1"], cfg["target_top1"], 0.05),
        ("top5", metrics["top5"], cfg["target_top5"], 0.05),
        ("top10", metrics["top10"], cfg["target_top10"], 0.05),
    ]

    bad = []
    for name, got, target, tol in checks:
        if abs(float(got) - float(target)) > tol:
            bad.append((name, got, target))

    if bad:
        print("\n[ERROR] Generated query table does not match final Table S17 metrics.")
        for name, got, target in bad:
            print(f"  {name}: got {got}, target {target}")
        raise SystemExit("Stop: do not use generated S20/S21 until metrics match.")


def latex_escape(x):
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


def write_tex(table, path, caption, label):
    headers = [
        "InChIKey",
        "Formula",
        "Candidates from MS-FINDER",
        "Candidates retained by TC-TopoRT",
        "MS-FINDER rank",
        "Rank after TC-TopoRT",
        "Status",
    ]

    lines = []
    lines.append(r"\begin{landscape}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{longtable}{p{4.2cm}p{2.0cm}rrrrp{2.2cm}}")
    lines.append(rf"\caption{{{latex_escape(caption)}}}")
    lines.append(rf"\label{{{label}}}\\")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{r}{Continued on next page}\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for _, row in table.iterrows():
        vals = [latex_escape(row[c]) for c in FINAL_COLUMNS]
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\end{longtable}")
    lines.append(r"\end{landscape}")
    path.write_text("\n".join(lines) + "\n")


def main():
    for _, cfg in DATASETS.items():
        print(f"\n===== {cfg['label']} =====")
        print("[SOURCE]", cfg["candidate_file"].relative_to(ROOT))
        print(
            f"[CONFIG] threshold={cfg['threshold']} guard_k={cfg['guard_k']} "
            f"tau={cfg['tau']} alpha={cfg['alpha']}"
        )

        df = pd.read_csv(cfg["candidate_file"])
        cols = get_columns(df)
        print("[COLUMNS]", cols)

        df = add_abs_delta(df, cols)
        table, metrics = evaluate_guarded_soft(df, cols, cfg)

        check_metrics(metrics, cfg)

        table = table.sort_values(["Formula", "InChIKey"], kind="mergesort").reset_index(drop=True)
        table.to_csv(cfg["out_csv"], index=False)
        write_tex(table, cfg["out_tex"], cfg["caption"], cfg["tex_label"])

        print(
            f"[FINAL] rows={metrics['n']} before={metrics['before']} after={metrics['after']} "
            f"reduction={metrics['reduction']:.2f}% retention={metrics['retention']:.2f}% "
            f"top1={metrics['top1']:.2f}% top5={metrics['top5']:.2f}% top10={metrics['top10']:.2f}%"
        )
        print("[WROTE]", cfg["out_csv"].relative_to(ROOT))
        print("[WROTE]", cfg["out_tex"].relative_to(ROOT))
        print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

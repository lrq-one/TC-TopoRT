from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "gwn/final_paper_tables"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {
        "Dataset": "MetaboBase",
        "summary": ROOT / "gwn/experiments_candidate_filtering/metabobase_evaluable45_rank_guard_soft_eval_seed42/rank_guard_soft_rerank_summary.csv",
        "method": "rank_guard_filter_soft_th60.0_g3_tau75.17_alpha1.5",
        "T": 60.0,
        "g": 3,
        "tau": 75.17,
        "alpha": 1.5,
    },
    {
        "Dataset": "RIKEN-PlaSMA",
        "summary": ROOT / "gwn/experiments_candidate_filtering/riken_exact85_rank_guard_soft_eval_seed42/rank_guard_soft_rerank_summary.csv",
        "method": "rank_guard_filter_soft_th50.0_g2_tau25.66_alpha2.0",
        "T": 50.0,
        "g": 2,
        "tau": 25.66,
        "alpha": 2.0,
    },
]

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

def fmt(x, nd=2):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        v = float(x)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.{nd}f}"
    except Exception:
        return str(x)

rows = []
for cfg in CONFIGS:
    if not cfg["summary"].exists():
        raise FileNotFoundError(cfg["summary"])

    df = pd.read_csv(cfg["summary"])
    sel = df[df["method"].astype(str).eq(cfg["method"])]
    if len(sel) != 1:
        print(df[["method"]].to_string(index=False))
        raise RuntimeError(f"Cannot uniquely find method: {cfg['method']}")

    r = sel.iloc[0]
    rows.append({
        "Dataset": cfg["Dataset"],
        "Selected method": "RT-aware guarded soft rerank",
        "T (s)": cfg["T"],
        "g": cfg["g"],
        "tau (s)": cfg["tau"],
        "alpha": cfg["alpha"],
        "Candidate retention rule": r"$|\Delta RT| \leq T$ or MS-FINDER rank $\leq g$",
        "Reranking score": r"$r_{\mathrm{MSFINDER}} + \alpha |\Delta RT|/\tau$",
        "Tie-breaking": "MS-FINDER rank, then MS-FINDER score",
        "Queries": int(r["n_queries"]),
        "Candidates before": int(r["n_candidate_rows_before"]),
        "Candidates after": int(r["n_candidate_rows_after"]),
        "Reduction (%)": float(r["candidate_reduction_pct"]),
        "True retained (%)": float(r["true_retention_pct"]),
        "Top-1 (%)": float(r["top1_after_pct"]),
        "Top-5 (%)": float(r["top5_after_pct"]),
        "Top-10 (%)": float(r["top10_after_pct"]),
    })

out = pd.DataFrame(rows)

csv_path = OUT / "Table_S22_candidate_filtering_parameters.csv"
tex_path = OUT / "Table_S22_candidate_filtering_parameters.tex"
text_path = OUT / "Text_S6_candidate_filtering_parameters.tex"

out.to_csv(csv_path, index=False)

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\scriptsize")
lines.append(r"\caption{Candidate filtering and guarded soft reranking parameters used in the formula-level candidate filtering experiments.}")
lines.append(r"\label{tab:s22-candidate-filtering-parameters}")
lines.append(r"\setlength{\tabcolsep}{3.0pt}")
lines.append(r"\renewcommand{\arraystretch}{1.10}")
lines.append(r"\begin{tabular}{lccccccccc}")
lines.append(r"\toprule")
lines.append(
    r"Dataset & $T$ (s) & $g$ & $\tau$ (s) & $\alpha$ & "
    r"Queries & Before & After & Reduction (\%) & True retained (\%) \\"
)
lines.append(r"\midrule")
for _, r in out.iterrows():
    vals = [
        tex_escape(r["Dataset"]),
        fmt(r["T (s)"]),
        fmt(r["g"]),
        fmt(r["tau (s)"]),
        fmt(r["alpha"]),
        fmt(r["Queries"]),
        fmt(r["Candidates before"]),
        fmt(r["Candidates after"]),
        fmt(r["Reduction (%)"]),
        fmt(r["True retained (%)"]),
    ]
    lines.append(" & ".join(vals) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append("")
lines.append(r"\vspace{2pt}")
lines.append(r"\begin{minipage}{0.95\linewidth}")
lines.append(r"\scriptsize")
lines.append(
    r"For each candidate, $\Delta RT$ denotes the absolute difference between the predicted candidate RT "
    r"and the query RT. A candidate was retained if $|\Delta RT| \leq T$ or its original MS-FINDER rank "
    r"was no larger than the guard parameter $g$. Retained candidates were reranked by "
    r"$r_{\mathrm{MSFINDER}} + \alpha |\Delta RT|/\tau$, where a smaller value indicates a better rank. "
    r"Ties were resolved by the original MS-FINDER rank and then by the MS-FINDER score."
)
lines.append(r"\end{minipage}")
lines.append(r"\end{table}")

tex_path.write_text("\n".join(lines) + "\n")

text = r"""
\noindent\textbf{Text S6. Candidate Filtering and Guarded Soft Reranking Parameters.}
For each query, TC-TopoRT first computed the absolute RT deviation between each candidate and the query RT.
Candidates were retained if $|\Delta RT| \leq T$ or if their original MS-FINDER rank was no larger than the
guard parameter $g$. The retained candidates were then reranked using the hybrid score
$r_{\mathrm{MSFINDER}} + \alpha |\Delta RT|/\tau$, where a smaller score indicates a better rank.
The dataset-specific values of $T$, $g$, $\tau$, and $\alpha$ are summarized in Table~\ref{tab:s22-candidate-filtering-parameters}.
"""
text_path.write_text(text.strip() + "\n")

print("[WROTE]", csv_path.relative_to(ROOT))
print("[WROTE]", tex_path.relative_to(ROOT))
print("[WROTE]", text_path.relative_to(ROOT))
print(out.to_string(index=False))

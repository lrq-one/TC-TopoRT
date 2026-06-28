from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "gwn/final_paper_tables"
OUT.mkdir(parents=True, exist_ok=True)

TABLE8_CANDIDATES = [
    ROOT / "gwn/final_paper_tables/Table_8_transfer_learning_effectiveness.csv",
    ROOT / "gwn/final_paper_tables/Table_S16_transfer_learning_effectiveness.csv",
    ROOT / "gwn/experiments_transfer_effectiveness/external_transfer_vs_scratch_effectiveness/external_transfer_vs_scratch_effectiveness_summary.csv",
]

CSV_OUT = OUT / "Table_S23_external_transfer_finetuning_details.csv"
TEX_OUT = OUT / "Table_S23_external_transfer_finetuning_details.tex"
TEXT_OUT = OUT / "Text_S7_external_transfer_finetuning_details.tex"

CONFIG = {
    "Initialization": "SMRT-pretrained TC-TopoRT checkpoint",
    "Entry point": "gwn/experiments_transfer_effectiveness/external_transfer_all10.py",
    "Fine-tuning script": "gwn/experiments_transfer_effectiveness/external_train_tcdv_transfer_or_scratch.py",
    "Aggregation script": "gwn/experiments_transfer_effectiveness/external_stack_fixed_raw_autoselect.py",
    "Protocol": "fixed raw AutoSelect external transfer",
    "Epochs": 150,
    "Batch size": 8,
    "Eval batch size": 64,
    "Learning rate": "1e-4",
    "Weight decay": "1e-2",
    "Optimizer": "AdamW",
    "L2-SP": "Not used",
    "Output setting": "fixed_raw_autoselect_all10_cvseed1",
}

def norm(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def pick_col(df, names):
    nmap = {norm(c): c for c in df.columns}
    for name in names:
        k = norm(name)
        if k in nmap:
            return nmap[k]
    for c in df.columns:
        nc = norm(c)
        if any(norm(name) in nc for name in names):
            return c
    return None

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

def fmt_float(x):
    if pd.isna(x) or str(x).strip() == "":
        return ""
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)

def find_table8():
    for p in TABLE8_CANDIDATES:
        if p.exists():
            return p, pd.read_csv(p)
    raise FileNotFoundError("Cannot find transfer-vs-scratch summary table.")

table_path, df = find_table8()
print("[SOURCE TABLE]", table_path.relative_to(ROOT))
print("[COLUMNS]", list(df.columns))

dataset_col = pick_col(df, ["dataset_name", "Dataset", "dataset"])
n_col = pick_col(df, ["n", "N", "num_molecules"])
transfer_col = pick_col(df, ["transfer_mae", "Transfer MAE (s)", "TL MAE", "Fine-tuned MAE"])
scratch_col = pick_col(df, ["scratch_mae", "Scratch MAE (s)", "From-scratch MAE"])
improve_col = pick_col(df, ["mae_improvement", "MAE improvement", "Improvement"])

if dataset_col is None:
    raise RuntimeError("Cannot find dataset column in Table 8 source.")

rows = []
for _, r in df.iterrows():
    row = {
        "Dataset": r[dataset_col],
        "N": r[n_col] if n_col else "",
        "Initialization": CONFIG["Initialization"],
        "Protocol": CONFIG["Protocol"],
        "Epochs": CONFIG["Epochs"],
        "Batch size": CONFIG["Batch size"],
        "Eval batch size": CONFIG["Eval batch size"],
        "Learning rate": CONFIG["Learning rate"],
        "Weight decay": CONFIG["Weight decay"],
        "Optimizer": CONFIG["Optimizer"],
        "L2-SP": CONFIG["L2-SP"],
        "Entry point": CONFIG["Entry point"],
        "Fine-tuning script": CONFIG["Fine-tuning script"],
        "Aggregation script": CONFIG["Aggregation script"],
        "Output setting": CONFIG["Output setting"],
        "Transfer MAE (s)": fmt_float(r[transfer_col]) if transfer_col else "",
        "Scratch MAE (s)": fmt_float(r[scratch_col]) if scratch_col else "",
        "MAE improvement (s)": fmt_float(r[improve_col]) if improve_col else "",
    }
    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(CSV_OUT, index=False)

# LaTeX table: keep it reproducibility-oriented, not too wide in the main columns.
lines = []
lines.append(r"\begin{landscape}")
lines.append(r"\scriptsize")
lines.append(r"\setlength{\tabcolsep}{2.5pt}")
lines.append(r"\renewcommand{\arraystretch}{1.08}")
lines.append(r"\begin{longtable}{p{3.0cm}p{0.8cm}p{2.4cm}p{1.0cm}p{1.0cm}p{1.0cm}p{1.1cm}p{1.1cm}p{1.0cm}p{1.2cm}p{3.8cm}}")
lines.append(r"\caption{External transfer fine-tuning settings used for the all10 external RT benchmarks.}")
lines.append(r"\label{tab:s23-external-transfer-finetuning-details}\\")
lines.append(r"\toprule")
lines.append(
    r"Dataset & $N$ & Initialization & Epochs & Batch & Eval batch & LR & Weight decay & Optimizer & L2-SP & Entry point \\"
)
lines.append(r"\midrule")
lines.append(r"\endfirsthead")
lines.append(r"\toprule")
lines.append(
    r"Dataset & $N$ & Initialization & Epochs & Batch & Eval batch & LR & Weight decay & Optimizer & L2-SP & Entry point \\"
)
lines.append(r"\midrule")
lines.append(r"\endhead")
lines.append(r"\midrule")
lines.append(r"\multicolumn{11}{r}{Continued on next page}\\")
lines.append(r"\endfoot")
lines.append(r"\bottomrule")
lines.append(r"\endlastfoot")

for _, r in out.iterrows():
    vals = [
        tex_escape(r["Dataset"]),
        tex_escape(r["N"]),
        tex_escape(r["Initialization"]),
        tex_escape(r["Epochs"]),
        tex_escape(r["Batch size"]),
        tex_escape(r["Eval batch size"]),
        tex_escape(r["Learning rate"]),
        tex_escape(r["Weight decay"]),
        tex_escape(r["Optimizer"]),
        tex_escape(r["L2-SP"]),
        tex_escape(r["Entry point"]),
    ]
    lines.append(" & ".join(vals) + r" \\")

lines.append(r"\end{longtable}")
lines.append(r"\end{landscape}")
TEX_OUT.write_text("\n".join(lines) + "\n")

text = r"""
\noindent\textbf{Text S7. External Transfer Fine-Tuning Details.}
For the all10 external RT benchmarks, the transfer-learning experiments were run using the paper-facing
external transfer workflow. The entry point was
\texttt{gwn/experiments\_transfer\_effectiveness/external\_transfer\_all10.py}, which calls the fixed
raw AutoSelect transfer workflow and the fine-tuning script
\texttt{external\_train\_tcdv\_transfer\_or\_scratch.py}. The fine-tuning stage used AdamW with a learning
rate of $1\times10^{-4}$, weight decay of $1\times10^{-2}$, a batch size of 8, an evaluation batch size of 64,
and a maximum of 150 epochs. L2-SP regularization was not used in this final all10 external transfer protocol.
The per-dataset settings are summarized in Table~\ref{tab:s23-external-transfer-finetuning-details}.
"""
TEXT_OUT.write_text(text.strip() + "\n")

print("[WROTE]", CSV_OUT.relative_to(ROOT))
print("[WROTE]", TEX_OUT.relative_to(ROOT))
print("[WROTE]", TEXT_OUT.relative_to(ROOT))
print()
print(out.to_string(index=False))

# hard validation: avoid the old wrong 45.3 LR problem
if out["Learning rate"].astype(str).str.contains("45.3").any():
    raise SystemExit("[ERROR] learning rate was incorrectly parsed as 45.3")
if out["Learning rate"].isna().any() or out["Learning rate"].astype(str).str.strip().eq("").any():
    raise SystemExit("[ERROR] missing learning rate")
if out["L2-SP"].isna().any() or out["L2-SP"].astype(str).str.strip().eq("").any():
    raise SystemExit("[ERROR] missing L2-SP status")

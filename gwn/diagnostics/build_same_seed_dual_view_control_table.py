from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()

SEEDS = [1, 5, 79, 123, 256]

SEED_FILES = {
    1:   ROOT / "gwn/results_OOF_DualView_Stack_v1/test_predictions_audited.csv",
    5:   ROOT / "gwn/results_OOF_DualView_Stack_seed5/test_predictions.csv",
    79:  ROOT / "gwn/results_OOF_DualView_Stack_seed79/test_predictions.csv",
    123: ROOT / "gwn/results_OOF_DualView_Stack_seed123/test_predictions.csv",
    256: ROOT / "gwn/results_OOF_DualView_Stack_seed256/test_predictions.csv",
}

ID_COL = "Source_Index"
Y_COL = "Actual_RT"
ORIG_COL = "Origin_Test_Pred"
TAUT_COL = "Taut_Test_Pred"
FINAL_COL = "Final_Pred"

OUT_DIR = ROOT / "gwn/final_paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "Table_S26_same_seed_dual_view_controls.csv"
OUT_DETAIL = OUT_DIR / "Table_S26_same_seed_dual_view_controls_detail.csv"
OUT_TEX = OUT_DIR / "Table_S26_same_seed_dual_view_controls.tex"


def mae(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(p - y)))


def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1))


def fmt(vals):
    m, s = mean_std(vals)
    return f"{m:.3f} ± {s:.3f}", m, s


def tex_escape(s):
    s = str(s)
    repl = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "±": r"$\pm$",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def load_seed_files():
    dfs = {}
    for seed, path in SEED_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing seed prediction file: {path}")

        df = pd.read_csv(path)
        need = [ID_COL, Y_COL, ORIG_COL, TAUT_COL, FINAL_COL]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}\ncolumns={df.columns.tolist()}")

        df = df[need].copy()
        for c in [Y_COL, ORIG_COL, TAUT_COL, FINAL_COL]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=[Y_COL, ORIG_COL, TAUT_COL, FINAL_COL])
        df = df.sort_values(ID_COL).reset_index(drop=True)
        dfs[seed] = df

    return dfs


def add_control(rows, detail, control, description, seed_to_mae):
    vals = [seed_to_mae[s] for s in SEEDS]
    text, m, sd = fmt(vals)

    rows.append({
        "Control": control,
        "Description": description,
        "Replicates": len(vals),
        "MAE (s)": text,
        "MAE_mean": m,
        "MAE_std": sd,
    })

    for s in SEEDS:
        detail.append({
            "Control": control,
            "Replicate": f"seed{s}",
            "MAE": seed_to_mae[s],
        })


def find_existing_shuffled_result():
    """
    Try to reuse the already reported shuffled-pairing result, e.g.
    25.275 ± 0.045 and optionally +0.220 ± 0.054.
    This does not recompute the wrong molecule-level random shuffle.
    """
    candidates = []

    for p in (ROOT / "gwn/final_paper_tables").rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".tex", ".csv", ".txt"}:
            continue

        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue

        low = text.lower()
        if "shuff" not in low:
            continue

        for line in text.splitlines():
            l = line.strip()
            if "shuff" not in l.lower():
                continue

            # match: 25.275 ± 0.045, 25.275 \pm 0.045, +0.220 ± 0.054
            m = re.search(r"([+-]?\d+\.\d+)\s*(?:±|\\pm|\$\\pm\$|\+/-)\s*(\d+\.\d+)", l)
            if m:
                candidates.append({
                    "file": str(p.relative_to(ROOT)),
                    "line": l,
                    "mean": float(m.group(1)),
                    "std": float(m.group(2)),
                })

    # Prefer realistic shuffled-pairing MAE around 25, not identity-destroyed 101.
    realistic = [c for c in candidates if 20 <= c["mean"] <= 35]
    increase = [c for c in candidates if 0 <= c["mean"] <= 5 and "minus" in c["line"].lower()]

    shuffled = realistic[0] if realistic else None
    delta = increase[0] if increase else None
    return shuffled, delta, candidates


def write_tex(out):
    with open(OUT_TEX, "w") as f:
        f.write("\\begin{table}[!htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Post-hoc same-seed dual-view fusion controls on the SMRT test set.}\n")
        f.write("\\label{tab:s26-same-seed-dual-view-controls}\n")
        f.write("\\sitableformat\n")
        f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llcl@{}}\n")
        f.write("\\toprule\n")
        f.write("Control & Description & Replicates & MAE (s) \\\\\n")
        f.write("\\midrule\n")
        for _, r in out.iterrows():
            f.write(
                f"{tex_escape(r['Control'])} & "
                f"{tex_escape(r['Description'])} & "
                f"{tex_escape(r['Replicates'])} & "
                f"{tex_escape(r['MAE (s)'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular*}\n")
        f.write("\\end{table}\n")


def main():
    dfs = load_seed_files()

    rows = []
    detail = []

    # 1. Original single view
    add_control(
        rows,
        detail,
        "Original single view",
        "Original branch only.",
        {s: mae(dfs[s][Y_COL].values, dfs[s][ORIG_COL].values) for s in SEEDS},
    )

    # 2. Same-seed original+original duplicate mean
    # This is exactly equal to original single view, but explicitly reports the same-seed same-view control.
    add_control(
        rows,
        detail,
        "Same-seed original+original mean control",
        "Duplicate same-seed original-view prediction averaged with itself.",
        {
            s: mae(
                dfs[s][Y_COL].values,
                0.5 * (dfs[s][ORIG_COL].values + dfs[s][ORIG_COL].values),
            )
            for s in SEEDS
        },
    )

    # 3. Tautomer single view
    add_control(
        rows,
        detail,
        "Strict tautomer single view",
        "Strict tautomer-canonical branch only.",
        {s: mae(dfs[s][Y_COL].values, dfs[s][TAUT_COL].values) for s in SEEDS},
    )

    # 4. Same-seed taut+taut duplicate mean
    add_control(
        rows,
        detail,
        "Same-seed tautomer+tautomer mean control",
        "Duplicate same-seed tautomer-view prediction averaged with itself.",
        {
            s: mae(
                dfs[s][Y_COL].values,
                0.5 * (dfs[s][TAUT_COL].values + dfs[s][TAUT_COL].values),
            )
            for s in SEEDS
        },
    )

    # 5. Same-seed original+tautomer paired mean fusion
    add_control(
        rows,
        detail,
        "Same-seed original+tautomer paired mean fusion",
        "Mean of original-view and tautomer-view predictions from the same seed.",
        {
            s: mae(
                dfs[s][Y_COL].values,
                0.5 * (dfs[s][ORIG_COL].values + dfs[s][TAUT_COL].values),
            )
            for s in SEEDS
        },
    )

    # 6. OOF Huber stacked fusion
    add_control(
        rows,
        detail,
        "OOF Huber stacked fusion",
        "Final no-leak OOF Huber prediction-level stacker.",
        {s: mae(dfs[s][Y_COL].values, dfs[s][FINAL_COL].values) for s in SEEDS},
    )

    # 7. Existing shuffled-pairing result, if already present in final_paper_tables
    shuffled, delta, candidates = find_existing_shuffled_result()
    if shuffled is not None:
        rows.append({
            "Control": "Shuffled tautomer pairing",
            "Description": "Previously reported control with molecule-wise original--tautomer pairing disrupted.",
            "Replicates": 5,
            "MAE (s)": f"{shuffled['mean']:.3f} ± {shuffled['std']:.3f}",
            "MAE_mean": shuffled["mean"],
            "MAE_std": shuffled["std"],
        })

    if delta is not None:
        rows.append({
            "Control": "Shuffle minus paired",
            "Description": "MAE increase after disrupting original--tautomer pairing.",
            "Replicates": 5,
            "MAE (s)": f"{delta['mean']:+.3f} ± {delta['std']:.3f}",
            "MAE_mean": delta["mean"],
            "MAE_std": delta["std"],
        })

    out = pd.DataFrame(rows)
    det = pd.DataFrame(detail)

    out.to_csv(OUT_CSV, index=False)
    det.to_csv(OUT_DETAIL, index=False)
    write_tex(out)

    print("===== Table S26 same-seed controls =====")
    print(out[["Control", "Description", "Replicates", "MAE (s)"]].to_string(index=False))

    print("\n===== detail =====")
    print(det.to_string(index=False))

    print("\n===== wrote =====")
    print(OUT_CSV.relative_to(ROOT))
    print(OUT_DETAIL.relative_to(ROOT))
    print(OUT_TEX.relative_to(ROOT))

    if shuffled is None:
        print("\n[NOTE] 没有自动从 gwn/final_paper_tables 里找到 25.xx 的 shuffled-pairing 表项。")
        print("       如果你已有 Table S12 的 tex/csv，把路径贴我，我再把它并入 S26。")
    else:
        print("\n===== shuffled source =====")
        print(shuffled["file"])
        print(shuffled["line"])


if __name__ == "__main__":
    main()

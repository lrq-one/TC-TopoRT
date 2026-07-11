from pathlib import Path
from collections import Counter
import re
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
except Exception as e:
    raise RuntimeError(
        "This script requires RDKit in the current environment. "
        "Please run it in your lrq_q environment."
    ) from e


ROOT = Path(".").resolve()
OUT_DIR = ROOT / "gwn/final_paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_STATS_CSV = OUT_DIR / "Table_S27_tautomer_change_statistics.csv"
OUT_STATS_TEX = OUT_DIR / "Table_S27_tautomer_change_statistics.tex"

OUT_TYPE_CSV = OUT_DIR / "Table_S28_tautomer_change_type_statistics.csv"
OUT_TYPE_TEX = OUT_DIR / "Table_S28_tautomer_change_type_statistics.tex"

OUT_EXAMPLES_CSV = OUT_DIR / "Table_S29_representative_tautomer_examples.csv"
OUT_EXAMPLES_TEX = OUT_DIR / "Table_S29_representative_tautomer_examples.tex"

OUT_DETAIL_CSV = OUT_DIR / "tautomer_change_detail_all_molecules.csv"


# ---------------------------------------------------------------------
# Robust path resolution
# ---------------------------------------------------------------------
def resolve_path(candidates):
    for p in candidates:
        p = ROOT / p
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find any of these files:\n" + "\n".join(str(ROOT / x) for x in candidates)
    )


PATHS = {
    "SMRT train": {
        "origin": resolve_path([
            "data/SMRT_train.csv",
            "gwn/data/SMRT_train.csv",
        ]),
        "taut": resolve_path([
            "data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv",
            "gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv",
        ]),
    },
    "SMRT test": {
        "origin": resolve_path([
            "data/SMRT_test.csv",
            "gwn/data/SMRT_test.csv",
        ]),
        "taut": resolve_path([
            "data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv",
            "gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv",
        ]),
    },
}


# ---------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def pick_smiles_col(df, mode):
    cols = list(df.columns)

    if mode == "origin":
        priorities = [
            "Orig_SMILES",
            "Original_SMILES",
            "origin_smiles",
            "orig_smiles",
            "SMILES",
            "smiles",
            "smile",
            "canonical_smiles",
            "Canonical_SMILES",
        ]
    else:
        priorities = [
            "Taut_SMILES",
            "Tautomer_SMILES",
            "taut_smiles",
            "tautomer_smiles",
            "strict_tautomer_smiles",
            "Strict_Tautomer_SMILES",
            "SMILES",
            "smiles",
            "smile",
            "canonical_smiles",
            "Canonical_SMILES",
        ]

    for c in priorities:
        if c in cols:
            return c

    for c in cols:
        nc = norm(c)
        if "smiles" in nc:
            return c

    raise RuntimeError(f"Cannot identify SMILES column for {mode}; columns={cols}")


def pick_optional_col(df, candidates):
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    for c in cols:
        nc = norm(c)
        for pat in candidates:
            if norm(pat) == nc:
                return c
    return None


# ---------------------------------------------------------------------
# RDKit helpers
# ---------------------------------------------------------------------
SMARTS = {
    "carbonyl": "[CX3]=[OX1]",
    "enol_like": "[OX2H][CX3]=[CX3]",
    "amide": "[NX3][CX3](=[OX1])",
    "imidic_acid_like": "[OX2H][CX3]=[NX2]",
    "imine": "[CX3]=[NX2]",
    "enamine_like": "[NX3][CX3]=[CX3]",
    "aromatic_nh": "[nH]",
    "aromatic_n": "[n]",
}

PATS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}


def mol_from_smiles(s):
    if pd.isna(s):
        return None
    try:
        return Chem.MolFromSmiles(str(s))
    except Exception:
        return None


def canon_smiles(s):
    mol = mol_from_smiles(s)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def mol_formula(mol):
    if mol is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


def count_pat(mol, name):
    if mol is None or PATS[name] is None:
        return 0
    try:
        return len(mol.GetSubstructMatches(PATS[name]))
    except Exception:
        return 0


def formal_charge_signature(mol):
    if mol is None:
        return None
    return tuple(sorted([a.GetFormalCharge() for a in mol.GetAtoms() if a.GetFormalCharge() != 0]))


def hetero_h_count(mol):
    if mol is None:
        return 0
    total = 0
    for a in mol.GetAtoms():
        if a.GetAtomicNum() in {7, 8, 15, 16}:
            total += int(a.GetTotalNumHs())
    return total


def aromatic_hetero_h_count(mol):
    if mol is None:
        return 0
    total = 0
    for a in mol.GetAtoms():
        if a.GetIsAromatic() and a.GetAtomicNum() in {7, 8, 16}:
            total += int(a.GetTotalNumHs())
    return total


def bond_signature(mol):
    if mol is None:
        return None
    sig = []
    for b in mol.GetBonds():
        a1 = b.GetBeginAtom().GetAtomicNum()
        a2 = b.GetEndAtom().GetAtomicNum()
        pair = tuple(sorted([a1, a2]))
        sig.append((pair[0], pair[1], str(b.GetBondType()), int(b.GetIsAromatic())))
    return tuple(sorted(sig))


def classify_change(orig_mol, taut_mol):
    """
    Rule-based, representation-level categories.
    These are not experimental tautomer-population labels.
    """
    if orig_mol is None or taut_mol is None:
        return "Invalid SMILES / parse issue"

    f1 = mol_formula(orig_mol)
    f2 = mol_formula(taut_mol)
    if f1 != f2:
        return "Formula not preserved / parse issue"

    c1 = {k: count_pat(orig_mol, k) for k in SMARTS}
    c2 = {k: count_pat(taut_mol, k) for k in SMARTS}

    # Ordered from more specific to more generic.
    if (
        c1["amide"] != c2["amide"]
        or c1["imidic_acid_like"] != c2["imidic_acid_like"]
    ):
        return "Amide/imidic-acid-like canonicalization"

    if (
        c1["carbonyl"] != c2["carbonyl"]
        or c1["enol_like"] != c2["enol_like"]
    ):
        return "Carbonyl/enol-like canonicalization"

    if (
        c1["imine"] != c2["imine"]
        or c1["enamine_like"] != c2["enamine_like"]
    ):
        return "Imine/enamine-like canonicalization"

    if (
        c1["aromatic_nh"] != c2["aromatic_nh"]
        or aromatic_hetero_h_count(orig_mol) != aromatic_hetero_h_count(taut_mol)
    ):
        return "Heteroaromatic proton relocation"

    if hetero_h_count(orig_mol) != hetero_h_count(taut_mol):
        return "Heteroatom proton relocation"

    if formal_charge_signature(orig_mol) != formal_charge_signature(taut_mol):
        return "Charge/protonation representation change"

    if bond_signature(orig_mol) != bond_signature(taut_mol):
        return "Bond-order/proton relocation"

    return "Other representation-level tautomer canonicalization"


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------
def latex_escape_text(s):
    s = str(s)
    repl = {
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


def latex_smiles(s):
    s = str(s)
    # SMILES generally do not contain braces; detokenize is safest for %, #, [, ], =, etc.
    s = s.replace("\\", "/")
    s = s.replace("{", "(").replace("}", ")")
    return r"\texttt{\detokenize{" + s + "}}"


def write_stats_tex(df):
    with open(OUT_STATS_TEX, "w") as f:
        f.write("\\begin{table}[!htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Strict tautomer-canonical representation change statistics for the SMRT data.}\n")
        f.write("\\label{tab:tautomer-change-statistics}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Total & Changed & Unchanged & Changed (\\%) & Formula preserved among changed & Invalid \\\\\n")
        f.write("\\midrule\n")
        for _, r in df.iterrows():
            f.write(
                f"{latex_escape_text(r['Dataset'])} & "
                f"{int(r['Total'])} & "
                f"{int(r['Changed'])} & "
                f"{int(r['Unchanged'])} & "
                f"{float(r['Changed (%)']):.2f} & "
                f"{int(r['Formula-preserved changed'])}/{int(r['Changed'])} & "
                f"{int(r['Invalid SMILES'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def write_type_tex(df):
    with open(OUT_TYPE_TEX, "w") as f:
        f.write("\\begin{table}[!htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Rule-based change-type summary for strict tautomer-canonical representation changes.}\n")
        f.write("\\label{tab:tautomer-change-type-statistics}\n")
        f.write("\\begin{tabular}{llrr}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Change type & Count & Among changed (\\%) \\\\\n")
        f.write("\\midrule\n")
        for _, r in df.iterrows():
            f.write(
                f"{latex_escape_text(r['Dataset'])} & "
                f"{latex_escape_text(r['Change type'])} & "
                f"{int(r['Count'])} & "
                f"{float(r['Among changed (%)']):.2f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def write_examples_tex(df):
    with open(OUT_EXAMPLES_TEX, "w") as f:
        f.write("\\begin{table}[!htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Representative examples of strict tautomer-canonical representation changes. "
                "The examples illustrate representation-level canonicalization and do not imply dominant solution-phase tautomer populations.}\n")
        f.write("\\label{tab:representative-tautomer-examples}\n")
        f.write("\\scriptsize\n")
        f.write("\\begin{tabular}{llllp{4.4cm}}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Formula & Change type & Original SMILES & Strict tautomer-canonical SMILES \\\\\n")
        f.write("\\midrule\n")
        for _, r in df.iterrows():
            f.write(
                f"{latex_escape_text(r['Dataset'])} & "
                f"{latex_escape_text(r['Formula'])} & "
                f"{latex_escape_text(r['Change type'])} & "
                f"{latex_smiles(r['Original SMILES'])} & "
                f"{latex_smiles(r['Strict tautomer-canonical SMILES'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------
def build_dataset_detail(dataset_name, origin_path, taut_path):
    odf = pd.read_csv(origin_path)
    tdf = pd.read_csv(taut_path)

    o_col = pick_smiles_col(odf, "origin")
    t_col = pick_smiles_col(tdf, "taut")

    if len(odf) != len(tdf):
        raise RuntimeError(
            f"{dataset_name}: row counts differ. origin={len(odf)}, taut={len(tdf)}. "
            "The strict tautomer files should be in origin order."
        )

    rt_col = pick_optional_col(odf, ["RT", "rt", "retention_time", "RetentionTime", "retention time"])
    id_col = pick_optional_col(odf, ["Source_Index", "source_index", "index", "ID", "id"])

    rows = []
    for i in range(len(odf)):
        orig = str(odf.iloc[i][o_col])
        taut = str(tdf.iloc[i][t_col])

        omol = mol_from_smiles(orig)
        tmol = mol_from_smiles(taut)

        orig_can = canon_smiles(orig)
        taut_can = canon_smiles(taut)

        invalid = int(omol is None or tmol is None)
        changed = bool(orig_can != taut_can) if invalid == 0 else False

        f1 = mol_formula(omol)
        f2 = mol_formula(tmol)
        formula_preserved = bool(f1 == f2) if invalid == 0 else False

        change_type = classify_change(omol, tmol) if changed else "Unchanged"

        row = {
            "Dataset": dataset_name,
            "Row": i,
            "ID": odf.iloc[i][id_col] if id_col else i,
            "RT": odf.iloc[i][rt_col] if rt_col else "",
            "Original SMILES": orig,
            "Strict tautomer-canonical SMILES": taut,
            "Original canonical SMILES": orig_can,
            "Strict tautomer canonical SMILES": taut_can,
            "Formula": f1 if f1 is not None else "",
            "Formula tautomer": f2 if f2 is not None else "",
            "Formula preserved": formula_preserved,
            "Changed": changed,
            "Invalid SMILES": invalid,
            "Change type": change_type,
            "Original SMILES length": len(orig),
            "Tautomer SMILES length": len(taut),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def choose_examples(detail, max_examples=12):
    changed = detail[
        (detail["Changed"] == True)
        & (detail["Invalid SMILES"] == 0)
        & (detail["Formula preserved"] == True)
    ].copy()

    if changed.empty:
        return changed

    # Prefer test-set examples, readable SMILES, and diverse change types.
    changed["dataset_priority"] = changed["Dataset"].map({"SMRT test": 0, "SMRT train": 1}).fillna(2)
    changed["length_sum"] = changed["Original SMILES length"] + changed["Tautomer SMILES length"]

    changed = changed.sort_values(
        ["dataset_priority", "Change type", "length_sum", "Row"],
        ascending=[True, True, True, True],
    )

    picked = []
    seen_types = set()

    # one per type first
    for _, r in changed.iterrows():
        t = r["Change type"]
        if t in seen_types:
            continue
        picked.append(r)
        seen_types.add(t)
        if len(picked) >= max_examples:
            break

    # fill remaining with short readable examples
    if len(picked) < max_examples:
        picked_keys = {(r["Dataset"], int(r["Row"])) for r in picked}
        rest = changed.sort_values(["dataset_priority", "length_sum", "Row"])
        for _, r in rest.iterrows():
            key = (r["Dataset"], int(r["Row"]))
            if key in picked_keys:
                continue
            picked.append(r)
            picked_keys.add(key)
            if len(picked) >= max_examples:
                break

    out = pd.DataFrame(picked)
    keep = [
        "Dataset",
        "ID",
        "RT",
        "Formula",
        "Change type",
        "Original SMILES",
        "Strict tautomer-canonical SMILES",
    ]
    return out[keep].reset_index(drop=True)


def main():
    print("===== input files =====")
    for name, pp in PATHS.items():
        print(name)
        print("  origin:", pp["origin"].relative_to(ROOT))
        print("  taut:  ", pp["taut"].relative_to(ROOT))

    details = []
    stats_rows = []
    type_rows = []

    for name, pp in PATHS.items():
        df = build_dataset_detail(name, pp["origin"], pp["taut"])
        details.append(df)

        total = len(df)
        invalid = int(df["Invalid SMILES"].sum())
        changed = int(df["Changed"].sum())
        unchanged = int(total - changed)
        formula_preserved_changed = int(
            df[(df["Changed"] == True) & (df["Formula preserved"] == True)].shape[0]
        )

        stats_rows.append({
            "Dataset": name,
            "Total": total,
            "Changed": changed,
            "Unchanged": unchanged,
            "Changed (%)": 100.0 * changed / total if total else np.nan,
            "Formula-preserved changed": formula_preserved_changed,
            "Invalid SMILES": invalid,
        })

        sub = df[df["Changed"] == True].copy()
        cnt = Counter(sub["Change type"].tolist())
        for change_type, n in cnt.most_common():
            type_rows.append({
                "Dataset": name,
                "Change type": change_type,
                "Count": int(n),
                "Among changed (%)": 100.0 * n / changed if changed else np.nan,
            })

    detail = pd.concat(details, ignore_index=True)
    stats = pd.DataFrame(stats_rows)
    types = pd.DataFrame(type_rows)
    examples = choose_examples(detail, max_examples=12)

    detail.to_csv(OUT_DETAIL_CSV, index=False)
    stats.to_csv(OUT_STATS_CSV, index=False)
    types.to_csv(OUT_TYPE_CSV, index=False)
    examples.to_csv(OUT_EXAMPLES_CSV, index=False)

    write_stats_tex(stats)
    write_type_tex(types)
    write_examples_tex(examples)

    print("\n===== Table S27 tautomer change statistics =====")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n===== Table S28 change-type statistics =====")
    print(types.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n===== Table S29 representative examples =====")
    show = examples.copy()
    for c in ["Original SMILES", "Strict tautomer-canonical SMILES"]:
        show[c] = show[c].astype(str).map(lambda x: x[:80] + ("..." if len(x) > 80 else ""))
    print(show.to_string(index=False))

    print("\n===== wrote =====")
    for p in [
        OUT_STATS_CSV, OUT_STATS_TEX,
        OUT_TYPE_CSV, OUT_TYPE_TEX,
        OUT_EXAMPLES_CSV, OUT_EXAMPLES_TEX,
        OUT_DETAIL_CSV,
    ]:
        print(p.relative_to(ROOT))

    print("\n===== SI wording reminder =====")
    print(
        "Use wording such as: the changed fraction reflects representation-level "
        "strict tautomer canonicalization under the RDKit rule set, not experimentally "
        "dominant solution-phase tautomer populations."
    )


if __name__ == "__main__":
    main()

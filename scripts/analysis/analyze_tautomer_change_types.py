#!/usr/bin/env python3
"""Summarize rule-based changes introduced by strict tautomer standardization.

The categories are representation-level labels used for the final SI. They do
not describe solution-phase populations, pH-specific microspecies, or ESI
tautomer equilibria.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


REPO_ROOT = Path(__file__).resolve().parents[2]

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
PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in SMARTS.items()}


def normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def pick_smiles_column(frame: pd.DataFrame, mode: str) -> str:
    if mode == "original":
        priorities = (
            "Orig_SMILES",
            "Original_SMILES",
            "origin_smiles",
            "orig_smiles",
            "SMILES",
            "smiles",
            "smile",
            "canonical_smiles",
            "Canonical_SMILES",
        )
    else:
        priorities = (
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
        )
    for column in priorities:
        if column in frame.columns:
            return column
    for column in frame.columns:
        if "smiles" in normalized_name(column):
            return column
    raise ValueError(f"Cannot identify the {mode} SMILES column: {list(frame.columns)}")


def pick_optional_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    normalized = {normalized_name(column): column for column in frame.columns}
    return next(
        (normalized[normalized_name(candidate)] for candidate in candidates
         if normalized_name(candidate) in normalized),
        None,
    )


def mol_from_smiles(value: object) -> Chem.Mol | None:
    if pd.isna(value):
        return None
    try:
        return Chem.MolFromSmiles(str(value))
    except Exception:
        return None


def canonical_smiles(value: object) -> str | None:
    molecule = mol_from_smiles(value)
    if molecule is None:
        return None
    try:
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def molecular_formula(molecule: Chem.Mol | None) -> str | None:
    if molecule is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(molecule)
    except Exception:
        return None


def pattern_count(molecule: Chem.Mol | None, name: str) -> int:
    pattern = PATTERNS[name]
    if molecule is None or pattern is None:
        return 0
    try:
        return len(molecule.GetSubstructMatches(pattern))
    except Exception:
        return 0


def formal_charge_signature(molecule: Chem.Mol | None) -> tuple[int, ...] | None:
    if molecule is None:
        return None
    return tuple(
        sorted(
            atom.GetFormalCharge()
            for atom in molecule.GetAtoms()
            if atom.GetFormalCharge() != 0
        )
    )


def hetero_h_count(molecule: Chem.Mol | None) -> int:
    if molecule is None:
        return 0
    return sum(
        int(atom.GetTotalNumHs())
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() in {7, 8, 15, 16}
    )


def aromatic_hetero_h_count(molecule: Chem.Mol | None) -> int:
    if molecule is None:
        return 0
    return sum(
        int(atom.GetTotalNumHs())
        for atom in molecule.GetAtoms()
        if atom.GetIsAromatic() and atom.GetAtomicNum() in {7, 8, 16}
    )


def bond_signature(
    molecule: Chem.Mol | None,
) -> tuple[tuple[int, int, str, int], ...] | None:
    if molecule is None:
        return None
    signature = []
    for bond in molecule.GetBonds():
        atom_pair = sorted(
            (bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum())
        )
        signature.append(
            (atom_pair[0], atom_pair[1], str(bond.GetBondType()), int(bond.GetIsAromatic()))
        )
    return tuple(sorted(signature))


def classify_change(original: Chem.Mol | None, standardized: Chem.Mol | None) -> str:
    """Apply the ordered rule set used for the final SI change categories."""
    if original is None or standardized is None:
        return "Invalid SMILES / parse issue"
    if molecular_formula(original) != molecular_formula(standardized):
        return "Formula not preserved / parse issue"

    original_counts = {name: pattern_count(original, name) for name in SMARTS}
    standardized_counts = {name: pattern_count(standardized, name) for name in SMARTS}

    if (
        original_counts["amide"] != standardized_counts["amide"]
        or original_counts["imidic_acid_like"]
        != standardized_counts["imidic_acid_like"]
    ):
        return "Amide/imidic-acid-like canonicalization"
    if (
        original_counts["carbonyl"] != standardized_counts["carbonyl"]
        or original_counts["enol_like"] != standardized_counts["enol_like"]
    ):
        return "Carbonyl/enol-like canonicalization"
    if (
        original_counts["imine"] != standardized_counts["imine"]
        or original_counts["enamine_like"] != standardized_counts["enamine_like"]
    ):
        return "Imine/enamine-like canonicalization"
    if (
        original_counts["aromatic_nh"] != standardized_counts["aromatic_nh"]
        or aromatic_hetero_h_count(original) != aromatic_hetero_h_count(standardized)
    ):
        return "Heteroaromatic proton relocation"
    if hetero_h_count(original) != hetero_h_count(standardized):
        return "Heteroatom proton relocation"
    if formal_charge_signature(original) != formal_charge_signature(standardized):
        return "Charge/protonation representation change"
    if bond_signature(original) != bond_signature(standardized):
        return "Bond-order/proton relocation"
    return "Other representation-level tautomer canonicalization"


def build_dataset_detail(
    dataset_name: str, original_path: Path, standardized_path: Path
) -> pd.DataFrame:
    original_frame = pd.read_csv(original_path)
    standardized_frame = pd.read_csv(standardized_path)
    if len(original_frame) != len(standardized_frame):
        raise ValueError(
            f"{dataset_name}: paired row counts differ "
            f"({len(original_frame)} versus {len(standardized_frame)})."
        )

    original_column = pick_smiles_column(original_frame, "original")
    standardized_column = pick_smiles_column(standardized_frame, "standardized")
    rt_column = pick_optional_column(
        original_frame, ("RT", "rt", "retention_time", "RetentionTime", "retention time")
    )
    id_column = pick_optional_column(
        original_frame, ("Source_Index", "source_index", "index", "ID", "id")
    )

    rows = []
    for row_index in range(len(original_frame)):
        original_text = str(original_frame.iloc[row_index][original_column])
        standardized_text = str(standardized_frame.iloc[row_index][standardized_column])
        original_molecule = mol_from_smiles(original_text)
        standardized_molecule = mol_from_smiles(standardized_text)
        original_canonical = canonical_smiles(original_text)
        standardized_canonical = canonical_smiles(standardized_text)
        invalid = int(original_molecule is None or standardized_molecule is None)
        changed = bool(original_canonical != standardized_canonical) if not invalid else False
        original_formula = molecular_formula(original_molecule)
        standardized_formula = molecular_formula(standardized_molecule)
        formula_preserved = (
            bool(original_formula == standardized_formula) if not invalid else False
        )
        rows.append(
            {
                "Dataset": dataset_name,
                "Row": row_index,
                "ID": original_frame.iloc[row_index][id_column]
                if id_column
                else row_index,
                "RT": original_frame.iloc[row_index][rt_column] if rt_column else "",
                "Original SMILES": original_text,
                "Strict tautomer-canonical SMILES": standardized_text,
                "Original canonical SMILES": original_canonical,
                "Strict tautomer canonical SMILES": standardized_canonical,
                "Formula": original_formula if original_formula is not None else "",
                "Formula tautomer": (
                    standardized_formula if standardized_formula is not None else ""
                ),
                "Formula preserved": formula_preserved,
                "Changed": changed,
                "Invalid SMILES": invalid,
                "Change type": (
                    classify_change(original_molecule, standardized_molecule)
                    if changed
                    else "Unchanged"
                ),
                "Original SMILES length": len(original_text),
                "Tautomer SMILES length": len(standardized_text),
            }
        )
    return pd.DataFrame(rows)


def choose_examples(detail: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    changed = detail[
        detail["Changed"]
        & (detail["Invalid SMILES"] == 0)
        & detail["Formula preserved"]
    ].copy()
    if changed.empty:
        return changed

    changed["dataset_priority"] = (
        changed["Dataset"].map({"SMRT test": 0, "SMRT train": 1}).fillna(2)
    )
    changed["length_sum"] = (
        changed["Original SMILES length"] + changed["Tautomer SMILES length"]
    )
    changed = changed.sort_values(
        ["dataset_priority", "Change type", "length_sum", "Row"],
        ascending=[True, True, True, True],
    )

    selected = []
    seen_types: set[str] = set()
    for _, row in changed.iterrows():
        change_type = str(row["Change type"])
        if change_type not in seen_types:
            selected.append(row)
            seen_types.add(change_type)
        if len(selected) >= max_examples:
            break

    if len(selected) < max_examples:
        selected_keys = {(row["Dataset"], int(row["Row"])) for row in selected}
        for _, row in changed.sort_values(
            ["dataset_priority", "length_sum", "Row"]
        ).iterrows():
            key = (row["Dataset"], int(row["Row"]))
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            if len(selected) >= max_examples:
                break

    columns = [
        "Dataset",
        "ID",
        "RT",
        "Formula",
        "Change type",
        "Original SMILES",
        "Strict tautomer-canonical SMILES",
    ]
    return pd.DataFrame(selected)[columns].reset_index(drop=True)


def latex_escape(value: object) -> str:
    text = str(value)
    for source, replacement in {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }.items():
        text = text.replace(source, replacement)
    return text


def latex_smiles(value: object) -> str:
    text = str(value).replace("\\", "/").replace("{", "(").replace("}", ")")
    return r"\texttt{\detokenize{" + text + "}}"


def write_stats_tex(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Strict tautomer-canonical representation change statistics for the SMRT data.}",
        r"\label{tab:tautomer-change-statistics}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Total & Changed & Unchanged & Changed (\%) & Formula preserved among changed & Invalid \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{latex_escape(row['Dataset'])} & {int(row['Total'])} & "
            f"{int(row['Changed'])} & {int(row['Unchanged'])} & "
            f"{float(row['Changed (%)']):.2f} & "
            f"{int(row['Formula-preserved changed'])}/{int(row['Changed'])} & "
            f"{int(row['Invalid SMILES'])} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_types_tex(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Rule-based change-type summary for strict tautomer-canonical representation changes.}",
        r"\label{tab:tautomer-change-type-statistics}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Dataset & Change type & Count & Among changed (\%) \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{latex_escape(row['Dataset'])} & {latex_escape(row['Change type'])} & "
            f"{int(row['Count'])} & {float(row['Among changed (%)']):.2f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_examples_tex(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{Representative examples of strict tautomer-canonical "
            r"representation changes. The examples illustrate representation-level "
            r"canonicalization and do not imply dominant solution-phase tautomer populations.}"
        ),
        r"\label{tab:representative-tautomer-examples}",
        r"\scriptsize",
        r"\begin{tabular}{llllp{4.4cm}}",
        r"\toprule",
        r"Dataset & Formula & Change type & Original SMILES & Strict tautomer-canonical SMILES \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{latex_escape(row['Dataset'])} & {latex_escape(row['Formula'])} & "
            f"{latex_escape(row['Change type'])} & "
            f"{latex_smiles(row['Original SMILES'])} & "
            f"{latex_smiles(row['Strict tautomer-canonical SMILES'])} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-train",
        type=Path,
        default=REPO_ROOT / "gwn/data/SMRT_train.csv",
    )
    parser.add_argument(
        "--original-test",
        type=Path,
        default=REPO_ROOT / "gwn/data/SMRT_test.csv",
    )
    parser.add_argument(
        "--standardized-train",
        type=Path,
        default=(
            REPO_ROOT
            / "gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv"
        ),
    )
    parser.add_argument(
        "--standardized-test",
        type=Path,
        default=(
            REPO_ROOT
            / "gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/analysis/tautomer_change_types",
    )
    parser.add_argument("--max-examples", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_examples < 1:
        raise ValueError("--max-examples must be positive.")

    inputs = (
        ("SMRT train", args.original_train, args.standardized_train),
        ("SMRT test", args.original_test, args.standardized_test),
    )
    details = []
    statistics = []
    type_statistics = []
    for dataset_name, original_path, standardized_path in inputs:
        detail = build_dataset_detail(dataset_name, original_path, standardized_path)
        details.append(detail)
        total = len(detail)
        changed = int(detail["Changed"].sum())
        statistics.append(
            {
                "Dataset": dataset_name,
                "Total": total,
                "Changed": changed,
                "Unchanged": total - changed,
                "Changed (%)": 100.0 * changed / total if total else np.nan,
                "Formula-preserved changed": int(
                    detail[detail["Changed"] & detail["Formula preserved"]].shape[0]
                ),
                "Invalid SMILES": int(detail["Invalid SMILES"].sum()),
            }
        )
        counter = Counter(detail.loc[detail["Changed"], "Change type"].tolist())
        for change_type, count in counter.most_common():
            type_statistics.append(
                {
                    "Dataset": dataset_name,
                    "Change type": change_type,
                    "Count": int(count),
                    "Among changed (%)": (
                        100.0 * count / changed if changed else np.nan
                    ),
                }
            )

    detail_frame = pd.concat(details, ignore_index=True)
    statistics_frame = pd.DataFrame(statistics)
    types_frame = pd.DataFrame(type_statistics)
    examples_frame = choose_examples(detail_frame, args.max_examples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "detail": args.output_dir / "tautomer_change_detail_all_molecules.csv",
        "statistics_csv": args.output_dir / "Table_S27_tautomer_change_statistics.csv",
        "statistics_tex": args.output_dir / "Table_S27_tautomer_change_statistics.tex",
        "types_csv": args.output_dir / "Table_S28_tautomer_change_type_statistics.csv",
        "types_tex": args.output_dir / "Table_S28_tautomer_change_type_statistics.tex",
        "examples_csv": args.output_dir / "Table_S29_representative_tautomer_examples.csv",
        "examples_tex": args.output_dir / "Table_S29_representative_tautomer_examples.tex",
    }
    detail_frame.to_csv(outputs["detail"], index=False)
    statistics_frame.to_csv(outputs["statistics_csv"], index=False)
    types_frame.to_csv(outputs["types_csv"], index=False)
    examples_frame.to_csv(outputs["examples_csv"], index=False)
    write_stats_tex(statistics_frame, outputs["statistics_tex"])
    write_types_tex(types_frame, outputs["types_tex"])
    write_examples_tex(examples_frame, outputs["examples_tex"])

    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


RDLogger.DisableLog("rdApp.*")

SMARTS = {
    "carbonyl": "[CX3]=[OX1]",
    "enol_like": "[OX2H][CX3]=[CX3]",
    "amide": "[NX3][CX3](=[OX1])",
    "imidic_acid_like": "[OX2H][CX3]=[NX2]",
    "imine": "[CX3]=[NX2]",
    "enamine_like": "[NX3][CX3]=[CX3]",
    "aromatic_nh": "[nH]",
}
PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in SMARTS.items()}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(column).strip().lower() for column in out.columns]
    if "smile" in out.columns and "smiles" not in out.columns:
        out = out.rename(columns={"smile": "smiles"})
    return out


def canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def formula(mol: Chem.Mol) -> str:
    return rdMolDescriptors.CalcMolFormula(mol)


def pattern_count(mol: Chem.Mol, name: str) -> int:
    pattern = PATTERNS[name]
    return len(mol.GetSubstructMatches(pattern)) if pattern is not None else 0


def hetero_h_count(mol: Chem.Mol) -> int:
    return int(
        sum(
            atom.GetTotalNumHs()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() in {7, 8, 15, 16}
        )
    )


def aromatic_hetero_h_count(mol: Chem.Mol) -> int:
    return int(
        sum(
            atom.GetTotalNumHs()
            for atom in mol.GetAtoms()
            if atom.GetIsAromatic() and atom.GetAtomicNum() in {7, 8, 16}
        )
    )


def charge_signature(mol: Chem.Mol) -> tuple[int, ...]:
    return tuple(
        sorted(
            atom.GetFormalCharge()
            for atom in mol.GetAtoms()
            if atom.GetFormalCharge() != 0
        )
    )


def bond_signature(mol: Chem.Mol) -> tuple[tuple[int, int, str, int], ...]:
    signature = []
    for bond in mol.GetBonds():
        atomic_numbers = sorted(
            [
                bond.GetBeginAtom().GetAtomicNum(),
                bond.GetEndAtom().GetAtomicNum(),
            ]
        )
        signature.append(
            (
                atomic_numbers[0],
                atomic_numbers[1],
                str(bond.GetBondType()),
                int(bond.GetIsAromatic()),
            )
        )
    return tuple(sorted(signature))


def classify_change(origin: Chem.Mol, tautomer: Chem.Mol) -> str:
    counts_origin = {name: pattern_count(origin, name) for name in SMARTS}
    counts_tautomer = {name: pattern_count(tautomer, name) for name in SMARTS}

    if (
        counts_origin["amide"] != counts_tautomer["amide"]
        or counts_origin["imidic_acid_like"]
        != counts_tautomer["imidic_acid_like"]
    ):
        return "Amide/imidic-acid-like canonicalization"
    if (
        counts_origin["carbonyl"] != counts_tautomer["carbonyl"]
        or counts_origin["enol_like"] != counts_tautomer["enol_like"]
    ):
        return "Carbonyl/enol-like canonicalization"
    if (
        counts_origin["imine"] != counts_tautomer["imine"]
        or counts_origin["enamine_like"] != counts_tautomer["enamine_like"]
    ):
        return "Imine/enamine-like canonicalization"
    if (
        counts_origin["aromatic_nh"] != counts_tautomer["aromatic_nh"]
        or aromatic_hetero_h_count(origin)
        != aromatic_hetero_h_count(tautomer)
    ):
        return "Heteroaromatic proton relocation"
    if hetero_h_count(origin) != hetero_h_count(tautomer):
        return "Heteroatom proton relocation"
    if charge_signature(origin) != charge_signature(tautomer):
        return "Charge/protonation representation change"
    if bond_signature(origin) != bond_signature(tautomer):
        return "Bond-order/proton relocation"
    return "Other representation-level tautomer canonicalization"


def load_aligned(
    dataset: str,
    origin_path: Path,
    tautomer_path: Path,
) -> pd.DataFrame:
    origin = normalize(pd.read_csv(origin_path, engine="python"))
    tautomer = normalize(pd.read_csv(tautomer_path, engine="python"))

    for path, frame in [(origin_path, origin), (tautomer_path, tautomer)]:
        if not {"smiles", "rt"}.issubset(frame.columns):
            raise ValueError(f"{path}: expected smile(s) and rt columns")

    origin["rt"] = pd.to_numeric(origin["rt"], errors="raise")
    tautomer["rt"] = pd.to_numeric(tautomer["rt"], errors="raise")
    origin = origin.loc[origin["rt"] > 300.0].copy()
    tautomer = tautomer.loc[tautomer["rt"] > 300.0].copy()
    if len(origin) != len(tautomer):
        raise RuntimeError(
            f"{dataset}: retained row counts differ: "
            f"origin={len(origin)}, tautomer={len(tautomer)}"
        )

    rows = []
    for retained_row, ((_, o_row), (_, t_row)) in enumerate(
        zip(origin.iterrows(), tautomer.iterrows())
    ):
        o_smiles = str(o_row["smiles"])
        t_smiles = str(t_row["smiles"])
        o_mol = Chem.MolFromSmiles(o_smiles)
        t_mol = Chem.MolFromSmiles(t_smiles)
        if o_mol is None or t_mol is None:
            continue
        if not np.isclose(float(o_row["rt"]), float(t_row["rt"]), atol=1e-8):
            raise RuntimeError(f"{dataset}: RT mismatch at retained row {retained_row}")

        changed = canonical(o_mol) != canonical(t_mol)
        formula_preserved = formula(o_mol) == formula(t_mol)
        rows.append(
            {
                "dataset": dataset,
                "retained_row": retained_row,
                "rt": float(o_row["rt"]),
                "original_smiles": o_smiles,
                "strict_tautomer_smiles": t_smiles,
                "formula": formula(o_mol),
                "changed": changed,
                "formula_preserved": formula_preserved,
                "change_type": (
                    classify_change(o_mol, t_mol) if changed else "Unchanged"
                ),
            }
        )

    detail = pd.DataFrame(rows)
    if len(detail) == 0:
        raise RuntimeError(f"{dataset}: no retained aligned molecules")
    return detail


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild strict-tautomer representation-change counts and "
            "rule-based change-type statistics for retained SMRT data."
        )
    )
    parser.add_argument("--origin_train", default="gwn/data/SMRT_train.csv")
    parser.add_argument("--origin_test", default="gwn/data/SMRT_test.csv")
    parser.add_argument(
        "--tautomer_train",
        default=(
            "gwn/data_taut_strict_origin_order/"
            "SMRT_train_tautomer_strict.csv"
        ),
    )
    parser.add_argument(
        "--tautomer_test",
        default=(
            "gwn/data_taut_strict_origin_order/"
            "SMRT_test_tautomer_strict.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/results/paper_tables/tautomer_changes",
    )
    parser.add_argument(
        "--verify_paper_counts",
        type=int,
        choices=[0, 1],
        default=1,
    )
    args = parser.parse_args()

    detail = pd.concat(
        [
            load_aligned(
                "SMRT train",
                Path(args.origin_train),
                Path(args.tautomer_train),
            ),
            load_aligned(
                "SMRT test",
                Path(args.origin_test),
                Path(args.tautomer_test),
            ),
        ],
        ignore_index=True,
    )

    summary_rows = []
    type_rows = []
    for dataset, subset in detail.groupby("dataset", sort=False):
        total = int(len(subset))
        changed = int(subset["changed"].sum())
        preserved = int((subset["changed"] & subset["formula_preserved"]).sum())
        summary_rows.append(
            {
                "dataset": dataset,
                "total": total,
                "changed": changed,
                "unchanged": total - changed,
                "changed_percent": 100.0 * changed / total,
                "formula_preserved_changed": preserved,
                "invalid_smiles": 0,
            }
        )

        counts = Counter(subset.loc[subset["changed"], "change_type"].astype(str))
        for change_type, count in counts.most_common():
            type_rows.append(
                {
                    "dataset": dataset,
                    "change_type": change_type,
                    "count": int(count),
                    "among_changed_percent": 100.0 * count / changed,
                }
            )

    summary = pd.DataFrame(summary_rows)
    change_types = pd.DataFrame(type_rows)

    if args.verify_paper_counts:
        expected = {
            "SMRT train": (70182, 37724),
            "SMRT test": (7798, 4242),
        }
        observed = {
            str(row["dataset"]): (int(row["total"]), int(row["changed"]))
            for _, row in summary.iterrows()
        }
        if observed != expected:
            raise RuntimeError(
                f"Retained/changed counts differ from the paper: {observed}"
            )
        if not np.array_equal(
            summary["formula_preserved_changed"].to_numpy(dtype=int),
            summary["changed"].to_numpy(dtype=int),
        ):
            raise RuntimeError(
                "At least one changed representation did not preserve formula"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "tautomer_change_detail.csv", index=False)
    summary.to_csv(out_dir / "tautomer_change_statistics.csv", index=False)
    change_types.to_csv(
        out_dir / "tautomer_change_type_statistics.csv",
        index=False,
    )

    print("\nRepresentation-change statistics")
    print(summary.to_string(index=False))
    print("\nRule-based change types")
    print(change_types.to_string(index=False))
    print(f"\nSaved to {out_dir}")
    print(
        "\nThese categories describe deterministic representation-level "
        "canonicalization, not dominant solution-phase tautomer populations."
    )


if __name__ == "__main__":
    main()

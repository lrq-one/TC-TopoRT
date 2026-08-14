#!/usr/bin/env python3
"""Audit paired SMRT views and representation collisions after standardization."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "train_records": 70182,
    "test_records": 7798,
    "train_changed": 37724,
    "test_changed": 4242,
    "distinct_standardized_structures": 77973,
    "collision_groups": 7,
    "collision_records": 14,
    "cross_split_collision_groups": 3,
}


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.astype(str).str.lower().str.startswith("unnamed")].copy()


def find_column(frame: pd.DataFrame, aliases: tuple[str, ...], label: str) -> str:
    names = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        if alias in names:
            return names[alias]
    raise ValueError(f"Cannot find {label}; columns={list(frame.columns)}")


@lru_cache(maxsize=200000)
def molecular_facts(smiles: str) -> tuple[bool, str, str, int]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False, "", "", -1
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    return True, canonical, formula, charge


def paired_split(split: str, original_path: Path, standardized_path: Path) -> pd.DataFrame:
    original = clean(pd.read_csv(original_path))
    standardized = clean(pd.read_csv(standardized_path))
    if len(original) != len(standardized):
        raise ValueError(
            f"{split}: paired row-count mismatch ({len(original)} vs {len(standardized)})."
        )
    original_smiles = find_column(original, ("smile", "smiles"), "original SMILES")
    standardized_smiles = find_column(
        standardized, ("smile", "smiles"), "standardized SMILES"
    )
    original_rt = pd.to_numeric(
        original[find_column(original, ("rt", "retention_time"), "original RT")],
        errors="raise",
    ).to_numpy(float)
    standardized_rt = pd.to_numeric(
        standardized[
            find_column(standardized, ("rt", "retention_time"), "standardized RT")
        ],
        errors="raise",
    ).to_numpy(float)
    if not np.allclose(original_rt, standardized_rt, atol=1e-8, rtol=0.0):
        raise ValueError(f"{split}: RT labels or row order differ between paired views.")

    rows = []
    for index, (original_text, standardized_text, rt) in enumerate(
        zip(
            original[original_smiles].astype(str),
            standardized[standardized_smiles].astype(str),
            original_rt,
        )
    ):
        original_ok, original_canonical, original_formula, original_charge = molecular_facts(
            original_text
        )
        standardized_ok, standardized_canonical, standardized_formula, standardized_charge = (
            molecular_facts(standardized_text)
        )
        valid = original_ok and standardized_ok
        rows.append(
            {
                "split": split,
                "row_index": index,
                "rt": float(rt),
                "original_smiles": original_text,
                "standardized_smiles": standardized_text,
                "original_canonical_smiles": original_canonical,
                "standardized_canonical_smiles": standardized_canonical,
                "parse_success": valid,
                "tautomer_changed": valid and original_canonical != standardized_canonical,
                "formula_preserved": valid and original_formula == standardized_formula,
                "formal_charge_preserved": valid and original_charge == standardized_charge,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-train", type=Path, default=REPO_ROOT / "gwn/data/SMRT_train.csv")
    parser.add_argument("--original-test", type=Path, default=REPO_ROOT / "gwn/data/SMRT_test.csv")
    parser.add_argument(
        "--standardized-train",
        type=Path,
        default=REPO_ROOT / "gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv",
    )
    parser.add_argument(
        "--standardized-test",
        type=Path,
        default=REPO_ROOT / "gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "artifacts/analysis/tautomer_collisions"
    )
    parser.add_argument("--verify-paper-scope", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train = paired_split("train", args.original_train, args.standardized_train)
    test = paired_split("test", args.original_test, args.standardized_test)
    audit = pd.concat([train, test], ignore_index=True)
    valid = audit[audit["parse_success"]].copy()

    group_sizes = valid.groupby("standardized_canonical_smiles").agg(
        n_records=("row_index", "size"),
        n_original_structures=("original_canonical_smiles", "nunique"),
    )
    collision_keys = group_sizes.index[group_sizes["n_original_structures"] >= 2]
    collision_members = valid[valid["standardized_canonical_smiles"].isin(collision_keys)].copy()
    collision_groups = []
    for index, (key, group) in enumerate(
        collision_members.groupby("standardized_canonical_smiles", sort=True), start=1
    ):
        rt_range = float(group["rt"].max() - group["rt"].min())
        collision_groups.append(
            {
                "collision_group_id": f"C{index:04d}",
                "standardized_canonical_smiles": key,
                "n_records": int(len(group)),
                "n_unique_original_structures": int(group["original_canonical_smiles"].nunique()),
                "splits_present": ";".join(sorted(group["split"].unique())),
                "cross_split_collision": int(group["split"].nunique()) > 1,
                "rt_range_seconds": rt_range,
                "rt_range_gt_10_seconds": rt_range > 10.0,
                "rt_range_gt_30_seconds": rt_range > 30.0,
                "rt_range_gt_60_seconds": rt_range > 60.0,
            }
        )
    group_table = pd.DataFrame(collision_groups)
    summary = {
        "train_records": int(len(train)),
        "test_records": int(len(test)),
        "train_changed": int(train["tautomer_changed"].sum()),
        "test_changed": int(test["tautomer_changed"].sum()),
        "train_changed_percent": 100.0 * float(train["tautomer_changed"].mean()),
        "test_changed_percent": 100.0 * float(test["tautomer_changed"].mean()),
        "distinct_standardized_structures": int(valid["standardized_canonical_smiles"].nunique()),
        "collision_groups": int(len(group_table)),
        "collision_records": int(len(collision_members)),
        "cross_split_collision_groups": int(group_table["cross_split_collision"].sum()),
        "representation_scope_note": (
            "Standardization is a representation-level normalization; these files do not "
            "encode pH-specific microspecies, solution populations, or ESI equilibria."
        ),
    }
    if args.verify_paper_scope:
        differences = {key: (summary[key], value) for key, value in EXPECTED.items() if summary[key] != value}
        if differences:
            raise AssertionError(f"Collision audit differs from locked paper values: {differences}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "paired_representation_audit.csv", index=False)
    group_table.to_csv(args.output_dir / "collision_groups.csv", index=False)
    collision_members.to_csv(args.output_dir / "collision_members.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


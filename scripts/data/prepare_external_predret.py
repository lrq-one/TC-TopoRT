#!/usr/bin/env python3
"""Prepare aligned original/standardized tables for the ten external datasets."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[2]
TAUTOMER = rdMolStandardize.TautomerEnumerator()
try:
    TAUTOMER.SetMaxTautomers(128)
    TAUTOMER.SetMaxTransforms(128)
except Exception:
    pass


@lru_cache(maxsize=100000)
def prepare_structure(smiles: str) -> dict:
    text = str(smiles)
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return {
            "canonical": text, "tautomer": text, "formula": "", "inchikey": "",
            "changed": 0, "fallback": 1, "reason": "parse_failed",
        }
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    inchikey = Chem.MolToInchiKey(mol)
    try:
        standardized_mol = TAUTOMER.Canonicalize(mol)
        standardized = Chem.MolToSmiles(
            standardized_mol, canonical=True, isomericSmiles=True
        )
        if rdMolDescriptors.CalcMolFormula(standardized_mol) != formula or (
            standardized_mol.GetNumHeavyAtoms() != mol.GetNumHeavyAtoms()
        ):
            raise ValueError("formula_or_heavy_atom_change")
        return {
            "canonical": canonical,
            "tautomer": standardized if standardized != canonical else text,
            "formula": formula,
            "inchikey": inchikey,
            "changed": int(standardized != canonical),
            "fallback": 0,
            "reason": "ok",
        }
    except Exception as error:
        return {
            "canonical": canonical, "tautomer": text, "formula": formula,
            "inchikey": inchikey, "changed": 0, "fallback": 1,
            "reason": f"exception:{type(error).__name__}",
        }


def find_column(frame: pd.DataFrame, aliases: tuple[str, ...], label: str) -> str:
    names = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        if alias in names:
            return names[alias]
    raise ValueError(f"Cannot find {label}; columns={list(frame.columns)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Combined CSV with dataset_name, SMILES, and experimental RT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "gwn/paper_analysis_stage4_external",
        help="Local ignored directory used by the final transfer entry points.",
    )
    parser.add_argument(
        "--smrt-train",
        type=Path,
        help="Optional retained SMRT train CSV for exact canonical-structure overlap audit.",
    )
    parser.add_argument("--verify-dataset-counts", action="store_true")
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    dataset_col = find_column(source, ("dataset_name", "dataset"), "dataset name")
    smiles_col = find_column(source, ("smiles", "smile"), "SMILES")
    rt_col = find_column(source, ("rt", "retention_time", "experimental_rt"), "RT")
    source[rt_col] = pd.to_numeric(source[rt_col], errors="raise")

    smrt_structures: set[str] = set()
    if args.smrt_train:
        smrt = pd.read_csv(args.smrt_train)
        smrt_smiles = find_column(smrt, ("smiles", "smile"), "SMRT SMILES")
        smrt_structures = {
            prepare_structure(value)["canonical"] for value in smrt[smrt_smiles].astype(str)
        }

    rows = []
    for index, row in source.reset_index(drop=True).iterrows():
        facts = prepare_structure(str(row[smiles_col]))
        dataset_name = str(row[dataset_col])
        record_id = str(row.get("record_id", f"{dataset_name}_{index}"))
        rows.append(
            {
                "stage4_index": index,
                "dataset_group": "predret10",
                "dataset_name": dataset_name,
                "source_file": str(args.input),
                "source_row": index,
                "record_id": record_id,
                "name": row.get("name", ""),
                "origin_smiles": str(row[smiles_col]),
                "taut_smiles": facts["tautomer"],
                "rt": float(row[rt_col]),
                "formula": facts["formula"],
                "inchikey": facts["inchikey"],
                "canonical_smiles": facts["canonical"],
                "taut_changed": facts["changed"],
                "taut_fallback": facts["fallback"],
                "taut_reason": facts["reason"],
                "smrt_exact_overlap": int(facts["canonical"] in smrt_structures),
            }
        )
    metadata = pd.DataFrame(rows)

    if args.verify_dataset_counts:
        expected = pd.read_csv(REPO_ROOT / "configs/external_datasets.csv").set_index("dataset_name")["n"]
        actual = metadata.groupby("dataset_name").size()
        if expected.to_dict() != actual.to_dict():
            raise AssertionError(
                f"External dataset counts differ. Expected {expected.to_dict()}, found {actual.to_dict()}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output_dir / "external_predret10_stage4_meta.csv", index=False)
    pd.DataFrame({"smile": metadata["origin_smiles"], "rt": 999.0}).to_csv(
        args.output_dir / "temp_external_predret10_origin.csv", index=False
    )
    pd.DataFrame({"smile": metadata["taut_smiles"], "rt": 999.0}).to_csv(
        args.output_dir / "temp_external_predret10_taut.csv", index=False
    )
    metadata[["stage4_index", "dataset_name", "origin_smiles", "taut_smiles", "taut_changed", "taut_fallback", "taut_reason"]].to_csv(
        args.output_dir / "external_predret10_tautomer_audit.csv", index=False
    )
    print(metadata.groupby("dataset_name").size().to_string())
    print(f"Saved aligned transfer inputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Audit paired SMRT view alignment, split overlap, and optional OOF coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[2]


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.astype(str).str.lower().str.startswith("unnamed")].copy()


def column(frame: pd.DataFrame, aliases: tuple[str, ...], label: str) -> str:
    names = {str(item).strip().lower(): item for item in frame.columns}
    for alias in aliases:
        if alias in names:
            return names[alias]
    raise ValueError(f"Cannot find {label}; columns={list(frame.columns)}")


def inchikey(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToInchiKey(mol) if mol is not None else ""


def check_pair(original_path: Path, standardized_path: Path, split: str) -> tuple[dict, set[str]]:
    original = clean(pd.read_csv(original_path))
    standardized = clean(pd.read_csv(standardized_path))
    original_smiles = column(original, ("smile", "smiles"), "original SMILES")
    standardized_smiles = column(standardized, ("smile", "smiles"), "standardized SMILES")
    original_rt = pd.to_numeric(original[column(original, ("rt",), "original RT")], errors="raise")
    standardized_rt = pd.to_numeric(
        standardized[column(standardized, ("rt",), "standardized RT")], errors="raise"
    )
    aligned = len(original) == len(standardized) and np.allclose(
        original_rt, standardized_rt, atol=1e-8, rtol=0.0
    )
    if not aligned:
        raise AssertionError(f"{split} paired views do not have identical row counts and RT order.")
    if "orig_smile" in standardized.columns:
        source_smiles_match = bool(
            (original[original_smiles].astype(str).to_numpy() == standardized["orig_smile"].astype(str).to_numpy()).all()
        )
    else:
        source_smiles_match = None
    keys = {key for key in original[original_smiles].astype(str).map(inchikey) if key}
    return {
        "records": int(len(original)),
        "paired_rt_and_row_order": True,
        "standardized_orig_smile_matches_original": source_smiles_match,
        "standardized_unique_structures": int(standardized[standardized_smiles].nunique()),
    }, keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-train", type=Path, default=REPO_ROOT / "gwn/data/SMRT_train.csv")
    parser.add_argument("--original-test", type=Path, default=REPO_ROOT / "gwn/data/SMRT_test.csv")
    parser.add_argument("--standardized-train", type=Path, default=REPO_ROOT / "gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv")
    parser.add_argument("--standardized-test", type=Path, default=REPO_ROOT / "gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv")
    parser.add_argument("--oof-prediction", action="append", type=Path, default=[], help="Optional OOF CSVs to audit for one prediction per training row.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts/analysis/pairing_noleakage.json")
    args = parser.parse_args()

    train, train_keys = check_pair(args.original_train, args.standardized_train, "train")
    test, test_keys = check_pair(args.original_test, args.standardized_test, "test")
    oof_checks = []
    for path in args.oof_prediction:
        frame = pd.read_csv(path)
        identifier = next((name for name in ["Source_Index", "source_index", "row_index"] if name in frame.columns), None)
        if identifier is None:
            raise ValueError(f"{path}: no source-row identifier column")
        oof_checks.append(
            {
                "path": str(path.resolve()),
                "rows": int(len(frame)),
                "unique_source_rows": int(frame[identifier].nunique()),
                "one_prediction_per_source_row": int(frame[identifier].nunique()) == len(frame),
            }
        )
    result = {
        "train_pair": train,
        "test_pair": test,
        "train_test_inchikey_overlap": int(len(train_keys & test_keys)),
        "oof_checks": oof_checks,
        "interpretation": "OOF predictions must be generated only for held-out folds; this audit checks alignment and unique OOF row coverage without retraining.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


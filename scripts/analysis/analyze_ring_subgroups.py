#!/usr/bin/env python3
"""Compare Full and No2Cell on overlapping ring-context subgroups."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def ring_flags(smiles: str) -> dict[str, bool]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {name: False for name in GROUPS}
    rings = list(mol.GetRingInfo().AtomRings())
    return {
        "acyclic molecules": not rings,
        "ring-containing molecules": bool(rings),
        "aromatic-ring molecules": any(all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring) for ring in rings),
        "heterocycle-containing molecules": any(any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring) for ring in rings),
        "multi-ring molecules": len(rings) >= 2,
    }


GROUPS = [
    "acyclic molecules", "ring-containing molecules", "aromatic-ring molecules",
    "heterocycle-containing molecules", "multi-ring molecules",
]


def find_smiles(frame: pd.DataFrame) -> str:
    for column in ["SMILES", "smiles", "Original_SMILES", "original_smiles"]:
        if column in frame.columns:
            return column
    raise ValueError("Full prediction file has no SMILES column.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--no2cell", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/ring_subgroups.csv"))
    args = parser.parse_args()
    full = pd.read_csv(args.full)
    no2 = pd.read_csv(args.no2cell)
    if len(full) != len(no2):
        raise ValueError("Full and No2Cell row counts differ.")
    for frame, label in [(full, "Full"), (no2, "No2Cell")]:
        missing = {"Actual_RT", "Final_Pred"} - set(frame.columns)
        if missing:
            raise ValueError(f"{label} file is missing columns: {sorted(missing)}")
    actual = pd.to_numeric(full["Actual_RT"], errors="raise").to_numpy(float)
    if not np.allclose(actual, pd.to_numeric(no2["Actual_RT"], errors="raise"), atol=1e-8):
        raise ValueError("Full and No2Cell target rows are not aligned.")
    flags = pd.DataFrame([ring_flags(value) for value in full[find_smiles(full)]])
    rows = []
    for group in GROUPS:
        mask = flags[group].to_numpy(bool)
        full_mae = float(np.abs(actual[mask] - pd.to_numeric(full.loc[mask, "Final_Pred"]).to_numpy(float)).mean())
        no2_mae = float(np.abs(actual[mask] - pd.to_numeric(no2.loc[mask, "Final_Pred"]).to_numpy(float)).mean())
        rows.append({"group": group, "N": int(mask.sum()), "full_mae_seconds": full_mae, "no2cell_mae_seconds": no2_mae, "delta_no2cell_minus_full_seconds": no2_mae - full_mae})
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))
    print("Note: ring-context groups overlap by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

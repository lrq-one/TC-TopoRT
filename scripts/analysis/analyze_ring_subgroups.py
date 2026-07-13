#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def require_columns(df: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def ring_context(smiles: str) -> dict[str, bool | int]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    atom_rings = list(mol.GetRingInfo().AtomRings())
    ring_count = len(atom_rings)
    has_aromatic_ring = any(
        all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring)
        for ring in atom_rings
    )
    has_heterocycle = any(
        any(mol.GetAtomWithIdx(atom_idx).GetAtomicNum() != 6 for atom_idx in ring)
        for ring in atom_rings
    )

    return {
        "ring_count": ring_count,
        "acyclic": ring_count == 0,
        "ring_containing": ring_count >= 1,
        "aromatic_ring": has_aromatic_ring,
        "heterocycle": has_heterocycle,
        "multi_ring": ring_count >= 2,
    }


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(y - pred)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the seed-5 ring-context subgroup comparison between "
            "full TC-TopoRT and the model without explicit ring 2-cells."
        )
    )
    parser.add_argument(
        "--full_predictions",
        default="artifacts/results/smrt/seed5/test_predictions.csv",
    )
    parser.add_argument(
        "--no2cell_predictions",
        default=(
            "artifacts/results/structural_ablation/"
            "no2cell_seed5/test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/results/paper_tables/subgroups/ring_context",
    )
    parser.add_argument(
        "--verify_paper_counts",
        type=int,
        choices=[0, 1],
        default=1,
        help="Check the five subgroup sizes reported for the retained SMRT test set.",
    )
    args = parser.parse_args()

    full_path = Path(args.full_predictions)
    no2_path = Path(args.no2cell_predictions)
    for path in [full_path, no2_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    full = pd.read_csv(full_path)
    no2 = pd.read_csv(no2_path)
    required = {"SMILES", "Actual_RT", "Final_Pred"}
    require_columns(full, required, full_path)
    require_columns(no2, required, no2_path)

    if len(full) != len(no2):
        raise RuntimeError(
            f"Prediction row counts differ: full={len(full)}, no2cell={len(no2)}"
        )
    if not np.allclose(
        full["Actual_RT"].to_numpy(dtype=float),
        no2["Actual_RT"].to_numpy(dtype=float),
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError("Full/no2cell RT labels are not row-aligned")

    full_keys = [canonical_smiles(x) for x in full["SMILES"].astype(str)]
    no2_keys = [canonical_smiles(x) for x in no2["SMILES"].astype(str)]
    mismatch = sum(a != b for a, b in zip(full_keys, no2_keys))
    if mismatch:
        raise RuntimeError(
            f"Full/no2cell molecular rows are not aligned; mismatches={mismatch}"
        )

    context = pd.DataFrame(
        [ring_context(smiles) for smiles in full["SMILES"].astype(str)]
    )
    y = full["Actual_RT"].to_numpy(dtype=float)
    full_pred = full["Final_Pred"].to_numpy(dtype=float)
    no2_pred = no2["Final_Pred"].to_numpy(dtype=float)

    groups = [
        ("Acyclic molecules", context["acyclic"].to_numpy(dtype=bool)),
        (
            "Ring-containing molecules",
            context["ring_containing"].to_numpy(dtype=bool),
        ),
        (
            "Aromatic-ring molecules",
            context["aromatic_ring"].to_numpy(dtype=bool),
        ),
        (
            "Heterocycle-containing molecules",
            context["heterocycle"].to_numpy(dtype=bool),
        ),
        ("Multi-ring molecules", context["multi_ring"].to_numpy(dtype=bool)),
    ]

    rows: list[dict[str, float | int | str]] = []
    for group, mask in groups:
        n = int(np.sum(mask))
        if n == 0:
            raise RuntimeError(f"Empty ring-context subgroup: {group}")
        full_mae = mae(y[mask], full_pred[mask])
        no2_mae = mae(y[mask], no2_pred[mask])
        rows.append(
            {
                "group": group,
                "n": n,
                "full_mae": full_mae,
                "without_ring_2cells_mae": no2_mae,
                "delta_mae": no2_mae - full_mae,
            }
        )

    table = pd.DataFrame(rows)
    if args.verify_paper_counts:
        expected = {
            "Acyclic molecules": 4,
            "Ring-containing molecules": 7794,
            "Aromatic-ring molecules": 7748,
            "Heterocycle-containing molecules": 7693,
            "Multi-ring molecules": 7747,
        }
        observed = dict(zip(table["group"], table["n"]))
        bad = {
            group: (observed.get(group), count)
            for group, count in expected.items()
            if observed.get(group) != count
        }
        if bad:
            raise RuntimeError(
                "Ring-context counts do not match the reported retained SMRT "
                f"test set: {bad}. Use --verify_paper_counts 0 only for a "
                "different dataset or RDKit definition."
            )

    detail = pd.concat(
        [
            full[["SMILES", "Actual_RT"]].reset_index(drop=True),
            context.reset_index(drop=True),
            pd.DataFrame(
                {
                    "full_pred": full_pred,
                    "without_ring_2cells_pred": no2_pred,
                    "full_abs_error": np.abs(y - full_pred),
                    "without_ring_2cells_abs_error": np.abs(y - no2_pred),
                }
            ),
        ],
        axis=1,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "ring_context_subgroup_table.csv", index=False)
    detail.to_csv(out_dir / "ring_context_assignments_and_errors.csv", index=False)

    print(table.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

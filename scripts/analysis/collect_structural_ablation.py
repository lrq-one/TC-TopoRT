#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_final_metrics(path: Path) -> dict[str, float | str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload.get("test_final")
    if not isinstance(block, dict):
        raise ValueError(f"{path}: missing test_final metric block")
    return {
        "selected_stacker": str(payload.get("selected_stacker", "")),
        "mae": float(block["mae"]),
        "medae": float(block["medae"]),
        "rmse": float(block["rmse"]),
        "r2": float(block["r2"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect full, no-ring-2-cell, no-CWN, and atom-bond GNN structural results."
    )
    parser.add_argument(
        "--full_metrics",
        default="artifacts/results/smrt/seed5/final_metrics.json",
    )
    parser.add_argument(
        "--no2cell_metrics",
        default="artifacts/results/structural_ablation/no2cell_seed5/final_metrics.json",
    )
    parser.add_argument(
        "--cwn0_metrics",
        default="artifacts/results/structural_ablation/cwn0_seed5/final_metrics.json",
    )
    parser.add_argument(
        "--atom_bond_metrics",
        default="artifacts/results/atom_bond_gnn/seed5/metrics.csv",
    )
    parser.add_argument(
        "--out_csv",
        default="artifacts/results/paper_tables/ablation/structural_ablation_seed5.csv",
    )
    parser.add_argument("--require_huber", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    full_path = Path(args.full_metrics)
    no2_path = Path(args.no2cell_metrics)
    cwn0_path = Path(args.cwn0_metrics)
    atom_path = Path(args.atom_bond_metrics)

    for path in [full_path, no2_path, cwn0_path, atom_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    rows: list[dict[str, float | int | str]] = []
    for variant, max_ring_size, cwn_layers, path in [
        ("Full TC-TopoRT", 6, 6, full_path),
        ("w/o explicit ring 2-cells", 2, 6, no2_path),
        ("w/o CWN message passing", 6, 0, cwn0_path),
    ]:
        result = load_final_metrics(path)
        if args.require_huber and result["selected_stacker"] != "huber_stack":
            raise RuntimeError(
                f"{variant}: expected selected_stacker=huber_stack, "
                f"found {result['selected_stacker']!r}"
            )
        rows.append(
            {
                "variant": variant,
                "max_ring_size": max_ring_size,
                "cwn_layers": cwn_layers,
                "source": str(path),
                **result,
            }
        )

    atom = pd.read_csv(atom_path)
    target = atom[atom["Method"].astype(str).eq("AtomBondGNN OOF Huber stack")]
    if len(target) != 1:
        raise RuntimeError(
            f"{atom_path}: expected exactly one 'AtomBondGNN OOF Huber stack' row"
        )
    atom_row = target.iloc[0]
    rows.append(
        {
            "variant": "Conventional atom-bond GNN",
            "max_ring_size": np.nan,
            "cwn_layers": np.nan,
            "source": str(atom_path),
            "selected_stacker": "huber_stack",
            "mae": float(atom_row["MAE"]),
            "medae": float(atom_row["MedAE"]),
            "rmse": float(atom_row["RMSE"]),
            "r2": float(atom_row["R2"]),
        }
    )

    table = pd.DataFrame(rows)
    full_mae = float(table.loc[table["variant"].eq("Full TC-TopoRT"), "mae"].iloc[0])
    table["delta_mae_vs_full"] = table["mae"] - full_mae

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    print(table.to_string(index=False))
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()

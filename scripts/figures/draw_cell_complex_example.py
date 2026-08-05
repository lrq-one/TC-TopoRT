#!/usr/bin/env python3
"""Draw a concrete 0/1/2-cell lifting example for a cyclic molecule.

The output is intended as a supplementary explanatory figure. It shows the
same molecule as (A) a chemical structure, (B) atom 0-cells and bond 1-cells,
and (C) explicit ring 2-cells overlaid on the molecular graph.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw

RDLogger.DisableLog("rdApp.*")


def ring_cells(mol: Chem.Mol, max_ring_size: int) -> list[tuple[int, ...]]:
    rings = [tuple(r) for r in mol.GetRingInfo().AtomRings() if 3 <= len(r) <= max_ring_size]
    return sorted(rings, key=lambda r: (len(r), r))


def coordinates(mol: Chem.Mol) -> dict[int, tuple[float, float]]:
    work = Chem.Mol(mol)
    AllChem.Compute2DCoords(work)
    conf = work.GetConformer()
    return {
        i: (float(conf.GetAtomPosition(i).x), float(conf.GetAtomPosition(i).y))
        for i in range(work.GetNumAtoms())
    }


def draw_graph(
    ax,
    mol: Chem.Mol,
    pos: dict[int, tuple[float, float]],
    rings: list[tuple[int, ...]],
    show_cells: bool,
) -> None:
    if show_cells:
        for cell_index, ring in enumerate(rings):
            points = np.array([pos[i] for i in ring], dtype=float)
            center = points.mean(axis=0)
            # Shrink slightly so adjacent or fused cells remain visually separable.
            points = center + 0.88 * (points - center)
            patch = Polygon(
                points,
                closed=True,
                alpha=0.22,
                linewidth=1.4,
                edgecolor="black",
            )
            ax.add_patch(patch)
            ax.text(
                center[0],
                center[1],
                f"2-cell {cell_index + 1}",
                ha="center",
                va="center",
                fontsize=9,
            )

    for bond_index, bond in enumerate(mol.GetBonds()):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        xi, yi = pos[i]
        xj, yj = pos[j]
        ax.plot([xi, xj], [yi, yj], linewidth=2.0, zorder=2)
        mx, my = (xi + xj) / 2.0, (yi + yj) / 2.0
        ax.text(
            mx,
            my,
            f"e{bond_index}",
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, pad=0.2),
            zorder=4,
        )

    for atom_index, atom in enumerate(mol.GetAtoms()):
        x, y = pos[atom_index]
        ax.scatter([x], [y], s=240, edgecolors="black", linewidths=1.0, zorder=5)
        ax.text(
            x,
            y,
            f"{atom.GetSymbol()}\nv{atom_index}",
            ha="center",
            va="center",
            fontsize=8,
            zorder=6,
        )

    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smiles",
        default="Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        help="Cyclic molecule used only for the explanatory lifting example.",
    )
    parser.add_argument("--name", default="Caffeine")
    parser.add_argument("--max_ring_size", type=int, default=6)
    parser.add_argument(
        "--out",
        default="artifacts/figures/cell_complex_example.png",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {args.smiles!r}")
    rings = ring_cells(mol, args.max_ring_size)
    if not rings:
        raise RuntimeError("The selected molecule has no ring eligible for a 2-cell")
    pos = coordinates(mol)

    # Chemical structure panel with atom indices matching graph panels.
    labelled = Chem.Mol(mol)
    for atom in labelled.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    structure = Draw.MolToImage(labelled, size=(900, 700), kekulize=True)

    fig = plt.figure(figsize=(15.5, 5.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.1], wspace=0.08)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    ax_a.imshow(structure)
    ax_a.axis("off")
    ax_a.set_title(
        f"A. Chemical structure: {args.name}",
        loc="left",
        fontweight="bold",
    )

    draw_graph(ax_b, mol, pos, rings, show_cells=False)
    ax_b.set_title(
        "B. 0-cells (atoms) and 1-cells (bonds)",
        loc="left",
        fontweight="bold",
    )

    draw_graph(ax_c, mol, pos, rings, show_cells=True)
    ax_c.set_title(
        "C. Ring 2-cells added to the complex",
        loc="left",
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.02,
        f"The lifted complex contains {mol.GetNumAtoms()} atom 0-cells, "
        f"{mol.GetNumBonds()} bond 1-cells, and {len(rings)} ring 2-cells "
        f"(ring size <= {args.max_ring_size}).",
        ha="center",
        fontsize=10,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(
        f"0-cells={mol.GetNumAtoms()}, 1-cells={mol.GetNumBonds()}, "
        f"2-cells={len(rings)}"
    )


if __name__ == "__main__":
    main()

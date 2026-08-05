#!/usr/bin/env python3
"""Pairwise isomer-discrimination analysis for TC-TopoRT.

This analysis addresses the reviewer request for controlled evidence on
formula-matched positional or constitutional isomers and stereoisomers. It
compares pairwise RT ordering and RT-difference prediction across the full
TC-TopoRT model and structural controls using the same SMRT test molecules.

The script deliberately calls one category ``positional-like`` because the
assignment is structure-based and heuristic: same molecular formula, same
Murcko scaffold, different constitutional graph, similar fingerprints, and
matching ring signature. The label should not be interpreted as a manually
curated positional-isomer ontology.
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

SMILES_CANDIDATES = ("SMILES", "smile", "smiles")
RT_CANDIDATES = ("Actual_RT", "actual_rt", "rt", "RT")
PRED_CANDIDATES = (
    "Final_Pred",
    "final_pred",
    "prediction",
    "pred_rt",
    "Predicted_RT",
)


def pick_column(df: pd.DataFrame, candidates: Iterable[str], path: Path) -> str:
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(
        f"{path}: none of {list(candidates)} found; columns={list(df.columns)}"
    )


def ring_signature(mol: Chem.Mol) -> str:
    rings = list(mol.GetRingInfo().AtomRings())
    sizes = sorted(len(r) for r in rings)
    aromatic = sum(
        1
        for r in rings
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
    )
    hetero = sum(
        1
        for r in rings
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in r)
    )
    return (
        f"n={len(rings)};sizes={','.join(map(str, sizes))};"
        f"arom={aromatic};hetero={hetero}"
    )


def molecule_table(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    smiles_col = pick_column(df, SMILES_CANDIDATES, path)
    rt_col = pick_column(df, RT_CANDIDATES, path)
    pred_col = pick_column(df, PRED_CANDIDATES, path)

    records: list[dict[str, object]] = []
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for source_row, row in df.reset_index(drop=True).iterrows():
        smi = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(
                f"{path}: invalid SMILES at row {source_row}: {smi!r}"
            )
        iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        noniso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
        records.append(
            {
                "source_row": int(source_row),
                "smiles": smi,
                "canon_iso": iso,
                "canon_noniso": noniso,
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "scaffold": scaffold,
                "ring_signature": ring_signature(mol),
                "actual_rt": float(row[rt_col]),
                "pred_rt": float(row[pred_col]),
                "mol": mol,
                "fingerprint": fpgen.GetFingerprint(mol),
            }
        )
    out = pd.DataFrame(records)
    if out["canon_iso"].duplicated().any():
        dup = out.loc[
            out["canon_iso"].duplicated(keep=False),
            "canon_iso",
        ].head().tolist()
        raise RuntimeError(
            f"{path}: duplicate canonical isomeric structures found; "
            f"examples={dup}. Deduplicate or provide a unique molecule "
            "identifier before pair analysis."
        )
    return out


def align_control(
    reference: pd.DataFrame,
    control_df: pd.DataFrame,
    path: Path,
    model_name: str,
) -> np.ndarray:
    control = molecule_table(control_df, path)
    by_key = control.set_index("canon_iso")
    reference_keys = set(reference["canon_iso"])
    missing = [k for k in reference["canon_iso"] if k not in by_key.index]
    extra = [k for k in by_key.index if k not in reference_keys]
    if missing or extra:
        raise RuntimeError(
            f"{model_name}: molecular set mismatch; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    aligned = by_key.loc[reference["canon_iso"]].reset_index()
    if not np.allclose(
        reference["actual_rt"].to_numpy(float),
        aligned["actual_rt"].to_numpy(float),
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{model_name}: RT labels do not align with the full-model file"
        )
    return aligned["pred_rt"].to_numpy(float)


def classify_pair(
    a: pd.Series,
    b: pd.Series,
    similarity: float,
    positional_similarity: float,
) -> dict[str, bool]:
    stereoisomer = (
        a["canon_noniso"] == b["canon_noniso"]
        and a["canon_iso"] != b["canon_iso"]
    )
    constitutional = a["canon_noniso"] != b["canon_noniso"]
    same_scaffold = bool(a["scaffold"]) and a["scaffold"] == b["scaffold"]
    same_ring_signature = a["ring_signature"] == b["ring_signature"]
    positional_like = (
        constitutional
        and same_scaffold
        and same_ring_signature
        and similarity >= positional_similarity
    )
    ring_topology_challenge = (
        constitutional and a["ring_signature"] != b["ring_signature"]
    )
    return {
        "constitutional_isomer": constitutional,
        "stereoisomer": stereoisomer,
        "positional_like": positional_like,
        "ring_topology_challenge": ring_topology_challenge,
    }


def ordering_correct(
    actual_delta: float,
    predicted_delta: float,
    tie_tolerance: float,
) -> float:
    if abs(actual_delta) <= tie_tolerance:
        return math.nan
    if abs(predicted_delta) <= tie_tolerance:
        return 0.0
    return float(np.sign(actual_delta) == np.sign(predicted_delta))


def build_pairs(
    molecules: pd.DataFrame,
    model_predictions: dict[str, np.ndarray],
    *,
    min_rt_gap: float,
    positional_similarity: float,
    max_formula_group: int,
    tie_tolerance: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for formula, group in molecules.groupby("formula", sort=True):
        group = group.sort_values("canon_iso").reset_index(drop=True)
        if len(group) < 2:
            continue
        if len(group) > max_formula_group:
            raise RuntimeError(
                f"Formula {formula} has {len(group)} molecules, exceeding "
                f"--max_formula_group={max_formula_group}. Increase the limit "
                "only after checking pair-count inflation."
            )
        for i, j in itertools.combinations(range(len(group)), 2):
            a = group.iloc[i]
            b = group.iloc[j]
            actual_delta = float(a["actual_rt"] - b["actual_rt"])
            if abs(actual_delta) < min_rt_gap:
                continue
            similarity = float(
                DataStructs.TanimotoSimilarity(
                    a["fingerprint"],
                    b["fingerprint"],
                )
            )
            classes = classify_pair(
                a,
                b,
                similarity,
                positional_similarity,
            )
            base: dict[str, object] = {
                "formula": formula,
                "a_source_row": int(a["source_row"]),
                "b_source_row": int(b["source_row"]),
                "a_smiles": a["smiles"],
                "b_smiles": b["smiles"],
                "a_canon_iso": a["canon_iso"],
                "b_canon_iso": b["canon_iso"],
                "a_actual_rt": float(a["actual_rt"]),
                "b_actual_rt": float(b["actual_rt"]),
                "actual_delta": actual_delta,
                "actual_abs_gap": abs(actual_delta),
                "fingerprint_similarity": similarity,
                "same_scaffold": (
                    bool(a["scaffold"])
                    and a["scaffold"] == b["scaffold"]
                ),
                "a_scaffold": a["scaffold"],
                "b_scaffold": b["scaffold"],
                "a_ring_signature": a["ring_signature"],
                "b_ring_signature": b["ring_signature"],
                **classes,
            }
            for model_name, predictions in model_predictions.items():
                pred_a = float(predictions[int(a["source_row"])])
                pred_b = float(predictions[int(b["source_row"])])
                pred_delta = pred_a - pred_b
                prefix = (
                    model_name.lower()
                    .replace("-", "_")
                    .replace(" ", "_")
                )
                base[f"{prefix}_a_pred"] = pred_a
                base[f"{prefix}_b_pred"] = pred_b
                base[f"{prefix}_pred_delta"] = pred_delta
                base[f"{prefix}_ordering_correct"] = ordering_correct(
                    actual_delta,
                    pred_delta,
                    tie_tolerance,
                )
                base[f"{prefix}_delta_mae"] = abs(
                    pred_delta - actual_delta
                )
                base[f"{prefix}_gap_mae"] = abs(
                    abs(pred_delta) - abs(actual_delta)
                )
            rows.append(base)
    if not rows:
        raise RuntimeError(
            "No formula-matched pairs passed the RT-gap criterion"
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    frame: pd.DataFrame,
    value_col: str,
    *,
    cluster_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    clean = frame[[cluster_col, value_col]].dropna()
    if clean.empty:
        return math.nan, math.nan, math.nan
    point = float(clean[value_col].mean())
    clusters = clean[cluster_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        parts = [clean[clean[cluster_col] == cluster] for cluster in sampled]
        boot = pd.concat(parts, ignore_index=True)
        values.append(float(boot[value_col].mean()))
    low, high = np.quantile(values, [0.025, 0.975])
    return point, float(low), float(high)


def category_masks(pairs: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    return [
        (
            "All formula-matched pairs",
            pd.Series(True, index=pairs.index),
        ),
        (
            "Constitutional isomers",
            pairs["constitutional_isomer"].astype(bool),
        ),
        (
            "Positional-like constitutional isomers",
            pairs["positional_like"].astype(bool),
        ),
        (
            "Stereoisomers",
            pairs["stereoisomer"].astype(bool),
        ),
        (
            "Ring-topology challenge pairs",
            pairs["ring_topology_challenge"].astype(bool),
        ),
    ]


def summarize_pairs(
    pairs: pd.DataFrame,
    model_names: list[str],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category, mask in category_masks(pairs):
        subset = pairs.loc[mask].copy()
        if subset.empty:
            continue
        for model_name in model_names:
            prefix = (
                model_name.lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            accuracy, accuracy_low, accuracy_high = bootstrap_ci(
                subset,
                f"{prefix}_ordering_correct",
                cluster_col="formula",
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
            )
            delta_mae, delta_low, delta_high = bootstrap_ci(
                subset,
                f"{prefix}_delta_mae",
                cluster_col="formula",
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed + 1,
            )
            rows.append(
                {
                    "category": category,
                    "model": model_name,
                    "n_pairs": int(len(subset)),
                    "n_formulas": int(subset["formula"].nunique()),
                    "ordering_accuracy": accuracy,
                    "ordering_accuracy_ci_low": accuracy_low,
                    "ordering_accuracy_ci_high": accuracy_high,
                    "rt_difference_mae_s": delta_mae,
                    "rt_difference_mae_ci_low_s": delta_low,
                    "rt_difference_mae_ci_high_s": delta_high,
                    "median_actual_rt_gap_s": float(
                        subset["actual_abs_gap"].median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def choose_cases(pairs: pd.DataFrame, n_cases: int) -> pd.DataFrame:
    full_col = "full_tc_toport_ordering_correct"
    no2_col = "without_ring_2_cells_ordering_correct"
    candidates = pairs.copy()
    if full_col in candidates and no2_col in candidates:
        candidates["priority"] = (
            5.0 * (candidates[full_col] == 1.0).astype(float)
            + 5.0 * (candidates[no2_col] == 0.0).astype(float)
            + candidates["positional_like"].astype(float)
            + candidates["ring_topology_challenge"].astype(float)
            + np.log1p(candidates["actual_abs_gap"])
        )
    else:
        candidates["priority"] = np.log1p(
            candidates["actual_abs_gap"]
        )
    candidates = candidates.sort_values(
        ["priority", "actual_abs_gap", "fingerprint_similarity"],
        ascending=[False, False, False],
    )
    return candidates.head(n_cases).copy()


def draw_cases(cases: pd.DataFrame, output: Path) -> None:
    molecules: list[Chem.Mol] = []
    legends: list[str] = []
    for case_index, row in cases.reset_index(drop=True).iterrows():
        for side in ("a", "b"):
            mol = Chem.MolFromSmiles(str(row[f"{side}_smiles"]))
            if mol is None:
                continue
            molecules.append(mol)
            legend = (
                f"Case {case_index + 1}{side.upper()} | {row['formula']}\n"
                f"Observed RT {row[f'{side}_actual_rt']:.1f} s"
            )
            if f"full_tc_toport_{side}_pred" in row:
                legend += (
                    f" | Full "
                    f"{row[f'full_tc_toport_{side}_pred']:.1f} s"
                )
            if f"without_ring_2_cells_{side}_pred" in row:
                legend += (
                    f" | no-2-cell "
                    f"{row[f'without_ring_2_cells_{side}_pred']:.1f} s"
                )
            legends.append(legend)
    if not molecules:
        return
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=2,
        subImgSize=(520, 300),
        legends=legends,
        useSVG=False,
        returnPNG=False,
    )
    image.save(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--atom_bond_predictions",
        default="",
        help="Optional atom-bond GNN test-prediction CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/results/paper_tables/isomer_discrimination",
    )
    parser.add_argument("--min_rt_gap", type=float, default=10.0)
    parser.add_argument("--positional_similarity", type=float, default=0.45)
    parser.add_argument("--max_formula_group", type=int, default=100)
    parser.add_argument("--tie_tolerance", type=float, default=1e-8)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260805)
    parser.add_argument("--n_cases", type=int, default=6)
    args = parser.parse_args()

    full_path = Path(args.full_predictions)
    no2_path = Path(args.no2cell_predictions)
    for path in (full_path, no2_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    full_df = pd.read_csv(full_path)
    molecules = molecule_table(full_df, full_path)
    model_predictions: dict[str, np.ndarray] = {
        "Full TC-TopoRT": molecules["pred_rt"].to_numpy(float)
    }
    model_predictions["Without ring 2-cells"] = align_control(
        molecules,
        pd.read_csv(no2_path),
        no2_path,
        "Without ring 2-cells",
    )

    if args.atom_bond_predictions:
        atom_path = Path(args.atom_bond_predictions)
        if not atom_path.is_file():
            raise FileNotFoundError(atom_path)
        model_predictions["Atom-bond GNN"] = align_control(
            molecules,
            pd.read_csv(atom_path),
            atom_path,
            "Atom-bond GNN",
        )

    pairs = build_pairs(
        molecules,
        model_predictions,
        min_rt_gap=args.min_rt_gap,
        positional_similarity=args.positional_similarity,
        max_formula_group=args.max_formula_group,
        tie_tolerance=args.tie_tolerance,
    )
    summary = summarize_pairs(
        pairs,
        list(model_predictions),
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    cases = choose_cases(pairs, args.n_cases)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_dir / "isomer_pair_records.csv", index=False)
    summary.to_csv(out_dir / "isomer_pair_summary.csv", index=False)
    cases.to_csv(out_dir / "representative_isomer_pairs.csv", index=False)
    draw_cases(cases, out_dir / "representative_isomer_pairs.png")

    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")
    print(
        "Interpretation note: 'positional-like' is a reproducible structural "
        "heuristic, not a manually curated positional-isomer label."
    )


if __name__ == "__main__":
    main()

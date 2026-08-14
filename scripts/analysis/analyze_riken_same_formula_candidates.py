#!/usr/bin/env python3
"""Audit the final matched Full-versus-No2Cell RIKEN candidate scope."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")
EXPECTED = {
    "queries": 85,
    "candidate_rows": 5044,
    "unique_candidate_structures": 4692,
    "queries_with_true_candidate": 85,
    "same_formula_structurally_distinct_queries": 84,
    "ring_signature_different_queries": 81,
    "strict_controlled_stereoisomer_pairs": 0,
}
EXPECTED_TARGET_METRICS = {
    "Full": {"MAE": 20.27229668112362, "RMSE": 28.35530691716036, "MedAE": 11.20703125, "Bias": 4.273510921702665},
    "No2Cell": {"MAE": 21.97721737132353, "RMSE": 29.14362704075065, "MedAE": 16.29736328125, "Bias": 5.011690027573529},
}


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


@lru_cache(maxsize=10000)
def stereo_facts(smiles: str) -> tuple[str, str, int]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "", "", 0
    iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    noniso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    assigned_centers = sum(
        label in {"R", "S"}
        for _, label in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    )
    assigned_bonds = sum(
        bond.GetStereo()
        not in {Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY}
        for bond in mol.GetBonds()
    )
    return iso, noniso, int(assigned_centers + assigned_bonds)


def load_target_metrics(path: Path, model: str) -> dict[str, float]:
    frame = pd.read_csv(path)
    required = {"view", "split", "MAE", "RMSE", "MedAE", "bias"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing target-metric columns: {missing}")
    row = frame[frame["view"].eq("dualview_avg") & frame["split"].eq("riken_exact85_test")]
    if len(row) != 1:
        raise ValueError(f"{model}: expected one dualview_avg/riken_exact85_test row in {path}")
    value = row.iloc[0]
    return {
        "MAE": float(value["MAE"]),
        "RMSE": float(value["RMSE"]),
        "MedAE": float(value["MedAE"]),
        "Bias": float(value["bias"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Matched Full/No2Cell candidate CSV."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/analysis/riken_same_formula")
    )
    parser.add_argument("--full-target-metrics", type=Path)
    parser.add_argument("--no2cell-target-metrics", type=Path)
    parser.add_argument("--verify-paper-scope", action="store_true")
    parser.add_argument("--verify-target-metrics", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.input, low_memory=False)
    required = {
        "query_id",
        "candidate_uid",
        "candidate_rank",
        "candidate_smiles_canon",
        "derived_formula",
        "ring_signature",
        "is_true",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    frame["is_true"] = as_bool(frame["is_true"])

    same_formula_queries = 0
    ring_different_queries = 0
    strict_stereo_pairs = 0
    query_rows = []
    for query_id, group in frame.groupby("query_id", sort=True):
        true_rows = group[group["is_true"]].sort_values("candidate_rank", kind="stable")
        if true_rows.empty:
            query_rows.append({"query_id": query_id, "true_candidate_present": False})
            continue
        true = true_rows.iloc[0]
        false = group[
            (~group["is_true"])
            & group["derived_formula"].astype(str).eq(str(true["derived_formula"]))
            & group["candidate_smiles_canon"].astype(str).ne(str(true["candidate_smiles_canon"]))
        ].copy()
        same_formula_queries += int(not false.empty)
        ring_different = false[
            false["ring_signature"].astype(str).ne(str(true["ring_signature"]))
        ]
        ring_different_queries += int(not ring_different.empty)

        true_iso, true_noniso, true_assigned = stereo_facts(str(true["candidate_smiles_canon"]))
        for smiles in false["candidate_smiles_canon"].astype(str):
            false_iso, false_noniso, false_assigned = stereo_facts(smiles)
            strict_stereo_pairs += int(
                bool(true_noniso)
                and true_noniso == false_noniso
                and true_iso != false_iso
                and true_assigned > 0
                and false_assigned > 0
            )
        query_rows.append(
            {
                "query_id": query_id,
                "true_candidate_present": True,
                "n_candidates": int(len(group)),
                "n_distinct_same_formula_false": int(len(false)),
                "n_ring_signature_different_false": int(len(ring_different)),
            }
        )

    true_frame = frame[frame["is_true"]].sort_values(
        ["query_id", "candidate_rank"], kind="stable"
    ).drop_duplicates("query_id", keep="first")
    if bool(args.full_target_metrics) != bool(args.no2cell_target_metrics):
        raise ValueError("Provide both --full-target-metrics and --no2cell-target-metrics, or neither.")
    target_metrics = None
    if args.full_target_metrics:
        target_metrics = {
            "Full": load_target_metrics(args.full_target_metrics, "Full"),
            "No2Cell": load_target_metrics(args.no2cell_target_metrics, "No2Cell"),
        }
    summary = {
        "queries": int(frame["query_id"].nunique()),
        "candidate_rows": int(len(frame)),
        "unique_candidate_structures": int(frame["candidate_smiles_canon"].nunique()),
        "queries_with_true_candidate": int(true_frame["query_id"].nunique()),
        "same_formula_structurally_distinct_queries": same_formula_queries,
        "ring_signature_different_queries": ring_different_queries,
        "strict_controlled_stereoisomer_pairs": strict_stereo_pairs,
        "target_domain_metrics": target_metrics,
        "target_metric_provenance": (
            "Separate transfer-learning metric tables; candidate-level prediction columns are not substituted."
            if target_metrics is not None
            else "Not evaluated: pass the two target metric tables explicitly."
        ),
    }
    if args.verify_paper_scope:
        differences = {key: (summary[key], value) for key, value in EXPECTED.items() if summary[key] != value}
        if differences:
            raise AssertionError(f"Scope differs from locked paper values: {differences}")
    if args.verify_target_metrics:
        if target_metrics is None:
            raise ValueError("--verify-target-metrics requires both target metric files.")
        differences = []
        for model, expected in EXPECTED_TARGET_METRICS.items():
            for metric, value in expected.items():
                if not np.isclose(target_metrics[model][metric], value, atol=1e-10):
                    differences.append(f"{model}/{metric}={target_metrics[model][metric]}")
        if differences:
            raise AssertionError("Target metrics differ from locked values: " + "; ".join(differences))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(query_rows).to_csv(args.output_dir / "query_scope.csv", index=False)
    if target_metrics is not None:
        pd.DataFrame(
            [{"model": name, **values} for name, values in target_metrics.items()]
        ).to_csv(args.output_dir / "target_domain_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

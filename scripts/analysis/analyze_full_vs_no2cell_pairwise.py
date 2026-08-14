#!/usr/bin/env python3
"""Query-clustered same-formula Full-versus-No2Cell discrimination analysis."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from functools import lru_cache
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


RDLogger.DisableLog("rdApp.*")
EPS = 1e-9
EXPECTED_PRIMARY = {
    "All_strict_same_formula_pairs": (84, 4888, 80.6436746990945, 77.21913915822014),
    "Ring_signature_different_pairs": (81, 2844, 82.27523769970544, 81.06598797249947),
    "MS_Top10_false_competitors": (84, 768, 76.9510582010582, 73.08201058201058),
    "Ring_different_MS_Top10": (72, 386, 77.66589506172838, 77.10813492063492),
}


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def exact_sign_test(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.0
    lower = min(positive, negative)
    return min(1.0, 2.0 * sum(comb(n, i) * 0.5**n for i in range(lower + 1)))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)  # type: ignore[arg-type]
    adjusted: dict[str, float] = {}
    previous = 0.0
    for index, name in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - index) * pvalues[name]))
        adjusted[name] = previous
    return adjusted


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    means = np.empty(n, dtype=float)
    for index in range(n):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


enumerator = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=10000)
def strict_tautomer_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return ""
    try:
        mol = enumerator.Canonicalize(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def score(true_error: float, false_error: float) -> float:
    margin = false_error - true_error
    return 1.0 if margin > EPS else 0.0 if margin < -EPS else 0.5


def build_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query_id, group in frame.groupby("query_id", sort=True):
        true_rows = group[group["is_true"]].sort_values(
            ["candidate_rank", "candidate_uid"], kind="stable"
        )
        if true_rows.empty:
            raise ValueError(f"{query_id}: no true candidate")
        true = true_rows.iloc[0]
        true_tautomer = strict_tautomer_smiles(str(true["candidate_smiles_canon"]))
        false_rows = group[
            (~group["is_true"])
            & group["derived_formula"].astype(str).eq(str(true["derived_formula"]))
            & group["candidate_smiles_canon"].astype(str).ne(str(true["candidate_smiles_canon"]))
        ].copy()
        false_rows["tautomer_signature"] = false_rows["candidate_smiles_canon"].astype(str).map(
            strict_tautomer_smiles
        )
        false_rows = false_rows[false_rows["tautomer_signature"].ne(true_tautomer)]
        for _, false in false_rows.iterrows():
            full_margin = float(false["abs_rt_delta_full"] - true["abs_rt_delta_full"])
            no2_margin = float(false["abs_rt_delta_no2"] - true["abs_rt_delta_no2"])
            full_score = score(float(true["abs_rt_delta_full"]), float(false["abs_rt_delta_full"]))
            no2_score = score(float(true["abs_rt_delta_no2"]), float(false["abs_rt_delta_no2"]))
            rows.append(
                {
                    "query_id": query_id,
                    "true_candidate_uid": str(true["candidate_uid"]),
                    "false_candidate_uid": str(false["candidate_uid"]),
                    "true_candidate_rank_ms": int(true["candidate_rank"]),
                    "false_candidate_rank_ms": int(false["candidate_rank"]),
                    "formula": str(true["derived_formula"]),
                    "true_smiles": str(true["candidate_smiles_canon"]),
                    "false_smiles": str(false["candidate_smiles_canon"]),
                    "ring_signature_different": str(false["ring_signature"]) != str(true["ring_signature"]),
                    "full_margin_s": full_margin,
                    "no2_margin_s": no2_margin,
                    "full_pairwise_score": full_score,
                    "no2_pairwise_score": no2_score,
                    "direct_pair_outcome": (
                        "Full_better" if full_score > no2_score else "No2Cell_better" if full_score < no2_score else "Same"
                    ),
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError("No strict same-formula pairs were found.")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Matched Full/No2Cell candidate CSV."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/analysis/riken_pairwise")
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--verify-paper-results", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.input, low_memory=False)
    required = {
        "query_id", "candidate_uid", "candidate_rank", "candidate_smiles_canon",
        "derived_formula", "ring_signature", "is_true", "abs_rt_delta_full", "abs_rt_delta_no2",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    frame["is_true"] = as_bool(frame["is_true"])
    for column in ["candidate_rank", "abs_rt_delta_full", "abs_rt_delta_no2"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    pairs = build_pairs(frame)
    subsets = OrderedDict(
        [
            ("All_strict_same_formula_pairs", np.ones(len(pairs), dtype=bool)),
            ("Ring_signature_different_pairs", pairs["ring_signature_different"].to_numpy(bool)),
            ("Ring_signature_same_pairs", ~pairs["ring_signature_different"].to_numpy(bool)),
            ("MS_Top5_false_competitors", pairs["false_candidate_rank_ms"].to_numpy(int) <= 5),
            ("MS_Top10_false_competitors", pairs["false_candidate_rank_ms"].to_numpy(int) <= 10),
            (
                "Ring_different_MS_Top5",
                pairs["ring_signature_different"].to_numpy(bool)
                & (pairs["false_candidate_rank_ms"].to_numpy(int) <= 5),
            ),
            (
                "Ring_different_MS_Top10",
                pairs["ring_signature_different"].to_numpy(bool)
                & (pairs["false_candidate_rank_ms"].to_numpy(int) <= 10),
            ),
        ]
    )
    rng = np.random.default_rng(args.bootstrap_seed)
    summaries = []
    querywise_tables = []
    pvalues = {}
    for name, mask in subsets.items():
        subset = pairs.loc[mask].copy()
        if subset.empty:
            continue
        querywise = subset.groupby("query_id", as_index=False).agg(
            n_pairs=("query_id", "size"),
            full_query_pairwise_score=("full_pairwise_score", "mean"),
            no2_query_pairwise_score=("no2_pairwise_score", "mean"),
            full_query_mean_margin_s=("full_margin_s", "mean"),
            no2_query_mean_margin_s=("no2_margin_s", "mean"),
        )
        querywise["subset"] = name
        difference = (
            querywise["full_query_pairwise_score"] - querywise["no2_query_pairwise_score"]
        ).to_numpy(float)
        positive = int((difference > EPS).sum())
        negative = int((difference < -EPS).sum())
        ci_low, ci_high = bootstrap_mean_ci(difference, rng, args.bootstrap_replicates)
        pvalues[name] = exact_sign_test(positive, negative)
        outcomes = subset["direct_pair_outcome"].value_counts()
        summaries.append(
            {
                "Subset": name,
                "Queries": int(querywise["query_id"].nunique()),
                "Pairs": int(len(subset)),
                "Full_query_macro_accuracy_pct": 100.0 * float(querywise["full_query_pairwise_score"].mean()),
                "No2Cell_query_macro_accuracy_pct": 100.0 * float(querywise["no2_query_pairwise_score"].mean()),
                "Macro_difference_pp": 100.0 * float(difference.mean()),
                "Macro_difference_95CI_low_pp": 100.0 * ci_low,
                "Macro_difference_95CI_high_pp": 100.0 * ci_high,
                "Full_better_queries": positive,
                "Same_queries": int(len(difference) - positive - negative),
                "No2Cell_better_queries": negative,
                "Exact_sign_p_unadjusted": pvalues[name],
                "Full_better_pairs": int(outcomes.get("Full_better", 0)),
                "Same_pairs": int(outcomes.get("Same", 0)),
                "No2Cell_better_pairs": int(outcomes.get("No2Cell_better", 0)),
            }
        )
        querywise_tables.append(querywise)

    adjusted = holm_adjust(pvalues)
    for row in summaries:
        row["Holm_adjusted_p"] = adjusted[row["Subset"]]
    summary = pd.DataFrame(summaries)
    if args.verify_paper_results:
        errors = []
        indexed = summary.set_index("Subset")
        for name, (queries, pair_count, full, no2) in EXPECTED_PRIMARY.items():
            row = indexed.loc[name]
            if int(row["Queries"]) != queries or int(row["Pairs"]) != pair_count:
                errors.append(f"{name}: scope {int(row['Queries'])}/{int(row['Pairs'])}")
            if not np.isclose(row["Full_query_macro_accuracy_pct"], full, atol=1e-10):
                errors.append(f"{name}: Full macro {row['Full_query_macro_accuracy_pct']}")
            if not np.isclose(row["No2Cell_query_macro_accuracy_pct"], no2, atol=1e-10):
                errors.append(f"{name}: No2Cell macro {row['No2Cell_query_macro_accuracy_pct']}")
        if errors:
            raise AssertionError("Pairwise results differ from locked paper values: " + "; ".join(errors))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "strict_same_formula_pairs.csv", index=False)
    summary.to_csv(args.output_dir / "pairwise_summary.csv", index=False)
    pd.concat(querywise_tables, ignore_index=True).to_csv(
        args.output_dir / "querywise_scores.csv", index=False
    )
    protocol = {
        "analysis_unit": "query",
        "success": "true candidate has smaller absolute RT error than the false candidate",
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "interpretation": (
            "Numerically positive Full-minus-No2Cell differences are not universal "
            "isomer-resolution claims; the reported confidence intervals span zero."
        ),
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

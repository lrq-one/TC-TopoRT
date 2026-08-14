#!/usr/bin/env python3
"""Apply the final-paper retention-time candidate filter.

The threshold for each predictor is calibrated on a fixed development set
held out from the calibration-model fit and frozen as ``3 * MAE_dev``.
Candidates with a missing prediction are retained; otherwise a candidate is
retained when its absolute RT error is no greater than the frozen threshold.
Filtering never changes the original MS-FINDER ordering.

Frozen candidate-level inputs are distributed in the author Figshare archive,
not in this source-code repository.  See ``data/README.md`` for placement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "candidate_filtering.yaml"

DATASET_ALIASES = {
    "metabobase": "metabobase",
    "riken": "riken_plasma",
    "riken-plasma": "riken_plasma",
    "riken_plasma": "riken_plasma",
}

COLUMN_ALIASES = {
    "query_id": ("query_id", "query_identifier", "query", "spectrum_id"),
    "candidate_id": (
        "candidate_id",
        "candidate_uid",
        "candidate_identifier",
        "candidate_inchikey",
        "inchikey",
        "candidate_smiles",
        "smiles",
    ),
    "rank": (
        "original_candidate_rank",
        "candidate_rank",
        "msfinder_rank",
        "initial_rank",
        "rank",
    ),
    "experimental_rt": (
        "experimental_rt",
        "experimental_rt_seconds",
        "rt_sec",
        "query_rt_sec",
        "exp_rt",
        "rt",
    ),
    "predicted_rt": (
        "predicted_rt",
        "predicted_rt_seconds",
        "candidate_pred_rt",
        "pred_rt",
        "prediction",
    ),
    "is_true": ("is_true", "true_candidate", "is_true_candidate", "true_flag"),
}


@dataclass(frozen=True)
class InputSchema:
    """Resolved input columns for one candidate table."""

    query_id: str
    candidate_id: str | None
    rank: str
    experimental_rt: str
    predicted_rt: str
    is_true: str


def _first_present(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    available = set(columns)
    return next((name for name in aliases if name in available), None)


def resolve_schema(frame: pd.DataFrame) -> InputSchema:
    """Resolve supported public/archive column names without guessing values."""

    resolved = {
        key: _first_present(frame.columns, aliases)
        for key, aliases in COLUMN_ALIASES.items()
    }
    missing = [
        key
        for key in ("query_id", "rank", "experimental_rt", "predicted_rt", "is_true")
        if resolved[key] is None
    ]
    if missing:
        choices = {key: list(COLUMN_ALIASES[key]) for key in missing}
        raise ValueError(
            "Candidate input is missing required fields "
            f"{missing}. Accepted column names: {choices}"
        )
    return InputSchema(**resolved)  # type: ignore[arg-type]


def _coerce_true_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="raise")
        invalid = ~values.isin([0, 1])
        if invalid.any():
            raise ValueError("True-candidate flags must contain only 0/1 or booleans.")
        return values.fillna(0).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    unknown = normalized.dropna()[~normalized.dropna().isin(mapping)]
    if not unknown.empty:
        raise ValueError(f"Unrecognized true-candidate flag: {unknown.iloc[0]!r}")
    return normalized.map(mapping).fillna(False).astype(bool)


def filter_candidates(
    frame: pd.DataFrame,
    threshold_seconds: float,
    schema: InputSchema | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Filter candidates and return candidate, query, and dataset summaries."""

    if frame.empty:
        raise ValueError("Candidate input is empty.")
    if not math.isfinite(threshold_seconds) or threshold_seconds < 0:
        raise ValueError("The frozen threshold must be a finite non-negative number.")

    schema = schema or resolve_schema(frame)
    work = pd.DataFrame(index=frame.index)
    work["query_id"] = frame[schema.query_id].astype("string")
    if work["query_id"].isna().any() or (work["query_id"].str.len() == 0).any():
        raise ValueError("Query identifiers must not be missing or empty.")

    if schema.candidate_id is None:
        work["candidate_id"] = [f"row_{index + 1}" for index in range(len(frame))]
    else:
        work["candidate_id"] = frame[schema.candidate_id].astype("string")

    work["original_candidate_rank"] = pd.to_numeric(frame[schema.rank], errors="raise")
    if work["original_candidate_rank"].isna().any() or (
        work["original_candidate_rank"] <= 0
    ).any():
        raise ValueError("Original candidate ranks must be positive and non-missing.")

    work["experimental_rt"] = pd.to_numeric(
        frame[schema.experimental_rt], errors="coerce"
    )
    if work["experimental_rt"].isna().any():
        raise ValueError("Experimental RT must be present and numeric for every record.")

    work["predicted_rt"] = pd.to_numeric(frame[schema.predicted_rt], errors="coerce")
    work["is_true"] = _coerce_true_flag(frame[schema.is_true])
    work["_input_order"] = range(len(work))
    work["_query_order"] = work.groupby("query_id", sort=False)["_input_order"].transform("min")

    # A stable sort makes the preserved MS-FINDER ordering explicit.
    work = work.sort_values(
        ["_query_order", "original_candidate_rank", "_input_order"], kind="stable"
    ).reset_index(drop=True)
    work["abs_rt_delta"] = (work["experimental_rt"] - work["predicted_rt"]).abs()
    work["retained"] = work["predicted_rt"].isna() | (
        work["abs_rt_delta"] <= float(threshold_seconds)
    )

    position = work.groupby("query_id", sort=False)["retained"].cumsum()
    work["rank_after_filtering"] = position.where(work["retained"]).astype("Int64")

    rows: list[dict[str, Any]] = []
    for query_id, group in work.groupby("query_id", sort=False):
        true_rows = group[group["is_true"]]
        if true_rows.empty:
            raise ValueError(f"Query {query_id!r} has no true candidate in the input list.")
        true_original_rank = float(true_rows["original_candidate_rank"].min())
        retained_true = true_rows[true_rows["retained"]]
        true_retained = not retained_true.empty
        true_final_rank = (
            int(retained_true["rank_after_filtering"].min()) if true_retained else pd.NA
        )
        rows.append(
            {
                "query_id": query_id,
                "initial_candidate_count": int(len(group)),
                "retained_candidate_count": int(group["retained"].sum()),
                "true_retained": bool(true_retained),
                "FN": int(not true_retained),
                "true_original_rank": true_original_rank,
                "true_rank_after_filtering": true_final_rank,
                "Top1": bool(true_retained and true_final_rank <= 1),
                "Top5": bool(true_retained and true_final_rank <= 5),
                "Top10": bool(true_retained and true_final_rank <= 10),
            }
        )

    query_summary = pd.DataFrame(rows)
    before = int(len(work))
    after = int(work["retained"].sum())
    queries = int(len(query_summary))
    true_retained_count = int(query_summary["true_retained"].sum())
    dataset_summary: dict[str, Any] = {
        "queries": queries,
        "candidate_records_before": before,
        "candidate_records_after": after,
        "candidate_reduction_percent": 100.0 * (before - after) / before,
        "candidate_reduction_denominator": "candidate_records_before",
        "true_retained_count": true_retained_count,
        "true_retained_percent": 100.0 * true_retained_count / queries,
        "false_negatives": int(query_summary["FN"].sum()),
        "Top1_count": int(query_summary["Top1"].sum()),
        "Top5_count": int(query_summary["Top5"].sum()),
        "Top10_count": int(query_summary["Top10"].sum()),
        "retention_and_topk_denominator": "queries",
        "threshold_seconds": float(threshold_seconds),
    }

    candidate_output = work[
        [
            "query_id",
            "candidate_id",
            "original_candidate_rank",
            "experimental_rt",
            "predicted_rt",
            "abs_rt_delta",
            "retained",
            "is_true",
            "rank_after_filtering",
        ]
    ].copy()
    return candidate_output, query_summary, dataset_summary


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return config


def _dataset_threshold(config: dict[str, Any], dataset: str, method: str) -> float:
    try:
        threshold = config["development"][dataset]["methods"][method]["threshold_seconds"]
    except KeyError as error:
        raise KeyError(f"Missing configuration key for {dataset}/{method}: {error}") from error
    return float(threshold)


def _default_input(config: dict[str, Any], dataset: str) -> Path:
    value = config.get("inputs", {}).get(dataset)
    if not value:
        value = f"data/local/candidate_filtering/{dataset}_candidate_predictions.csv"
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def run_one(
    dataset: str,
    method: str,
    input_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Final candidate-level input not found: {input_path}\n"
            "Download the frozen author-generated candidate table from the TC-TopoRT "
            "Figshare archive and place it as described in data/README.md, or pass --input."
        )
    threshold = _dataset_threshold(config, dataset, method)
    frame = pd.read_csv(input_path)
    candidates, queries, summary = filter_candidates(frame, threshold)
    summary.update({"dataset": dataset, "method": method, "input": str(input_path)})

    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(dataset_dir / "candidate_level_filtering.csv", index=False)
    queries.to_csv(dataset_dir / "query_level_summary.csv", index=False)
    with (dataset_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Final filtering YAML configuration."
    )
    parser.add_argument(
        "--dataset",
        choices=["metabobase", "riken_plasma", "riken", "riken-plasma", "all"],
        required=True,
        help="Candidate dataset to process.",
    )
    parser.add_argument("--method", default="tc_toport", help="Predictor key in the configuration.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Candidate CSV for a single dataset; omit to use the configured local path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "results" / "candidate_filtering",
        help="Directory for generated candidate/query/dataset summaries.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)

    if args.dataset == "all":
        if args.input is not None:
            raise ValueError("--input can only be used when processing one dataset.")
        datasets = ["metabobase", "riken_plasma"]
    else:
        datasets = [DATASET_ALIASES[args.dataset]]

    output_dir = args.output_dir.expanduser().resolve()
    summaries = []
    for dataset in datasets:
        input_path = args.input.expanduser().resolve() if args.input else _default_input(config, dataset)
        summary = run_one(
            dataset,
            args.method,
            input_path,
            output_dir,
            config,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(2) from error

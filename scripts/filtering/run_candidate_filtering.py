#!/usr/bin/env python3
"""
Reproduce the final TC-TopoRT candidate-filtering results.

Retention rule:
    |predicted RT - experimental RT| <= T
    OR original MS-FINDER rank <= g

Soft-reranking score:
    MS-FINDER rank + alpha * |Delta RT| / tau

Tie-breaking:
    hybrid score ascending,
    original MS-FINDER rank ascending,
    original MS-FINDER score descending.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUT_DIR = (
    "artifacts/results/candidate_filtering"
)

CONFIGS: list[dict[str, Any]] = [
    {
        "dataset": "MetaboBase",
        "slug": "metabobase",
        "input": (
            'data/candidate_filtering/metabobase_candidate_predictions.csv'
        ),
        "group_columns": ["s10_row", "query_id"],
        "threshold_sec": 60.0,
        "guard_k": 3,
        "tau": 75.17,
        "alpha": 1.5,
        "reference": {
            "queries": 45,
            "before": 3023,
            "after": 933,
            "reduction": 69.136619,
            "true_retention": 93.333333,
            "top1_before": 44.444444,
            "top5_before": 75.555556,
            "top10_before": 84.444444,
            "top1_after": 55.555556,
            "top5_after": 82.222222,
            "top10_after": 88.888889,
            "false_negatives": 3,
        },
        "abcort": {
            "after": 1864,
            "reduction": 38.35,
            "top1": 51.11,
            "top5": 73.33,
            "top10": 82.22,
        },
    },
    {
        "dataset": "RIKEN-PlaSMA",
        "slug": "riken_plasma",
        "input": (
            'data/candidate_filtering/riken_candidate_predictions.csv'
        ),
        "group_columns": ["s11_row", "query_id"],
        "threshold_sec": 50.0,
        "guard_k": 2,
        "tau": 25.66,
        "alpha": 2.0,
        "reference": {
            "queries": 85,
            "before": 5044,
            "after": 2712,
            "reduction": 46.233148,
            "true_retention": 97.647059,
            "top1_before": 47.058824,
            "top5_before": 70.588235,
            "top10_before": 82.352941,
            "top1_after": 54.117647,
            "top5_after": 77.647059,
            "top10_after": 89.411765,
            "false_negatives": 2,
        },
        "abcort": {
            "after": 3608,
            "reduction": 28.46,
            "top1": 52.94,
            "top5": 76.47,
            "top10": 83.53,
        },
    },
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return (REPO_ROOT / path).resolve()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {"1", "true", "t", "yes", "y"}
    )


def topk(rank: float | int | None, k: int) -> bool:
    if rank is None or pd.isna(rank):
        return False

    return float(rank) <= float(k)


def detect_group_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str:
    for column in candidates:
        if column in frame.columns:
            return column

    raise KeyError(
        "No query-group column was found. "
        f"Tried {candidates}; available columns are "
        f"{list(frame.columns)}"
    )


def prepare_input(
    csv_path: Path,
    group_candidates: list[str],
) -> tuple[pd.DataFrame, str]:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    frame = pd.read_csv(
        csv_path,
        low_memory=False,
    )

    required = {
        "candidate_rank",
        "abs_rt_delta",
        "is_true",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise KeyError(
            f"{csv_path} is missing columns: {missing}"
        )

    group_col = detect_group_column(
        frame,
        group_candidates,
    )

    frame = frame.copy()

    frame["_source_row"] = np.arange(
        len(frame),
        dtype=int,
    )

    frame["candidate_rank"] = pd.to_numeric(
        frame["candidate_rank"],
        errors="raise",
    )

    frame["abs_rt_delta"] = pd.to_numeric(
        frame["abs_rt_delta"],
        errors="raise",
    )

    if "candidate_score" not in frame.columns:
        frame["candidate_score"] = 0.0

    frame["candidate_score"] = pd.to_numeric(
        frame["candidate_score"],
        errors="coerce",
    ).fillna(0.0)

    frame["is_true"] = bool_series(
        frame["is_true"]
    )

    return frame, group_col


def evaluate(
    frame: pd.DataFrame,
    group_col: str,
    threshold_sec: float,
    guard_k: int,
    tau: float,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    result = frame.copy()

    result["retained"] = (
        result["abs_rt_delta"].le(
            threshold_sec
        )
        | result["candidate_rank"].le(
            guard_k
        )
    )

    result["hybrid_score"] = (
        result["candidate_rank"]
        + alpha
        * result["abs_rt_delta"]
        / tau
    )

    result["rank_after"] = np.nan

    query_rows: list[dict[str, Any]] = []

    for query_key, sub in result.groupby(
        group_col,
        sort=True,
        dropna=False,
    ):
        sub = sub.copy()

        original = sub.sort_values(
            [
                "candidate_rank",
                "candidate_score",
            ],
            ascending=[
                True,
                False,
            ],
            kind="mergesort",
        )

        true_before_rows = original[
            original["is_true"]
        ]

        if true_before_rows.empty:
            true_rank_before = np.nan
            true_abs_rt_delta = np.nan
        else:
            true_rank_before = float(
                true_before_rows[
                    "candidate_rank"
                ].min()
            )

            true_abs_rt_delta = float(
                true_before_rows.iloc[0][
                    "abs_rt_delta"
                ]
            )

        retained = sub[
            sub["retained"]
        ].copy()

        retained = retained.sort_values(
            [
                "hybrid_score",
                "candidate_rank",
                "candidate_score",
            ],
            ascending=[
                True,
                True,
                False,
            ],
            kind="mergesort",
        ).reset_index(drop=True)

        retained["rank_after"] = np.arange(
            1,
            len(retained) + 1,
            dtype=int,
        )

        rank_map = dict(
            zip(
                retained["_source_row"],
                retained["rank_after"],
            )
        )

        mask = result["_source_row"].isin(
            rank_map
        )

        result.loc[
            mask,
            "rank_after",
        ] = result.loc[
            mask,
            "_source_row",
        ].map(rank_map)

        true_after_rows = retained[
            retained["is_true"]
        ]

        if true_after_rows.empty:
            true_rank_after = np.nan
        else:
            true_rank_after = float(
                true_after_rows[
                    "rank_after"
                ].min()
            )

        query_rows.append(
            {
                "query_key": query_key,
                "n_candidates_before": int(
                    len(sub)
                ),
                "n_candidates_after": int(
                    len(retained)
                ),
                "true_rank_before": (
                    true_rank_before
                ),
                "true_rank_after": (
                    true_rank_after
                ),
                "true_retained_after": bool(
                    not true_after_rows.empty
                ),
                "true_abs_rt_delta": (
                    true_abs_rt_delta
                ),
                "top1_before": topk(
                    true_rank_before, 1
                ),
                "top5_before": topk(
                    true_rank_before, 5
                ),
                "top10_before": topk(
                    true_rank_before, 10
                ),
                "top1_after": topk(
                    true_rank_after, 1
                ),
                "top5_after": topk(
                    true_rank_after, 5
                ),
                "top10_after": topk(
                    true_rank_after, 10
                ),
            }
        )

    query_table = pd.DataFrame(
        query_rows
    )

    before = int(
        query_table[
            "n_candidates_before"
        ].sum()
    )

    after = int(
        query_table[
            "n_candidates_after"
        ].sum()
    )

    summary = {
        "n_queries": int(
            len(query_table)
        ),
        "n_candidate_rows_before": before,
        "n_candidate_rows_after": after,
        "candidate_reduction_pct": (
            100.0
            * (before - after)
            / max(before, 1)
        ),
        "true_retention_pct": (
            100.0
            * query_table[
                "true_retained_after"
            ].mean()
        ),
        "top1_before_pct": (
            100.0
            * query_table[
                "top1_before"
            ].mean()
        ),
        "top5_before_pct": (
            100.0
            * query_table[
                "top5_before"
            ].mean()
        ),
        "top10_before_pct": (
            100.0
            * query_table[
                "top10_before"
            ].mean()
        ),
        "top1_after_pct": (
            100.0
            * query_table[
                "top1_after"
            ].mean()
        ),
        "top5_after_pct": (
            100.0
            * query_table[
                "top5_after"
            ].mean()
        ),
        "top10_after_pct": (
            100.0
            * query_table[
                "top10_after"
            ].mean()
        ),
        "false_negatives": int(
            (
                ~query_table[
                    "true_retained_after"
                ]
            ).sum()
        ),
    }

    result = result.sort_values(
        [
            group_col,
            "candidate_rank",
            "candidate_score",
        ],
        ascending=[
            True,
            True,
            False,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return result, query_table, summary


def validate_reference(
    dataset: str,
    summary: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    exact_checks = {
        "n_queries": "queries",
        "n_candidate_rows_before": "before",
        "n_candidate_rows_after": "after",
        "false_negatives": "false_negatives",
    }

    for summary_key, reference_key in exact_checks.items():
        actual = int(summary[summary_key])
        expected = int(reference[reference_key])

        if actual != expected:
            raise RuntimeError(
                f"{dataset}: {summary_key} "
                f"{actual} != {expected}"
            )

    float_checks = {
        "candidate_reduction_pct": "reduction",
        "true_retention_pct": "true_retention",
        "top1_before_pct": "top1_before",
        "top5_before_pct": "top5_before",
        "top10_before_pct": "top10_before",
        "top1_after_pct": "top1_after",
        "top5_after_pct": "top5_after",
        "top10_after_pct": "top10_after",
    }

    for summary_key, reference_key in float_checks.items():
        actual = float(summary[summary_key])
        expected = float(reference[reference_key])

        if not np.isclose(
            actual,
            expected,
            atol=1e-5,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"{dataset}: {summary_key} "
                f"{actual} != {expected}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metabobase_csv",
        default=CONFIGS[0]["input"],
    )
    parser.add_argument(
        "--riken_csv",
        default=CONFIGS[1]["input"],
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--dry_run",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--skip_reference_validation",
        type=int,
        choices=[0, 1],
        default=0,
    )

    args = parser.parse_args()

    inputs = {
        "MetaboBase": resolve_path(
            args.metabobase_csv
        ),
        "RIKEN-PlaSMA": resolve_path(
            args.riken_csv
        ),
    }

    out_dir = resolve_path(
        args.out_dir
    )

    print("=" * 88)
    print(
        "TC-TopoRT final candidate-filtering workflow"
    )
    print("=" * 88)

    for config in CONFIGS:
        print()
        print(config["dataset"])
        print(
            " input:",
            inputs[config["dataset"]],
        )
        print(
            " T/g/tau/alpha:",
            config["threshold_sec"],
            config["guard_k"],
            config["tau"],
            config["alpha"],
        )

    print()
    print("output:", out_dir)

    if args.dry_run:
        print()
        print(
            "DRY RUN: no filtering was executed."
        )
        return

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_rows = []
    main_table_rows = []
    parameter_rows = []

    for config in CONFIGS:
        dataset = config["dataset"]

        frame, group_col = prepare_input(
            inputs[dataset],
            config["group_columns"],
        )

        ranked, queries, summary = evaluate(
            frame=frame,
            group_col=group_col,
            threshold_sec=float(
                config["threshold_sec"]
            ),
            guard_k=int(
                config["guard_k"]
            ),
            tau=float(
                config["tau"]
            ),
            alpha=float(
                config["alpha"]
            ),
        )

        if not args.skip_reference_validation:
            validate_reference(
                dataset,
                summary,
                config["reference"],
            )

        dataset_dir = (
            out_dir / config["slug"]
        )
        dataset_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        ranked.to_csv(
            dataset_dir
            / "candidate_reranking.csv",
            index=False,
        )

        queries.to_csv(
            dataset_dir
            / "query_filtering_metrics.csv",
            index=False,
        )

        selected_rows.append(
            {
                "Dataset": dataset,
                "T (s)": config[
                    "threshold_sec"
                ],
                "g": config["guard_k"],
                "tau (s)": config["tau"],
                "alpha": config["alpha"],
                "Queries": summary[
                    "n_queries"
                ],
                "Candidates before": summary[
                    "n_candidate_rows_before"
                ],
                "Candidates after": summary[
                    "n_candidate_rows_after"
                ],
                "Reduction (%)": summary[
                    "candidate_reduction_pct"
                ],
                "True retained (%)": summary[
                    "true_retention_pct"
                ],
                "False negatives": summary[
                    "false_negatives"
                ],
                "Top-1 (%)": summary[
                    "top1_after_pct"
                ],
                "Top-5 (%)": summary[
                    "top5_after_pct"
                ],
                "Top-10 (%)": summary[
                    "top10_after_pct"
                ],
            }
        )

        main_table_rows.append(
            {
                "Dataset": dataset,
                "Method": (
                    "MS-FINDER only / No RT"
                ),
                "Queries": summary[
                    "n_queries"
                ],
                "Initial candidates": summary[
                    "n_candidate_rows_before"
                ],
                "Retained candidates": summary[
                    "n_candidate_rows_before"
                ],
                "Reduction (%)": 0.0,
                "Top-1 (%)": summary[
                    "top1_before_pct"
                ],
                "Top-5 (%)": summary[
                    "top5_before_pct"
                ],
                "Top-10 (%)": summary[
                    "top10_before_pct"
                ],
                "True retained (%)": 100.0,
                "False negatives": 0,
            }
        )

        abcort = config["abcort"]

        main_table_rows.append(
            {
                "Dataset": dataset,
                "Method": "ABCoRT-TL",
                "Queries": summary[
                    "n_queries"
                ],
                "Initial candidates": summary[
                    "n_candidate_rows_before"
                ],
                "Retained candidates": abcort[
                    "after"
                ],
                "Reduction (%)": abcort[
                    "reduction"
                ],
                "Top-1 (%)": abcort["top1"],
                "Top-5 (%)": abcort["top5"],
                "Top-10 (%)": abcort["top10"],
                "True retained (%)": np.nan,
                "False negatives": np.nan,
            }
        )

        main_table_rows.append(
            {
                "Dataset": dataset,
                "Method": "TC-TopoRT",
                "Queries": summary[
                    "n_queries"
                ],
                "Initial candidates": summary[
                    "n_candidate_rows_before"
                ],
                "Retained candidates": summary[
                    "n_candidate_rows_after"
                ],
                "Reduction (%)": summary[
                    "candidate_reduction_pct"
                ],
                "Top-1 (%)": summary[
                    "top1_after_pct"
                ],
                "Top-5 (%)": summary[
                    "top5_after_pct"
                ],
                "Top-10 (%)": summary[
                    "top10_after_pct"
                ],
                "True retained (%)": summary[
                    "true_retention_pct"
                ],
                "False negatives": summary[
                    "false_negatives"
                ],
            }
        )

        parameter_rows.append(
            {
                "Dataset": dataset,
                "Selected method": (
                    "RT-aware guarded soft rerank"
                ),
                "T (s)": config[
                    "threshold_sec"
                ],
                "g": config["guard_k"],
                "tau (s)": config["tau"],
                "alpha": config["alpha"],
                "Candidate retention rule": (
                    "|Delta RT| <= T or "
                    "MS-FINDER rank <= g"
                ),
                "Reranking score": (
                    "MS-FINDER rank + "
                    "alpha * |Delta RT| / tau"
                ),
                "Tie-breaking": (
                    "MS-FINDER rank, then "
                    "MS-FINDER score"
                ),
                "Queries": summary[
                    "n_queries"
                ],
                "Candidates before": summary[
                    "n_candidate_rows_before"
                ],
                "Candidates after": summary[
                    "n_candidate_rows_after"
                ],
                "Reduction (%)": summary[
                    "candidate_reduction_pct"
                ],
                "True retained (%)": summary[
                    "true_retention_pct"
                ],
                "Top-1 (%)": summary[
                    "top1_after_pct"
                ],
                "Top-5 (%)": summary[
                    "top5_after_pct"
                ],
                "Top-10 (%)": summary[
                    "top10_after_pct"
                ],
            }
        )

        print()
        print(f"[{dataset}]")
        print(
            "candidates:",
            summary[
                "n_candidate_rows_before"
            ],
            "->",
            summary[
                "n_candidate_rows_after"
            ],
        )
        print(
            "reduction:",
            f"{summary['candidate_reduction_pct']:.6f}%",
        )
        print(
            "true retained:",
            f"{summary['true_retention_pct']:.6f}%",
        )
        print(
            "Top-1/5/10:",
            f"{summary['top1_after_pct']:.6f}",
            f"{summary['top5_after_pct']:.6f}",
            f"{summary['top10_after_pct']:.6f}",
        )
        print(
            "false negatives:",
            summary["false_negatives"],
        )

    selected = pd.DataFrame(
        selected_rows
    )

    main_table = pd.DataFrame(
        main_table_rows
    )

    parameters = pd.DataFrame(
        parameter_rows
    )

    selected.to_csv(
        out_dir
        / "candidate_filtering_selected_summary.csv",
        index=False,
    )

    main_table.to_csv(
        out_dir
        / "Table_3_candidate_filtering_main.csv",
        index=False,
    )

    parameters.to_csv(
        out_dir
        / "Table_S22_candidate_filtering_parameters.csv",
        index=False,
    )

    print()
    print("=" * 88)
    print("FINAL TABLE")
    print("=" * 88)
    print(
        main_table.to_string(index=False)
    )

    print()
    print(
        "PASS: final candidate-filtering "
        "results reproduced."
    )
    print("[SAVE]", out_dir)


if __name__ == "__main__":
    main()

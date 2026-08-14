#!/usr/bin/env python3
"""Audit the operational RT < 300 s scope in the complete public SMRT table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED = {"total_records": 80038, "early_records": 2058}


def find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str:
    names = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        if alias in names:
            return names[alias]
    raise ValueError(f"None of {aliases} was found in columns {list(frame.columns)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Complete 80,038-record SMRT CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/smrt_early_rt_coverage.json"),
    )
    parser.add_argument("--verify-paper-scope", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    rt_column = find_column(
        frame, ("rt", "retention_time", "retention time", "retentiontime")
    )
    rt = pd.to_numeric(frame[rt_column], errors="coerce")
    if rt.isna().any():
        raise ValueError(f"RT contains {int(rt.isna().sum())} missing/non-numeric values.")

    early = int((rt < 300.0).sum())
    summary = {
        "input": str(args.input.resolve()),
        "total_records": int(len(frame)),
        "early_records_rt_lt_300_seconds": early,
        "early_percent": 100.0 * early / len(frame),
        "retained_or_later_records_rt_ge_300_seconds": int((rt >= 300.0).sum()),
        "interpretation": (
            "Operational early-eluting/non-retained subset under the SMRT RPLC method; "
            "not a universal definition of highly polar metabolites."
        ),
        "predictive_benchmark_reported_for_early_subset": False,
    }
    if args.verify_paper_scope:
        if len(frame) != EXPECTED["total_records"] or early != EXPECTED["early_records"]:
            raise AssertionError(f"Computed scope does not match the locked paper values: {summary}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd


WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
POSIX_PRIVATE_PREFIXES = ("/home/", "/Users/", "/mnt/", "/tmp/")


def sanitized_value(value: object) -> tuple[object, bool]:
    if value is None or pd.isna(value):
        return value, False
    text = str(value)
    stripped = text.strip()

    if WINDOWS_ABSOLUTE.match(stripped):
        return PureWindowsPath(stripped).name, True
    if stripped.startswith(POSIX_PRIVATE_PREFIXES):
        return PurePosixPath(stripped).name, True
    return value, False


def sanitize_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    result = frame.copy()
    replacements = 0
    for column in result.select_dtypes(include=["object", "string"]).columns:
        cleaned = []
        for value in result[column].tolist():
            new_value, changed = sanitized_value(value)
            cleaned.append(new_value)
            replacements += int(changed)
        result[column] = cleaned
    return result, replacements


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove machine-local absolute path prefixes from candidate-level CSVs "
            "while preserving basenames and all analysis columns."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/candidate_filtering/metabobase_candidate_predictions.csv",
            "data/candidate_filtering/riken_candidate_predictions.csv",
        ],
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/data/candidate_filtering_sanitized",
    )
    parser.add_argument(
        "--in_place",
        type=int,
        choices=[0, 1],
        default=0,
        help="Overwrite input files only when explicitly set to 1.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not args.in_place:
        out_dir.mkdir(parents=True, exist_ok=True)

    for input_value in args.inputs:
        input_path = Path(input_value)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        frame = pd.read_csv(input_path, low_memory=False)
        sanitized, replacements = sanitize_frame(frame)
        output_path = input_path if args.in_place else out_dir / input_path.name
        sanitized.to_csv(output_path, index=False)

        print(
            f"{input_path}: rows={len(frame)}, columns={len(frame.columns)}, "
            f"path values sanitized={replacements}, output={output_path}"
        )


if __name__ == "__main__":
    main()

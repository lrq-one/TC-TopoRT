#!/usr/bin/env python3
"""Prepare a leakage-free full-range SMRT sensitivity split.

The current paper benchmark retains compounds with RT > 300 s. This script
adds the excluded early-retention region without changing the established
train/test assignment of the retained compounds:

1. Parse the official SMRT SDF and read the RETENTION_TIME property.
2. Canonicalize and deduplicate structures.
3. Preserve the current retained train/test membership by canonical structure.
4. Allocate previously excluded compounds (RT <= threshold) to train/test using
   a deterministic stratified split with the same approximate train:test ratio.
5. Write CSV files compatible with the existing TC-TopoRT training entry point.

The resulting split is an extended sensitivity split, not an official split
provided by the original SMRT release. Manuscript text and reviewer responses
should state this explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import StratifiedShuffleSplit

RDLogger.DisableLog("rdApp.*")

SMILES_CANDIDATES = ("smile", "SMILES", "smiles")
RT_CANDIDATES = ("rt", "RT", "RETENTION_TIME")


def pick_column(df: pd.DataFrame, candidates: tuple[str, ...], path: Path) -> str:
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(f"{path}: none of {candidates} found; columns={list(df.columns)}")


def canon_iso(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def read_existing(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    smi_col = pick_column(df, SMILES_CANDIDATES, path)
    rt_col = pick_column(df, RT_CANDIDATES, path)
    out = pd.DataFrame(
        {
            "smile": df[smi_col].astype(str),
            "rt": pd.to_numeric(df[rt_col], errors="raise").astype(float),
            "split": split,
        }
    )
    out["canon_iso"] = [canon_iso(s) for s in out["smile"]]
    if out["canon_iso"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate canonical structures in retained split")
    return out


def get_rt_property(mol: Chem.Mol, preferred: str) -> float:
    names = [preferred, "RETENTION_TIME", "retention_time", "RT", "rt"]
    for name in names:
        if mol.HasProp(name):
            value = mol.GetProp(name)
            try:
                return float(value)
            except ValueError:
                continue
    available = list(mol.GetPropNames())
    raise KeyError(f"No numeric RT property found; tried={names}, available={available}")


def load_sdf(path: Path, rt_property: str) -> tuple[pd.DataFrame, dict[str, int]]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    records: list[dict[str, object]] = []
    invalid = 0
    missing_rt = 0
    for source_index, mol in enumerate(supplier):
        if mol is None:
            invalid += 1
            continue
        try:
            Chem.SanitizeMol(mol)
            rt = get_rt_property(mol, rt_property)
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            records.append(
                {
                    "source_index": source_index,
                    "smile": smiles,
                    "canon_iso": smiles,
                    "rt": float(rt),
                }
            )
        except KeyError:
            missing_rt += 1
        except Exception:
            invalid += 1
    if not records:
        raise RuntimeError(f"No valid SMRT records were read from {path}")
    frame = pd.DataFrame(records)
    audit = {
        "sdf_records_valid": len(frame),
        "sdf_records_invalid": invalid,
        "sdf_records_missing_rt": missing_rt,
    }
    return frame, audit


def deduplicate(
    frame: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    conflicting = 0
    duplicate_records = 0
    for key, group in frame.groupby("canon_iso", sort=False):
        rts = group["rt"].to_numpy(float)
        duplicate_records += max(0, len(group) - 1)
        if float(np.max(rts) - np.min(rts)) > tolerance:
            conflicting += 1
        first = group.iloc[0]
        rows.append(
            {
                "source_index": int(first["source_index"]),
                "smile": str(first["smile"]),
                "canon_iso": key,
                "rt": float(np.mean(rts)),
                "n_replicates": int(len(group)),
                "rt_range": float(np.max(rts) - np.min(rts)),
            }
        )
    return pd.DataFrame(rows), {
        "duplicate_sdf_records_collapsed": duplicate_records,
        "duplicate_structures_with_rt_range_above_tolerance": conflicting,
    }


def quantile_labels(values: np.ndarray, max_bins: int = 10) -> np.ndarray:
    series = pd.Series(values)
    for bins in range(min(max_bins, len(series)), 1, -1):
        try:
            labels = pd.qcut(series, q=bins, labels=False, duplicates="drop")
            labels = np.asarray(labels, dtype=int)
            counts = np.bincount(labels)
            if len(counts) >= 2 and int(counts.min()) >= 2:
                return labels
        except Exception:
            continue
    return np.zeros(len(series), dtype=int)


def split_early(
    early: pd.DataFrame,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if len(early) < 2:
        raise RuntimeError("Fewer than two early-retention molecules are available")
    n_test = max(1, int(round(len(early) * test_fraction)))
    n_test = min(n_test, len(early) - 1)
    labels = quantile_labels(early["rt"].to_numpy(float))
    if len(np.unique(labels)) >= 2:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=n_test,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(np.zeros(len(early)), labels))
        method = "StratifiedShuffleSplit over early-RT quantile bins"
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(early))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        method = "deterministic random split (quantile stratification unavailable)"
    return early.iloc[train_idx].copy(), early.iloc[test_idx].copy(), method


def write_smrt_csv(frame: pd.DataFrame, path: Path) -> None:
    out = frame[["smile", "rt"]].reset_index(drop=True)
    out.to_csv(path, index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", required=True, help="Official SMRT_dataset.sdf path")
    parser.add_argument("--retained_train_csv", default="gwn/data/SMRT_train.csv")
    parser.add_argument("--retained_test_csv", default="gwn/data/SMRT_test.csv")
    parser.add_argument("--out_dir", default="artifacts/data/smrt_full_range")
    parser.add_argument("--rt_property", default="RETENTION_TIME")
    parser.add_argument("--early_threshold", type=float, default=300.0)
    parser.add_argument("--early_test_fraction", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=20260805)
    parser.add_argument("--duplicate_rt_tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    sdf_path = Path(args.sdf)
    train_path = Path(args.retained_train_csv)
    test_path = Path(args.retained_test_csv)
    for path in (sdf_path, train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    retained_train = read_existing(train_path, "retained_train")
    retained_test = read_existing(test_path, "retained_test")
    retained = pd.concat([retained_train, retained_test], ignore_index=True)
    overlap = set(retained_train["canon_iso"]) & set(retained_test["canon_iso"])
    if overlap:
        raise RuntimeError(f"Retained train/test structural overlap: {len(overlap)}")

    full_raw, audit_sdf = load_sdf(sdf_path, args.rt_property)
    full, audit_dedup = deduplicate(full_raw, args.duplicate_rt_tolerance)
    full_map = full.set_index("canon_iso")

    missing_retained = [
        key for key in retained["canon_iso"] if key not in full_map.index
    ]
    if missing_retained:
        raise RuntimeError(
            f"{len(missing_retained)} retained structures were not found in the SDF; "
            f"examples={missing_retained[:5]}"
        )

    # Use the established retained CSV labels exactly, while confirming that the
    # source SDF labels agree.
    matched = full_map.loc[retained["canon_iso"]].reset_index()
    max_label_diff = float(
        np.max(
            np.abs(
                matched["rt"].to_numpy(float)
                - retained["rt"].to_numpy(float)
            )
        )
    )
    if max_label_diff > args.duplicate_rt_tolerance:
        raise RuntimeError(
            "Retained CSV/SDF RT mismatch: maximum absolute difference="
            f"{max_label_diff} s"
        )

    retained_keys = set(retained["canon_iso"])
    early = full[
        (full["rt"] <= args.early_threshold)
        & (~full["canon_iso"].isin(retained_keys))
    ].copy()
    unassigned_late = full[
        (full["rt"] > args.early_threshold)
        & (~full["canon_iso"].isin(retained_keys))
    ].copy()
    if len(unassigned_late):
        raise RuntimeError(
            f"Found {len(unassigned_late)} unassigned structures with RT > threshold. "
            "This indicates that the retained CSVs do not reproduce the intended "
            "retained benchmark."
        )

    early_train, early_test, split_method = split_early(
        early,
        args.early_test_fraction,
        args.split_seed,
    )
    full_train = pd.concat(
        [
            retained_train[["smile", "rt", "canon_iso"]],
            early_train[["smile", "rt", "canon_iso"]],
        ],
        ignore_index=True,
    )
    full_test = pd.concat(
        [
            retained_test[["smile", "rt", "canon_iso"]],
            early_test[["smile", "rt", "canon_iso"]],
        ],
        ignore_index=True,
    )
    train_test_overlap = set(full_train["canon_iso"]) & set(full_test["canon_iso"])
    if train_test_overlap:
        raise RuntimeError(
            f"Full-range train/test structural overlap: {len(train_test_overlap)}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_smrt_csv(full_train, out_dir / "SMRT_full_train.csv")
    write_smrt_csv(full_test, out_dir / "SMRT_full_test.csv")
    write_smrt_csv(early_train, out_dir / "SMRT_early_train.csv")
    write_smrt_csv(early_test, out_dir / "SMRT_early_test.csv")

    assignment = pd.concat(
        [
            retained_train.assign(range_group="retained"),
            retained_test.assign(range_group="retained"),
            early_train.assign(split="early_train", range_group="early"),
            early_test.assign(split="early_test", range_group="early"),
        ],
        ignore_index=True,
    )
    assignment.to_csv(
        out_dir / "SMRT_full_range_split_assignments.csv",
        index=False,
    )

    audit = {
        **audit_sdf,
        **audit_dedup,
        "unique_valid_structures": int(len(full)),
        "early_threshold_s": args.early_threshold,
        "retained_train_n": int(len(retained_train)),
        "retained_test_n": int(len(retained_test)),
        "early_total_n": int(len(early)),
        "early_train_n": int(len(early_train)),
        "early_test_n": int(len(early_test)),
        "full_train_n": int(len(full_train)),
        "full_test_n": int(len(full_test)),
        "split_seed": args.split_seed,
        "early_split_method": split_method,
        "retained_csv_sdf_max_rt_difference_s": max_label_diff,
        "train_test_structure_overlap": int(len(train_test_overlap)),
        "scope_note": (
            "Extended full-range sensitivity split preserving the established "
            "retained-compound assignments; early compounds were newly allocated "
            "and this is not an official source split."
        ),
    }
    with open(
        out_dir / "SMRT_full_range_audit.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, indent=2)

    print(json.dumps(audit, indent=2))
    print(f"\nPrepared full-range inputs under {out_dir}")
    print(
        "Next: run scripts/data/build_strict_tautomer_views.py on "
        "SMRT_full_train.csv and SMRT_full_test.csv, then train with "
        "scripts/training/run_smrt_full_range_single_seed.sh."
    )


if __name__ == "__main__":
    main()

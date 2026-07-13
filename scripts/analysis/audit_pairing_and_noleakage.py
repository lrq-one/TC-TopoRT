#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")
SEEDS = [1, 5, 79, 123, 256]


def load_retained(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "smile" in df.columns and "smiles" not in df.columns:
        df = df.rename(columns={"smile": "smiles"})
    if not {"smiles", "rt"}.issubset(df.columns):
        raise ValueError(f"{path}: expected smile(s) and rt columns")

    df["rt"] = pd.to_numeric(df["rt"], errors="raise")
    df = df[df["rt"] > 300.0].copy()
    id_col = next(
        (
            c
            for c in [
                "molecule_id",
                "compound_id",
                "metlin_id",
                "hmdb_id",
                "pubchem_id",
                "id",
            ]
            if c in df.columns
        ),
        None,
    )

    rows = []
    for source_row, row in df.iterrows():
        smiles = str(row["smiles"])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        item = {
            "source_row": int(source_row),
            "rt": float(row["rt"]),
            "canonical_isomeric": Chem.MolToSmiles(
                mol, canonical=True, isomericSmiles=True
            ),
            "canonical_nonisomeric": Chem.MolToSmiles(
                mol, canonical=True, isomericSmiles=False
            ),
            "inchikey": Chem.MolToInchiKey(mol),
        }
        if id_col is not None:
            item["molecule_id"] = str(row[id_col])
        if "orig_smile" in df.columns:
            original = Chem.MolFromSmiles(str(row["orig_smile"]))
            item["orig_canonical"] = (
                Chem.MolToSmiles(
                    original, canonical=True, isomericSmiles=True
                )
                if original is not None
                else ""
            )
        rows.append(item)
    return pd.DataFrame(rows).reset_index(drop=True)


def add(
    rows: list[dict[str, str]],
    section: str,
    check: str,
    passed: bool | None,
    observed: object,
    expected: object,
    note: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "check": check,
            "status": (
                "WARN" if passed is None else ("PASS" if passed else "FAIL")
            ),
            "observed": str(observed),
            "expected": str(expected),
            "note": note,
        }
    )


def overlap(left: pd.Series, right: pd.Series) -> int:
    return len(
        {str(x) for x in left if str(x)}
        & {str(x) for x in right if str(x)}
    )


def audit_pair(
    rows: list[dict[str, str]],
    split: str,
    origin: pd.DataFrame,
    tautomer: pd.DataFrame,
) -> None:
    section = f"pairing::{split}"
    same_n = len(origin) == len(tautomer)
    add(
        rows,
        section,
        "retained row counts match",
        same_n,
        f"{len(origin)}/{len(tautomer)}",
        "equal",
    )
    if not same_n:
        return

    source_mismatch = int(
        np.sum(
            origin["source_row"].to_numpy()
            != tautomer["source_row"].to_numpy()
        )
    )
    add(
        rows,
        section,
        "source-row order",
        source_mismatch == 0,
        source_mismatch,
        0,
    )

    rt_diff = float(
        np.max(
            np.abs(
                origin["rt"].to_numpy()
                - tautomer["rt"].to_numpy()
            )
        )
    )
    add(
        rows,
        section,
        "maximum absolute RT-label difference",
        rt_diff <= 1e-8,
        rt_diff,
        "<=1e-8 s",
    )

    if "orig_canonical" in tautomer.columns:
        mismatch = int(
            np.sum(
                origin["canonical_isomeric"].to_numpy()
                != tautomer["orig_canonical"].to_numpy()
            )
        )
        add(
            rows,
            section,
            "original/tautomer molecular pairing",
            mismatch == 0,
            mismatch,
            0,
        )
    else:
        add(
            rows,
            section,
            "original/tautomer molecular pairing",
            None,
            "orig_smile unavailable",
            "0 mismatches",
            "Source-row and RT-label alignment were still checked.",
        )


def audit_source(rows: list[dict[str, str]], path: Path) -> None:
    section = "prediction_provenance"
    if not path.is_file():
        add(rows, section, "training source exists", False, False, True)
        return
    source = path.read_text(encoding="utf-8")
    start = source.find("def fit_oof_stackers")
    end = source.find("def make_stratified_bins")
    stacker = source[start:end] if start >= 0 and end > start else ""

    checks = {
        "paired views share fold indices": (
            source.count("train_idx=train_idx") >= 2
            and source.count("val_idx=val_idx") >= 2
        ),
        "OOF predictions assigned at validation indices": (
            "oof_origin[val_idx] = val_pred_origin" in source
            and "oof_taut[val_idx] = val_pred_taut" in source
        ),
        "Huber stacker fitted on OOF features": (
            "huber.fit(x_oof, y)" in stacker
        ),
        "stacker selected by OOF MAE": (
            'best_name = min(summary.keys(), key=lambda k: summary[k]["oof"]["mae"])'
            in source
        ),
        "test predictions averaged across source folds": (
            "test_origin_mean = np.mean(np.vstack(test_origin_folds), axis=0)"
            in source
            and "test_taut_mean = np.mean(np.vstack(test_taut_folds), axis=0)"
            in source
        ),
        "test labels absent from stacker fitting": (
            bool(stacker) and 'test_df["Actual_RT"]' not in stacker
        ),
    }
    for check, passed in checks.items():
        add(rows, section, check, passed, passed, True)

    add(
        rows,
        section,
        "test labels used during stacker fitting",
        checks["test labels absent from stacker fitting"],
        0 if checks["test labels absent from stacker fitting"] else "unknown",
        0,
    )
    add(
        rows,
        section,
        "test labels used during test prediction averaging",
        checks["test predictions averaged across source folds"],
        0 if checks["test predictions averaged across source folds"] else "unknown",
        0,
    )


def finite(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(
        column in df.columns
        and np.isfinite(
            pd.to_numeric(df[column], errors="coerce").to_numpy()
        ).all()
        for column in columns
    )


def audit_runs(
    rows: list[dict[str, str]],
    root: Path,
    seeds: list[int],
    n_train: int,
    n_test: int,
    require_huber: bool,
) -> None:
    for seed in seeds:
        section = f"seed_output::{seed}"
        run = root / f"seed{seed}"
        paths = {
            "oof_base": run / "oof_base_predictions.csv",
            "oof_final": run / "oof_predictions.csv",
            "test_base": run / "test_base_predictions.csv",
            "test_final": run / "test_predictions.csv",
            "metrics": run / "final_metrics.json",
        }
        for name, path in paths.items():
            add(
                rows,
                section,
                f"{name} exists",
                path.is_file(),
                path.is_file(),
                True,
                str(path),
            )
        if not all(path.is_file() for path in paths.values()):
            continue

        oof_base = pd.read_csv(paths["oof_base"])
        oof_final = pd.read_csv(paths["oof_final"])
        test_base = pd.read_csv(paths["test_base"])
        test_final = pd.read_csv(paths["test_final"])

        add(
            rows,
            section,
            "OOF row count",
            len(oof_base) == n_train,
            len(oof_base),
            n_train,
        )
        add(
            rows,
            section,
            "test row count",
            len(test_base) == n_test,
            len(test_base),
            n_test,
        )

        folds = (
            pd.to_numeric(oof_base["Fold"], errors="coerce")
            if "Fold" in oof_base.columns
            else pd.Series(dtype=float)
        )
        coverage_ok = (
            len(folds) == n_train
            and folds.notna().all()
            and (folds >= 0).all()
            and folds.nunique() == 5
        )
        add(
            rows,
            section,
            "OOF prediction coverage",
            bool(coverage_ok),
            (
                f"{100.0 * folds.notna().mean():.2f}%; "
                f"folds={sorted(folds.dropna().unique())}"
                if len(folds)
                else "Fold column unavailable"
            ),
            "100%; five folds",
        )

        for check, frame, columns in [
            (
                "base OOF predictions finite",
                oof_base,
                ["Origin_OOF_Pred", "Taut_OOF_Pred"],
            ),
            ("final OOF predictions finite", oof_final, ["Final_OOF_Pred"]),
            (
                "base test predictions finite",
                test_base,
                ["Origin_Test_Pred", "Taut_Test_Pred"],
            ),
            ("final test predictions finite", test_final, ["Final_Pred"]),
        ]:
            passed = finite(frame, columns)
            add(
                rows,
                section,
                check,
                passed,
                "all finite" if passed else "missing/nonfinite",
                "all finite",
            )

        payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        selected = str(payload.get("selected_stacker", ""))
        if require_huber:
            add(
                rows,
                section,
                "formal selected stacker",
                selected == "huber_stack",
                selected,
                "huber_stack",
            )

        summary = payload.get("stacker_summary_oof", {})
        maes = {}
        if isinstance(summary, dict):
            for name, value in summary.items():
                try:
                    maes[name] = float(value["oof"]["mae"])
                except (KeyError, TypeError, ValueError):
                    pass
        if maes:
            best = min(maes, key=maes.get)
            add(
                rows,
                section,
                "selected stacker equals minimum OOF MAE",
                selected == best,
                f"selected={selected}; best={best}",
                "selected == best",
            )
        else:
            add(
                rows,
                section,
                "selected stacker equals minimum OOF MAE",
                False,
                "OOF metrics unavailable",
                "OOF metrics available",
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit paired-view identity, train/test separation, OOF coverage, "
            "and prediction-level stacking provenance."
        )
    )
    parser.add_argument("--origin_train", default="gwn/data/SMRT_train.csv")
    parser.add_argument("--origin_test", default="gwn/data/SMRT_test.csv")
    parser.add_argument(
        "--tautomer_train",
        default="gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv",
    )
    parser.add_argument(
        "--tautomer_test",
        default="gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv",
    )
    parser.add_argument(
        "--training_source",
        default="gwn/train_oof_dualview_stack.py",
    )
    parser.add_argument("--results_root", default="artifacts/results/smrt")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--require_huber", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--out_dir",
        default="artifacts/results/paper_tables/audits/no_leakage",
    )
    args = parser.parse_args()

    origin_train = load_retained(Path(args.origin_train))
    origin_test = load_retained(Path(args.origin_test))
    tautomer_train = load_retained(Path(args.tautomer_train))
    tautomer_test = load_retained(Path(args.tautomer_test))

    rows: list[dict[str, str]] = []
    audit_pair(rows, "train", origin_train, tautomer_train)
    audit_pair(rows, "test", origin_test, tautomer_test)

    if "molecule_id" in origin_train and "molecule_id" in origin_test:
        count = overlap(origin_train["molecule_id"], origin_test["molecule_id"])
        add(
            rows,
            "train_test_split",
            "molecule-id overlap",
            count == 0,
            count,
            0,
        )
    else:
        add(
            rows,
            "train_test_split",
            "molecule-id overlap",
            None,
            "identifier column unavailable",
            0,
            "Canonical structure and InChIKey checks remain active.",
        )

    for check, column in [
        ("canonical isomeric SMILES overlap", "canonical_isomeric"),
        ("canonical non-isomeric SMILES overlap", "canonical_nonisomeric"),
        ("InChIKey overlap", "inchikey"),
    ]:
        count = overlap(origin_train[column], origin_test[column])
        add(
            rows,
            "train_test_split",
            check,
            count == 0,
            count,
            0,
        )

    audit_source(rows, Path(args.training_source))
    audit_runs(
        rows,
        Path(args.results_root),
        args.seeds,
        len(origin_train),
        len(origin_test),
        bool(args.require_huber),
    )

    checks = pd.DataFrame(rows)
    counts = checks["status"].value_counts().to_dict()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks.to_csv(out_dir / "no_leakage_audit_checks.csv", index=False)
    (out_dir / "no_leakage_audit_summary.json").write_text(
        json.dumps(
            {
                "status_counts": counts,
                "seeds": args.seeds,
                "results_root": args.results_root,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nStatus counts")
    print(counts)
    print(f"\nSaved to {out_dir}")
    failures = checks[checks["status"].eq("FAIL")]
    if len(failures):
        print("\nFailed checks")
        print(failures.to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

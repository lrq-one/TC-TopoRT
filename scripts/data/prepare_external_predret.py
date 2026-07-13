#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_strict_tautomer_views import strict_tautomer_view  # noqa: E402


DATASET_ALIASES = {
    "FEM-short": "FEM_short_73",
    "FEM_short": "FEM_short_73",
    "FEM_short_73": "FEM_short_73",
    "UniToyama-Atlantis": "UniToyama_Atlantis_143",
    "UniToyama_Atlantis": "UniToyama_Atlantis_143",
    "UniToyama_Atlantis_143": "UniToyama_Atlantis_143",
    "FEM-long": "FEM_long_412",
    "FEM_long": "FEM_long_412",
    "FEM_long_412": "FEM_long_412",
    "Eawag-XBridgeC18": "Eawag_XBridgeC18_364",
    "Eawag_XBridgeC18": "Eawag_XBridgeC18_364",
    "Eawag_XBridgeC18_364": "Eawag_XBridgeC18_364",
    "LIFE-old": "LIFE_old_194",
    "LIFE_old": "LIFE_old_194",
    "LIFE_old_194": "LIFE_old_194",
    "MTBLS87": "MTBLS87_147",
    "MTBLS87_147": "MTBLS87_147",
    "LIFE-new": "LIFE_new_184",
    "LIFE_new": "LIFE_new_184",
    "LIFE_new_184": "LIFE_new_184",
    "Cao-HILIC": "Cao_HILIC_116",
    "Cao_HILIC": "Cao_HILIC_116",
    "Cao_HILIC_116": "Cao_HILIC_116",
    "IPB-Halle": "IPB_Halle_82",
    "IPB_Halle": "IPB_Halle_82",
    "IPB_Halle_82": "IPB_Halle_82",
    "FEM-lipids": "FEM_lipids_72",
    "FEM_lipids": "FEM_lipids_72",
    "FEM_lipids_72": "FEM_lipids_72",
}


def first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    lower = {str(column).strip().lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def normalize_dataset_name(value: object) -> str:
    text = str(value).strip()
    return DATASET_ALIASES.get(text, text)


def molecule_metadata(smiles: str) -> dict[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid SMILES")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = ""
    return {
        "canonical_smiles": canonical,
        "formula": formula,
        "inchikey": inchikey,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a standardized combined PredRet export into the three "
            "processed tables consumed by the TC-TopoRT external workflows."
        )
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help=(
            "Combined external table containing dataset_name, SMILES and RT. "
            "Accepted SMILES aliases: smiles, smile, canonical_smiles."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/data/external",
    )
    parser.add_argument(
        "--manifest",
        default="configs/external_datasets.csv",
    )
    parser.add_argument(
        "--strict_counts",
        type=int,
        choices=[0, 1],
        default=1,
        help="Require the retained counts to match the paper manifest.",
    )
    parser.add_argument(
        "--dummy_rt",
        type=float,
        default=1000.0,
        help=(
            "Dummy RT written to graph-construction CSVs. It must exceed 300 s "
            "because SMRTComplexDataset applies the SMRT retained-compound filter. "
            "Actual external RT values remain in the metadata table."
        ),
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if args.dummy_rt <= 300.0:
        raise ValueError("--dummy_rt must be greater than 300 s")

    source = pd.read_csv(input_csv)
    dataset_col = first_existing(source, ["dataset_name", "dataset", "system"])
    smiles_col = first_existing(source, ["smiles", "smile", "canonical_smiles"])
    rt_col = first_existing(source, ["rt", "retention_time", "retention time"])

    missing = [
        name
        for name, column in [
            ("dataset_name", dataset_col),
            ("smiles", smiles_col),
            ("rt", rt_col),
        ]
        if column is None
    ]
    if missing:
        raise ValueError(f"{input_csv}: missing required columns {missing}")

    record_id_col = first_existing(source, ["record_id", "id", "index"])
    name_col = first_existing(source, ["name", "compound_name", "compound"])

    rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []

    for source_row, record in source.iterrows():
        dataset_name = normalize_dataset_name(record[dataset_col])
        smiles = str(record[smiles_col]).strip()
        try:
            rt = float(record[rt_col])
        except (TypeError, ValueError):
            invalid_rows.append({"source_row": source_row, "reason": "invalid_rt"})
            continue
        if not np.isfinite(rt):
            invalid_rows.append({"source_row": source_row, "reason": "nonfinite_rt"})
            continue

        try:
            metadata = molecule_metadata(smiles)
        except Exception:
            invalid_rows.append({"source_row": source_row, "reason": "invalid_smiles"})
            continue

        tautomer = strict_tautomer_view(smiles)
        taut_smiles = str(tautomer["new_smile"])

        rows.append(
            {
                "dataset_name": dataset_name,
                "record_id": (
                    str(record[record_id_col])
                    if record_id_col is not None
                    else f"{dataset_name}:{source_row}"
                ),
                "name": (
                    str(record[name_col])
                    if name_col is not None and not pd.isna(record[name_col])
                    else ""
                ),
                "origin_smiles": smiles,
                "taut_smiles": taut_smiles,
                "rt": rt,
                "formula": metadata["formula"],
                "inchikey": metadata["inchikey"],
                "canonical_smiles": metadata["canonical_smiles"],
                "taut_changed": int(tautomer["real_changed"]),
                "smrt_exact_overlap": 0,
                "source_row": int(source_row),
            }
        )

    meta = pd.DataFrame(rows)
    if meta.empty:
        raise RuntimeError("No valid external rows were retained")

    manifest = pd.read_csv(manifest_path)
    expected_order = manifest["dataset_name"].astype(str).tolist()
    expected_counts = dict(
        zip(
            manifest["dataset_name"].astype(str),
            manifest["n"].astype(int),
        )
    )

    unknown = sorted(set(meta["dataset_name"]) - set(expected_order))
    if unknown:
        raise ValueError(f"Input contains datasets not present in the manifest: {unknown}")

    meta["dataset_name"] = pd.Categorical(
        meta["dataset_name"], categories=expected_order, ordered=True
    )
    meta = meta.sort_values(["dataset_name", "source_row"], kind="mergesort")
    meta = meta.reset_index(drop=True)
    meta["dataset_name"] = meta["dataset_name"].astype(str)
    meta.insert(0, "stage4_index", np.arange(len(meta), dtype=int))

    observed_counts = meta.groupby("dataset_name", sort=False).size().to_dict()
    count_rows = []
    mismatches = []
    for dataset_name in expected_order:
        expected = int(expected_counts[dataset_name])
        observed = int(observed_counts.get(dataset_name, 0))
        count_rows.append(
            {
                "dataset_name": dataset_name,
                "expected_n": expected,
                "observed_n": observed,
                "matches": expected == observed,
            }
        )
        if expected != observed:
            mismatches.append((dataset_name, expected, observed))

    if args.strict_counts and mismatches:
        message = "; ".join(
            f"{name}: expected {expected}, observed {observed}"
            for name, expected, observed in mismatches
        )
        raise RuntimeError(f"External dataset count mismatch: {message}")

    origin = pd.DataFrame(
        {
            "smile": meta["origin_smiles"],
            "rt": float(args.dummy_rt),
        }
    )
    tautomer = pd.DataFrame(
        {
            "smile": meta["taut_smiles"],
            "rt": float(args.dummy_rt),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_dir / "external_predret10_stage4_meta.csv", index=False)
    origin.to_csv(out_dir / "temp_external_predret10_origin.csv", index=False)
    tautomer.to_csv(out_dir / "temp_external_predret10_taut.csv", index=False)
    pd.DataFrame(count_rows).to_csv(out_dir / "external_dataset_count_audit.csv", index=False)
    pd.DataFrame(invalid_rows).to_csv(out_dir / "external_invalid_rows.csv", index=False)

    print("Prepared external TC-TopoRT inputs")
    print(pd.DataFrame(count_rows).to_string(index=False))
    print(f"Invalid rows excluded: {len(invalid_rows)}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()

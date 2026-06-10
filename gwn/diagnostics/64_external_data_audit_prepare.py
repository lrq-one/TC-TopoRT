import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_csv(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path} shape={df.shape}")


def safe_mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def safe_mol_from_inchi(inchi):
    if pd.isna(inchi):
        return None
    try:
        return Chem.MolFromInchi(str(inchi), sanitize=True)
    except Exception:
        return None


def mol_to_smiles(mol):
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def mol_to_formula(mol):
    if mol is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


def mol_to_inchikey(mol):
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def normalize_rt(x):
    try:
        v = float(x)
    except Exception:
        return np.nan
    if not np.isfinite(v):
        return np.nan
    return v


def choose_first_valid_smiles(row, cols):
    for c in cols:
        if c in row.index:
            val = row[c]
            if pd.notna(val) and str(val).strip() and str(val).strip().lower() != "nan":
                mol = safe_mol_from_smiles(val)
                if mol is not None:
                    return str(val), mol, c
    return None, None, None


def standardize_record(
    dataset_group,
    dataset_name,
    source_file,
    source_row,
    smiles=None,
    inchi=None,
    rt=None,
    name=None,
    formula_raw=None,
    inchikey_raw=None,
    extra=None,
):
    mol = None
    structure_source = None

    if smiles is not None and pd.notna(smiles) and str(smiles).strip():
        mol = safe_mol_from_smiles(smiles)
        structure_source = "smiles"

    if mol is None and inchi is not None and pd.notna(inchi) and str(inchi).strip():
        mol = safe_mol_from_inchi(inchi)
        structure_source = "inchi"

    canonical_smiles = mol_to_smiles(mol)
    formula = mol_to_formula(mol)
    inchikey = mol_to_inchikey(mol)

    row = {
        "dataset_group": dataset_group,
        "dataset_name": dataset_name,
        "source_file": str(source_file),
        "source_row": int(source_row) if source_row is not None else -1,
        "record_id": f"{dataset_name}_{source_row}",
        "name": name,
        "smiles_raw": smiles,
        "inchi_raw": inchi,
        "structure_source": structure_source,
        "valid_rdkit": int(mol is not None),
        "canonical_smiles": canonical_smiles,
        "formula": formula,
        "formula_raw": formula_raw,
        "inchikey": inchikey,
        "inchikey_raw": inchikey_raw,
        "inchikey14": inchikey[:14] if isinstance(inchikey, str) and len(inchikey) >= 14 else None,
        "rt": normalize_rt(rt),
    }

    if extra:
        row.update(extra)

    return row


def load_predret10_txt(txt_dir):
    txt_dir = Path(txt_dir)
    rows = []

    for path in sorted(txt_dir.glob("*.txt")):
        dataset_name = path.stem
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            df = pd.read_csv(path)

        df.columns = [str(c).strip().lower() for c in df.columns]

        if "smiles" not in df.columns or "rt" not in df.columns:
            print(f"[WARN] skip {path}, columns={df.columns.tolist()}")
            continue

        for i, r in df.iterrows():
            rows.append(standardize_record(
                dataset_group="predret10",
                dataset_name=dataset_name,
                source_file=path,
                source_row=i,
                smiles=r.get("smiles"),
                rt=r.get("rt"),
                extra={"rt_unit_assumed": "seconds_from_txt"},
            ))

    return pd.DataFrame(rows)


def load_metabobase(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] missing {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    rows = []

    for i, r in df.iterrows():
        smi, mol, smi_col = choose_first_valid_smiles(r, ["computedSMILES", "SMILES", "smiles"])
        if smi is None:
            smi = r.get("SMILES", None)

        rows.append(standardize_record(
            dataset_group="raw_libraries",
            dataset_name="MetaboBase",
            source_file=path,
            source_row=i,
            smiles=smi,
            inchi=r.get("InChI", None),
            rt=r.get("RT", None),
            name=r.get("NAME", None),
            formula_raw=r.get("Formula", None),
            inchikey_raw=r.get("InChIKey", None),
            extra={
                "rt_unit_assumed": "seconds",
                "smiles_column_used": smi_col,
                "has_msms": int("Peaks" in df.columns and pd.notna(r.get("Peaks", None))),
                "precursor_type": r.get("Precursor_type", None),
                "ion_mode": r.get("Ion_mode", None),
            },
        ))

    return pd.DataFrame(rows)


def load_riken_mona(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] missing {path}")
        return pd.DataFrame()

    xl = pd.ExcelFile(path)
    sheet = "mona_filter_result" if "mona_filter_result" in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)

    rows = []

    for i, r in df.iterrows():
        smi, mol, smi_col = choose_first_valid_smiles(r, ["smiles", "SMILES"])
        if smi is None:
            smi = r.get("smiles", r.get("SMILES", None))

        rt = r.get("RT", None)
        if pd.isna(rt) and "retention time" in df.columns:
            # retention time in this file appears to be minutes, RT appears to be seconds when present.
            rt_min = r.get("retention time")
            try:
                rt = float(rt_min) * 60.0
            except Exception:
                rt = np.nan

        rows.append(standardize_record(
            dataset_group="raw_libraries",
            dataset_name="RIKEN_MONA",
            source_file=path,
            source_row=i,
            smiles=smi,
            inchi=r.get("InChI", None),
            rt=rt,
            name=r.get("compound_name", None),
            formula_raw=r.get("molecular formula", None),
            inchikey_raw=r.get("InChIKey", r.get("inchikey", None)),
            extra={
                "rt_unit_assumed": "seconds_from_RT_column",
                "smiles_column_used": smi_col,
                "has_msms": int("spectrum" in df.columns and pd.notna(r.get("spectrum", None))),
                "column": r.get("column", None),
                "ion_mode": r.get("ionization mode", None),
                "precursor_type": r.get("precursor type", None),
            },
        ))

    return pd.DataFrame(rows)


def load_massbank(path, dataset_name):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] missing {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    rows = []

    for i, r in df.iterrows():
        rows.append(standardize_record(
            dataset_group="raw_libraries",
            dataset_name=dataset_name,
            source_file=path,
            source_row=i,
            inchi=r.get("InChI", None),
            rt=r.get("RT", None),
            extra={
                "rt_unit_assumed": "seconds_or_original_file",
                "has_msms": 0,
            },
        ))

    return pd.DataFrame(rows)


def load_smrt_meta(train_csv, test_csv):
    rows = []

    for split, path in [("smrt_train", train_csv), ("smrt_test", test_csv)]:
        df = pd.read_csv(path, engine="python")
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "smile" in df.columns and "smiles" not in df.columns:
            df = df.rename(columns={"smile": "smiles"})
        if "smiles" not in df.columns or "rt" not in df.columns:
            continue
        df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
        df = df[df["rt"] > 300.0].copy()

        for i, r in df.iterrows():
            mol = safe_mol_from_smiles(r["smiles"])
            rows.append({
                "smrt_split": split,
                "smrt_source_row": int(i),
                "smrt_canonical_smiles": mol_to_smiles(mol),
                "smrt_formula": mol_to_formula(mol),
                "smrt_inchikey": mol_to_inchikey(mol),
                "smrt_inchikey14": mol_to_inchikey(mol)[:14] if mol_to_inchikey(mol) else None,
            })

    return pd.DataFrame(rows)


def dataset_audit_summary(df):
    rows = []

    for dataset_name, sub in df.groupby("dataset_name", dropna=False):
        valid = sub[sub["valid_rdkit"] == 1].copy()
        rt_valid = valid[pd.to_numeric(valid["rt"], errors="coerce").notna()].copy()

        rows.append({
            "dataset_name": dataset_name,
            "dataset_group": sub["dataset_group"].iloc[0] if len(sub) else None,
            "raw_rows": int(len(sub)),
            "valid_rdkit_rows": int(len(valid)),
            "valid_rdkit_ratio": float(len(valid) / max(len(sub), 1)),
            "rt_valid_rows": int(len(rt_valid)),
            "rt_valid_ratio": float(len(rt_valid) / max(len(sub), 1)),
            "unique_canonical_smiles": int(valid["canonical_smiles"].nunique(dropna=True)),
            "unique_inchikey": int(valid["inchikey"].nunique(dropna=True)),
            "unique_formula": int(valid["formula"].nunique(dropna=True)),
            "rt_min": float(rt_valid["rt"].min()) if len(rt_valid) else np.nan,
            "rt_median": float(rt_valid["rt"].median()) if len(rt_valid) else np.nan,
            "rt_max": float(rt_valid["rt"].max()) if len(rt_valid) else np.nan,
            "has_msms_rows": int(valid["has_msms"].fillna(0).sum()) if "has_msms" in valid.columns else 0,
        })

    return pd.DataFrame(rows).sort_values(["dataset_group", "dataset_name"])


def formula_pool_summary(df, min_pool_sizes=(2, 5, 10)):
    rows = []
    valid = df[(df["valid_rdkit"] == 1) & df["formula"].notna() & pd.to_numeric(df["rt"], errors="coerce").notna()].copy()

    for dataset_name, sub in valid.groupby("dataset_name", dropna=False):
        counts = sub.groupby("formula").size().rename("pool_n").reset_index()
        tmp = sub.merge(counts, on="formula", how="left")

        for min_pool_size in min_pool_sizes:
            eligible = tmp[tmp["pool_n"] >= min_pool_size]
            rows.append({
                "dataset_name": dataset_name,
                "dataset_group": sub["dataset_group"].iloc[0] if len(sub) else None,
                "min_pool_size": int(min_pool_size),
                "total_valid_rt_rows": int(len(sub)),
                "eligible_queries": int(len(eligible)),
                "eligible_query_ratio": float(len(eligible) / max(len(sub), 1)),
                "unique_formulas_total": int(sub["formula"].nunique()),
                "unique_formulas_eligible": int(eligible["formula"].nunique()) if len(eligible) else 0,
                "pool_n_mean_for_eligible": float(eligible["pool_n"].mean()) if len(eligible) else np.nan,
                "pool_n_median_for_eligible": float(eligible["pool_n"].median()) if len(eligible) else np.nan,
                "pool_n_max": int(tmp["pool_n"].max()) if len(tmp) else 0,
            })

    return pd.DataFrame(rows).sort_values(["dataset_group", "dataset_name", "min_pool_size"])


def duplicate_summary(df):
    valid = df[df["valid_rdkit"] == 1].copy()
    rows = []

    for dataset_name, sub in valid.groupby("dataset_name", dropna=False):
        rows.append({
            "dataset_name": dataset_name,
            "rows": int(len(sub)),
            "duplicate_canonical_rows": int(len(sub) - sub["canonical_smiles"].nunique(dropna=True)),
            "duplicate_inchikey_rows": int(len(sub) - sub["inchikey"].nunique(dropna=True)),
            "duplicate_formula_rows": int(len(sub) - sub["formula"].nunique(dropna=True)),
        })

    return pd.DataFrame(rows).sort_values("dataset_name")


def smrt_overlap_summary(external_df, smrt_df):
    ext = external_df[external_df["valid_rdkit"] == 1].copy()

    smrt_can = set(smrt_df["smrt_canonical_smiles"].dropna())
    smrt_ik = set(smrt_df["smrt_inchikey"].dropna())
    smrt_ik14 = set(smrt_df["smrt_inchikey14"].dropna())

    detail_rows = []
    summary_rows = []

    for dataset_name, sub in ext.groupby("dataset_name", dropna=False):
        exact_can = sub["canonical_smiles"].isin(smrt_can)
        exact_ik = sub["inchikey"].isin(smrt_ik)
        ik14 = sub["inchikey14"].isin(smrt_ik14)

        summary_rows.append({
            "dataset_name": dataset_name,
            "valid_rows": int(len(sub)),
            "canonical_smiles_overlap_with_smrt": int(exact_can.sum()),
            "inchikey_overlap_with_smrt": int(exact_ik.sum()),
            "inchikey14_overlap_with_smrt": int(ik14.sum()),
            "canonical_overlap_ratio": float(exact_can.mean()) if len(sub) else np.nan,
            "inchikey_overlap_ratio": float(exact_ik.mean()) if len(sub) else np.nan,
            "inchikey14_overlap_ratio": float(ik14.mean()) if len(sub) else np.nan,
        })

        tmp = sub[exact_can | exact_ik | ik14].copy()
        if len(tmp):
            tmp["overlap_canonical"] = exact_can[exact_can | exact_ik | ik14].astype(int).values
            tmp["overlap_inchikey"] = exact_ik[exact_can | exact_ik | ik14].astype(int).values
            tmp["overlap_inchikey14"] = ik14[exact_can | exact_ik | ik14].astype(int).values
            detail_rows.append(tmp)

    summary = pd.DataFrame(summary_rows).sort_values("dataset_name")
    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return summary, detail


def write_readme(out_dir):
    text = """External data audit and preparation

Inputs:
- external_data/predret10/smiles_rt_txt/*.txt
- external_data/raw_libraries/MetaboBase.csv
- external_data/raw_libraries/RIKEN_MONA.xlsx
- external_data/raw_libraries/MassBank1.csv
- external_data/raw_libraries/MassBank2.csv

Outputs:
- external_predret10_clean.csv
- external_libraries_clean.csv
- external_all_clean.csv
- external_dataset_audit_summary.csv
- external_formula_pool_summary.csv
- external_duplicate_summary.csv
- external_smrt_overlap_summary.csv
- external_smrt_overlap_details.csv

This script does not run model inference. It only checks whether external datasets are valid and whether they have enough formula-level candidate pools for Stage 2B and Stage 4 experiments.
"""
    path = Path(out_dir) / "README_outputs.txt"
    path.write_text(text, encoding="utf-8")
    print(f"[SAVE] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="paper_analysis_external")
    ap.add_argument("--predret10_txt_dir", default="external_data/predret10/smiles_rt_txt")
    ap.add_argument("--raw_library_dir", default="external_data/raw_libraries")
    ap.add_argument("--smrt_train_csv", default="data/SMRT_train.csv")
    ap.add_argument("--smrt_test_csv", default="data/SMRT_test.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print("=== External data audit and preparation ===")
    print("out_dir:", out_dir)

    print("\n=== Load PredRet10 smiles-RT txt files ===")
    predret10 = load_predret10_txt(args.predret10_txt_dir)
    print("predret10:", predret10.shape)

    print("\n=== Load raw libraries ===")
    raw_dir = Path(args.raw_library_dir)

    metabobase = load_metabobase(raw_dir / "MetaboBase.csv")
    print("MetaboBase:", metabobase.shape)

    riken = load_riken_mona(raw_dir / "RIKEN_MONA.xlsx")
    print("RIKEN_MONA:", riken.shape)

    mb1 = load_massbank(raw_dir / "MassBank1.csv", "MassBank1")
    print("MassBank1:", mb1.shape)

    mb2 = load_massbank(raw_dir / "MassBank2.csv", "MassBank2")
    print("MassBank2:", mb2.shape)

    libraries = pd.concat([metabobase, riken, mb1, mb2], ignore_index=True)
    all_df = pd.concat([predret10, libraries], ignore_index=True)

    save_csv(predret10, out_dir / "external_predret10_clean.csv")
    save_csv(libraries, out_dir / "external_libraries_clean.csv")
    save_csv(all_df, out_dir / "external_all_clean.csv")

    print("\n=== Dataset audit summary ===")
    audit = dataset_audit_summary(all_df)
    save_csv(audit, out_dir / "external_dataset_audit_summary.csv")
    print(audit.to_string(index=False))

    print("\n=== Formula pool summary ===")
    pool = formula_pool_summary(all_df, min_pool_sizes=(2, 5, 10))
    save_csv(pool, out_dir / "external_formula_pool_summary.csv")
    print(pool.to_string(index=False))

    print("\n=== Duplicate summary ===")
    dup = duplicate_summary(all_df)
    save_csv(dup, out_dir / "external_duplicate_summary.csv")
    print(dup.to_string(index=False))

    print("\n=== SMRT overlap audit ===")
    smrt = load_smrt_meta(args.smrt_train_csv, args.smrt_test_csv)
    save_csv(smrt, out_dir / "smrt_train_test_structure_meta.csv")

    overlap_summary, overlap_detail = smrt_overlap_summary(all_df, smrt)
    save_csv(overlap_summary, out_dir / "external_smrt_overlap_summary.csv")
    save_csv(overlap_detail, out_dir / "external_smrt_overlap_details.csv")
    print(overlap_summary.to_string(index=False))

    write_readme(out_dir)

    print("\n✅ Done. Outputs are in:", out_dir)


if __name__ == "__main__":
    main()

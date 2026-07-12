import os
import argparse
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

TAUT = rdMolStandardize.TautomerEnumerator()

# 尽量避免极端分子枚举太久
try:
    TAUT.SetMaxTautomers(128)
except Exception:
    pass

try:
    TAUT.SetMaxTransforms(128)
except Exception:
    pass


def safe_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return mol
    except Exception:
        return None


def canon_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def formula(mol):
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return ""


def strict_tautomer_view(smiles):
    """
    只统计真正的 tautomer canonical 改变。

    不用 raw SMILES 和 tautomer SMILES 直接比较，
    而是先把原始 mol canonicalize，再和 tautomer canonical 比较。
    这样可以避免把普通 SMILES 重排误判为 tautomer changed。
    """
    original = str(smiles)
    mol = safe_mol(original)

    if mol is None:
        return {
            "new_smile": original,
            "orig_canon": original,
            "taut_canon": original,
            "raw_changed": 0,
            "real_changed": 0,
            "formula_same": 0,
            "heavy_same": 0,
            "fallback": 1,
            "reason": "parse_failed",
        }

    try:
        orig_canon = canon_smiles(mol)
        orig_formula = formula(mol)
        orig_heavy = mol.GetNumHeavyAtoms()

        taut_mol = TAUT.Canonicalize(mol)
        if taut_mol is None:
            raise RuntimeError("tautomer_none")

        taut_canon = canon_smiles(taut_mol)
        taut_formula = formula(taut_mol)
        taut_heavy = taut_mol.GetNumHeavyAtoms()

        # 互变异构应该保持分子式和重原子数
        formula_same = int(orig_formula == taut_formula)
        heavy_same = int(orig_heavy == taut_heavy)

        if not formula_same or not heavy_same:
            return {
                "new_smile": original,
                "orig_canon": orig_canon,
                "taut_canon": taut_canon,
                "raw_changed": int(taut_canon != original),
                "real_changed": 0,
                "formula_same": formula_same,
                "heavy_same": heavy_same,
                "fallback": 1,
                "reason": "formula_or_heavy_changed",
            }

        real_changed = int(taut_canon != orig_canon)

        # 关键：没有真正 tautomer 改变时，保持官方原始 SMILES，不强行 canonicalize
        if real_changed:
            new_smile = taut_canon
        else:
            new_smile = original

        return {
            "new_smile": new_smile,
            "orig_canon": orig_canon,
            "taut_canon": taut_canon,
            "raw_changed": int(taut_canon != original),
            "real_changed": real_changed,
            "formula_same": formula_same,
            "heavy_same": heavy_same,
            "fallback": 0,
            "reason": "ok",
        }

    except Exception as e:
        return {
            "new_smile": original,
            "orig_canon": original,
            "taut_canon": original,
            "raw_changed": 0,
            "real_changed": 0,
            "formula_same": 0,
            "heavy_same": 0,
            "fallback": 1,
            "reason": f"exception:{type(e).__name__}",
        }


def build_one(input_csv, output_csv, audit_csv):
    df = pd.read_csv(input_csv)
    assert "smile" in df.columns, f"{input_csv} must have column smile"

    rows = []
    new_smiles = []

    for idx, s in enumerate(df["smile"].astype(str).tolist()):
        info = strict_tautomer_view(s)
        new_smiles.append(info["new_smile"])

        rows.append({
            "idx": idx,
            "orig_smile": s,
            "new_smile": info["new_smile"],
            "orig_canon": info["orig_canon"],
            "taut_canon": info["taut_canon"],
            "raw_changed": info["raw_changed"],
            "real_changed": info["real_changed"],
            "formula_same": info["formula_same"],
            "heavy_same": info["heavy_same"],
            "fallback": info["fallback"],
            "reason": info["reason"],
        })

    out = df.copy()
    out["orig_smile"] = df["smile"].astype(str)
    out["smile"] = new_smiles

    audit = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False)
    audit.to_csv(audit_csv, index=False)

    print("\n", output_csv)
    print("rows:", len(out))
    print("raw_changed:", int(audit["raw_changed"].sum()), "/", len(out), "ratio:", float(audit["raw_changed"].mean()))
    print("real_tautomer_changed:", int(audit["real_changed"].sum()), "/", len(out), "ratio:", float(audit["real_changed"].mean()))
    print("fallback:", int(audit["fallback"].sum()), "/", len(out), "ratio:", float(audit["fallback"].mean()))

    ex = audit[audit["real_changed"] == 1].head(10)
    if len(ex) > 0:
        print("\nexamples real_changed:")
        for _, r in ex.iterrows():
            print("ORIG:", r["orig_smile"])
            print("TAUT:", r["new_smile"])
            print("---")



def main():
    ap = argparse.ArgumentParser(
        description="Build strict tautomer canonical SMRT CSVs without changing RT labels or train/test split."
    )
    ap.add_argument(
        "--train_csv",
        default="gwn/data/SMRT_train.csv",
        help="Input official SMRT train CSV. Must contain column 'smile'.",
    )
    ap.add_argument(
        "--test_csv",
        default="gwn/data/SMRT_test.csv",
        help="Input official SMRT test CSV. Must contain column 'smile'.",
    )
    ap.add_argument(
        "--out_dir",
        default="artifacts/data/strict_tautomer_generated",
        help="Output directory for generated strict tautomer CSVs. Default avoids overwriting final paper data.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    build_one(
        args.train_csv,
        os.path.join(args.out_dir, "SMRT_train_tautomer_strict.csv"),
        os.path.join(args.out_dir, "SMRT_train_tautomer_strict_audit.csv"),
    )

    build_one(
        args.test_csv,
        os.path.join(args.out_dir, "SMRT_test_tautomer_strict.csv"),
        os.path.join(args.out_dir, "SMRT_test_tautomer_strict_audit.csv"),
    )


if __name__ == "__main__":
    main()

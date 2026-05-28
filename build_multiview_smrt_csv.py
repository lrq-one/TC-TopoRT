import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


TAUT_ENUM = rdMolStandardize.TautomerEnumerator()
UNCHARGER = rdMolStandardize.Uncharger()
REIONIZER = rdMolStandardize.Reionizer()


def safe_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def mol_to_smiles_or_original(mol, original):
    try:
        if mol is None:
            return str(original), 0
        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if not smi:
            return str(original), 0
        changed = int(smi != str(original))
        return smi, changed
    except Exception:
        return str(original), 0


def tautomer_view(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return str(smiles), 0

    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        taut = TAUT_ENUM.Canonicalize(mol)
        return mol_to_smiles_or_original(taut, smiles)
    except Exception:
        return str(smiles), 0


def protocharge_view(smiles):
    """
    这不是严格 pKa protomer 枚举。
    这是 RDKit 的 charge/protonation proxy view：
    Cleanup + FragmentParent + Uncharger + Reionizer。
    用来给模型一个更标准的电荷/离子化表达。
    """
    mol = safe_mol(smiles)
    if mol is None:
        return str(smiles), 0

    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = UNCHARGER.uncharge(mol)
        mol = REIONIZER.reionize(mol)
        return mol_to_smiles_or_original(mol, smiles)
    except Exception:
        return str(smiles), 0


def build_one(input_csv, output_csv, view_name):
    df = pd.read_csv(input_csv)
    assert "smile" in df.columns, f"{input_csv} must contain column 'smile'"

    new_smiles = []
    changed = []

    for s in df["smile"].astype(str).tolist():
        if view_name == "tautomer":
            ns, ch = tautomer_view(s)
        elif view_name == "protocharge":
            ns, ch = protocharge_view(s)
        else:
            raise ValueError(view_name)

        # 最后再检查一次，失败就退回原 SMILES，保证行数和标签不变
        if safe_mol(ns) is None:
            ns, ch = s, 0

        new_smiles.append(ns)
        changed.append(ch)

    out = df.copy()
    out["orig_smile"] = df["smile"].astype(str)
    out["smile"] = new_smiles
    out[f"{view_name}_changed"] = changed
    out["view_name"] = view_name

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(output_csv)
    print("rows:", len(out))
    print("changed:", sum(changed), "/", len(out), "ratio:", sum(changed) / max(len(out), 1))


def main():
    train_csv = "./SMRT_data/data/SMRT_train.csv"
    test_csv = "./SMRT_data/data/SMRT_test.csv"

    build_one(train_csv, "./SMRT_data/data/SMRT_train_tautomer.csv", "tautomer")
    build_one(test_csv, "./SMRT_data/data/SMRT_test_tautomer.csv", "tautomer")

    build_one(train_csv, "./SMRT_data/data/SMRT_train_protocharge.csv", "protocharge")
    build_one(test_csv, "./SMRT_data/data/SMRT_test_protocharge.csv", "protocharge")


if __name__ == "__main__":
    main()

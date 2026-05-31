import argparse
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen


def norm_pred(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    smi = cols.get("smiles", "SMILES")
    y = cols.get("actual_rt", None) or cols.get("y", None) or cols.get("y_true", None)
    p = cols.get("predicted_rt", None) or cols.get("y_pred", None) or cols.get("pred", None)

    if y is None or p is None:
        raise ValueError(f"bad columns in {path}: {df.columns.tolist()}")

    out = df[[smi, y, p]].copy()
    out.columns = ["SMILES", "y", "pred"]
    out["SMILES"] = out["SMILES"].astype(str)
    out["y"] = out["y"].astype(float)
    out["pred"] = out["pred"].astype(float)
    out["abs_err"] = (out["y"] - out["pred"]).abs()
    return out


def safe_mol(smi):
    try:
        return Chem.MolFromSmiles(str(smi))
    except Exception:
        return None


def flags_from_smiles(smi):
    s = str(smi)
    mol = safe_mol(s)

    out = {}

    # 字符串 motif：对你现在 worst list 最直接
    out["imidic_NC_O"] = int(("N=C(O)" in s) or ("C(O)=N" in s) or ("C(=N)O" in s) or ("N=C(C)O" in s))
    out["amide"] = int(("C(=O)N" in s) or ("NC(=O)" in s))
    out["sulfonamide"] = int(("S(=O)(=O)N" in s) or ("NS(=O)(=O)" in s))
    out["sulfone"] = int("S(=O)(=O)" in s)
    out["cf3"] = int("C(F)(F)F" in s)
    out["halogen"] = int(("Cl" in s) or ("Br" in s) or ("F" in s) or ("I" in s))
    out["piperazine_like"] = int(("N1CCN" in s) or ("N2CCN" in s) or ("N3CCN" in s))
    out["morpholine_like"] = int(("N1CCOCC1" in s) or ("N2CCOCC2" in s) or ("OCCN" in s))
    out["many_lower_hetero_arom"] = int((s.count("n") + s.count("o") + s.count("s")) >= 3)

    if mol is None:
        out.update({
            "mol_ok": 0,
            "ring_count": 0,
            "arom_ring_count": 0,
            "hetero_count": 0,
            "rot_bonds": 0,
            "tpsa": 0.0,
            "hba": 0,
            "hbd": 0,
            "logp": 0.0,
            "mw": 0.0,
            "polar_lipophilic_conflict": 0,
            "many_rings": 0,
            "high_tpsa": 0,
            "many_hba": 0,
            "flexible": 0,
        })
        return out

    hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    arom_ring_count = rdMolDescriptors.CalcNumAromaticRings(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)

    out.update({
        "mol_ok": 1,
        "ring_count": ring_count,
        "arom_ring_count": arom_ring_count,
        "hetero_count": hetero,
        "rot_bonds": rot_bonds,
        "tpsa": tpsa,
        "hba": hba,
        "hbd": hbd,
        "logp": logp,
        "mw": mw,
        "polar_lipophilic_conflict": int((logp >= 3.0) and (tpsa >= 70.0)),
        "many_rings": int(ring_count >= 4),
        "high_tpsa": int(tpsa >= 80.0),
        "many_hba": int(hba >= 6),
        "flexible": int(rot_bonds >= 6),
    })

    return out


def add_flags(df):
    rows = [flags_from_smiles(s) for s in df["SMILES"]]
    f = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), f.reset_index(drop=True)], axis=1)


def enrichment_table(df, hard_thr):
    df = df.copy()
    df["hard"] = (df["abs_err"] >= hard_thr).astype(int)

    flag_cols = [
        "imidic_NC_O",
        "amide",
        "sulfonamide",
        "sulfone",
        "cf3",
        "halogen",
        "piperazine_like",
        "morpholine_like",
        "many_lower_hetero_arom",
        "polar_lipophilic_conflict",
        "many_rings",
        "high_tpsa",
        "many_hba",
        "flexible",
    ]

    rows = []
    hard = df[df["hard"] == 1]
    easy = df[df["hard"] == 0]

    for c in flag_cols:
        ph = hard[c].mean() if len(hard) else 0.0
        pe = easy[c].mean() if len(easy) else 0.0
        ratio = (ph + 1e-6) / (pe + 1e-6)
        support = int(hard[c].sum()) if len(hard) else 0
        rows.append({
            "motif": c,
            "hard_rate": ph,
            "easy_rate": pe,
            "enrichment": ratio,
            "hard_support": support,
        })

    tab = pd.DataFrame(rows).sort_values(["enrichment", "hard_support"], ascending=[False, False])
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pred", required=True)
    ap.add_argument("--val_pred", required=True)
    ap.add_argument("--test_pred", required=True)
    ap.add_argument("--out_dir", default="results_TopoCellRT_CWNReplace_orig/2d_hard_motif_analysis")
    ap.add_argument("--hard_thr", type=float, default=80.0)
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    train = add_flags(norm_pred(args.train_pred))
    val = add_flags(norm_pred(args.val_pred))
    test = add_flags(norm_pred(args.test_pred))

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    trainval = pd.concat([train, val], axis=0).reset_index(drop=True)

    tab_tv = enrichment_table(trainval, args.hard_thr)
    tab_test = enrichment_table(test, args.hard_thr)

    trainval.to_csv(f"{args.out_dir}/trainval_with_2d_flags.csv", index=False)
    test.to_csv(f"{args.out_dir}/test_with_2d_flags.csv", index=False)
    tab_tv.to_csv(f"{args.out_dir}/trainval_motif_enrichment.csv", index=False)
    tab_test.to_csv(f"{args.out_dir}/test_motif_enrichment_diagnostic.csv", index=False)

    print("\n=== Train+Val hard motif enrichment, hard_thr =", args.hard_thr, "===")
    print(tab_tv.to_string(index=False))

    print("\n=== Test motif enrichment diagnostic only ===")
    print(tab_test.to_string(index=False))

    print("\nsaved to:", args.out_dir)


if __name__ == "__main__":
    main()

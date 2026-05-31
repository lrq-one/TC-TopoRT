import argparse
import os
import re
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
    out.columns = ["SMILES", "Actual_RT", "Predicted_RT"]
    out["SMILES"] = out["SMILES"].astype(str)
    out["Actual_RT"] = out["Actual_RT"].astype(float)
    out["Predicted_RT"] = out["Predicted_RT"].astype(float)
    out["Abs_Error"] = (out["Actual_RT"] - out["Predicted_RT"]).abs()
    return out


def safe_mol(smi):
    try:
        return Chem.MolFromSmiles(str(smi))
    except Exception:
        return None


def chem_features(smi):
    s = str(smi)
    mol = safe_mol(s)

    imidic = bool(re.search(r"N=C\(O\)|C\(O\)=N|C\(=N\)O|N=C\(C\)O", s))
    sulfonyl = "S(=O)(=O)" in s
    cf3 = "C(F)(F)F" in s
    halogen = ("Cl" in s) or ("Br" in s) or ("F" in s) or ("I" in s)
    cyclic_amine = bool(re.search(r"N[123456789]CC|N[123456789]CCC|N[123456789]CCCC|N[123456789]CCN|N[123456789]CCO", s))
    lower_hetero_arom_count = s.count("n") + s.count("o") + s.count("s")

    if mol is None:
        return {
            "mol_ok": 0,
            "imidic": int(imidic),
            "sulfonyl": int(sulfonyl),
            "cf3": int(cf3),
            "halogen": int(halogen),
            "cyclic_amine": int(cyclic_amine),
            "lower_hetero_arom_count": lower_hetero_arom_count,
            "rings": 0,
            "arom_rings": 0,
            "hetero": 0,
            "hba": 0,
            "hbd": 0,
            "tpsa": 0.0,
            "logp": 0.0,
            "rot": 0,
            "mw": 0.0,
            "chem_score": 0.0,
            "family_hit": 0,
            "family_reason": "bad_mol",
        }

    rings = rdMolDescriptors.CalcNumRings(mol)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    logp = Crippen.MolLogP(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    mw = Descriptors.MolWt(mol)

    poly_hetero_ring = (hetero >= 6) and (rings >= 3)
    polycyclic = rings >= 4
    polar_lipophilic_conflict = (logp >= 3.0) and (tpsa >= 70.0)
    high_tpsa = tpsa >= 80.0
    many_hba = hba >= 6
    flexible = rot >= 6

    reasons = []

    # 核心大类 1：复杂多杂原子多环/稠合杂环
    if poly_hetero_ring and (polycyclic or arom_rings >= 3):
        reasons.append("poly_hetero_polycyclic")

    # 核心大类 2：互变异构敏感 + 多环/多杂原子
    if imidic and rings >= 3 and hetero >= 4:
        reasons.append("tautomeric_imidic_ring")

    # 核心大类 3：磺酰胺/砜 + 多环/多杂原子
    if sulfonyl and rings >= 3 and hetero >= 5:
        reasons.append("sulfonyl_complex")

    # 核心大类 4：环胺/柔性胺 + 复杂杂环/互变异构
    if cyclic_amine and rings >= 3 and (imidic or hetero >= 6 or sulfonyl):
        reasons.append("cyclic_amine_complex")

    # 核心大类 5：极性-疏水冲突，RT 可能受构象/暴露面积影响
    if polar_lipophilic_conflict and rings >= 3 and hetero >= 4:
        reasons.append("polar_lipophilic_conflict")

    # 核心大类 6：卤素/CF3 + 复杂杂环
    if halogen and rings >= 3 and hetero >= 5 and (imidic or sulfonyl or arom_rings >= 3):
        reasons.append("halogenated_complex_heterocycle")

    chem_score = 0.0
    chem_score += 2.0 * poly_hetero_ring
    chem_score += 1.5 * polycyclic
    chem_score += 1.5 * imidic
    chem_score += 1.2 * polar_lipophilic_conflict
    chem_score += 1.0 * cyclic_amine
    chem_score += 1.0 * sulfonyl
    chem_score += 0.6 * halogen
    chem_score += 0.6 * high_tpsa
    chem_score += 0.5 * many_hba
    chem_score += 0.4 * flexible
    chem_score += min(lower_hetero_arom_count, 5) * 0.2

    family_hit = int(len(reasons) > 0)

    return {
        "mol_ok": 1,
        "imidic": int(imidic),
        "sulfonyl": int(sulfonyl),
        "cf3": int(cf3),
        "halogen": int(halogen),
        "cyclic_amine": int(cyclic_amine),
        "lower_hetero_arom_count": lower_hetero_arom_count,
        "rings": rings,
        "arom_rings": arom_rings,
        "hetero": hetero,
        "hba": hba,
        "hbd": hbd,
        "tpsa": tpsa,
        "logp": logp,
        "rot": rot,
        "mw": mw,
        "chem_score": chem_score,
        "family_hit": family_hit,
        "family_reason": "|".join(reasons) if reasons else "none",
    }


def add_chem(df):
    feats = pd.DataFrame([chem_features(s) for s in df["SMILES"]])
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)


def select_family(df, split, max_n, min_score):
    df = df.copy()
    df["split"] = split

    sel = df[(df["family_hit"] == 1) & (df["chem_score"] >= min_score)].copy()
    sel = sel.sort_values(["chem_score", "rings", "hetero"], ascending=False)

    if max_n > 0 and len(sel) > max_n:
        sel = sel.head(max_n)

    return sel


def report(name, df, sel):
    print(f"\n=== {name} ===")
    print("all N:", len(df))
    print("selected N:", len(sel))
    if len(sel) == 0:
        return

    print("selected Abs_Error stats:")
    print(sel["Abs_Error"].agg(["mean", "median", "max"]).to_string())
    print(">80:", int((sel["Abs_Error"] > 80).sum()))
    print(">100:", int((sel["Abs_Error"] > 100).sum()))
    print(">200:", int((sel["Abs_Error"] > 200).sum()))
    print("\nfamily_reason counts:")
    print(sel["family_reason"].value_counts().head(20).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pred", required=True)
    ap.add_argument("--val_pred", required=True)
    ap.add_argument("--test_pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_dir", default="results_TopoCellRT_CWNReplace_orig/chem_3d_family_analysis")
    ap.add_argument("--min_score", type=float, default=4.5)
    ap.add_argument("--max_train", type=int, default=2500)
    ap.add_argument("--max_val", type=int, default=800)
    ap.add_argument("--max_test", type=int, default=800)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train = add_chem(norm_pred(args.train_pred))
    val = add_chem(norm_pred(args.val_pred))
    test = add_chem(norm_pred(args.test_pred))

    tr = select_family(train, "train", args.max_train, args.min_score)
    va = select_family(val, "val", args.max_val, args.min_score)
    te = select_family(test, "test", args.max_test, args.min_score)

    report("TRAIN", train, tr)
    report("VAL", val, va)
    report("TEST diagnostic only", test, te)

    all_sel = pd.concat([tr, va, te], axis=0)
    all_sel = all_sel.drop_duplicates("SMILES")
    all_sel = all_sel.sort_values(["split", "chem_score"], ascending=[True, False])

    all_sel.to_csv(args.out, index=False)
    train.to_csv(f"{args.out_dir}/train_chem_family_all.csv", index=False)
    val.to_csv(f"{args.out_dir}/val_chem_family_all.csv", index=False)
    test.to_csv(f"{args.out_dir}/test_chem_family_all.csv", index=False)

    print("\nsaved selection:", args.out)
    print("total selected unique:", len(all_sel))
    print("\nTop 30 selected TEST diagnostic:")
    cols = ["SMILES", "Actual_RT", "Predicted_RT", "Abs_Error", "chem_score", "family_reason", "rings", "hetero", "tpsa", "logp"]
    print(te.head(30)[cols].to_string(index=False))


if __name__ == "__main__":
    main()

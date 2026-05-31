import argparse
import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def norm_pred(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    smi = cols.get("smiles", "SMILES")
    y = cols.get("actual_rt", None) or cols.get("y", None) or cols.get("y_true", None)
    p = cols.get("predicted_rt", None) or cols.get("y_pred", None) or cols.get("pred", None)

    if y is None or p is None:
        raise ValueError(f"bad columns in {path}: {df.columns.tolist()}")

    out = df[[smi, y, p]].copy()
    out.columns = ["SMILES", "y", "base_pred"]
    out["SMILES"] = out["SMILES"].astype(str)
    out["y"] = out["y"].astype(float)
    out["base_pred"] = out["base_pred"].astype(float)
    out["abs_err"] = (out["y"] - out["base_pred"]).abs()
    return out


def mol_from_smi(s):
    try:
        return Chem.MolFromSmiles(str(s))
    except Exception:
        return None


def featurize_one(smi, base_pred):
    s = str(smi)
    mol = mol_from_smi(s)

    # string motif flags
    feats = {
        "base_pred": float(base_pred),
        "base_pred_low": float(base_pred < 650),
        "base_pred_high": float(base_pred > 1050),
        "imidic_NC_O": float(("N=C(O)" in s) or ("C(O)=N" in s) or ("C(=N)O" in s) or ("N=C(C)O" in s)),
        "amide": float(("C(=O)N" in s) or ("NC(=O)" in s)),
        "sulfonamide": float(("S(=O)(=O)N" in s) or ("NS(=O)(=O)" in s)),
        "sulfone": float("S(=O)(=O)" in s),
        "cf3": float("C(F)(F)F" in s),
        "halogen": float(("Cl" in s) or ("Br" in s) or ("F" in s) or ("I" in s)),
        "piperazine_like": float(("N1CCN" in s) or ("N2CCN" in s) or ("N3CCN" in s)),
        "morpholine_like": float(("N1CCOCC1" in s) or ("N2CCOCC2" in s) or ("OCCN" in s)),
        "lower_hetero_arom_count": float(s.count("n") + s.count("o") + s.count("s")),
        "smiles_len": float(len(s)),
    }

    if mol is None:
        extra = {
            "mol_ok": 0.0,
            "mw": 0.0,
            "logp": 0.0,
            "tpsa": 0.0,
            "hba": 0.0,
            "hbd": 0.0,
            "rot_bonds": 0.0,
            "ring_count": 0.0,
            "arom_ring_count": 0.0,
            "hetero_count": 0.0,
            "heavy_atoms": 0.0,
            "fraction_csp3": 0.0,
            "polar_lipophilic_conflict": 0.0,
            "many_rings": 0.0,
            "high_tpsa": 0.0,
            "many_hba": 0.0,
            "flexible": 0.0,
        }
        feats.update(extra)
        return feats

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    heavy = mol.GetNumHeavyAtoms()
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    feats.update({
        "mol_ok": 1.0,
        "mw": float(mw),
        "logp": float(logp),
        "tpsa": float(tpsa),
        "hba": float(hba),
        "hbd": float(hbd),
        "rot_bonds": float(rot),
        "ring_count": float(rings),
        "arom_ring_count": float(arom_rings),
        "hetero_count": float(hetero),
        "heavy_atoms": float(heavy),
        "fraction_csp3": float(fsp3),
        "polar_lipophilic_conflict": float((logp >= 3.0) and (tpsa >= 70.0)),
        "many_rings": float(rings >= 4),
        "high_tpsa": float(tpsa >= 80.0),
        "many_hba": float(hba >= 6),
        "flexible": float(rot >= 6),
    })
    return feats


def featurize_df(df):
    rows = []
    for smi, bp in zip(df["SMILES"].values, df["base_pred"].values):
        rows.append(featurize_one(smi, bp))
    f = pd.DataFrame(rows)
    return f


def topk_report(df, score_col, name, ks=(100, 200, 400, 600, 800, 1000), hard_thr=80.0):
    print(f"\n=== {name} top-K by selector score ===")
    d = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    rows = []
    for k in ks:
        if k > len(d):
            continue
        sub = d.head(k)
        rows.append({
            "K": k,
            "mean_abs_err": sub["abs_err"].mean(),
            "median_abs_err": sub["abs_err"].median(),
            "max_abs_err": sub["abs_err"].max(),
            f"hard>={hard_thr}_count": int((sub["abs_err"] >= hard_thr).sum()),
            f"precision@{k}": float((sub["abs_err"] >= hard_thr).mean()),
            ">100": int((sub["abs_err"] > 100).sum()),
            ">200": int((sub["abs_err"] > 200).sum()),
        })
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    return rep


def eval_auc(df, score_col, hard_thr, name):
    y = (df["abs_err"].values >= hard_thr).astype(int)
    s = df[score_col].values
    if len(np.unique(y)) < 2:
        print(name, "AUC/AP skipped, only one class")
        return
    print(
        f"{name}: AUC={roc_auc_score(y, s):.4f}, AP={average_precision_score(y, s):.4f}, "
        f"hard_rate={y.mean():.4f}, N={len(y)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pred", required=True)
    ap.add_argument("--val_pred", required=True)
    ap.add_argument("--test_pred", required=True)
    ap.add_argument("--existing_3d_npz", default="data/unimol_012_hard_k2n1_np1compat.npz")
    ap.add_argument("--out_dir", default="results_TopoCellRT_CWNReplace_orig/2d_hardness_selector")
    ap.add_argument("--hard_thr", type=float, default=80.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train = norm_pred(args.train_pred)
    val = norm_pred(args.val_pred)
    test = norm_pred(args.test_pred)

    print("rows:", len(train), len(val), len(test))
    print("hard_thr:", args.hard_thr)
    print("train hard:", int((train["abs_err"] >= args.hard_thr).sum()))
    print("val hard:", int((val["abs_err"] >= args.hard_thr).sum()))
    print("test hard:", int((test["abs_err"] >= args.hard_thr).sum()))

    print("\nFeaturizing RDKit descriptors...")
    xtr_df = featurize_df(train)
    xva_df = featurize_df(val)
    xte_df = featurize_df(test)

    feat_cols = xtr_df.columns.tolist()
    xtr = xtr_df.values.astype(np.float32)
    xva = xva_df.values.astype(np.float32)
    xte = xte_df.values.astype(np.float32)

    ytr = (train["abs_err"].values >= args.hard_thr).astype(int)
    yva = (val["abs_err"].values >= args.hard_thr).astype(int)
    yte = (test["abs_err"].values >= args.hard_thr).astype(int)

    # 用 train 训练；val/test 只评估
    clf = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=10,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )

    print("\nTraining 2D-hardness selector...")
    clf.fit(xtr, ytr)

    val["risk_score"] = clf.predict_proba(xva)[:, 1]
    test["risk_score"] = clf.predict_proba(xte)[:, 1]
    train["risk_score"] = clf.predict_proba(xtr)[:, 1]

    print("\n=== Selector ranking quality ===")
    eval_auc(train, "risk_score", args.hard_thr, "train")
    eval_auc(val, "risk_score", args.hard_thr, "val")
    eval_auc(test, "risk_score", args.hard_thr, "test_diagnostic")

    rep_val = topk_report(val, "risk_score", "VAL", hard_thr=args.hard_thr)
    rep_test = topk_report(test, "risk_score", "TEST diagnostic", hard_thr=args.hard_thr)

    # 和旧 3D hard set 比 overlap
    try:
        z = np.load(args.existing_3d_npz, allow_pickle=False)
        s3d = set(z["smiles"].astype(str).tolist())
        for k in [100, 200, 400, 600, 800, 1000]:
            sel = set(test.sort_values("risk_score", ascending=False).head(k)["SMILES"].astype(str))
            print(f"test top{k} selector overlap existing_3d:", len(sel & s3d), "/", k)
    except Exception as e:
        print("existing 3D overlap skipped:", e)

    train_out = pd.concat([train.reset_index(drop=True), xtr_df.add_prefix("feat_").reset_index(drop=True)], axis=1)
    val_out = pd.concat([val.reset_index(drop=True), xva_df.add_prefix("feat_").reset_index(drop=True)], axis=1)
    test_out = pd.concat([test.reset_index(drop=True), xte_df.add_prefix("feat_").reset_index(drop=True)], axis=1)

    train_out.to_csv(f"{args.out_dir}/train_2d_hardness_scores.csv", index=False)
    val_out.to_csv(f"{args.out_dir}/val_2d_hardness_scores.csv", index=False)
    test_out.to_csv(f"{args.out_dir}/test_2d_hardness_scores.csv", index=False)
    rep_val.to_csv(f"{args.out_dir}/val_topk_report.csv", index=False)
    rep_test.to_csv(f"{args.out_dir}/test_topk_report_diagnostic.csv", index=False)

    # 给后续 3D 提取用：正式 no-leak 选择文件
    selected = test.sort_values("risk_score", ascending=False).head(600).copy()
    selected["split"] = "test"
    selected.to_csv(f"{args.out_dir}/test_selector_top600_for_unimol.csv", index=False)

    print("\nsaved:", args.out_dir)
    print("selected test top600 csv:", f"{args.out_dir}/test_selector_top600_for_unimol.csv")

    # feature importance
    imp = pd.DataFrame({"feature": feat_cols, "importance": clf.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(f"{args.out_dir}/feature_importance.csv", index=False)
    print("\nTop feature importance:")
    print(imp.head(25).to_string(index=False))


if __name__ == "__main__":
    main()

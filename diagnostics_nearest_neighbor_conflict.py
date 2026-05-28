import os
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


TRAIN_CSV = "./SMRT_data/data/SMRT_train.csv"
TEST_CSV  = "./SMRT_data/data/SMRT_test.csv"
WORST_CSV = "results/TopoCellRT/worst_test_top300.csv"
OUT_DIR   = "results/TopoCellRT"


tautomer_enumerator = rdMolStandardize.TautomerEnumerator()


def get_mol(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canon_smiles(smiles):
    mol = get_mol(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def tautomer_smiles(smiles):
    mol = get_mol(smiles)
    if mol is None:
        return ""
    try:
        tmol = tautomer_enumerator.Canonicalize(mol)
        return Chem.MolToSmiles(tmol, isomericSmiles=False)
    except Exception:
        return ""


def fp(smiles):
    mol = get_mol(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find any of {candidates} in columns={df.columns.tolist()}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    worst = pd.read_csv(WORST_CSV)

    train_smiles_col = find_col(train, ["smile", "smiles", "SMILES"])
    train_rt_col = find_col(train, ["rt", "RT", "y", "retention_time"])

    print("train columns:", train.columns.tolist())
    print("using train smiles:", train_smiles_col, "rt:", train_rt_col)

    train = train[[train_smiles_col, train_rt_col]].copy()
    train.columns = ["smiles", "rt"]
    train["rt"] = train["rt"].astype(float)

    print("building train fingerprints...")
    train["fp"] = [fp(s) for s in train["smiles"]]
    train = train[train["fp"].notna()].reset_index(drop=True)

    fps = list(train["fp"])
    rts = train["rt"].values
    smiles_list = train["smiles"].tolist()

    rows = []

    print("analyzing worst test top300...")
    for idx, row in worst.iterrows():
        smi = row["smiles"]
        qfp = fp(smi)
        if qfp is None:
            continue

        sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, fps), dtype=float)
        top_idx = sims.argsort()[::-1][:20]

        top_sims = sims[top_idx]
        top_rts = rts[top_idx]
        top_smiles = [smiles_list[i] for i in top_idx]

        high_mask_075 = top_sims >= 0.75
        high_mask_065 = top_sims >= 0.65
        high_mask_055 = top_sims >= 0.55

        def safe_stats(mask):
            if mask.sum() == 0:
                return {
                    "n": 0,
                    "rt_mean": np.nan,
                    "rt_std": np.nan,
                    "rt_min": np.nan,
                    "rt_max": np.nan,
                    "rt_range": np.nan,
                    "frac_early": np.nan,
                    "frac_late": np.nan,
                }
            vals = top_rts[mask]
            return {
                "n": int(mask.sum()),
                "rt_mean": float(vals.mean()),
                "rt_std": float(vals.std()),
                "rt_min": float(vals.min()),
                "rt_max": float(vals.max()),
                "rt_range": float(vals.max() - vals.min()),
                "frac_early": float((vals < 750).mean()),
                "frac_late": float((vals > 1050).mean()),
            }

        s075 = safe_stats(high_mask_075)
        s065 = safe_stats(high_mask_065)
        s055 = safe_stats(high_mask_055)

        y_true = float(row["y_true"])
        y_pred = float(row["y_pred"])
        abs_err = float(row["abs_err"])

        # 最近邻是否支持真实值，还是支持模型预测值
        nn_rt = float(top_rts[0])
        nn_sim = float(top_sims[0])
        nn_support_true = abs(nn_rt - y_true)
        nn_support_pred = abs(nn_rt - y_pred)

        conflict_065 = (
            s065["n"] >= 3
            and s065["rt_range"] == s065["rt_range"]
            and s065["rt_range"] >= 350
        )

        rows.append({
            "rank": idx + 1,
            "smiles": smi,
            "canon_smiles": canon_smiles(smi),
            "tautomer_smiles": tautomer_smiles(smi),
            "y_true": y_true,
            "y_pred": y_pred,
            "signed_err": float(row["signed_err"]),
            "abs_err": abs_err,

            "nn_sim": nn_sim,
            "nn_rt": nn_rt,
            "nn_smiles": top_smiles[0],
            "nn_abs_to_true": nn_support_true,
            "nn_abs_to_pred": nn_support_pred,

            "n075": s075["n"],
            "rt_range075": s075["rt_range"],
            "frac_early075": s075["frac_early"],
            "frac_late075": s075["frac_late"],

            "n065": s065["n"],
            "rt_mean065": s065["rt_mean"],
            "rt_std065": s065["rt_std"],
            "rt_min065": s065["rt_min"],
            "rt_max065": s065["rt_max"],
            "rt_range065": s065["rt_range"],
            "frac_early065": s065["frac_early"],
            "frac_late065": s065["frac_late"],
            "conflict_065": conflict_065,

            "n055": s055["n"],
            "rt_range055": s055["rt_range"],
            "frac_early055": s055["frac_early"],
            "frac_late055": s055["frac_late"],

            "top5_sims": ";".join([f"{x:.3f}" for x in top_sims[:5]]),
            "top5_rts": ";".join([f"{x:.1f}" for x in top_rts[:5]]),
            "top5_smiles": " || ".join(top_smiles[:5]),
        })

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "worst_test_top300_nn_conflict.csv")
    out.to_csv(out_path, index=False)

    print("\nSaved:", out_path)

    print("\n=== Summary ===")
    print("N worst analyzed:", len(out))
    print("nn_sim >= 0.75:", int((out["nn_sim"] >= 0.75).sum()))
    print("nn_sim >= 0.65:", int((out["nn_sim"] >= 0.65).sum()))
    print("conflict_065:", int(out["conflict_065"].sum()))

    print("\n=== Top 30 by abs_err with NN info ===")
    cols = [
        "rank", "smiles", "y_true", "y_pred", "abs_err",
        "nn_sim", "nn_rt", "nn_abs_to_true", "nn_abs_to_pred",
        "n065", "rt_min065", "rt_max065", "rt_range065",
        "frac_early065", "frac_late065", "conflict_065",
        "top5_sims", "top5_rts"
    ]
    print(out[cols].head(30).to_string(index=False))

    print("\n=== Most conflicted neighbors ===")
    print(out.sort_values("rt_range065", ascending=False)[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()

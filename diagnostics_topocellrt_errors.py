import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

from data_topocellrt import TopoCellRTTrainDataset, TopoCellRTTestDataset
from model_topocellrt import TopoCellRTNet


SEED = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR = "results/TopoCellRT"
BEST_MODEL = "model_dict/best_model_TopoCellRT.pkl"


SMARTS = {
    "cf3": "[CX4](F)(F)F",
    "sulfonamide": "S(=O)(=O)N",
    "amide": "C(=O)N",
    "urea": "NC(=O)N",
    "piperazine": "N1CCNCC1",
    "morpholine": "O1CCNCC1",
    "n_hetero_aromatic": "n",
    "halogen": "[F,Cl,Br,I]",
}


def mol_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "valid_mol": 0,
            "molwt": np.nan,
            "logp": np.nan,
            "tpsa": np.nan,
            "hba": np.nan,
            "hbd": np.nan,
            "rotb": np.nan,
            "rings": np.nan,
            "aromatic_rings": np.nan,
            "aliphatic_rings": np.nan,
            "hetero_count": np.nan,
            "heavy_count": np.nan,
            "halogen_count": np.nan,
            **{k: 0 for k in SMARTS},
        }

    atoms = list(mol.GetAtoms())
    hetero_count = sum(1 for a in atoms if a.GetAtomicNum() not in [1, 6])
    heavy_count = mol.GetNumHeavyAtoms()
    halogen_count = sum(1 for a in atoms if a.GetSymbol() in ["F", "Cl", "Br", "I"])

    out = {
        "valid_mol": 1,
        "molwt": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "rotb": Lipinski.NumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "hetero_count": hetero_count,
        "heavy_count": heavy_count,
        "halogen_count": halogen_count,
    }

    for name, smarts in SMARTS.items():
        q = Chem.MolFromSmarts(smarts)
        out[name] = int(mol.HasSubstructMatch(q)) if q is not None else 0

    return out


@torch.no_grad()
def collect_predictions(model, loader):
    rows = []
    model.eval()

    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch).view(-1).detach().cpu().numpy()
        y = batch.y.view(-1).detach().cpu().numpy()

        smiles_list = batch.smiles
        hard_flag = batch.hard_flag.view(-1).detach().cpu().numpy() if hasattr(batch, "hard_flag") else np.zeros_like(y)

        for i, (yt, yp, smi, hf) in enumerate(zip(y, pred, smiles_list, hard_flag)):
            row = {
                "smiles": smi,
                "y_true": float(yt),
                "y_pred": float(yp),
                "signed_err": float(yp - yt),
                "abs_err": float(abs(yp - yt)),
                "hard_flag": float(hf),
            }
            row.update(mol_features(smi))
            rows.append(row)

    return pd.DataFrame(rows)


def add_bins(df):
    df["rt_bin"] = pd.cut(
        df["y_true"],
        bins=[0, 600, 750, 900, 1050, 1200, 2000],
        labels=["<600", "600-750", "750-900", "900-1050", "1050-1200", ">1200"],
    )
    df["pred_bin"] = pd.cut(
        df["y_pred"],
        bins=[0, 600, 750, 900, 1050, 1200, 2000],
        labels=["<600", "600-750", "750-900", "900-1050", "1050-1200", ">1200"],
    )

    df["direction"] = np.where(df["signed_err"] > 0, "over_pred", "under_pred")
    df["tail_100"] = df["abs_err"] > 100
    df["tail_200"] = df["abs_err"] > 200

    df["multi_ring"] = df["rings"] >= 3
    df["multi_aromatic"] = df["aromatic_rings"] >= 2
    df["many_hetero"] = df["hetero_count"] >= 4
    df["large_mol"] = df["molwt"] >= 450
    df["polar"] = df["tpsa"] >= 100
    df["hydrophobic"] = df["logp"] >= 4

    return df


def summarize_group(df, col):
    g = df.groupby(col, dropna=False).agg(
        n=("abs_err", "size"),
        mae=("abs_err", "mean"),
        medae=("abs_err", "median"),
        p95=("abs_err", lambda x: np.quantile(x, 0.95)),
        tail100=("tail_100", "sum"),
        tail200=("tail_200", "sum"),
        mean_signed=("signed_err", "mean"),
    ).reset_index()
    g["tail100_rate"] = g["tail100"] / g["n"]
    g["tail200_rate"] = g["tail200"] / g["n"]
    return g.sort_values(["mae", "tail200_rate"], ascending=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_all = TopoCellRTTrainDataset("./SMRT_data/reload/SMRT_train")
    train_len = len(train_all)
    train_len2 = int(train_len * 0.9)
    val_len = train_len - train_len2
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(train_all, [train_len2, val_len], generator=generator)

    test_dataset = TopoCellRTTestDataset("./SMRT_data/reload/SMRT_test")

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = TopoCellRTNet().to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))

    val_df = add_bins(collect_predictions(model, val_loader))
    test_df = add_bins(collect_predictions(model, test_loader))

    val_df.to_csv(os.path.join(OUT_DIR, "val_predictions_chem.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test_predictions_chem.csv"), index=False)

    # worst cases
    val_df.sort_values("abs_err", ascending=False).head(300).to_csv(
        os.path.join(OUT_DIR, "worst_val_top300.csv"), index=False
    )
    test_df.sort_values("abs_err", ascending=False).head(300).to_csv(
        os.path.join(OUT_DIR, "worst_test_top300.csv"), index=False
    )

    group_cols = [
        "rt_bin",
        "pred_bin",
        "direction",
        "hard_flag",
        "multi_ring",
        "multi_aromatic",
        "many_hetero",
        "large_mol",
        "polar",
        "hydrophobic",
        "cf3",
        "sulfonamide",
        "amide",
        "urea",
        "piperazine",
        "morpholine",
        "halogen",
    ]

    with open(os.path.join(OUT_DIR, "error_group_report.txt"), "w") as f:
        for name, df in [("VAL", val_df), ("TEST", test_df)]:
            f.write(f"\n\n========== {name} ==========\n")
            f.write(f"MAE={df.abs_err.mean():.4f}, MedAE={df.abs_err.median():.4f}, ")
            f.write(f"P95={df.abs_err.quantile(0.95):.4f}, P99={df.abs_err.quantile(0.99):.4f}, ")
            f.write(f">100={int((df.abs_err>100).sum())}/{len(df)}, >200={int((df.abs_err>200).sum())}/{len(df)}\n")

            for col in group_cols:
                f.write(f"\n--- group by {col} ---\n")
                f.write(summarize_group(df, col).to_string(index=False))
                f.write("\n")

    print("Saved:")
    print(os.path.join(OUT_DIR, "val_predictions_chem.csv"))
    print(os.path.join(OUT_DIR, "test_predictions_chem.csv"))
    print(os.path.join(OUT_DIR, "worst_val_top300.csv"))
    print(os.path.join(OUT_DIR, "worst_test_top300.csv"))
    print(os.path.join(OUT_DIR, "error_group_report.txt"))

    print("\nTop 20 TEST worst:")
    cols = [
        "smiles", "y_true", "y_pred", "signed_err", "abs_err",
        "rings", "aromatic_rings", "hetero_count", "halogen_count",
        "molwt", "logp", "tpsa", "rt_bin", "pred_bin",
        "cf3", "sulfonamide", "amide", "urea", "piperazine", "morpholine",
    ]
    print(test_df.sort_values("abs_err", ascending=False)[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

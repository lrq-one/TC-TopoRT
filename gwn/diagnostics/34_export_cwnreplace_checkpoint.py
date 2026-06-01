import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from rdkit import Chem

from mp.complex import ComplexBatch
from mp.smrt_dataset import SMRTComplexDataset
from net.topocellrt_cwn_replace import TopoCellRTCWNReplace


def complex_collate_fn(batch):
    return ComplexBatch.from_complex_list(batch)


def load_valid_smiles(csv_path):
    df = pd.read_csv(csv_path, engine="python")
    df.columns = [str(c).lower().strip() for c in df.columns]

    if "smile" in df.columns and "smiles" not in df.columns:
        df.rename(columns={"smile": "smiles"}, inplace=True)

    if "smiles" not in df.columns or "rt" not in df.columns:
        df = pd.read_csv(csv_path, sep=r"\s+", names=["smiles", "rt"], header=0, engine="python")

    df = df[df["rt"] > 300.0].copy()

    valid_smiles = []
    valid_rt = []

    for _, row in df.iterrows():
        smi = row.get("smiles", None)
        rt = row.get("rt", None)

        if pd.isna(smi):
            continue

        try:
            rt = float(rt)
        except Exception:
            continue

        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        valid_smiles.append(str(smi))
        valid_rt.append(rt)

    return valid_smiles, valid_rt


@torch.no_grad()
def export_predictions(model, loader, smiles_list, save_path, device):
    model.eval()

    preds_list = []
    targets_list = []

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y.view(-1)
        pred = model(batch)
        if isinstance(pred, tuple):
            pred = pred[0]

        preds_list.append(pred.view(-1).detach().cpu())
        targets_list.append(targets.detach().cpu())

    preds = torch.cat(preds_list).numpy()
    targets = torch.cat(targets_list).numpy()

    n = min(len(smiles_list), len(preds), len(targets))

    out = pd.DataFrame({
        "SMILES": smiles_list[:n],
        "Actual_RT": targets[:n],
        "Predicted_RT": preds[:n],
        "Abs_Error": np.abs(targets[:n] - preds[:n]),
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out.to_csv(save_path, index=False)

    print("saved:", save_path)
    print("N:", len(out))
    print("MAE:", float(out["Abs_Error"].mean()))
    print("MedAE:", float(out["Abs_Error"].median()))
    print("P95:", float(out["Abs_Error"].quantile(0.95)))
    print("P99:", float(out["Abs_Error"].quantile(0.99)))
    print(">100:", int((out["Abs_Error"] > 100).sum()))
    print(">200:", int((out["Abs_Error"] > 200).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_seed", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_ring_size", type=int, default=6)
    ap.add_argument("--cwn_layers", type=int, default=6)
    ap.add_argument("--cwn_hidden", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print("\n=== loading train full dataset ===")
    dataset_train_full = SMRTComplexDataset(
        args.train_root,
        args.train_csv,
        args.max_ring_size,
        use_edge_features=True,
    )

    total_train_len = len(dataset_train_full)
    train_len = int(0.9 * total_train_len)
    val_len = total_train_len - train_len

    train_set, val_set = random_split(
        dataset_train_full,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.split_seed),
    )

    print("\n=== loading test dataset ===")
    test_set = SMRTComplexDataset(
        args.test_root,
        args.test_csv,
        args.max_ring_size,
        use_edge_features=True,
    )

    train_valid_smiles, _ = load_valid_smiles(args.train_csv)
    test_valid_smiles, _ = load_valid_smiles(args.test_csv)

    train_smiles = [train_valid_smiles[i] for i in train_set.indices]
    val_smiles = [train_valid_smiles[i] for i in val_set.indices]

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=0,
    )

    model = TopoCellRTCWNReplace(
        emb_dim=256,
        cwn_layers=args.cwn_layers,
        cwn_hidden=args.cwn_hidden,
        max_dim=2,
        drop_ratio=0.0,
    ).to(device)

    print("loading ckpt:", args.ckpt)
    state = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)

    export_predictions(
        model,
        train_loader,
        train_smiles,
        os.path.join(args.out_dir, "base_train_predictions.csv"),
        device,
    )

    export_predictions(
        model,
        val_loader,
        val_smiles,
        os.path.join(args.out_dir, "base_val_predictions.csv"),
        device,
    )

    export_predictions(
        model,
        test_loader,
        test_valid_smiles,
        os.path.join(args.out_dir, "base_test_predictions.csv"),
        device,
    )


if __name__ == "__main__":
    main()

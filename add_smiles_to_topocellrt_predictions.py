import os
import torch
import pandas as pd
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from data_topocellrt import TopoCellRTTrainDataset, TopoCellRTTestDataset


def get_smiles_from_loader(loader):
    smiles = []
    y_list = []

    for data in loader:
        if hasattr(data, "smiles"):
            if isinstance(data.smiles, (list, tuple)):
                smiles.extend(list(data.smiles))
            else:
                smiles.append(data.smiles)
        else:
            raise RuntimeError("Data batch has no smiles attribute.")

        y_list.extend(data.y.view(-1).detach().cpu().numpy().tolist())

    return smiles, y_list


def patch_file(pred_path, smiles, y_from_dataset, out_path):
    df = pd.read_csv(pred_path)

    if len(df) != len(smiles):
        raise RuntimeError(
            f"Length mismatch: {pred_path}, df={len(df)}, smiles={len(smiles)}"
        )

    # 检查 y 顺序是否一致
    if "y_true" in df.columns:
        y_predfile = df["y_true"].astype(float).values
    elif "Actual_RT" in df.columns:
        y_predfile = df["Actual_RT"].astype(float).values
    else:
        raise RuntimeError(f"No y column in {pred_path}: {df.columns.tolist()}")

    y_dataset = pd.Series(y_from_dataset).astype(float).values
    max_diff = abs(y_predfile - y_dataset).max()
    mean_diff = abs(y_predfile - y_dataset).mean()

    print("\npatch:", pred_path)
    print("rows:", len(df))
    print("max y diff:", max_diff)
    print("mean y diff:", mean_diff)

    if max_diff > 1e-4:
        raise RuntimeError(
            f"y_true order mismatch for {pred_path}; max_diff={max_diff}. "
            f"不要强行加 SMILES。"
        )

    df.insert(0, "SMILES", smiles)
    df.to_csv(out_path, index=False)
    print("saved:", out_path)


if __name__ == "__main__":
    randint = 1
    batch_size = 64
    num_workers = 0

    train_root = os.environ.get("TOPOCELLRT_TRAIN_ROOT", "./SMRT_data/reload/SMRT_train")
    test_root = os.environ.get("TOPOCELLRT_TEST_ROOT", "./SMRT_data/reload/SMRT_test")

    dataset_train = TopoCellRTTrainDataset(train_root)
    train_len = len(dataset_train)
    train_len2 = int(train_len * 0.9)
    val_len = train_len - train_len2

    generator = torch.Generator().manual_seed(randint)
    train_dataset, val_dataset = random_split(
        dataset_train,
        [train_len2, val_len],
        generator=generator,
    )

    dataset_test = TopoCellRTTestDataset(test_root)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    val_smiles, val_y = get_smiles_from_loader(val_loader)
    test_smiles, test_y = get_smiles_from_loader(test_loader)

    os.makedirs("results/TopoCellRT_with_smiles", exist_ok=True)

    patch_file(
        "results/TopoCellRT/val_predictions.csv",
        val_smiles,
        val_y,
        "results/TopoCellRT_with_smiles/val_predictions.csv",
    )

    patch_file(
        "results/TopoCellRT/test_predictions.csv",
        test_smiles,
        test_y,
        "results/TopoCellRT_with_smiles/test_predictions.csv",
    )

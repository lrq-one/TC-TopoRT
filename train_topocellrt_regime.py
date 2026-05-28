import os
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics

from tqdm import tqdm
from torch import optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from data_topocellrt import TopoCellRTTrainDataset, TopoCellRTTestDataset
from model_topocellrt_regime import TopoCellRTNet


BATCH_SIZE = 64
TEST_BATCH = 64
NUM_WORKERS = 2
EPOCHS = 80
SEED = 1

BASE_CKPT = "model_dict/best_model_TopoCellRT.pkl"
BEST_CKPT = "model_dict/best_model_TopoCellRT_regime.pkl"
RESULT_DIR = "results/TopoCellRT_regime"

RT_BIN_EDGES = torch.tensor([600.0, 750.0, 900.0, 1050.0, 1200.0])
RT_BIN_CENTERS = torch.tensor([540.0, 675.0, 825.0, 975.0, 1125.0, 1320.0])


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_rt_bin(y, device):
    edges = RT_BIN_EDGES.to(device)
    return torch.bucketize(y.view(-1), edges).long()


def summarize(y_true, y_pred):
    err = torch.abs(y_true - y_pred)
    return {
        "mae": err.mean().item(),
        "medae": err.median().item(),
        "rmse": torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item(),
        "p95": torch.quantile(err, 0.95).item(),
        "p99": torch.quantile(err, 0.99).item(),
        "err100": int((err > 100).sum().item()),
        "err200": int((err > 200).sum().item()),
        "n": int(err.numel()),
    }


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    ys, ps = [], []

    for data in loader:
        data = data.to(device, non_blocking=True)
        pred = model(data).view(-1)
        ys.append(data.y.view(-1).detach().cpu())
        ps.append(pred.detach().cpu())

    return torch.cat(ys), torch.cat(ps)


@torch.no_grad()
def evaluate(model, loader, device):
    y_true, y_pred = collect_predictions(model, loader, device)
    mae = torch.abs(y_true - y_pred).mean()
    mre = torch.div(torch.abs(y_true - y_pred), y_true).mean()
    medae = torch.median(torch.abs(y_true - y_pred))
    medre = torch.median(torch.div(torch.abs(y_true - y_pred), y_true))
    r2 = torchmetrics.R2Score()(y_pred, y_true)
    return mae, mre, medae, medre, r2


def save_predictions(path, y_true, y_pred):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    y_true = y_true.detach().cpu().tolist()
    y_pred = y_pred.detach().cpu().tolist()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "abs_err"])
        for yt, yp in zip(y_true, y_pred):
            writer.writerow([yt, yp, abs(yt - yp)])


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_reg = 0.0
    total_cls = 0.0
    total_exp = 0.0

    bin_weight = torch.tensor([1.25, 1.00, 1.00, 1.05, 1.20, 1.80], device=device)
    centers = RT_BIN_CENTERS.to(device)

    for data in tqdm(loader):
        data = data.to(device, non_blocking=True)
        y = data.y.view(-1).float()

        pred, logits = model(data, return_aux=True)
        pred = pred.view(-1)

        y_bin = make_rt_bin(y, device)

        reg_each = F.smooth_l1_loss(pred, y, reduction="none")

        # 不再用 hard_flag；之前 hard_flag 几乎全是 1，没有区分度
        reg_w = torch.ones_like(reg_each)
        reg_w = reg_w + 0.15 * ((y_bin == 0) | (y_bin == 5)).float()
        reg_w = reg_w + 0.08 * ((y_bin == 4)).float()
        reg_loss = (reg_each * reg_w).mean()

        cls_loss = F.cross_entropy(logits, y_bin, weight=bin_weight)

        prob = torch.softmax(logits, dim=-1)
        expected_rt = (prob * centers.view(1, -1)).sum(dim=-1)
        exp_loss = F.smooth_l1_loss(expected_rt, y)

        loss = reg_loss + 1.0 * cls_loss + 0.05 * exp_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_reg += reg_loss.item()
        total_cls += cls_loss.item()
        total_exp += exp_loss.item()

    n = len(loader)
    return total_loss / n, total_reg / n, total_cls / n, total_exp / n


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs("model_dict", exist_ok=True)

    set_seed(SEED)

    print("loading train/val/test data")
    dataset_train = TopoCellRTTrainDataset("./SMRT_data/reload/SMRT_train")
    dataset_test = TopoCellRTTestDataset("./SMRT_data/reload/SMRT_test")

    train_len = len(dataset_train)
    train_len2 = int(train_len * 0.9)
    val_len = train_len - train_len2

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        dataset_train, [train_len2, val_len], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TEST_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=TEST_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("# train:", len(train_dataset), "# val:", len(val_dataset), "# test:", len(dataset_test))

    model = TopoCellRTNet().to(device)

    if os.path.exists(BASE_CKPT):
        missing, unexpected = model.load_state_dict(
            torch.load(BASE_CKPT, map_location=device),
            strict=False,
        )
        print("Loaded base checkpoint:", BASE_CKPT)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    else:
        print("WARNING: base checkpoint not found, training from scratch")

    bin_params = []
    base_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("rt_bin_head"):
            bin_params.append(p)
        else:
            base_params.append(p)

    optimizer = optim.AdamW(
        [
            {"params": base_params, "lr": 2e-6},
            {"params": bin_params, "lr": 5e-5},
        ],
        weight_decay=1e-2,
        amsgrad=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=2e-7,
    )

    best_val = 999999.0
    patience = 25
    bad_count = 0

    log_path = os.path.join(RESULT_DIR, "TopoCellRT_regime_result.txt")

    with open(log_path, "w") as f:
        for epoch in range(EPOCHS):
            train_loss, train_reg, train_cls, train_exp = train_one_epoch(
                model, train_loader, optimizer, device
            )
            scheduler.step()

            model.eval()
            train_mae, train_mre, train_medae, train_medre, train_r2 = evaluate(model, train_loader, device)
            val_mae, val_mre, val_medae, val_medre, val_r2 = evaluate(model, val_loader, device)

            line1 = (
                f"epoch:{epoch}\t"
                f"loss:{train_loss:.4f}\treg:{train_reg:.4f}\tcls:{train_cls:.4f}\texp:{train_exp:.4f}\t"
                f"train_mae:{train_mae.item():.4f}\ttrain_medAE:{train_medae.item():.4f}\ttrain_r2:{train_r2.item():.4f}"
            )

            line2 = (
                f"epoch:{epoch}\t"
                f"val_mae:{val_mae.item():.4f}\tval_mre:{val_mre.item():.6f}\t"
                f"val_medAE:{val_medae.item():.4f}\tval_medRE:{val_medre.item():.6f}\tval_r2:{val_r2.item():.4f}"
            )

            print(line1)
            print(line2)

            f.write(line1 + "\n")
            f.write(line2 + "\n")
            f.flush()

            if val_mae.item() < best_val:
                best_val = val_mae.item()
                bad_count = 0
                torch.save(model.state_dict(), BEST_CKPT)
                print("saved best:", BEST_CKPT, "val_mae:", best_val)
            else:
                bad_count += 1

            if bad_count >= patience:
                print("early stop at epoch", epoch)
                break

    if os.path.exists(BEST_CKPT):
        model.load_state_dict(torch.load(BEST_CKPT, map_location=device), strict=False)

    model.eval()
    val_true, val_pred = collect_predictions(model, val_loader, device)
    test_true, test_pred = collect_predictions(model, test_loader, device)

    val_stats = summarize(val_true, val_pred)
    test_stats = summarize(test_true, test_pred)

    print("VAL stats:", val_stats)
    print("TEST stats:", test_stats)

    save_predictions(os.path.join(RESULT_DIR, "val_predictions.csv"), val_true, val_pred)
    save_predictions(os.path.join(RESULT_DIR, "test_predictions.csv"), test_true, test_pred)

    with open(os.path.join(RESULT_DIR, "final_report.txt"), "w") as f:
        f.write("VAL stats:\n")
        f.write(str(val_stats) + "\n")
        f.write("TEST stats:\n")
        f.write(str(test_stats) + "\n")


if __name__ == "__main__":
    main()

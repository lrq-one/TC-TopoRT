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
from model_topocellrt_car import TopoCellRTNet

from train_topocellrt_car import (
    build_fp_cache,
    build_scaffold_groups,
    build_pair_banks,
    ConflictPairBatchSampler,
    get_batch_fps,
    car_contrastive_loss,
    freeze_batchnorm_stats,
)


BATCH_SIZE = 64
TEST_BATCH = 64
NUM_WORKERS = 2
SEED = 1

EPOCHS = int(os.getenv("EPOCHS", "20"))
BASE_LR = float(os.getenv("BASE_LR", "3e-7"))
CONTRAST_LR = float(os.getenv("CONTRAST_LR", "5e-5"))
LAMBDA_CAR_MAX = float(os.getenv("LAMBDA_CAR_MAX", "0.01"))
WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", "8"))

RESULT_DIR = "results/TopoCellRT_CARv3"
BEST_MODEL_PATH = "model_dict/best_model_TopoCellRT_CARv3.pkl"
PRETRAINED_PATH = "model_dict/best_model_TopoCellRT.pkl"


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def summarize(y_true, y_pred):
    err = torch.abs(y_true - y_pred)
    return {
        "mae": err.mean().item(),
        "medae": err.median().item(),
        "rmse": torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item(),
        "p95": torch.quantile(err, 0.95).item(),
        "p99": torch.quantile(err, 0.99).item(),
        "err_100": int((err > 100).sum().item()),
        "err_200": int((err > 200).sum().item()),
        "total": int(err.numel()),
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

    return mae, mre, medae, medre, r2, summarize(y_true, y_pred)


def save_predictions(path, y_true, y_pred):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    y_true = y_true.detach().cpu().tolist()
    y_pred = y_pred.detach().cpu().tolist()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "abs_err"])
        for yt, yp in zip(y_true, y_pred):
            writer.writerow([yt, yp, abs(yt - yp)])


def make_optimizer(model):
    contrast_params = list(model.contrast_proj.parameters())
    contrast_param_ids = {id(p) for p in contrast_params}

    base_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in contrast_param_ids
    ]

    optimizer = optim.AdamW(
        [
            {"params": base_params, "lr": BASE_LR},
            {"params": contrast_params, "lr": CONTRAST_LR},
        ],
        weight_decay=1e-2,
        amsgrad=True,
    )

    return optimizer


def train_one_epoch(model, reg_loader, pair_loader, optimizer, device, fp_cache, epoch):
    model.train()
    freeze_batchnorm_stats(model)

    warmup = min(1.0, epoch / float(max(WARMUP_EPOCHS, 1)))
    lambda_car = LAMBDA_CAR_MAX * warmup

    pair_iter = iter(pair_loader)

    total_reg = 0.0
    total_car = 0.0
    total_pos = 0
    total_neg = 0
    steps = 0

    for reg_batch in tqdm(reg_loader):
        reg_batch = reg_batch.to(device, non_blocking=True)

        # 1) 官方分布随机 batch：只做 RT 回归
        pred = model(reg_batch).view(-1)
        y = reg_batch.y.view(-1).float()
        reg_loss = F.smooth_l1_loss(pred, y)

        # 2) conflict pair batch：只做 CAR，不做回归
        try:
            pair_batch = next(pair_iter)
        except StopIteration:
            pair_iter = iter(pair_loader)
            pair_batch = next(pair_iter)

        pair_batch = pair_batch.to(device, non_blocking=True)
        _, z_pair = model(pair_batch, return_emb=True)

        pair_y = pair_batch.y.view(-1).float()
        pair_fp = get_batch_fps(pair_batch.smiles, fp_cache, device=z_pair.device)

        car_loss, pos_pairs, neg_pairs = car_contrastive_loss(
            z_pair,
            pair_y,
            pair_fp,
            pos_struct_min=0.45,
            pos_rt_delta=50.0,
            neg_struct_min=0.65,
            neg_rt_delta=300.0,
            pos_margin=0.75,
            neg_margin=0.15,
            neg_weight=1.5,
        )

        loss = reg_loss + lambda_car * car_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_reg += reg_loss.item()
        total_car += car_loss.item()
        total_pos += pos_pairs
        total_neg += neg_pairs
        steps += 1

    return {
        "reg_loss": total_reg / max(steps, 1),
        "car_loss": total_car / max(steps, 1),
        "pos_pairs": total_pos / max(steps, 1),
        "neg_pairs": total_neg / max(steps, 1),
        "lambda_car": lambda_car,
    }


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs("model_dict", exist_ok=True)

    set_seed(SEED)

    print("loading... get SMRT train data feature")
    dataset_train = TopoCellRTTrainDataset("./SMRT_data/reload/SMRT_train")

    train_len = len(dataset_train)
    train_len2 = int(train_len * 0.9)
    val_len = train_len - train_len2

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        dataset_train,
        [train_len2, val_len],
        generator=generator,
    )

    print("loading... get SMRT test data feature")
    dataset_test = TopoCellRTTestDataset("./SMRT_data/reload/SMRT_test")

    train_smiles = [train_dataset[i].smiles for i in range(len(train_dataset))]
    train_rts = [float(train_dataset[i].y.view(-1)[0]) for i in range(len(train_dataset))]

    print("building fp cache...")
    fp_cache = build_fp_cache(train_smiles)

    scaffold_groups = build_scaffold_groups(train_smiles)

    positive_pairs, conflict_pairs = build_pair_banks(
        train_smiles=train_smiles,
        train_rts=train_rts,
        scaffold_groups=scaffold_groups,
        fp_cache=fp_cache,
        seed=SEED,
        pos_struct_min=0.45,
        pos_rt_delta=50.0,
        neg_struct_min=0.65,
        neg_rt_delta=300.0,
        max_group_size=320,
        max_pairs_per_group=800,
    )

    print("positive pair bank:", len(positive_pairs))
    print("conflict pair bank:", len(conflict_pairs))

    # A. 正常随机 loader：保持官方训练分布，用于回归
    reg_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # B. pair loader：只用于 CAR 对比学习
    pair_sampler = ConflictPairBatchSampler(
        num_samples=len(train_dataset),
        conflict_pairs=conflict_pairs,
        positive_pairs=positive_pairs,
        batch_size=BATCH_SIZE,
        conflict_pairs_per_batch=6,
        positive_pairs_per_batch=4,
        seed=SEED,
        num_batches=len(reg_loader),
        shuffle=True,
    )

    pair_loader = DataLoader(
        train_dataset,
        batch_sampler=pair_sampler,
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

    model = TopoCellRTNet()

    if os.path.exists(PRETRAINED_PATH):
        missing, unexpected = model.load_state_dict(
            torch.load(PRETRAINED_PATH, map_location="cpu"),
            strict=False,
        )
        print("Loaded pretrained:", PRETRAINED_PATH)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    else:
        print("WARNING: pretrained not found, training from scratch")

    model.to(device)

    optimizer = make_optimizer(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(EPOCHS, 1),
        eta_min=1e-8,
    )

    best_val = 999999.0
    log_path = os.path.join(RESULT_DIR, "train_log.txt")

    with open(log_path, "w") as f:
        for epoch in range(EPOCHS):
            stats = train_one_epoch(
                model=model,
                reg_loader=reg_loader,
                pair_loader=pair_loader,
                optimizer=optimizer,
                device=device,
                fp_cache=fp_cache,
                epoch=epoch,
            )
            scheduler.step()

            val_mae, val_mre, val_medae, val_medre, val_r2, val_stats = evaluate(
                model,
                val_loader,
                device,
            )

            line = (
                f"epoch:{epoch}\t"
                f"val_mae:{val_mae.item():.6f}\t"
                f"val_mre:{val_mre.item():.6f}\t"
                f"val_medAE:{val_medae.item():.6f}\t"
                f"val_medRE:{val_medre.item():.6f}\t"
                f"val_r2:{val_r2.item():.6f}\t"
                f"reg_loss:{stats['reg_loss']:.6f}\t"
                f"car_loss:{stats['car_loss']:.6f}\t"
                f"pos_pairs:{stats['pos_pairs']:.3f}\t"
                f"neg_pairs:{stats['neg_pairs']:.3f}\t"
                f"lambda_car:{stats['lambda_car']:.6f}\t"
                f"p95:{val_stats['p95']:.6f}\t"
                f"p99:{val_stats['p99']:.6f}\t"
                f"err100:{val_stats['err_100']}\t"
                f"err200:{val_stats['err_200']}"
            )

            print(line)
            f.write(line + "\n")
            f.flush()

            if val_mae.item() < best_val:
                best_val = val_mae.item()
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print("saved best:", BEST_MODEL_PATH, "val_mae:", best_val)

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device), strict=False)

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

import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import torchmetrics
from model_topocellrt import TopoCellRTNet
from data_topocellrt import TopoCellRTTrainDataset, TopoCellRTTestDataset
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings
import random
import os
import csv
from torch import optim
import torch.nn.functional as F
warnings.filterwarnings("ignore")


class TopoCellRTTrainer(object):
    def __init__(self, model, lr, device):
        self.model = model

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, amsgrad=True, weight_decay=1e-2)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 150)
        self.device = device

    def train_one_epoch(self, data_loader, epoch):
        total_loss = 0.0
        for i, data in enumerate(tqdm(data_loader)):
            data.to(self.device)
            pred = self.model(data)
            y = data.y

            loss_each = F.smooth_l1_loss(
                pred.view(-1),
                y.view(-1),
                reduction="none",
            )

            if hasattr(data, "hard_flag"):
                w = 1.0 + 0.3 * data.hard_flag.view(-1).to(loss_each.device)
                loss = (loss_each * w).mean()
            else:
                loss = loss_each.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()
        print(total_loss / len(data_loader))
        return 0


class TopoCellRTEvaluator(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def collect_retention_predictions(self, data_loader):
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data in data_loader:
                data.to(self.device, non_blocking=True)
                y_hat = self.model(data)
                y_true.append(data.y.view(-1))
                y_pred.append(y_hat.view(-1))

            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)

        return y_true, y_pred

    def evaluate_retention(self, data_loader):
        y_true, y_pred = self.collect_retention_predictions(data_loader)

        mae = torch.abs(y_true - y_pred).mean()
        mre = torch.div(torch.abs(y_true - y_pred), y_true).mean()
        medAE = torch.median(torch.abs(y_true - y_pred))
        medRE = torch.median(torch.div(torch.abs(y_true - y_pred), y_true))

        score = torchmetrics.R2Score().to(self.device)
        r2 = score(y_pred, y_true)
        return mae, mre, medAE, medRE, r2


def set_reproducible_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def summarize_retention_errors(y_true, y_pred):
    abs_err = torch.abs(y_true - y_pred)
    p95 = torch.quantile(abs_err, 0.95).item()
    p99 = torch.quantile(abs_err, 0.99).item()
    err_100 = (abs_err > 100).sum().item()
    err_200 = (abs_err > 200).sum().item()
    total = abs_err.numel()
    return {
        "p95": p95,
        "p99": p99,
        "err_100": err_100,
        "err_200": err_200,
        "total": total,
    }


def save_retention_predictions(path, y_true, y_pred):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y_true = y_true.detach().cpu().tolist()
    y_pred = y_pred.detach().cpu().tolist()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "abs_err"])
        for yt, yp in zip(y_true, y_pred):
            writer.writerow([yt, yp, abs(yt - yp)])


if __name__ == '__main__':
    batch_size = 64
    num_works = 2
    lr = 1e-5
    epochs = 150
    test_batch = 64

    randint = 1

    set_reproducible_seed(randint)
    print("loading... get SMRT train data feature")
    dataset_train = TopoCellRTTrainDataset('./SMRT_data/reload/SMRT_train')

    train_len = dataset_train.__len__()
    train_len2 = int(dataset_train.__len__() * 0.9)

    val_len = train_len - train_len2
    generator = torch.Generator().manual_seed(randint)
    train_dataset, val_dataset = random_split(dataset_train, [train_len2, val_len], generator=generator)

    print("loading... get SMRT test data feature")
    dataset_test = TopoCellRTTestDataset('./SMRT_data/reload/SMRT_test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_works, pin_memory=True,
                              prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch, shuffle=False,
                            num_workers=num_works, pin_memory=True,
                            prefetch_factor=8, persistent_workers=True)
    test_loader = DataLoader(dataset_test, batch_size=test_batch, shuffle=False,
                             num_workers=num_works, pin_memory=True,
                             prefetch_factor=8, persistent_workers=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'use\r', device)
    print('-' * 100)
    print('# of training data samples:', len(train_dataset))
    print('# of validation data samples:', len(val_dataset))
    print('# of testing data samples:', len(dataset_test))
    print('-' * 100)
    print('Creating a model.')

    model = TopoCellRTNet()
    trainer = TopoCellRTTrainer(model, lr, device)
    evaluator = TopoCellRTEvaluator(model, device)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-' * 100)
    print('Start training.')

    model.to(device=device)

    val_mae_best = 999999.0
    best_model_path = './model_dict/best_model_TopoCellRT.pkl'
    results_dir = './results/TopoCellRT'
    os.makedirs(results_dir, exist_ok=True)

    with open('./results/TopoCellRT_result.txt', 'a') as f:
        for epoch in range(epochs):
            model.train()
            try:
                loss_training = trainer.train_one_epoch(train_loader, epoch)
                print(trainer.optimizer.param_groups[0]['lr'])
                model.eval()
                mae_train, mre_train, medAE_train, medRE_train, r2_train = evaluator.evaluate_retention(train_loader)
                val_mae, val_mre, val_medAE, val_medRE, val_r2 = evaluator.evaluate_retention(val_loader)
                print(f'epoch:{epoch}\ttrain_loss:{mae_train}\tmre_train:{mre_train}\tmedAE_train:{medAE_train}\tmedRE_train:{medRE_train}\tr2_train:{r2_train}')
                print(f'epoch:{epoch}\tval_mae:{val_mae}\tval_mre:{val_mre}\tval_medAE:{val_medAE}\tval_medRE:{val_medRE}\tval_r2:{val_r2}')
                f.write(f'epoch:{epoch}\tval_mae:{val_mae}\tval_mre:{val_mre}\tval_medAE:{val_medAE}\tval_medRE:{val_medRE}\tval_r2:{val_r2}\n')
                f.flush()

                if val_mae < val_mae_best:
                    torch.save(model.state_dict(), best_model_path)
                    val_mae_best = val_mae

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise Exception

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    val_true, val_pred = evaluator.collect_retention_predictions(val_loader)
    test_true, test_pred = evaluator.collect_retention_predictions(test_loader)

    val_stats = summarize_retention_errors(val_true, val_pred)
    test_stats = summarize_retention_errors(test_true, test_pred)

    print('Val error stats:', val_stats)
    print('Test error stats:', test_stats)

    save_retention_predictions(os.path.join(results_dir, 'val_predictions.csv'), val_true, val_pred)
    save_retention_predictions(os.path.join(results_dir, 'test_predictions.csv'), test_true, test_pred)

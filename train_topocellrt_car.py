import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import torchmetrics
from model_topocellrt_car import TopoCellRTNet
from data_topocellrt import TopoCellRTTrainDataset, TopoCellRTTestDataset
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Sampler
import warnings
import random
import os
import csv
from torch import optim
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
warnings.filterwarnings("ignore")


class ScaffoldBatchSampler(Sampler):
    def __init__(self, scaffold_to_indices, batch_size, groups_per_batch=16, samples_per_group=3, seed=0, shuffle=True):
        self.scaffold_to_indices = {k: list(v) for k, v in scaffold_to_indices.items()}
        self.batch_size = batch_size
        self.groups_per_batch = groups_per_batch
        self.samples_per_group = samples_per_group
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        pools = {k: list(v) for k, v in self.scaffold_to_indices.items()}
        for k in pools:
            if self.shuffle:
                rng.shuffle(pools[k])

        available = [k for k, v in pools.items() if len(v) > 0]

        while available:
            if self.shuffle:
                rng.shuffle(available)

            selected = available[:self.groups_per_batch]
            batch = []

            # 先从选中的 scaffold 里各取若干个
            for k in selected:
                take_n = min(self.samples_per_group, len(pools[k]))
                for _ in range(take_n):
                    batch.append(pools[k].pop())

            # 关键修复：selected 之后，必须立刻移除已经空的 scaffold
            available = [k for k in available if len(pools[k]) > 0]

            # 再随机补齐 batch；每次 pop 前都检查非空
            safety = 0
            while len(batch) < self.batch_size and available:
                safety += 1
                if safety > self.batch_size * 10:
                    break

                k = rng.choice(available)

                if len(pools[k]) == 0:
                    available = [kk for kk in available if len(pools[kk]) > 0]
                    continue

                batch.append(pools[k].pop())

                if len(pools[k]) == 0:
                    available = [kk for kk in available if len(pools[kk]) > 0]

            if batch:
                yield batch

    def __len__(self):
        total = sum(len(v) for v in self.scaffold_to_indices.values())
        return int(np.ceil(total / self.batch_size))


def smiles_to_morgan_fp(smiles, n_bits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(n_bits, dtype=torch.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float32)


def build_fp_cache(smiles_list, n_bits=2048, radius=2):
    cache = {}
    for s in tqdm(sorted(set(smiles_list))):
        cache[s] = smiles_to_morgan_fp(s, n_bits=n_bits, radius=radius)
    return cache


def compute_scaffold(smiles, fallback_key):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"NONE_{fallback_key}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold if scaffold else f"NONE_{fallback_key}"


def build_scaffold_groups(smiles_list):
    groups = {}
    for idx, s in enumerate(smiles_list):
        scaffold = compute_scaffold(s, idx)
        groups.setdefault(scaffold, []).append(idx)
    return groups


def get_batch_fps(smiles, fp_cache, device):
    if isinstance(smiles, (list, tuple)):
        fps = torch.stack([fp_cache[s] for s in smiles], dim=0)
    else:
        fps = fp_cache[str(smiles)].unsqueeze(0)
    return fps.to(device)


def tanimoto_matrix(fp):
    inter = fp @ fp.t()
    bits = fp.sum(dim=1, keepdim=True)
    union = bits + bits.t() - inter
    return inter / union.clamp_min(1.0)


def car_contrastive_loss(
    z,
    y,
    fp,
    pos_struct_min=0.45,
    pos_rt_delta=50.0,
    neg_struct_min=0.65,
    neg_rt_delta=300.0,
    pos_margin=0.75,
    neg_margin=0.15,
    neg_weight=1.5,
):
    bsz = z.size(0)
    if bsz <= 1:
        return torch.tensor(0.0, device=z.device), 0, 0

    cos = z @ z.t()
    struct_sim = tanimoto_matrix(fp)
    rt = y.view(-1, 1)
    rt_diff = torch.abs(rt - rt.t())

    eye = torch.eye(bsz, dtype=torch.bool, device=z.device)

    pos_mask = (struct_sim >= pos_struct_min) & (rt_diff <= pos_rt_delta) & (~eye)
    neg_mask = (struct_sim >= neg_struct_min) & (rt_diff >= neg_rt_delta) & (~eye)

    pos_loss = torch.tensor(0.0, device=z.device)
    neg_loss = torch.tensor(0.0, device=z.device)

    if pos_mask.any():
        pos_cos = cos[pos_mask]
        pos_loss = F.relu(pos_margin - pos_cos).pow(2).mean()

    if neg_mask.any():
        neg_cos = cos[neg_mask]
        neg_loss = F.relu(neg_cos - neg_margin).pow(2).mean()

    loss = pos_loss + neg_weight * neg_loss
    return loss, int(pos_mask.sum().item()), int(neg_mask.sum().item())


class TopoCellRTCARTrainer(object):
    def __init__(self, model, base_lr, contrast_lr, device, lambda_car_max=0.08, warmup_epochs=10):
        self.model = model
        self.device = device
        self.lambda_car_max = lambda_car_max
        self.warmup_epochs = warmup_epochs

        contrast_params = list(self.model.contrast_proj.parameters())
        contrast_param_ids = {id(p) for p in contrast_params}
        base_params = [p for p in self.model.parameters() if id(p) not in contrast_param_ids]

        self.optimizer = optim.AdamW(
            [
                {"params": base_params, "lr": base_lr},
                {"params": contrast_params, "lr": contrast_lr},
            ],
            amsgrad=True,
            weight_decay=1e-2,
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 60)

    def train_one_epoch(self, data_loader, epoch, fp_cache):
        total_reg_loss = 0.0
        total_car_loss = 0.0
        total_pos_pairs = 0
        total_neg_pairs = 0
        steps = 0

        warmup_factor = min(1.0, epoch / float(self.warmup_epochs))
        lambda_car = self.lambda_car_max * warmup_factor

        for _, data in enumerate(tqdm(data_loader)):
            data.to(self.device)
            pred, z = self.model(data, return_emb=True)
            y = data.y

            loss_each = F.smooth_l1_loss(
                pred.view(-1),
                y.view(-1),
                reduction="none",
            )

            if hasattr(data, "hard_flag"):
                w = 1.0 + 0.3 * data.hard_flag.view(-1).to(loss_each.device)
                reg_loss = (loss_each * w).mean()
            else:
                reg_loss = loss_each.mean()

            batch_fps = get_batch_fps(data.smiles, fp_cache, device=z.device)
            car_loss, pos_pairs, neg_pairs = car_contrastive_loss(z, y, batch_fps)

            loss = reg_loss + lambda_car * car_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_reg_loss += reg_loss.item()
            total_car_loss += car_loss.item()
            total_pos_pairs += pos_pairs
            total_neg_pairs += neg_pairs
            steps += 1

        self.scheduler.step()

        return {
            "reg_loss": total_reg_loss / max(steps, 1),
            "car_loss": total_car_loss / max(steps, 1),
            "pos_pairs": total_pos_pairs / max(steps, 1),
            "neg_pairs": total_neg_pairs / max(steps, 1),
            "lambda_car": lambda_car,
        }


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
    base_lr = 2e-6
    contrast_lr = 5e-5
    epochs = 60
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

    train_smiles = [train_dataset[i].smiles for i in range(len(train_dataset))]
    fp_cache = build_fp_cache(train_smiles)
    scaffold_groups = build_scaffold_groups(train_smiles)

    train_batch_sampler = ScaffoldBatchSampler(
        scaffold_groups,
        batch_size=batch_size,
        groups_per_batch=16,
        samples_per_group=3,
        seed=randint,
        shuffle=True,
    )

    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,
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
    pretrained_path = './model_dict/best_model_TopoCellRT.pkl'
    if os.path.exists(pretrained_path):
        missing, unexpected = model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        print('Loaded pretrained weights.')
        print('Missing keys:', len(missing), 'Unexpected keys:', len(unexpected))

    trainer = TopoCellRTCARTrainer(model, base_lr, contrast_lr, device)
    evaluator = TopoCellRTEvaluator(model, device)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-' * 100)
    print('Start training.')

    model.to(device=device)

    val_mae_best = 999999.0
    best_model_path = './model_dict/best_model_TopoCellRT_CAR.pkl'
    results_dir = './results/TopoCellRT_CAR'
    os.makedirs(results_dir, exist_ok=True)

    with open('./results/TopoCellRT_CAR_result.txt', 'a') as f:
        for epoch in range(epochs):
            model.train()
            try:
                train_stats = trainer.train_one_epoch(train_loader, epoch, fp_cache)
                print(trainer.optimizer.param_groups[0]['lr'])
                model.eval()
                mae_train, mre_train, medAE_train, medRE_train, r2_train = evaluator.evaluate_retention(train_loader)
                val_mae, val_mre, val_medAE, val_medRE, val_r2 = evaluator.evaluate_retention(val_loader)
                print(
                    f'epoch:{epoch}\ttrain_mae:{mae_train}\tval_mae:{val_mae}'
                    f'\tcar_loss:{train_stats["car_loss"]}\tpos_pairs:{train_stats["pos_pairs"]}'
                    f'\tneg_pairs:{train_stats["neg_pairs"]}\tlambda_car:{train_stats["lambda_car"]}'
                )
                print(
                    f'epoch:{epoch}\tval_mre:{val_mre}\tval_medAE:{val_medAE}\tval_medRE:{val_medRE}\tval_r2:{val_r2}'
                )
                f.write(
                    f'epoch:{epoch}\tval_mae:{val_mae}\tval_mre:{val_mre}\tval_medAE:{val_medAE}'
                    f'\tval_medRE:{val_medRE}\tval_r2:{val_r2}'
                    f'\tcar_loss:{train_stats["car_loss"]}\tpos_pairs:{train_stats["pos_pairs"]}'
                    f'\tneg_pairs:{train_stats["neg_pairs"]}\tlambda_car:{train_stats["lambda_car"]}\n'
                )
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
        model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)

    model.eval()
    val_true, val_pred = evaluator.collect_retention_predictions(val_loader)
    test_true, test_pred = evaluator.collect_retention_predictions(test_loader)

    val_stats = summarize_retention_errors(val_true, val_pred)
    test_stats = summarize_retention_errors(test_true, test_pred)

    print('Val error stats:', val_stats)
    print('Test error stats:', test_stats)

    save_retention_predictions(os.path.join(results_dir, 'val_predictions.csv'), val_true, val_pred)
    save_retention_predictions(os.path.join(results_dir, 'test_predictions.csv'), test_true, test_pred)

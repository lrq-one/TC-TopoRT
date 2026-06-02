import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from rdkit import Chem

from mp.complex import ComplexBatch
from mp.smrt_dataset import SMRTComplexDataset
from net.topocellrt_cwn_dualview import TopoCellRTCWNDualView


class Config:
    ORIGIN_TRAIN_CSV = os.getenv("ORIGIN_TRAIN_CSV", "data/SMRT_train.csv")
    ORIGIN_TEST_CSV = os.getenv("ORIGIN_TEST_CSV", "data/SMRT_test.csv")

    TAUT_TRAIN_CSV = os.getenv(
        "TAUT_TRAIN_CSV",
        "data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv",
    )
    TAUT_TEST_CSV = os.getenv(
        "TAUT_TEST_CSV",
        "data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv",
    )

    ORIGIN_TRAIN_ROOT = os.getenv("ORIGIN_TRAIN_ROOT", "smrt_cwn_data_train")
    ORIGIN_TEST_ROOT = os.getenv("ORIGIN_TEST_ROOT", "smrt_cwn_data_test")

    # 这里默认先复用你已经跑过的 taut cache，省时间。
    # 如果脚本检查 y 不一致，再换成新的 root 重新处理。
    TAUT_TRAIN_ROOT = os.getenv("TAUT_TRAIN_ROOT", "smrt_cwn_data_train_taut_strict")
    TAUT_TEST_ROOT = os.getenv("TAUT_TEST_ROOT", "smrt_cwn_data_test_taut_strict")

    ORIGIN_CKPT = os.getenv(
        "ORIGIN_CKPT",
        "results_TopoCellRT_CWNReplace_orig/checkpoints/best_model.pth",
    )
    TAUT_CKPT = os.getenv(
        "TAUT_CKPT",
        "results_TopoCellRT_CWNReplace_taut_strict/checkpoints/best_model.pth",
    )

    RESULT_DIR = os.getenv("RESULT_DIR", "results_TopoCellRT_CWNDualView_end2end")
    CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
    PLOT_DIR = os.path.join(RESULT_DIR, "plots")
    LOG_FILE = os.path.join(RESULT_DIR, "train.log")

    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    EPOCHS = int(os.getenv("EPOCHS", "150"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
    HEAD_LR = float(os.getenv("HEAD_LR", "1e-4"))
    BRANCH_LR = float(os.getenv("BRANCH_LR", "2e-5"))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-2"))

    SPLIT_SEED = int(os.getenv("SPLIT_SEED", "1"))
    MAX_RING_SIZE = int(os.getenv("MAX_RING_SIZE", "6"))
    USE_EDGE_ATTR = True
    HUBER_BETA = float(os.getenv("HUBER_BETA", "1.0"))

    FREEZE_EPOCHS = int(os.getenv("FREEZE_EPOCHS", "10"))
    AUX_LOSS_W = float(os.getenv("AUX_LOSS_W", "0.2"))
    GATE_PRIOR_W = float(os.getenv("GATE_PRIOR_W", "0.05"))
    PREF_LOSS_W = float(os.getenv("PREF_LOSS_W", "0.5"))
    PREF_TEMP = float(os.getenv("PREF_TEMP", "20.0"))

    # V3 default: do not update origin/taut branches, only train gate.
    TRAIN_BRANCHES = int(os.getenv("TRAIN_BRANCHES", "0")) == 1
    TRAIN_TAU = int(os.getenv("TRAIN_TAU", "0")) == 1

    CWN_LAYERS = int(os.getenv("CWN_LAYERS", "6"))
    CWN_HIDDEN = int(os.getenv("CWN_HIDDEN", "256"))
    SHARE_ENCODER = int(os.getenv("SHARE_ENCODER", "0")) == 1

    INIT_TAU = float(os.getenv("INIT_TAU", "5.0"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "5.0"))
    GATE_PRIOR_ALPHA = float(os.getenv("GATE_PRIOR_ALPHA", "0.704"))

    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 只跑少量 batch 做 smoke test，用于检查脚本，不正式训练
    MAX_TRAIN_STEPS = int(os.getenv("MAX_TRAIN_STEPS", "0"))
    MAX_VAL_STEPS = int(os.getenv("MAX_VAL_STEPS", "0"))


os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.PLOT_DIR, exist_ok=True)

logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format="%(message)s",
    filemode="a",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)


def log_json(data):
    logging.info(json.dumps(data, ensure_ascii=False))


def complex_collate_fn(batch):
    origin_list, taut_list = zip(*batch)
    return (
        ComplexBatch.from_complex_list(origin_list),
        ComplexBatch.from_complex_list(taut_list),
    )


class PairedComplexDataset(Dataset):
    def __init__(self, origin_dataset, taut_dataset, full_check=True):
        self.origin_dataset = origin_dataset
        self.taut_dataset = taut_dataset

        if len(origin_dataset) != len(taut_dataset):
            raise RuntimeError(
                f"origin/taut dataset length mismatch: {len(origin_dataset)} vs {len(taut_dataset)}"
            )

        self.n = len(origin_dataset)

        # 检查 y 是否严格一一对应。InMemory 数据集索引很快。
        if full_check:
            max_diff = 0.0
            bad_i = -1
            for i in range(self.n):
                yo = float(origin_dataset[i].y.view(-1)[0])
                yt = float(taut_dataset[i].y.view(-1)[0])
                d = abs(yo - yt)
                if d > max_diff:
                    max_diff = d
                    bad_i = i
                if d > 1e-6:
                    raise RuntimeError(
                        f"paired dataset y mismatch at i={i}: origin={yo}, taut={yt}"
                    )

            print(f"✅ PairedComplexDataset y check passed. N={self.n}, max_diff={max_diff}, bad_i={bad_i}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.origin_dataset[idx], self.taut_dataset[idx]


def load_valid_meta(csv_path):
    df = pd.read_csv(csv_path, engine="python")
    raw_cols = df.columns.tolist()
    df.columns = [str(c).lower().strip() for c in df.columns]

    if "smile" in df.columns and "smiles" not in df.columns:
        df.rename(columns={"smile": "smiles"}, inplace=True)

    if "smiles" not in df.columns or "rt" not in df.columns:
        raise ValueError(f"{csv_path} must contain smiles/rt, got {raw_cols}")

    df["rt"] = df["rt"].astype(float)
    df = df[df["rt"] > 300.0].copy()

    rows = []
    for source_idx, row in df.iterrows():
        smi = str(row["smiles"])
        rt = float(row["rt"])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        item = {
            "Source_Index": int(source_idx),
            "SMILES": smi,
            "Actual_RT_csv": rt,
        }

        if "orig_smile" in df.columns:
            item["Orig_SMILES"] = str(row["orig_smile"])
        else:
            item["Orig_SMILES"] = smi

        for c in ["raw_changed", "real_changed", "formula_same", "heavy_same", "fallback", "reason"]:
            if c in df.columns:
                item[c] = row[c]

        rows.append(item)

    return pd.DataFrame(rows)


def metric_dict(y, p, prefix):
    e = np.abs(y - p)
    return {
        f"{prefix}_mae": float(e.mean()),
        f"{prefix}_medae": float(np.median(e)),
        f"{prefix}_rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        f"{prefix}_r2": float(r2_score(y, p)),
        f"{prefix}_p95": float(np.percentile(e, 95)),
        f"{prefix}_p99": float(np.percentile(e, 99)),
        f"{prefix}_gt80": int((e > 80).sum()),
        f"{prefix}_gt100": int((e > 100).sum()),
        f"{prefix}_gt200": int((e > 200).sum()),
        f"{prefix}_n": int(len(e)),
    }


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def freeze_all_branches(model):
    set_requires_grad(model.origin_encoder, False)
    set_requires_grad(model.taut_encoder, False)

    if hasattr(model, "gate_mlp"):
        set_requires_grad(model.gate_mlp, True)
    if hasattr(model, "gate_delta"):
        set_requires_grad(model.gate_delta, True)

    model.tau.requires_grad = bool(Config.TRAIN_TAU)


def unfreeze_head_branches(model):
    if not Config.TRAIN_BRANCHES:
        freeze_all_branches(model)
        return

    # Optional experimental mode. Default should NOT use this.
    set_requires_grad(model.origin_encoder, False)
    set_requires_grad(model.taut_encoder, False)

    for enc in [model.origin_encoder, model.taut_encoder]:
        for name in [
            "trans_graph",
            "trans_add",
            "layerNorm_out",
            "trans_out",
            "global_proj",
            "global_gate",
            "out_lin",
        ]:
            if hasattr(enc, name):
                set_requires_grad(getattr(enc, name), True)

    if hasattr(model, "gate_mlp"):
        set_requires_grad(model.gate_mlp, True)
    if hasattr(model, "gate_delta"):
        set_requires_grad(model.gate_delta, True)

    model.tau.requires_grad = bool(Config.TRAIN_TAU)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def enforce_train_eval_mode(model, epoch):
    """
    Critical fix:
    requires_grad=False does not freeze BatchNorm running_mean/running_var.
    Since origin/taut branches are pretrained, keep them eval by default.
    """
    model.train()

    if not Config.TRAIN_BRANCHES:
        model.origin_encoder.eval()
        model.taut_encoder.eval()
        if hasattr(model, "gate_mlp"):
            model.gate_mlp.train()
        if hasattr(model, "gate_delta"):
            model.gate_delta.train()
        return

    if hasattr(model.origin_encoder, "cwn_adapter"):
        model.origin_encoder.cwn_adapter.eval()
    if hasattr(model.taut_encoder, "cwn_adapter"):
        model.taut_encoder.cwn_adapter.eval()

    if hasattr(model, "gate_mlp"):
        model.gate_mlp.train()
    if hasattr(model, "gate_delta"):
        model.gate_delta.train()


def load_branch_ckpt(model):
    if Config.ORIGIN_CKPT and os.path.exists(Config.ORIGIN_CKPT):
        print("=== Loading origin branch checkpoint ===")
        print(Config.ORIGIN_CKPT)
        state = torch.load(Config.ORIGIN_CKPT, map_location=Config.DEVICE)
        missing, unexpected = model.origin_encoder.load_state_dict(state, strict=False)
        print("origin missing:", missing)
        print("origin unexpected:", unexpected)
    else:
        print("⚠️ origin checkpoint not found:", Config.ORIGIN_CKPT)

    if Config.TAUT_CKPT and os.path.exists(Config.TAUT_CKPT):
        print("=== Loading taut branch checkpoint ===")
        print(Config.TAUT_CKPT)
        state = torch.load(Config.TAUT_CKPT, map_location=Config.DEVICE)
        missing, unexpected = model.taut_encoder.load_state_dict(state, strict=False)
        print("taut missing:", missing)
        print("taut unexpected:", unexpected)
    else:
        print("⚠️ taut checkpoint not found:", Config.TAUT_CKPT)


def make_optimizer(model, epoch):
    if (not Config.TRAIN_BRANCHES) or epoch <= Config.FREEZE_EPOCHS:
        freeze_all_branches(model)
    else:
        unfreeze_head_branches(model)

    gate_params = []
    branch_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("gate_mlp") or name.startswith("gate_delta") or name == "tau":
            gate_params.append(param)
        else:
            branch_params.append(param)

    params = []
    if branch_params:
        params.append({"params": branch_params, "lr": Config.BRANCH_LR})
    if gate_params:
        params.append({"params": gate_params, "lr": Config.HEAD_LR})

    optimizer = torch.optim.AdamW(
        params,
        weight_decay=Config.WEIGHT_DECAY,
        amsgrad=True,
    )
    return optimizer


def compute_loss(final_pred, aux, target, model):
    main_loss = F.smooth_l1_loss(final_pred.view(-1), target, beta=Config.HUBER_BETA)

    # Branch losses are only useful if branches are trainable.
    if Config.TRAIN_BRANCHES:
        origin_loss = F.smooth_l1_loss(aux["origin_pred"].view(-1), target, beta=Config.HUBER_BETA)
        taut_loss = F.smooth_l1_loss(aux["taut_pred"].view(-1), target, beta=Config.HUBER_BETA)
    else:
        origin_loss = torch.zeros_like(main_loss)
        taut_loss = torch.zeros_like(main_loss)

    gate_prior = model.gate_prior_loss(aux)

    # Preference target:
    # alpha_origin should be high if origin error < taut error.
    with torch.no_grad():
        err_o = torch.abs(aux["origin_pred"].view(-1) - target)
        err_t = torch.abs(aux["taut_pred"].view(-1) - target)

        # Soft target is more stable than hard 0/1.
        # If taut is much worse than origin, target -> 1.
        # If origin is much worse than taut, target -> 0.
        pref_target = torch.sigmoid((err_t - err_o) / Config.PREF_TEMP)

    alpha = aux["alpha_origin"].view(-1).clamp(1e-4, 1.0 - 1e-4)
    pref_loss = F.binary_cross_entropy(alpha, pref_target)

    loss = (
        main_loss
        + Config.AUX_LOSS_W * origin_loss
        + Config.AUX_LOSS_W * taut_loss
        + Config.GATE_PRIOR_W * gate_prior
        + Config.PREF_LOSS_W * pref_loss
    )

    return loss, {
        "main_loss": float(main_loss.detach().cpu()),
        "origin_loss": float(origin_loss.detach().cpu()),
        "taut_loss": float(taut_loss.detach().cpu()),
        "gate_prior": float(gate_prior.detach().cpu()),
        "pref_loss": float(pref_loss.detach().cpu()),
    }


def train_one_epoch(model, loader, optimizer, epoch):
    enforce_train_eval_mode(model, epoch)
    total_loss = 0.0
    total_mae = 0.0
    steps = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)

    for origin_batch, taut_batch in pbar:
        origin_batch = origin_batch.to(Config.DEVICE)
        taut_batch = taut_batch.to(Config.DEVICE)

        target = origin_batch.y.view(-1).float()

        optimizer.zero_grad()
        final_pred, aux = model(origin_batch, taut_batch, return_aux=True)
        loss, parts = compute_loss(final_pred, aux, target, model)
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        mae = F.l1_loss(final_pred.view(-1), target).item()
        total_loss += float(loss.item())
        total_mae += mae
        steps += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "mae": f"{mae:.2f}",
            "tau": f"{float(aux['tau'].detach().cpu()):.2f}",
            "alpha": f"{float(aux['alpha_origin'].mean().detach().cpu()):.3f}",
            "use": f"{float(aux['soft_use'].mean().detach().cpu()):.3f}",
        })

        if Config.MAX_TRAIN_STEPS > 0 and steps >= Config.MAX_TRAIN_STEPS:
            break

    return total_loss / max(steps, 1), total_mae / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, epoch, prefix="val"):
    model.eval()

    y_all = []
    final_all = []
    origin_all = []
    taut_all = []
    alpha_all = []
    use_all = []

    total_loss = 0.0
    steps = 0

    for origin_batch, taut_batch in loader:
        origin_batch = origin_batch.to(Config.DEVICE)
        taut_batch = taut_batch.to(Config.DEVICE)

        target = origin_batch.y.view(-1).float()
        final_pred, aux = model(origin_batch, taut_batch, return_aux=True)

        loss, _ = compute_loss(final_pred, aux, target, model)

        y_all.append(target.detach().cpu())
        final_all.append(final_pred.view(-1).detach().cpu())
        origin_all.append(aux["origin_pred"].view(-1).detach().cpu())
        taut_all.append(aux["taut_pred"].view(-1).detach().cpu())
        alpha_all.append(aux["alpha_origin"].view(-1).detach().cpu())
        use_all.append(aux["soft_use"].view(-1).detach().cpu())

        total_loss += float(loss.item())
        steps += 1

        if prefix == "val" and Config.MAX_VAL_STEPS > 0 and steps >= Config.MAX_VAL_STEPS:
            break

    y = torch.cat(y_all).numpy()
    pf = torch.cat(final_all).numpy()
    po = torch.cat(origin_all).numpy()
    pt = torch.cat(taut_all).numpy()
    alpha = torch.cat(alpha_all).numpy()
    use = torch.cat(use_all).numpy()

    m = {}
    m.update(metric_dict(y, pf, f"{prefix}_final"))
    m.update(metric_dict(y, po, f"{prefix}_origin"))
    m.update(metric_dict(y, pt, f"{prefix}_taut"))

    m[f"{prefix}_loss"] = total_loss / max(steps, 1)
    m[f"{prefix}_alpha_mean"] = float(alpha.mean())
    m[f"{prefix}_alpha_p10"] = float(np.percentile(alpha, 10))
    m[f"{prefix}_alpha_p90"] = float(np.percentile(alpha, 90))
    m[f"{prefix}_soft_use_mean"] = float(use.mean())
    m[f"{prefix}_soft_use_p90"] = float(np.percentile(use, 90))
    m[f"{prefix}_tau"] = float(model.tau.detach().cpu())

    return m


@torch.no_grad()
def export_predictions(model, loader, meta_df, save_path, split_name):
    model.eval()

    y_all = []
    final_all = []
    origin_all = []
    taut_all = []
    alpha_all = []
    use_all = []

    for origin_batch, taut_batch in tqdm(loader, desc=f"Export {split_name}", leave=False):
        origin_batch = origin_batch.to(Config.DEVICE)
        taut_batch = taut_batch.to(Config.DEVICE)

        target = origin_batch.y.view(-1).float()
        final_pred, aux = model(origin_batch, taut_batch, return_aux=True)

        y_all.append(target.detach().cpu())
        final_all.append(final_pred.view(-1).detach().cpu())
        origin_all.append(aux["origin_pred"].view(-1).detach().cpu())
        taut_all.append(aux["taut_pred"].view(-1).detach().cpu())
        alpha_all.append(aux["alpha_origin"].view(-1).detach().cpu())
        use_all.append(aux["soft_use"].view(-1).detach().cpu())

    y = torch.cat(y_all).numpy()
    pf = torch.cat(final_all).numpy()
    po = torch.cat(origin_all).numpy()
    pt = torch.cat(taut_all).numpy()
    alpha = torch.cat(alpha_all).numpy()
    use = torch.cat(use_all).numpy()

    n = min(len(meta_df), len(y))

    out = meta_df.iloc[:n].copy().reset_index(drop=True)
    out["Actual_RT"] = y[:n]
    out["Final_Pred"] = pf[:n]
    out["Origin_Pred"] = po[:n]
    out["Taut_Pred"] = pt[:n]
    out["Alpha_Origin"] = alpha[:n]
    out["Soft_Use"] = use[:n]

    out["Final_Abs_Error"] = np.abs(out["Actual_RT"] - out["Final_Pred"])
    out["Origin_Abs_Error"] = np.abs(out["Actual_RT"] - out["Origin_Pred"])
    out["Taut_Abs_Error"] = np.abs(out["Actual_RT"] - out["Taut_Pred"])
    out["Gain_vs_Origin"] = out["Origin_Abs_Error"] - out["Final_Abs_Error"]
    out["Gain_vs_Taut"] = out["Taut_Abs_Error"] - out["Final_Abs_Error"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out.to_csv(save_path, index=False)

    print(f"✅ exported {split_name}:", save_path)
    print(metric_dict(out["Actual_RT"].values, out["Final_Pred"].values, f"{split_name}_final"))
    print(metric_dict(out["Actual_RT"].values, out["Origin_Pred"].values, f"{split_name}_origin"))
    print(metric_dict(out["Actual_RT"].values, out["Taut_Pred"].values, f"{split_name}_taut"))


def main():
    print("=== DualView Config ===")
    for k, v in Config.__dict__.items():
        if k.isupper():
            print(k, "=", v)

    torch.manual_seed(Config.SPLIT_SEED)
    np.random.seed(Config.SPLIT_SEED)

    print("\n=== Loading origin train/test datasets ===")
    origin_train_full = SMRTComplexDataset(
        Config.ORIGIN_TRAIN_ROOT,
        Config.ORIGIN_TRAIN_CSV,
        Config.MAX_RING_SIZE,
        use_edge_features=Config.USE_EDGE_ATTR,
    )
    origin_test = SMRTComplexDataset(
        Config.ORIGIN_TEST_ROOT,
        Config.ORIGIN_TEST_CSV,
        Config.MAX_RING_SIZE,
        use_edge_features=Config.USE_EDGE_ATTR,
    )

    print("\n=== Loading taut train/test datasets ===")
    taut_train_full = SMRTComplexDataset(
        Config.TAUT_TRAIN_ROOT,
        Config.TAUT_TRAIN_CSV,
        Config.MAX_RING_SIZE,
        use_edge_features=Config.USE_EDGE_ATTR,
    )
    taut_test = SMRTComplexDataset(
        Config.TAUT_TEST_ROOT,
        Config.TAUT_TEST_CSV,
        Config.MAX_RING_SIZE,
        use_edge_features=Config.USE_EDGE_ATTR,
    )

    print("\n=== Building paired datasets ===")
    paired_train_full = PairedComplexDataset(origin_train_full, taut_train_full, full_check=True)
    paired_test = PairedComplexDataset(origin_test, taut_test, full_check=True)

    total_len = len(paired_train_full)
    train_len = int(0.9 * total_len)
    val_len = total_len - train_len

    train_set, val_set = random_split(
        paired_train_full,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(Config.SPLIT_SEED),
    )

    print("total/train/val/test:", total_len, train_len, val_len, len(paired_test))

    # metadata for export
    origin_meta_train = load_valid_meta(Config.ORIGIN_TRAIN_CSV)
    taut_meta_train = load_valid_meta(Config.TAUT_TRAIN_CSV)
    origin_meta_test = load_valid_meta(Config.ORIGIN_TEST_CSV)
    taut_meta_test = load_valid_meta(Config.TAUT_TEST_CSV)

    train_indices = list(train_set.indices)
    val_indices = list(val_set.indices)

    train_meta = origin_meta_train.iloc[train_indices].reset_index(drop=True)
    val_meta = origin_meta_train.iloc[val_indices].reset_index(drop=True)
    test_meta = origin_meta_test.reset_index(drop=True)

    # add taut smiles/audit to metadata
    train_meta["Taut_SMILES"] = taut_meta_train.iloc[train_indices]["SMILES"].values
    val_meta["Taut_SMILES"] = taut_meta_train.iloc[val_indices]["SMILES"].values
    test_meta["Taut_SMILES"] = taut_meta_test["SMILES"].values

    for c in ["raw_changed", "real_changed", "formula_same", "heavy_same", "fallback", "reason"]:
        if c in taut_meta_train.columns:
            train_meta[c] = taut_meta_train.iloc[train_indices][c].values
            val_meta[c] = taut_meta_train.iloc[val_indices][c].values
        if c in taut_meta_test.columns:
            test_meta[c] = taut_meta_test[c].values

    train_loader = DataLoader(
        train_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=complex_collate_fn,
        num_workers=Config.NUM_WORKERS,
    )
    train_export_loader = DataLoader(
        train_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        paired_test,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=0,
    )

    print("\n=== Init DualView model ===")
    model = TopoCellRTCWNDualView(
        emb_dim=256,
        cwn_layers=Config.CWN_LAYERS,
        cwn_hidden=Config.CWN_HIDDEN,
        max_dim=2,
        drop_ratio=0.0,
        share_encoder=Config.SHARE_ENCODER,
        init_tau=Config.INIT_TAU,
        temperature=Config.TEMPERATURE,
        gate_prior_alpha=Config.GATE_PRIOR_ALPHA,
    ).to(Config.DEVICE)

    load_branch_ckpt(model)

    best_val_mae = float("inf")
    best_epoch = -1
    bad_count = 0
    patience = 30
    optimizer = None
    current_stage = None

    print("\n=== Training ===")

    for epoch in range(1, Config.EPOCHS + 1):
        stage = "gate_only" if ((not Config.TRAIN_BRANCHES) or epoch <= Config.FREEZE_EPOCHS) else "head_finetune"

        if stage != current_stage:
            optimizer = make_optimizer(model, epoch)
            current_stage = stage
            print(f"\n=== Switch stage: {stage} at epoch {epoch} ===")
            print("trainable params:", count_trainable_params(model))

        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, epoch)
        val_metrics = evaluate(model, val_loader, epoch, prefix="val")

        row = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": train_loss,
            "train_mae": train_mae,
            **val_metrics,
        }
        log_json(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

        val_mae = val_metrics["val_final_mae"]

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            bad_count = 0
            torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
            print(f"✅ saved best model epoch={epoch}, val_final_mae={val_mae:.4f}")
        else:
            bad_count += 1

        if Config.MAX_TRAIN_STEPS > 0:
            print("Smoke-test mode detected. Stop after one epoch.")
            break

        if bad_count >= patience:
            print(f"Early stopping at epoch {epoch}. best_epoch={best_epoch}, best_val_mae={best_val_mae:.4f}")
            break

    print("\n=== Load best model and evaluate/export ===")
    best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    state = torch.load(best_path, map_location=Config.DEVICE)
    model.load_state_dict(state, strict=False)

    val_metrics = evaluate(model, val_loader, best_epoch, prefix="val")
    test_metrics = evaluate(model, test_loader, best_epoch, prefix="test")

    print("\n=== FINAL VAL ===")
    print(json.dumps(val_metrics, ensure_ascii=False, indent=2))
    print("\n=== FINAL TEST ===")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

    with open(os.path.join(Config.RESULT_DIR, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_mae": best_val_mae,
                "val": val_metrics,
                "test": test_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    export_predictions(
        model,
        train_export_loader,
        train_meta,
        os.path.join(Config.RESULT_DIR, "dualview_train_predictions.csv"),
        "train",
    )
    export_predictions(
        model,
        val_loader,
        val_meta,
        os.path.join(Config.RESULT_DIR, "dualview_val_predictions.csv"),
        "val",
    )
    export_predictions(
        model,
        test_loader,
        test_meta,
        os.path.join(Config.RESULT_DIR, "dualview_test_predictions.csv"),
        "test",
    )

    print("\n✅ done:", Config.RESULT_DIR)


if __name__ == "__main__":
    main()

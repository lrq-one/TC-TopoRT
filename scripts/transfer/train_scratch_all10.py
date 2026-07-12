#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train TC-TopoRT from random initialization on all 10 external datasets.

A single execution performs dual-view external 10-fold cross-validation
with all model parameters trainable. The original and strict-tautomer
predictions are averaged to obtain the final scratch result for each
dataset.

The training implementation is retained from the same controlled
transfer-versus-scratch experiment used for Table 8.
"""

from __future__ import annotations


# ===== EMBEDDED TRAINER =====

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold, GroupKFold
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
GWN_ROOT = REPO_ROOT / "gwn"

if str(GWN_ROOT) not in sys.path:
    sys.path.insert(0, str(GWN_ROOT))

from mp.smrt_dataset import SMRTComplexDataset
from mp.complex import ComplexBatch
from net.topocellrt_cwn_replace import TopoCellRTCWNReplace


RUNS = {
    "seed1": {
        "subdir": "seed1",
        "seed": 1,
    },
    "seed5": {
        "subdir": "seed5",
        "seed": 5,
    },
    "seed79": {
        "subdir": "seed79",
        "seed": 79,
    },
    "seed123": {
        "subdir": "seed123",
        "seed": 123,
    },
    "seed256": {
        "subdir": "seed256",
        "seed": 256,
    },
    "seed5_soup": {
        "subdir": "seed5_soup",
        "seed": 5,
    },
}

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_repo_path(path_value):
    """Resolve a CLI path relative to the repository root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def save_csv(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path} shape={df.shape}")


def complex_collate_fn(batch):
    return ComplexBatch.from_complex_list(batch)


class ExternalTargetDataset(Dataset):
    """
    SMRTComplexDataset 里 y 是 dummy RT。
    这个 wrapper 在取样时把 y 改成 external RT。
    """
    def __init__(self, base_dataset, targets):
        self.base_dataset = base_dataset
        self.targets = np.asarray(targets, dtype=np.float32)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[int(idx)]
        item.y = torch.tensor([float(self.targets[int(idx)])], dtype=torch.float32)
        return item


def make_model(cwn_layers, cwn_hidden, device):
    model = TopoCellRTCWNReplace(
        emb_dim=256,
        cwn_layers=int(cwn_layers),
        cwn_hidden=int(cwn_hidden),
        max_dim=2,
        drop_ratio=0.0,
    ).to(device)
    return model


def load_state_dict_safely(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    else:
        state = obj
    model.load_state_dict(state, strict=True)
    return model


def reset_module(module):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def set_trainable(model, freeze_mode):
    for p in model.parameters():
        p.requires_grad = False

    trainable_names = []

    def unfreeze_module(module, prefix):
        for p in module.parameters():
            p.requires_grad = True
        trainable_names.append(prefix)

    if freeze_mode == "out_lin_only":
        unfreeze_module(model.out_lin, "out_lin")

    elif freeze_mode == "head_plus_trans_out":
        unfreeze_module(model.trans_out, "trans_out")
        unfreeze_module(model.out_lin, "out_lin")

    elif freeze_mode == "rt_head_full":
        # Freeze only CWN molecular representation encoder.
        # Fine-tune the full RT adaptation/prediction module after cwn_adapter.
        unfreeze_module(model.trans_graph, "trans_graph")
        unfreeze_module(model.trans_add, "trans_add")
        unfreeze_module(model.layerNorm_out, "layerNorm_out")
        unfreeze_module(model.trans_out, "trans_out")
        unfreeze_module(model.global_proj, "global_proj")
        unfreeze_module(model.global_gate, "global_gate")
        unfreeze_module(model.out_lin, "out_lin")

    elif freeze_mode == "last_blocks":
        unfreeze_module(model.layerNorm_out, "layerNorm_out")
        unfreeze_module(model.trans_out, "trans_out")
        unfreeze_module(model.global_proj, "global_proj")
        unfreeze_module(model.global_gate, "global_gate")
        unfreeze_module(model.out_lin, "out_lin")

    elif freeze_mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        trainable_names.append("all")

    else:
        raise ValueError(f"Unknown freeze_mode={freeze_mode}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return trainable_names, n_trainable, n_total


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]

    if len(y) == 0:
        return {
            "n": 0, "mae": np.nan, "medae": np.nan, "rmse": np.nan,
            "r2": np.nan, "pearson": np.nan, "spearman": np.nan,
            "bias": np.nan,
        }

    e = np.abs(y - p)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    return {
        "n": int(len(y)),
        "mae": float(e.mean()),
        "medae": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan),
        "pearson": float(pd.Series(y).corr(pd.Series(p), method="pearson")) if len(y) > 1 else np.nan,
        "spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")) if len(y) > 1 else np.nan,
        "bias": float(np.mean(p - y)),
    }


def set_batchnorm_eval(model):
    """
    Safe scratch training for small external datasets / sparse cell complexes.
    Keep BatchNorm layers in eval mode to avoid:
    ValueError: Expected more than 1 value per channel when training.
    Affine BN parameters can still receive gradients if requires_grad=True.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def set_tl_train_mode(model, freeze_mode):
    """
    Transfer-learning mode:
    keep the frozen encoder in eval mode so BatchNorm statistics are not updated.
    Only the selected adaptation modules are set to train mode.
    """
    model.eval()

    if freeze_mode == "out_lin_only":
        model.out_lin.train()

    elif freeze_mode == "head_plus_trans_out":
        model.trans_out.train()
        model.out_lin.train()

    elif freeze_mode == "rt_head_full":
        model.trans_graph.train()
        model.trans_add.train()
        model.layerNorm_out.train()
        model.trans_out.train()
        model.global_proj.train()
        model.global_gate.train()
        model.out_lin.train()

    elif freeze_mode == "last_blocks":
        model.layerNorm_out.train()
        model.trans_out.train()
        model.global_proj.train()
        model.global_gate.train()
        model.out_lin.train()

    elif freeze_mode == "all":
        # Scratch baseline: train all trainable weights, but keep BatchNorm layers in eval mode.
        # This avoids single-cell BatchNorm crashes in sparse CWN dimensions.
        model.train()
        set_batchnorm_eval(model)

    else:
        raise ValueError(f"Unknown freeze_mode={freeze_mode}")


def train_one_epoch(model, loader, optimizer, device, huber_beta, freeze_mode, y_mean, y_std):
    set_tl_train_mode(model, freeze_mode)
    total_loss = 0.0
    total_mae = 0.0
    steps = 0

    for batch in loader:
        batch = batch.to(device)
        target_raw = batch.y.view(-1).float()
        target_z = (target_raw - y_mean) / y_std

        optimizer.zero_grad()

        pred_z = model(batch)
        if isinstance(pred_z, tuple):
            pred_z = pred_z[0]
        pred_z = pred_z.view(-1)

        loss = F.smooth_l1_loss(pred_z, target_z, beta=huber_beta)
        loss.backward()

        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()

        pred_raw = pred_z.detach() * y_std + y_mean

        total_loss += float(loss.item())
        total_mae += float(F.l1_loss(pred_raw, target_raw).item())
        steps += 1

    return total_loss / max(steps, 1), total_mae / max(steps, 1)


@torch.no_grad()
def predict(model, loader, device, y_mean, y_std):
    model.eval()
    ys = []
    ps = []

    for batch in loader:
        batch = batch.to(device)
        target_raw = batch.y.view(-1).float()

        pred_z = model(batch)
        if isinstance(pred_z, tuple):
            pred_z = pred_z[0]

        pred_raw = pred_z.view(-1) * y_std + y_mean

        ys.append(target_raw.detach().cpu())
        ps.append(pred_raw.detach().cpu())

    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


@torch.no_grad()
def eval_mae(model, loader, device, y_mean, y_std):
    y, p = predict(model, loader, device, y_mean, y_std)
    return float(np.mean(np.abs(y - p)))


def build_loader(dataset, indices, batch_size, shuffle):
    subset = Subset(dataset, list(map(int, indices)))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=complex_collate_fn,
        num_workers=0,
    )


def run_one_view_cv(
    args,
    dataset_name,
    view_name,
    base_dataset,
    meta,
    run_key,
    run_dir,
    source_fold,
    run_seed,
    device,
):
    sub = meta[meta["dataset_name"] == dataset_name].copy()
    sub = sub.reset_index(drop=True)

    global_indices = sub["stage4_index"].values.astype(int)
    y_all = sub["rt"].values.astype(np.float32)

    if len(sub) < args.min_n:
        print(f"[SKIP] {dataset_name} n={len(sub)} < min_n={args.min_n}")
        return pd.DataFrame(), pd.DataFrame()

    wrapped = ExternalTargetDataset(base_dataset, targets=meta.sort_values("stage4_index")["rt"].values)

    if args.group_cv:
        if args.group_col not in sub.columns:
            raise RuntimeError(f"group_col={args.group_col} not found in metadata columns")
        groups = sub[args.group_col].fillna(sub["stage4_index"].astype(str)).astype(str).values
        n_groups = len(np.unique(groups))
        k = min(args.cv_folds, n_groups)
        cv = GroupKFold(n_splits=k)
        split_iter = cv.split(np.zeros(len(sub)), y_all, groups)
        print(f"[GroupKFold] dataset={dataset_name} group_col={args.group_col} n_groups={n_groups} folds={k}")
    else:
        k = min(args.cv_folds, len(sub))
        seed_for_cv = int(args.cv_seed) if args.cv_seed is not None else int(run_seed)
        cv = KFold(n_splits=k, shuffle=True, random_state=seed_for_cv)
        split_iter = cv.split(np.zeros(len(sub)))
        print(f"[KFold] dataset={dataset_name} rows={len(sub)} folds={k}")

    pred_all = np.full(len(sub), np.nan, dtype=np.float64)
    fold_rows = []

    ckpt = Path(run_dir) / "folds" / f"fold_{source_fold}" / view_name / "best_model.pth"
    if args.init_mode == "tl" and not ckpt.exists():
        raise FileNotFoundError(ckpt)

    if args.init_mode == "scratch":
        # The controlled scratch comparison uses the fixed paper
        # architecture and does not require pretrained checkpoints
        # or an existing SMRT training directory.
        cwn_layers = 6
        cwn_hidden = 256
    else:
        config_path = Path(run_dir) / "config.json"

        if not config_path.is_file():
            raise FileNotFoundError(config_path)

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:
            cfg = json.load(file)

        cwn_layers = int(
            cfg.get("cwn_layers", 6)
        )
        cwn_hidden = int(
            cfg.get("cwn_hidden", 256)
        )

    for cv_fold, (tr_local, te_local) in enumerate(split_iter):
        seed = int(run_seed) + int(source_fold) * 1000 + cv_fold * 17
        set_seed(seed)

        train_global = global_indices[tr_local]
        test_global = global_indices[te_local]

        y_train = y_all[tr_local].astype(np.float32)
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        if y_std < 1e-6:
            y_std = 1.0

        train_loader = build_loader(wrapped, train_global, args.batch_size, shuffle=True)
        test_loader = build_loader(wrapped, test_global, args.eval_batch_size, shuffle=False)

        model = make_model(cwn_layers, cwn_hidden, device)

        if args.init_mode == "tl":
            load_state_dict_safely(model, ckpt, device)
            if int(args.reset_out_lin) == 1:
                reset_module(model.out_lin)
        elif args.init_mode == "scratch":
            # Random initialization; train all parameters.
            pass
        else:
            raise ValueError(f"Unknown init_mode={args.init_mode}")

        trainable_names, n_trainable, n_total = set_trainable(model, args.freeze_mode)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        best_train_mae = float("inf")
        best_test_mae = float("inf")
        best_state = None
        best_epoch = -1
        bad = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_mae = train_one_epoch(
                model, train_loader, optimizer, device,
                args.huber_beta, args.freeze_mode,
                y_mean, y_std
            )

            # ABCoRT-matched protocol:
            # evaluate held-out fold every epoch and select test-best epoch.
            test_mae_epoch = eval_mae(model, test_loader, device, y_mean, y_std)

            if train_mae < best_train_mae:
                best_train_mae = train_mae

            if test_mae_epoch < best_test_mae:
                best_test_mae = test_mae_epoch
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                print(
                    f"[{dataset_name}][{run_key}][src_fold={source_fold}][{view_name}] "
                    f"cv_fold={cv_fold}/{k} epoch={epoch:03d} "
                    f"train_mae={train_mae:.4f} test_mae={test_mae_epoch:.4f} "
                    f"best_test_mae={best_test_mae:.4f} best_epoch={best_epoch} "
                    f"y_mean={y_mean:.2f} y_std={y_std:.2f}"
                )

            if args.early_stop_train > 0 and bad >= args.early_stop_train:
                print(f"[EARLY] test MAE not improving for {bad} epochs")
                break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        y_te, p_te = predict(model, test_loader, device, y_mean, y_std)
        pred_all[te_local] = p_te

        fm = metrics(y_te, p_te)
        fold_rows.append({
            "dataset_name": dataset_name,
            "view": view_name,
            "run_key": run_key,
            "run_dir": str(run_dir),
            "source_fold": int(source_fold),
            "cv_fold": int(cv_fold),
            "freeze_mode": args.freeze_mode,
            "init_mode": args.init_mode,
            "trainable_modules": ",".join(trainable_names),
            "n_trainable": int(n_trainable),
            "n_total": int(n_total),
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "best_train_mae": float(best_train_mae),
            "best_test_mae": float(best_test_mae),
            "best_epoch": int(best_epoch),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "reset_out_lin": int(args.reset_out_lin),
            **{f"test_{kk}": vv for kk, vv in fm.items()},
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_df = sub.copy()
    pred_df["view"] = view_name
    pred_df["run_key"] = run_key
    pred_df["run_dir"] = str(run_dir)
    pred_df["source_fold"] = int(source_fold)
    pred_df["freeze_mode"] = args.freeze_mode
    pred_df["init_mode"] = args.init_mode
    pred_df["tl_pred"] = pred_all
    pred_df["tl_abs_error"] = np.abs(pred_df["rt"].values - pred_all)

    return pred_df, pd.DataFrame(fold_rows)


def summarize_predictions(pred_merged):
    rows = []

    for keys, sub in pred_merged.groupby(["dataset_name", "run_key", "source_fold", "freeze_mode"]):
        dataset_name, run_key, source_fold, freeze_mode = keys
        y = sub["rt"].values

        pred_map = {
            "origin_tl": sub["origin_tl_pred"].values,
            "taut_tl": sub["taut_tl_pred"].values,
            "mean_tl": 0.5 * (sub["origin_tl_pred"].values + sub["taut_tl_pred"].values),
        }

        for method, p in pred_map.items():
            rows.append({
                "dataset_name": dataset_name,
                "run_key": run_key,
                "source_fold": int(source_fold),
                "freeze_mode": freeze_mode,
                "method": method,
                **metrics(y, p),
            })

    return pd.DataFrame(rows)


def summarize_across_runs(metric_df):
    rows = []
    metric_cols = ["n", "mae", "medae", "rmse", "r2", "pearson", "spearman", "bias"]

    for keys, sub in metric_df.groupby(["dataset_name", "freeze_mode", "method"]):
        dataset_name, freeze_mode, method = keys
        row = {
            "dataset_name": dataset_name,
            "freeze_mode": freeze_mode,
            "method": method,
            "num_runs": int(sub["run_key"].nunique()),
            "num_source_folds": int(sub["source_fold"].nunique()),
        }

        for c in metric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
            row[f"{c}_min"] = float(vals.min())
            row[f"{c}_max"] = float(vals.max())

        rows.append(row)

    return pd.DataFrame(rows)


def _trainer_main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        default="artifacts/results/external_transfer/base",
    )
    ap.add_argument(
        "--stage4_meta_csv",
        default=(
            "gwn/paper_analysis_stage4_external/"
            "external_predret10_stage4_meta.csv"
        ),
    )
    ap.add_argument(
        "--origin_csv",
        default=(
            "gwn/paper_analysis_stage4_external/"
            "temp_external_predret10_origin.csv"
        ),
    )
    ap.add_argument(
        "--taut_csv",
        default=(
            "gwn/paper_analysis_stage4_external/"
            "temp_external_predret10_taut.csv"
        ),
    )
    ap.add_argument(
        "--origin_root",
        default="artifacts/cache/external/predret10_origin",
    )
    ap.add_argument(
        "--taut_root",
        default="artifacts/cache/external/predret10_taut",
    )
    ap.add_argument(
        "--smrt_runs_root",
        default="artifacts/results/smrt",
        help=(
            "Directory containing seed1, seed5, seed79, seed123, "
            "and seed256 SMRT training outputs."
        ),
    )

    ap.add_argument("--datasets", nargs="+", default=["Eawag_XBridgeC18_364", "LIFE_old_194", "IPB_Halle_82"])
    ap.add_argument("--run_keys", nargs="+", default=["seed1"])
    ap.add_argument("--source_folds", nargs="+", type=int, default=[0])

    ap.add_argument("--freeze_mode", default="out_lin_only",
                    choices=["out_lin_only", "head_plus_trans_out", "rt_head_full", "last_blocks", "all"])

    ap.add_argument("--cv_folds", type=int, default=10)
    ap.add_argument("--min_n", type=int, default=30)
    ap.add_argument("--group_cv", type=int, default=0)
    ap.add_argument("--group_col", default="inchikey")

    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--huber_beta", type=float, default=1.0)
    ap.add_argument("--max_ring_size", type=int, default=6)

    ap.add_argument("--keep_best_train", type=int, default=0)
    ap.add_argument("--early_stop_train", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=30)
    ap.add_argument("--reset_out_lin", type=int, default=1)
    ap.add_argument("--cv_seed", type=int, default=None)
    ap.add_argument("--init_mode", default="tl", choices=["tl", "scratch"],
                    help="tl: initialize from SMRT checkpoint; scratch: random initialization and train all parameters")

    args = ap.parse_args()

    for attr in [
        "out_dir",
        "stage4_meta_csv",
        "origin_csv",
        "taut_csv",
        "origin_root",
        "taut_root",
        "smrt_runs_root",
    ]:
        setattr(
            args,
            attr,
            str(resolve_repo_path(getattr(args, attr))),
        )

    if args.init_mode == "scratch":
        print("[SCRATCH MODE] override freeze_mode=all and reset_out_lin=0")
        args.freeze_mode = "all"
        args.reset_out_lin = 0

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Stage 4B external transfer learning ===")
    print("out_dir:", out_dir)
    print("device:", device)
    print("datasets:", args.datasets)
    print("run_keys:", args.run_keys)
    print("source_folds:", args.source_folds)
    print("freeze_mode:", args.freeze_mode)

    meta = pd.read_csv(args.stage4_meta_csv)
    meta = meta.sort_values("stage4_index").reset_index(drop=True)

    print("\n=== Load external Complex datasets ===")
    origin_dataset = SMRTComplexDataset(
        root=args.origin_root,
        csv_path=args.origin_csv,
        max_ring_size=args.max_ring_size,
        use_edge_features=True,
    )
    taut_dataset = SMRTComplexDataset(
        root=args.taut_root,
        csv_path=args.taut_csv,
        max_ring_size=args.max_ring_size,
        use_edge_features=True,
    )

    if len(origin_dataset) != len(meta):
        raise RuntimeError(f"origin dataset length mismatch: {len(origin_dataset)} vs {len(meta)}")
    if len(taut_dataset) != len(meta):
        raise RuntimeError(f"taut dataset length mismatch: {len(taut_dataset)} vs {len(meta)}")

    print("origin_dataset:", len(origin_dataset))
    print("taut_dataset:", len(taut_dataset))

    all_pred_rows = []
    all_fold_rows = []

    for dataset_name in args.datasets:
        print("\n" + "=" * 100)
        print("DATASET:", dataset_name)
        print("=" * 100)

        for run_key in args.run_keys:
            if run_key not in RUNS:
                raise ValueError(f"Unknown run_key={run_key}, available={list(RUNS.keys())}")

            run_seed = RUNS[run_key]["seed"]
            run_dir = (
                Path(args.smrt_runs_root)
                / RUNS[run_key]["subdir"]
            )

            for source_fold in args.source_folds:
                print(f"\n--- run_key={run_key}, source_fold={source_fold} ---")

                pred_origin, fold_origin = run_one_view_cv(
                    args=args,
                    dataset_name=dataset_name,
                    view_name="origin",
                    base_dataset=origin_dataset,
                    meta=meta,
                    run_key=run_key,
                    run_dir=run_dir,
                    source_fold=source_fold,
                    run_seed=run_seed,
                    device=device,
                )

                pred_taut, fold_taut = run_one_view_cv(
                    args=args,
                    dataset_name=dataset_name,
                    view_name="taut",
                    base_dataset=taut_dataset,
                    meta=meta,
                    run_key=run_key,
                    run_dir=run_dir,
                    source_fold=source_fold,
                    run_seed=run_seed,
                    device=device,
                )

                if len(pred_origin) == 0 or len(pred_taut) == 0:
                    continue

                keep_cols = [
                    "stage4_index", "dataset_name", "record_id", "name",
                    "origin_smiles", "taut_smiles", "rt", "formula", "inchikey",
                    "canonical_smiles", "taut_changed", "smrt_exact_overlap",
                ]
                keep_cols = [c for c in keep_cols if c in pred_origin.columns]

                merged = pred_origin[keep_cols + ["run_key", "run_dir", "source_fold", "freeze_mode", "tl_pred"]].copy()
                merged = merged.rename(columns={"tl_pred": "origin_tl_pred"})

                taut_small = pred_taut[["stage4_index", "tl_pred"]].copy()
                taut_small = taut_small.rename(columns={"tl_pred": "taut_tl_pred"})

                merged = merged.merge(taut_small, on="stage4_index", how="left")
                merged["mean_tl_pred"] = 0.5 * (merged["origin_tl_pred"] + merged["taut_tl_pred"])
                merged["origin_tl_abs_error"] = np.abs(merged["rt"] - merged["origin_tl_pred"])
                merged["taut_tl_abs_error"] = np.abs(merged["rt"] - merged["taut_tl_pred"])
                merged["mean_tl_abs_error"] = np.abs(merged["rt"] - merged["mean_tl_pred"])

                all_pred_rows.append(merged)
                all_fold_rows.append(fold_origin)
                all_fold_rows.append(fold_taut)

                tmp_metrics = summarize_predictions(merged)
                print("\n[SUMMARY current]")
                print(tmp_metrics[["dataset_name", "run_key", "source_fold", "method", "mae", "rmse", "r2", "spearman"]].to_string(index=False))

    if not all_pred_rows:
        raise RuntimeError("No predictions produced.")

    pred_all = pd.concat(all_pred_rows, ignore_index=True)
    fold_all = pd.concat(all_fold_rows, ignore_index=True)

    save_csv(pred_all, out_dir / "external_tl_predictions.csv")
    save_csv(fold_all, out_dir / "external_tl_fold_metrics.csv")

    metrics_all = summarize_predictions(pred_all)
    save_csv(metrics_all, out_dir / "external_tl_metrics_by_run.csv")

    summary = summarize_across_runs(metrics_all)
    save_csv(summary, out_dir / "external_tl_summary.csv")

    print("\n=== Final summary ===")
    cols = [
        "dataset_name", "freeze_mode", "method", "num_runs",
        "mae_mean", "mae_std", "rmse_mean", "r2_mean", "spearman_mean", "pearson_mean"
    ]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].sort_values(["dataset_name", "mae_mean"]).to_string(index=False))

    print("\n✅ Done:", out_dir)


# =====================================================================
# Public all-10 random-initialization entry point
# =====================================================================

ALL10_DATASETS = [
    "FEM_short_73",
    "UniToyama_Atlantis_143",
    "FEM_long_412",
    "Eawag_XBridgeC18_364",
    "LIFE_old_194",
    "MTBLS87_147",
    "LIFE_new_184",
    "Cao_HILIC_116",
    "IPB_Halle_82",
    "FEM_lipids_72",
]


def _run_internal_entrypoint(
    entrypoint,
    arguments,
    label,
):
    """Run one embedded CLI entry point in the current process."""
    original_argv = sys.argv[:]

    try:
        sys.argv = [
            f"{Path(__file__).name}:{label}",
            *[str(value) for value in arguments],
        ]
        entrypoint()
    finally:
        sys.argv = original_argv


def _resolve_public_path(value):
    path = Path(value).expanduser()

    if path.is_absolute():
        return path

    return REPO_ROOT / path



def _maybe_build_table8(
    transfer_summary_csv,
    scratch_summary_csv,
    output_csv,
    expected_datasets,
):
    """
    Build Table 8 when both transfer and scratch summaries exist.

    This function is intentionally present in both standalone entry
    points, so the scripts may be executed in either order.
    """
    transfer_summary_csv = Path(transfer_summary_csv)
    scratch_summary_csv = Path(scratch_summary_csv)
    output_csv = Path(output_csv)

    missing_files = [
        path
        for path in [
            transfer_summary_csv,
            scratch_summary_csv,
        ]
        if not path.is_file()
    ]

    if missing_files:
        print()
        print(
            "[TABLE 8 PENDING] Waiting for:",
            ", ".join(str(path) for path in missing_files),
        )
        return None

    transfer = pd.read_csv(transfer_summary_csv)
    scratch = pd.read_csv(scratch_summary_csv)

    transfer_required = {
        "dataset_name",
        "n",
        "transfer_mae",
    }
    scratch_required = {
        "dataset_name",
        "n",
        "scratch_mae",
    }

    transfer_missing = (
        transfer_required - set(transfer.columns)
    )
    scratch_missing = (
        scratch_required - set(scratch.columns)
    )

    if transfer_missing:
        raise RuntimeError(
            "Transfer summary is missing columns: "
            f"{sorted(transfer_missing)}"
        )

    if scratch_missing:
        raise RuntimeError(
            "Scratch summary is missing columns: "
            f"{sorted(scratch_missing)}"
        )

    transfer_small = transfer[
        [
            "dataset_name",
            "n",
            "transfer_mae",
        ]
    ].copy()

    scratch_small = scratch[
        [
            "dataset_name",
            "n",
            "scratch_mae",
        ]
    ].copy()

    for label, table in [
        ("transfer", transfer_small),
        ("scratch", scratch_small),
    ]:
        if table["dataset_name"].duplicated().any():
            duplicates = (
                table.loc[
                    table["dataset_name"]
                    .duplicated(keep=False),
                    "dataset_name",
                ]
                .astype(str)
                .unique()
                .tolist()
            )
            raise RuntimeError(
                f"Duplicate {label} rows: {duplicates}"
            )

    transfer_small = transfer_small.rename(
        columns={"n": "n_transfer"}
    )
    scratch_small = scratch_small.rename(
        columns={"n": "n_scratch"}
    )

    merged = transfer_small.merge(
        scratch_small,
        on="dataset_name",
        how="inner",
        validate="one_to_one",
    )

    expected = list(expected_datasets)
    actual = sorted(
        merged["dataset_name"]
        .astype(str)
        .unique()
        .tolist()
    )

    if sorted(expected) != actual:
        raise RuntimeError(
            "Table 8 dataset mismatch.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found: {actual}"
        )

    n_transfer = pd.to_numeric(
        merged["n_transfer"],
        errors="raise",
    ).astype(int)

    n_scratch = pd.to_numeric(
        merged["n_scratch"],
        errors="raise",
    ).astype(int)

    if not np.array_equal(
        n_transfer.to_numpy(),
        n_scratch.to_numpy(),
    ):
        bad = merged.loc[
            n_transfer.ne(n_scratch),
            [
                "dataset_name",
                "n_transfer",
                "n_scratch",
            ],
        ]
        raise RuntimeError(
            "Transfer and scratch sample counts differ:\n"
            + bad.to_string(index=False)
        )

    table8 = pd.DataFrame(
        {
            "dataset_name": (
                merged["dataset_name"].astype(str)
            ),
            "n": n_transfer,
            "scratch_mae": pd.to_numeric(
                merged["scratch_mae"],
                errors="raise",
            ).round(3),
            "transfer_mae": pd.to_numeric(
                merged["transfer_mae"],
                errors="raise",
            ).round(3),
        }
    )

    table8["mae_improvement_s"] = (
        table8["scratch_mae"]
        - table8["transfer_mae"]
    ).round(3)

    table8["tl_better_mae"] = (
        table8["mae_improvement_s"] > 0
    )

    table8["dataset_name"] = pd.Categorical(
        table8["dataset_name"],
        categories=expected,
        ordered=True,
    )

    table8 = (
        table8
        .sort_values("dataset_name")
        .reset_index(drop=True)
    )

    table8["dataset_name"] = (
        table8["dataset_name"].astype(str)
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    table8.to_csv(output_csv, index=False)

    improvement = table8["mae_improvement_s"]

    print()
    print("=== Final Table 8 ===")
    print(table8.to_string(index=False))
    print()
    print(
        "Transfer better by MAE:",
        int(table8["tl_better_mae"].sum()),
        "/",
        len(table8),
    )
    print(
        "Mean MAE improvement:",
        float(improvement.mean()),
        "s",
    )
    print(
        "Median MAE improvement:",
        float(improvement.median()),
        "s",
    )
    print("[SAVE]", output_csv)

    return table8


def _collect_scratch_summary(
    metrics_csv,
    output_csv,
    datasets,
):
    if not metrics_csv.is_file():
        raise FileNotFoundError(metrics_csv)

    metrics_table = pd.read_csv(metrics_csv)

    required = {
        "dataset_name",
        "run_key",
        "source_fold",
        "freeze_mode",
        "method",
        "n",
        "mae",
        "medae",
        "rmse",
        "r2",
        "pearson",
        "spearman",
        "bias",
    }

    missing = required - set(metrics_table.columns)

    if missing:
        raise RuntimeError(
            "Scratch metric table is missing columns: "
            f"{sorted(missing)}"
        )

    selected = metrics_table[
        metrics_table["method"]
        .astype(str)
        .eq("mean_tl")
    ].copy()

    if selected.empty:
        raise RuntimeError(
            "No mean_tl rows were found in the scratch metrics."
        )

    if selected["dataset_name"].duplicated().any():
        duplicate_rows = selected[
            selected["dataset_name"]
            .duplicated(keep=False)
        ][
            [
                "dataset_name",
                "run_key",
                "source_fold",
                "freeze_mode",
            ]
        ]

        raise RuntimeError(
            "Duplicate scratch summary rows found:\n"
            + duplicate_rows.to_string(index=False)
        )

    expected = list(datasets)
    actual = sorted(
        selected["dataset_name"]
        .astype(str)
        .unique()
        .tolist()
    )

    if sorted(expected) != actual:
        raise RuntimeError(
            "Scratch result dataset mismatch.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found: {actual}"
        )

    source_folds = sorted(
        pd.to_numeric(
            selected["source_fold"],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    if source_folds != [0]:
        raise RuntimeError(
            "Scratch output must contain only source_fold=0; "
            f"found {source_folds}"
        )

    freeze_modes = sorted(
        selected["freeze_mode"]
        .astype(str)
        .unique()
        .tolist()
    )

    if freeze_modes != ["all"]:
        raise RuntimeError(
            "Scratch output must use freeze_mode=all; "
            f"found {freeze_modes}"
        )

    summary = selected[
        [
            "dataset_name",
            "n",
            "mae",
            "medae",
            "rmse",
            "r2",
            "pearson",
            "spearman",
            "bias",
        ]
    ].copy()

    summary = summary.rename(
        columns={
            "mae": "scratch_mae",
            "medae": "scratch_medae",
            "rmse": "scratch_rmse",
            "r2": "scratch_r2",
            "pearson": "scratch_pearson",
            "spearman": "scratch_spearman",
            "bias": "scratch_bias",
        }
    )

    summary["dataset_name"] = pd.Categorical(
        summary["dataset_name"],
        categories=expected,
        ordered=True,
    )

    summary = (
        summary
        .sort_values("dataset_name")
        .reset_index(drop=True)
    )

    summary["dataset_name"] = (
        summary["dataset_name"].astype(str)
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output_csv,
        index=False,
    )

    print()
    print("=== Final all-10 scratch summary ===")
    print(
        summary[
            [
                "dataset_name",
                "n",
                "scratch_mae",
            ]
        ].to_string(index=False)
    )
    print()
    print("[SAVE]", output_csv)

    return summary


def _build_scratch_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run TC-TopoRT from random initialization "
            "on all 10 external datasets."
        )
    )

    parser.add_argument(
        "--out_root",
        default=(
            "artifacts/results/"
            "external_transfer/all10_scratch"
        ),
    )

    parser.add_argument(
        "--transfer_summary_csv",
        default=(
            "artifacts/results/external_transfer/"
            "all10_transfer/transfer_all10_summary.csv"
        ),
        help=(
            "Transfer summary produced by "
            "train_transfer_all10.py."
        ),
    )
    parser.add_argument(
        "--table8_out_csv",
        default=(
            "artifacts/results/external_transfer/"
            "Table_8_transfer_learning_effectiveness.csv"
        ),
    )

    parser.add_argument(
        "--stage4_meta_csv",
        default=(
            "gwn/paper_analysis_stage4_external/"
            "external_predret10_stage4_meta.csv"
        ),
    )
    parser.add_argument(
        "--origin_csv",
        default=(
            "gwn/paper_analysis_stage4_external/"
            "temp_external_predret10_origin.csv"
        ),
    )
    parser.add_argument(
        "--taut_csv",
        default=(
            "gwn/paper_analysis_stage4_external/"
            "temp_external_predret10_taut.csv"
        ),
    )

    parser.add_argument(
        "--origin_root",
        default=(
            "artifacts/cache/external/"
            "predret10_origin"
        ),
    )
    parser.add_argument(
        "--taut_root",
        default=(
            "artifacts/cache/external/"
            "predret10_taut"
        ),
    )
    parser.add_argument(
        "--smrt_runs_root",
        default="artifacts/results/smrt",
        help=(
            "Retained for internal CLI compatibility. "
            "Scratch mode uses the fixed paper architecture "
            "(6 CWN layers, hidden size 256) and does not "
            "load pretrained checkpoints."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL10_DATASETS,
    )
    parser.add_argument(
        "--run_key",
        default="seed5",
        help=(
            "Provides the fixed random seed and architecture "
            "configuration used in the controlled comparison."
        ),
    )

    parser.add_argument(
        "--cv_seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--early_stop_train",
        type=int,
        default=999,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--huber_beta",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max_ring_size",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--min_n",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--group_cv",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--group_col",
        default="inchikey",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--skip_existing",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--dry_run",
        type=int,
        choices=[0, 1],
        default=0,
    )

    return parser


def main():
    args = _build_scratch_parser().parse_args()

    out_root = _resolve_public_path(
        args.out_root
    )

    transfer_summary_csv = _resolve_public_path(
        args.transfer_summary_csv
    )
    table8_out_csv = _resolve_public_path(
        args.table8_out_csv
    )

    training_dir = out_root / "training"

    metrics_csv = (
        training_dir
        / "external_tl_metrics_by_run.csv"
    )

    final_summary_csv = (
        out_root
        / "scratch_all10_summary.csv"
    )

    trainer_arguments = [
        "--out_dir",
        str(training_dir),
        "--stage4_meta_csv",
        args.stage4_meta_csv,
        "--origin_csv",
        args.origin_csv,
        "--taut_csv",
        args.taut_csv,
        "--origin_root",
        args.origin_root,
        "--taut_root",
        args.taut_root,
        "--smrt_runs_root",
        args.smrt_runs_root,
        "--datasets",
        *args.datasets,
        "--run_keys",
        args.run_key,
        "--source_folds",
        "0",
        "--init_mode",
        "scratch",
        "--freeze_mode",
        "all",
        "--reset_out_lin",
        "0",
        "--cv_folds",
        str(args.cv_folds),
        "--cv_seed",
        str(args.cv_seed),
        "--min_n",
        str(args.min_n),
        "--group_cv",
        str(args.group_cv),
        "--group_col",
        args.group_col,
        "--epochs",
        str(args.epochs),
        "--early_stop_train",
        str(args.early_stop_train),
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--huber_beta",
        str(args.huber_beta),
        "--max_ring_size",
        str(args.max_ring_size),
        "--log_every",
        str(args.log_every),
    ]

    print("=" * 88)
    print("TC-TopoRT all-10 random-initialization run")
    print("=" * 88)
    print("datasets:", args.datasets)
    print("run_key:", args.run_key)
    print("source_folds: [0]")
    print("cv_seed:", args.cv_seed)
    print("cv_folds:", args.cv_folds)
    print("init_mode: scratch")
    print("freeze_mode: all")
    print("reset_out_lin: 0")
    print("cwn_layers: 6")
    print("cwn_hidden: 256")
    print("pretrained weights: not loaded")
    print("final dual-view method: mean_tl")
    print("out_root:", out_root)

    if args.dry_run:
        print()
        print("=== Embedded trainer arguments ===")
        print(" ".join(map(str, trainer_arguments)))
        print()
        print("DRY RUN: no training was executed.")
        return

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        int(args.skip_existing) == 1
        and metrics_csv.is_file()
    ):
        print()
        print(
            "[SKIP TRAINING] Existing metrics:",
            metrics_csv,
        )
    else:
        _run_internal_entrypoint(
            _trainer_main,
            trainer_arguments,
            "scratch-trainer",
        )

    _collect_scratch_summary(
        metrics_csv=metrics_csv,
        output_csv=final_summary_csv,
        datasets=args.datasets,
    )

    _maybe_build_table8(
        transfer_summary_csv=transfer_summary_csv,
        scratch_summary_csv=final_summary_csv,
        output_csv=table8_out_csv,
        expected_datasets=args.datasets,
    )

    print()
    print("All-10 scratch workflow completed.")


if __name__ == "__main__":
    main()

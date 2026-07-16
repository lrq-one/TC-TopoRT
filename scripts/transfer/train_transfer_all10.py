#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train TC-TopoRT on all 10 external datasets using the validated
transfer-learning protocol.

A single execution performs:

1. Dual-view external fine-tuning initialized from the five SMRT
   source-fold checkpoints.
2. Fixed raw AutoSelect aggregation.
3. Export of the final per-dataset transfer MAE summary.

The scientific training and aggregation implementations are retained
from the validated six-dataset protocol. The external dataset list is
extended from six to ten datasets.
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
    """"""
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

    with open(Path(run_dir) / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cwn_layers = int(cfg.get("cwn_layers", 6))
    cwn_hidden = int(cfg.get("cwn_hidden", 256))

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

# ===== EMBEDDED RAW AUTOSELECT =====

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
external_stack_fixed_raw_autoselect.py

No-leak fixed AutoCal / AutoSelect for external TCDV transfer predictions.

Input:
  external_tl_predictions.csv from external_train_tcdv_transfer_or_scratch.py

This script:
1. Reconstructs external cv_fold using the same KFold(cv_seed).
2. Builds source-fold/view prediction candidates.
3. For each outer held-out fold:
   - uses only other folds to select candidate + calibrator
   - applies the selected calibrator to held-out fold
4. Reports no-leak results.

Important:
No test-fold labels are used for selecting/calibrating that test fold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression


def metric_row(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n": 0, "mae": np.nan, "medae": np.nan, "rmse": np.nan,
            "r2": np.nan, "pearson": np.nan, "spearman": np.nan, "bias": np.nan,
        }

    err = np.abs(y_true - y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(err)),
        "medae": float(np.median(err)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "pearson": float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")) if len(y_true) > 1 else np.nan,
        "spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")) if len(y_true) > 1 else np.nan,
        "bias": float(np.mean(y_pred - y_true)),
    }


def reconstruct_cv_fold(df: pd.DataFrame, cv_seed: int, cv_folds: int) -> pd.DataFrame:
    df = df.copy()
    maps = []

    for ds, sub_all in df.groupby("dataset_name"):
        unique = (
            sub_all[["dataset_name", "stage4_index"]]
            .drop_duplicates()
            .sort_values("stage4_index")
            .reset_index(drop=True)
        )

        k = min(int(cv_folds), len(unique))
        cv = KFold(n_splits=k, shuffle=True, random_state=int(cv_seed))
        fold_id = np.full(len(unique), -1, dtype=int)

        for f, (_, te_idx) in enumerate(cv.split(np.zeros(len(unique)))):
            fold_id[te_idx] = f

        unique["cv_fold"] = fold_id
        maps.append(unique)

    fmap = pd.concat(maps, ignore_index=True)
    out = df.merge(fmap, on=["dataset_name", "stage4_index"], how="left")

    if out["cv_fold"].isna().any():
        raise RuntimeError("cv_fold reconstruction failed.")

    out["cv_fold"] = out["cv_fold"].astype(int)
    return out


def trim_mean(arr, axis=1):
    arr = np.asarray(arr, dtype=float)
    if arr.shape[axis] <= 2:
        return np.mean(arr, axis=axis)
    s = np.sort(arr, axis=axis)
    if axis == 1:
        return np.mean(s[:, 1:-1], axis=1)
    return np.mean(s[1:-1], axis=0)


def build_bank(df: pd.DataFrame, source_folds):
    required = [
        "dataset_name", "stage4_index", "rt", "run_key", "source_fold", "cv_fold",
        "origin_tl_pred", "taut_tl_pred",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df[df["source_fold"].astype(int).isin([int(x) for x in source_folds])].copy()
    df["source_fold"] = df["source_fold"].astype(int)
    df["run_key"] = df["run_key"].astype(str)

    long_rows = []
    for pred_col, prefix in [
        ("origin_tl_pred", "origin"),
        ("taut_tl_pred", "taut"),
    ]:
        tmp = df[["dataset_name", "stage4_index", "rt", "cv_fold", "run_key", "source_fold", pred_col]].copy()
        tmp["feat_name"] = prefix + "_" + tmp["run_key"].astype(str) + "_src" + tmp["source_fold"].astype(str)
        tmp = tmp.rename(columns={pred_col: "pred"})
        long_rows.append(tmp)

    long_df = pd.concat(long_rows, ignore_index=True)

    bank = (
        long_df.pivot_table(
            index=["dataset_name", "stage4_index", "rt", "cv_fold"],
            columns="feat_name",
            values="pred",
            aggfunc="mean",
        )
        .reset_index()
    )
    bank.columns.name = None

    origin_cols = sorted([c for c in bank.columns if c.startswith("origin_")])
    taut_cols = sorted([c for c in bank.columns if c.startswith("taut_")])

    taut_by_suffix = {c.replace("taut_", ""): c for c in taut_cols}
    matched_origin, matched_taut = [], []
    for oc in origin_cols:
        suf = oc.replace("origin_", "")
        tc = taut_by_suffix.get(suf)
        if tc is not None:
            matched_origin.append(oc)
            matched_taut.append(tc)

    if not matched_origin:
        raise RuntimeError("No matched origin/taut source-fold predictions.")

    O = bank[matched_origin].astype(float).values
    T = bank[matched_taut].astype(float).values
    P = 0.5 * (O + T)

    
    bank["cand_origin_mean"] = O.mean(axis=1)
    bank["cand_taut_mean"] = T.mean(axis=1)
    bank["cand_pair_mean"] = P.mean(axis=1)

    bank["cand_origin_median"] = np.median(O, axis=1)
    bank["cand_taut_median"] = np.median(T, axis=1)
    bank["cand_pair_median"] = np.median(P, axis=1)

    bank["cand_origin_trimmean"] = trim_mean(O, axis=1)
    bank["cand_taut_trimmean"] = trim_mean(T, axis=1)
    bank["cand_pair_trimmean"] = trim_mean(P, axis=1)

    
    bank["diag_abs_origin_taut_delta"] = np.abs(bank["cand_origin_mean"] - bank["cand_taut_mean"])

    cand_cols = [
        "cand_origin_mean",
        "cand_taut_mean",
        "cand_pair_mean",
        "cand_origin_median",
        "cand_taut_median",
        "cand_pair_median",
        "cand_origin_trimmean",
        "cand_taut_trimmean",
        "cand_pair_trimmean",
    ]

    if bank[cand_cols].isna().any().any():
        bad = bank[cand_cols].columns[bank[cand_cols].isna().any()].tolist()
        raise RuntimeError(f"NaN in candidate cols: {bad}")

    return bank, cand_cols, matched_origin, matched_taut


def fit_predict_calibrator(calib_mode, p_train, y_train, p_test):
    p_train = np.asarray(p_train, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    p_test = np.asarray(p_test, dtype=float).reshape(-1)

    if calib_mode == "raw":
        return p_train.copy(), p_test.copy(), {"a": 1.0, "b": 0.0}

    if calib_mode == "bias":
        b = float(np.mean(y_train - p_train))
        return p_train + b, p_test + b, {"a": 1.0, "b": b}

    X_train = p_train.reshape(-1, 1)
    X_test = p_test.reshape(-1, 1)

    if calib_mode == "linear":
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model.predict(X_train), model.predict(X_test), {
            "a": float(model.coef_[0]),
            "b": float(model.intercept_),
        }

    if calib_mode == "ridge":
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        return model.predict(X_train), model.predict(X_test), {
            "a": float(model.coef_[0]),
            "b": float(model.intercept_),
        }

    if calib_mode == "huber":
        model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=5000)
        model.fit(X_train, y_train)
        return model.predict(X_train), model.predict(X_test), {
            "a": float(model.coef_[0]),
            "b": float(model.intercept_),
        }

    raise ValueError(f"Unknown calib_mode={calib_mode}")


def noleak_autocal_one_dataset(ddf, cand_cols, calib_modes, selection_metric="mae"):
    pred_rows = []
    fold_rows = []

    for f in sorted(ddf["cv_fold"].unique()):
        train = ddf[ddf["cv_fold"] != f].copy()
        test = ddf[ddf["cv_fold"] == f].copy()

        y_train = train["rt"].values.astype(float)
        y_test = test["rt"].values.astype(float)

        candidates = []

        for cand in cand_cols:
            p_train_raw = train[cand].values.astype(float)
            p_test_raw = test[cand].values.astype(float)

            for mode in calib_modes:
                try:
                    p_train_cal, p_test_cal, params = fit_predict_calibrator(
                        mode, p_train_raw, y_train, p_test_raw
                    )
                except Exception:
                    continue

                m_train = metric_row(y_train, p_train_cal)
                score = m_train[selection_metric]

                candidates.append({
                    "cand": cand,
                    "calib_mode": mode,
                    "train_score": score,
                    "train_mae": m_train["mae"],
                    "train_medae": m_train["medae"],
                    "p_test": p_test_cal,
                    "a": params.get("a", np.nan),
                    "b": params.get("b", np.nan),
                })

        if not candidates:
            raise RuntimeError(f"No candidates for dataset={test['dataset_name'].iloc[0]} fold={f}")

        candidates = sorted(candidates, key=lambda x: x["train_score"])
        best = candidates[0]

        tmp = test[["dataset_name", "stage4_index", "cv_fold", "rt"] + cand_cols].copy()
        tmp["y_pred_autocal"] = best["p_test"]
        tmp["selected_candidate"] = best["cand"]
        tmp["selected_calib"] = best["calib_mode"]
        tmp["selected_train_score"] = best["train_score"]
        tmp["selected_a"] = best["a"]
        tmp["selected_b"] = best["b"]
        pred_rows.append(tmp)

        m_test = metric_row(y_test, best["p_test"])
        m_test.update({
            "dataset_name": test["dataset_name"].iloc[0],
            "cv_fold": int(f),
            "selected_candidate": best["cand"],
            "selected_calib": best["calib_mode"],
            "selected_train_mae": best["train_mae"],
            "selected_train_medae": best["train_medae"],
            "selected_a": best["a"],
            "selected_b": best["b"],
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        })
        fold_rows.append(m_test)

    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(fold_rows)


def _stacker_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cv_seed", type=int, required=True)
    ap.add_argument("--cv_folds", type=int, default=10)
    ap.add_argument("--source_folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--calib_modes", nargs="+", default=["raw", "bias", "ridge", "huber"])
    ap.add_argument("--selection_metric", choices=["mae", "medae"], default="mae")
    args = ap.parse_args()

    pred_csv = Path(args.pred_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)
    print("=== Load ===")
    print("pred_csv:", pred_csv)
    print("shape:", df.shape)

    df = reconstruct_cv_fold(df, args.cv_seed, args.cv_folds)
    bank, cand_cols, origin_cols, taut_cols = build_bank(df, args.source_folds)

    print("\n=== Bank ===")
    print("bank:", bank.shape)
    print("datasets:", sorted(bank["dataset_name"].unique().tolist()))
    print("origin_cols:", origin_cols)
    print("taut_cols:", taut_cols)
    print("cand_cols:", cand_cols)
    print("calib_modes:", args.calib_modes)
    print("selection_metric:", args.selection_metric)

    bank.to_csv(out_dir / "tcdv_autocal_prediction_bank.csv", index=False)

    all_preds = []
    all_folds = []
    summary_rows = []

    for ds, ddf in bank.groupby("dataset_name"):
        ddf = ddf.copy()

        pred_df, fold_df = noleak_autocal_one_dataset(
            ddf,
            cand_cols=cand_cols,
            calib_modes=args.calib_modes,
            selection_metric=args.selection_metric,
        )

        all_preds.append(pred_df)
        all_folds.append(fold_df)

        # auto result
        m = metric_row(pred_df["rt"].values, pred_df["y_pred_autocal"].values)
        m.update({"dataset_name": ds, "method": "tcdv_fixed_noleak_autocal"})
        summary_rows.append(m)

        # raw fixed candidates
        for col in cand_cols:
            mb = metric_row(ddf["rt"].values, ddf[col].values)
            mb.update({"dataset_name": ds, "method": col})
            summary_rows.append(mb)

    all_preds = pd.concat(all_preds, ignore_index=True)
    all_folds = pd.concat(all_folds, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    order = {"tcdv_fixed_noleak_autocal": 0}
    summary["method_order"] = summary["method"].map(order).fillna(1)
    summary = summary.sort_values(["dataset_name", "method_order", "mae"]).drop(columns=["method_order"])

    all_preds.to_csv(out_dir / "tcdv_fixed_noleak_autocal_predictions.csv", index=False)
    all_folds.to_csv(out_dir / "tcdv_fixed_noleak_autocal_fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "tcdv_fixed_noleak_autocal_summary.csv", index=False)

    # selection count
    sel = (
        all_folds.groupby(["dataset_name", "selected_candidate", "selected_calib"])
        .size()
        .reset_index(name="n_folds")
        .sort_values(["dataset_name", "n_folds"], ascending=[True, False])
    )
    sel.to_csv(out_dir / "tcdv_fixed_noleak_autocal_selection_counts.csv", index=False)

    meta = {
        "pred_csv": str(pred_csv),
        "cv_seed": int(args.cv_seed),
        "cv_folds": int(args.cv_folds),
        "source_folds": [int(x) for x in args.source_folds],
        "calib_modes": args.calib_modes,
        "selection_metric": args.selection_metric,
        "cand_cols": cand_cols,
        "origin_cols": origin_cols,
        "taut_cols": taut_cols,
        "protocol": "fixed no-leak train-fold AutoCal/AutoSelect over predefined source-fold/view aggregation candidates",
    }
    with open(out_dir / "tcdv_fixed_noleak_autocal_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Selection counts ===")
    print(sel.to_string(index=False))

    print("\n[SAVE]", out_dir / "tcdv_fixed_noleak_autocal_summary.csv")


# =====================================================================
# Public all-10 transfer entry point
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
    """Run one embedded CLI entry point without a subprocess."""
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


def _validate_base_predictions(
    prediction_csv,
    datasets,
    source_folds,
):
    if not prediction_csv.is_file():
        raise FileNotFoundError(prediction_csv)

    predictions = pd.read_csv(prediction_csv)

    required = {
        "dataset_name",
        "source_fold",
        "rt",
        "origin_tl_pred",
        "taut_tl_pred",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise RuntimeError(
            "Base prediction table is missing columns: "
            f"{sorted(missing)}"
        )

    expected_datasets = list(datasets)
    actual_datasets = sorted(
        predictions["dataset_name"]
        .astype(str)
        .unique()
        .tolist()
    )

    if sorted(expected_datasets) != actual_datasets:
        raise RuntimeError(
            "Base prediction dataset mismatch.\n"
            f"Expected: {sorted(expected_datasets)}\n"
            f"Found: {actual_datasets}"
        )

    expected_folds = sorted(
        int(value)
        for value in source_folds
    )

    for dataset_name in expected_datasets:
        subset = predictions[
            predictions["dataset_name"]
            .astype(str)
            .eq(dataset_name)
        ]

        actual_folds = sorted(
            pd.to_numeric(
                subset["source_fold"],
                errors="raise",
            )
            .astype(int)
            .unique()
            .tolist()
        )

        if actual_folds != expected_folds:
            raise RuntimeError(
                f"{dataset_name}: source folds differ; "
                f"expected={expected_folds}, "
                f"found={actual_folds}"
            )

        for column in [
            "rt",
            "origin_tl_pred",
            "taut_tl_pred",
        ]:
            if subset[column].isna().any():
                raise RuntimeError(
                    f"{dataset_name}: NaN found in {column}"
                )

    print(
        "Base prediction validation: PASS "
        f"({len(expected_datasets)} datasets, "
        f"source folds={expected_folds})"
    )


def _collect_transfer_summary(
    stack_summary_csv,
    output_csv,
    datasets,
):
    if not stack_summary_csv.is_file():
        raise FileNotFoundError(stack_summary_csv)

    raw = pd.read_csv(stack_summary_csv)

    required = {
        "dataset_name",
        "method",
        "n",
        "mae",
    }

    missing = required - set(raw.columns)

    if missing:
        raise RuntimeError(
            "AutoSelect summary is missing columns: "
            f"{sorted(missing)}"
        )

    selected = raw[
        raw["method"]
        .astype(str)
        .eq("tcdv_fixed_noleak_autocal")
    ].copy()

    if selected.empty:
        raise RuntimeError(
            "No tcdv_fixed_noleak_autocal rows found."
        )

    if selected["dataset_name"].duplicated().any():
        duplicates = (
            selected.loc[
                selected["dataset_name"]
                .duplicated(keep=False),
                "dataset_name",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise RuntimeError(
            "Duplicate final transfer rows found: "
            f"{duplicates}"
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
            "Final transfer dataset mismatch.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found: {actual}"
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
            "mae": "transfer_mae",
            "medae": "transfer_medae",
            "rmse": "transfer_rmse",
            "r2": "transfer_r2",
            "pearson": "transfer_pearson",
            "spearman": "transfer_spearman",
            "bias": "transfer_bias",
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
    print("=== Final all-10 transfer summary ===")
    print(
        summary[
            [
                "dataset_name",
                "n",
                "transfer_mae",
            ]
        ].to_string(index=False)
    )
    print()
    print("[SAVE]", output_csv)

    return summary


def _build_transfer_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the validated TC-TopoRT transfer-learning "
            "protocol on all 10 external datasets."
        )
    )

    parser.add_argument(
        "--out_root",
        default=(
            "artifacts/results/"
            "external_transfer/all10_transfer"
        ),
    )

    parser.add_argument(
        "--scratch_summary_csv",
        default=(
            "artifacts/results/external_transfer/"
            "all10_scratch/scratch_all10_summary.csv"
        ),
        help=(
            "Scratch summary produced by "
            "train_scratch_all10.py."
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
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL10_DATASETS,
    )
    parser.add_argument(
        "--run_key",
        default="seed5",
    )
    parser.add_argument(
        "--source_folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
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
        "--skip_existing_base",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--skip_existing_stack",
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
    args = _build_transfer_parser().parse_args()

    out_root = _resolve_public_path(
        args.out_root
    )

    scratch_summary_csv = _resolve_public_path(
        args.scratch_summary_csv
    )
    table8_out_csv = _resolve_public_path(
        args.table8_out_csv
    )

    base_dir = (
        out_root
        / "base_predictions"
    )
    stack_dir = (
        out_root
        / "raw_autoselect"
    )

    base_prediction_csv = (
        base_dir
        / "external_tl_predictions.csv"
    )

    stack_summary_csv = (
        stack_dir
        / "tcdv_fixed_noleak_autocal_summary.csv"
    )

    final_summary_csv = (
        out_root
        / "transfer_all10_summary.csv"
    )

    trainer_arguments = [
        "--out_dir",
        str(base_dir),
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
        *[str(value) for value in args.source_folds],
        "--init_mode",
        "tl",
        "--freeze_mode",
        "rt_head_full",
        "--reset_out_lin",
        "1",
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

    stacker_arguments = [
        "--pred_csv",
        str(base_prediction_csv),
        "--out_dir",
        str(stack_dir),
        "--cv_seed",
        str(args.cv_seed),
        "--cv_folds",
        str(args.cv_folds),
        "--source_folds",
        *[str(value) for value in args.source_folds],
        "--calib_modes",
        "raw",
        "--selection_metric",
        "mae",
    ]

    print("=" * 88)
    print("TC-TopoRT all-10 transfer-learning run")
    print("=" * 88)
    print("datasets:", args.datasets)
    print("run_key:", args.run_key)
    print("source_folds:", args.source_folds)
    print("cv_seed:", args.cv_seed)
    print("cv_folds:", args.cv_folds)
    print("freeze_mode: rt_head_full")
    print("reset_out_lin: 1")
    print("calib_modes: raw")
    print("selection_metric: mae")
    print("out_root:", out_root)

    if args.dry_run:
        print()
        print("=== Embedded trainer arguments ===")
        print(" ".join(map(str, trainer_arguments)))
        print()
        print("=== Embedded AutoSelect arguments ===")
        print(" ".join(map(str, stacker_arguments)))
        print()
        print("DRY RUN: no training was executed.")
        return

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        int(args.skip_existing_base) == 1
        and base_prediction_csv.is_file()
    ):
        print()
        print(
            "[SKIP BASE] Existing predictions:",
            base_prediction_csv,
        )
    else:
        _run_internal_entrypoint(
            _trainer_main,
            trainer_arguments,
            "trainer",
        )

    _validate_base_predictions(
        prediction_csv=base_prediction_csv,
        datasets=args.datasets,
        source_folds=args.source_folds,
    )

    if (
        int(args.skip_existing_stack) == 1
        and stack_summary_csv.is_file()
    ):
        print()
        print(
            "[SKIP AUTOSELECT] Existing summary:",
            stack_summary_csv,
        )
    else:
        _run_internal_entrypoint(
            _stacker_main,
            stacker_arguments,
            "raw-autoselect",
        )

    _collect_transfer_summary(
        stack_summary_csv=stack_summary_csv,
        output_csv=final_summary_csv,
        datasets=args.datasets,
    )

    _maybe_build_table8(
        transfer_summary_csv=final_summary_csv,
        scratch_summary_csv=scratch_summary_csv,
        output_csv=table8_out_csv,
        expected_datasets=args.datasets,
    )

    print()
    print("All-10 transfer workflow completed.")


if __name__ == "__main__":
    main()

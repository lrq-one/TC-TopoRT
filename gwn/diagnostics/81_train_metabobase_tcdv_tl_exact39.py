#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import math
import shutil
import inspect
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mp.smrt_dataset import SMRTComplexDataset
from mp.complex import ComplexBatch
from net.topocellrt_cwn_replace import TopoCellRTCWNReplace
import train_oof_dualview_stack as t

try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception as e:
    Chem = None
    rdMolStandardize = None
    print("[WARNING] RDKit unavailable:", e)


IN_DIR = Path("experiments_candidate_filtering/metabobase_tl_exact39")
OUT_DIR = Path("experiments_candidate_filtering/metabobase_tl_exact39_training")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SMRTComplexDataset filters rt <= 300. Shift MetaboBase labels upward during dataset build/training,
# then subtract this shift for reporting/prediction evaluation.
TARGET_SHIFT = 300.0


def complex_collate(batch):
    sig = inspect.signature(ComplexBatch.from_complex_list)
    kwargs = {}
    if "follow_batch" in sig.parameters:
        kwargs["follow_batch"] = []
    if "max_dim" in sig.parameters:
        kwargs["max_dim"] = 2
    return ComplexBatch.from_complex_list(batch, **kwargs)


def canon_smiles(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return ""
    if Chem is None:
        return s
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def tautomer_strict_smiles(s):
    s = canon_smiles(s)
    if not s or Chem is None or rdMolStandardize is None:
        return s
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s
    try:
        te = rdMolStandardize.TautomerEnumerator()
        tmol = te.Canonicalize(mol)
        return Chem.MolToSmiles(tmol, isomericSmiles=True)
    except Exception:
        return s


def prepare_view_csvs():
    train_origin_src = IN_DIR / "metabobase_train_exact39_for_model.csv"
    test_origin_src = IN_DIR / "metabobase_test_exact39_for_model.csv"

    train_origin = IN_DIR / "metabobase_train_exact39_origin_shift300_for_model.csv"
    test_origin = IN_DIR / "metabobase_test_exact39_origin_shift300_for_model.csv"
    train_taut = IN_DIR / "metabobase_train_exact39_taut_shift300_for_model.csv"
    test_taut = IN_DIR / "metabobase_test_exact39_taut_shift300_for_model.csv"

    # origin shifted
    for src, dst in [(train_origin_src, train_origin), (test_origin_src, test_origin)]:
        df = pd.read_csv(src)
        df["smiles"] = df["smiles"].map(canon_smiles)
        df = df[df["smiles"].astype(str).str.len() > 0].copy()
        df["rt_real"] = df["rt"].astype(float)
        df["rt"] = df["rt_real"] + TARGET_SHIFT
        df.to_csv(dst, index=False)

    # tautomer shifted
    for src, dst in [(train_origin_src, train_taut), (test_origin_src, test_taut)]:
        df = pd.read_csv(src)
        df["smiles"] = df["smiles"].map(tautomer_strict_smiles)
        df = df[df["smiles"].astype(str).str.len() > 0].copy()
        df["rt_real"] = df["rt"].astype(float)
        df["rt"] = df["rt_real"] + TARGET_SHIFT
        df.to_csv(dst, index=False)

    return train_origin, test_origin, train_taut, test_taut


def make_dataset(root, csv_path, rebuild=False):
    root = Path(root)
    if rebuild and root.exists():
        print("[rebuild] removing", root)
        shutil.rmtree(root)

    ds = SMRTComplexDataset(
        root=str(root),
        csv_path=str(csv_path),
        max_ring_size=6,
        use_edge_features=True,
        n_jobs=4,
        init_method="sum",
        include_down_adj=True,
    )
    return ds


def make_model(device):
    model = TopoCellRTCWNReplace(
        emb_dim=256,
        cwn_layers=6,
        cwn_hidden=256,
        max_dim=2,
        drop_ratio=0.0,
    )
    return model.to(device)


def load_pretrained(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("loaded:", ckpt_path)
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))
    if len(missing) > 0:
        print("missing sample:", missing[:10])
    if len(unexpected) > 0:
        print("unexpected sample:", unexpected[:10])


def move_batch(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def get_y(batch):
    for name in ["y", "label", "labels", "target"]:
        if hasattr(batch, name):
            y = getattr(batch, name)
            return y.view(-1).float()
    raise RuntimeError("Cannot find target y/label in ComplexBatch")


def forward_pred(model, batch):
    out = model(batch)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.view(-1).float()


def mae_np(y, p):
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    return float(np.mean(np.abs(y - p)))


def rmse_np(y, p):
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y - p) ** 2)))


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        y = get_y(batch)
        pred = forward_pred(model, batch)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p


def set_batchnorm_eval(model):
    """
    MetaboBase TL has very small batches / sparse cell dimensions.
    BatchNorm in CWN can receive tensors like [1, 256] and crash in training mode.
    Keep BN layers in eval mode while still training other parameters.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_one_epoch(model, loader, optimizer, device, loss_fn, grad_clip=5.0):
    model.train()
    set_batchnorm_eval(model)
    losses = []
    for batch in loader:
        batch = move_batch(batch, device)
        y = get_y(batch)
        pred = forward_pred(model, batch)

        loss = loss_fn(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))


def freeze_batchnorm_params(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(False)


def make_optimizer(model, lr_base, lr_head, weight_decay):
    freeze_batchnorm_params(model)
    head_keywords = ["out_lin", "global_proj", "global_gate", "trans_add", "trans_out"]
    base_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in head_keywords):
            head_params.append(p)
        else:
            base_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": base_params, "lr": lr_base},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )


def split_train_val(n, seed=42, val_frac=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(20, int(round(n * val_frac)))
    n_val = min(n_val, n - 1)
    val_idx = idx[:n_val].tolist()
    tr_idx = idx[n_val:].tolist()
    return tr_idx, val_idx


def train_view(view, train_ds, test_ds, ckpt_path, args, device):
    out_dir = OUT_DIR / f"seed{args.seed}" / view
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_idx, va_idx = split_train_val(len(train_ds), seed=args.seed, val_frac=args.val_frac)

    tr_loader = DataLoader(
        Subset(train_ds, tr_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=complex_collate,
    )
    va_loader = DataLoader(
        Subset(train_ds, va_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=complex_collate,
    )
    full_train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=complex_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=complex_collate,
    )

    model = make_model(device)
    load_pretrained(model, ckpt_path, device)

    optimizer = make_optimizer(
        model,
        lr_base=args.lr_base,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=12,
        min_lr=1e-7,
    )
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    best_val = math.inf
    best_epoch = -1
    bad = 0
    history = []

    print("\n" + "=" * 100)
    print(f"[train view] {view}")
    print("train subset:", len(tr_idx), "val subset:", len(va_idx), "test:", len(test_ds))
    print("ckpt:", ckpt_path)
    print("=" * 100)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, tr_loader, optimizer, device, loss_fn, grad_clip=args.grad_clip)

        y_val, p_val = predict_loader(model, va_loader, device)
        val_mae = mae_np(y_val, p_val)
        val_rmse = rmse_np(y_val, p_val)

        y_test, p_test = predict_loader(model, test_loader, device)
        test_mae = mae_np(y_test, p_test)
        test_rmse = rmse_np(y_test, p_test)

        scheduler.step(val_mae)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "test_mae_monitor": test_mae,
            "test_rmse_monitor": test_rmse,
            "lr_base": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        history.append(row)

        if val_mae < best_val - args.min_delta:
            best_val = val_mae
            best_epoch = epoch
            bad = 0
            torch.save(model.state_dict(), out_dir / "best_model.pth")
        else:
            bad += 1

        if epoch == 1 or epoch % args.print_every == 0 or bad == 0:
            print(
                f"[{view}] epoch {epoch:03d} "
                f"loss={train_loss:.4f} "
                f"val_mae={val_mae:.3f} val_rmse={val_rmse:.3f} "
                f"test_mae_monitor={test_mae:.3f} "
                f"best_val={best_val:.3f}@{best_epoch} bad={bad}"
            )

        if bad >= args.patience:
            print(f"[early stop] {view} epoch={epoch}, best={best_val:.3f}@{best_epoch}")
            break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "history.csv", index=False)

    # reload best
    model.load_state_dict(torch.load(out_dir / "best_model.pth", map_location=device), strict=True)

    y_train_shift, p_train_shift = predict_loader(model, full_train_loader, device)
    y_test_shift, p_test_shift = predict_loader(model, test_loader, device)

    # convert shifted RT back to real MetaboBase seconds
    y_train = y_train_shift - TARGET_SHIFT
    p_train = p_train_shift - TARGET_SHIFT
    y_test = y_test_shift - TARGET_SHIFT
    p_test = p_test_shift - TARGET_SHIFT

    train_metrics = {
        "view": view,
        "split": "train187",
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "MAE": mae_np(y_train, p_train),
        "RMSE": rmse_np(y_train, p_train),
        "MedAE": float(np.median(np.abs(y_train - p_train))),
        "bias": float(np.mean(p_train - y_train)),
    }
    test_metrics = {
        "view": view,
        "split": "exact39_test",
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "MAE": mae_np(y_test, p_test),
        "RMSE": rmse_np(y_test, p_test),
        "MedAE": float(np.median(np.abs(y_test - p_test))),
        "bias": float(np.mean(p_test - y_test)),
    }

    pred_train = pd.DataFrame({
        "view": view,
        "split": "train187",
        "actual_rt": y_train,
        "pred_rt": p_train,
        "pred_rt_shifted": p_train_shift,
        "abs_err": np.abs(y_train - p_train),
    })
    pred_test = pd.DataFrame({
        "view": view,
        "split": "exact39_test",
        "actual_rt": y_test,
        "pred_rt": p_test,
        "pred_rt_shifted": p_test_shift,
        "abs_err": np.abs(y_test - p_test),
    })

    pred_train.to_csv(out_dir / "train_predictions.csv", index=False)
    pred_test.to_csv(out_dir / "test_predictions.csv", index=False)

    return model, train_metrics, test_metrics, pred_train, pred_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--patience", type=int, default=45)
    ap.add_argument("--print_every", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.2)

    ap.add_argument("--lr_base", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--huber_beta", type=float, default=50.0)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--min_delta", type=float, default=0.05)

    ap.add_argument("--base_run", default="results_OOF_DualView_Stack_v1")
    ap.add_argument("--base_fold", type=int, default=0)
    ap.add_argument("--rebuild_dataset", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print("device:", device)

    train_origin_csv, test_origin_csv, train_taut_csv, test_taut_csv = prepare_view_csvs()

    ds_train_origin = make_dataset(
        OUT_DIR / "datasets" / "origin_train",
        train_origin_csv,
        rebuild=args.rebuild_dataset,
    )
    ds_test_origin = make_dataset(
        OUT_DIR / "datasets" / "origin_test",
        test_origin_csv,
        rebuild=args.rebuild_dataset,
    )
    ds_train_taut = make_dataset(
        OUT_DIR / "datasets" / "taut_train",
        train_taut_csv,
        rebuild=args.rebuild_dataset,
    )
    ds_test_taut = make_dataset(
        OUT_DIR / "datasets" / "taut_test",
        test_taut_csv,
        rebuild=args.rebuild_dataset,
    )

    print("origin train/test:", len(ds_train_origin), len(ds_test_origin))
    print("taut train/test:", len(ds_train_taut), len(ds_test_taut))

    base = Path(args.base_run) / "folds" / f"fold_{args.base_fold}"
    origin_ckpt = base / "origin" / "best_model.pth"
    taut_ckpt = base / "taut" / "best_model.pth"

    _, m1_train, m1_test, p1_train, p1_test = train_view(
        "origin",
        ds_train_origin,
        ds_test_origin,
        origin_ckpt,
        args,
        device,
    )
    _, m2_train, m2_test, p2_train, p2_test = train_view(
        "taut",
        ds_train_taut,
        ds_test_taut,
        taut_ckpt,
        args,
        device,
    )

    # average origin/taut prediction by row order
    avg_train_actual = p1_train["actual_rt"].to_numpy(float)
    avg_train_pred = 0.5 * (p1_train["pred_rt"].to_numpy(float) + p2_train["pred_rt"].to_numpy(float))
    avg_test_actual = p1_test["actual_rt"].to_numpy(float)
    avg_test_pred = 0.5 * (p1_test["pred_rt"].to_numpy(float) + p2_test["pred_rt"].to_numpy(float))

    avg_train = pd.DataFrame({
        "split": "train187",
        "actual_rt": avg_train_actual,
        "origin_pred": p1_train["pred_rt"].to_numpy(float),
        "taut_pred": p2_train["pred_rt"].to_numpy(float),
        "avg_pred": avg_train_pred,
        "abs_err": np.abs(avg_train_actual - avg_train_pred),
    })
    avg_test = pd.DataFrame({
        "split": "exact39_test",
        "actual_rt": avg_test_actual,
        "origin_pred": p1_test["pred_rt"].to_numpy(float),
        "taut_pred": p2_test["pred_rt"].to_numpy(float),
        "avg_pred": avg_test_pred,
        "abs_err": np.abs(avg_test_actual - avg_test_pred),
    })

    avg_dir = OUT_DIR / f"seed{args.seed}" / "dualview_avg"
    avg_dir.mkdir(parents=True, exist_ok=True)
    avg_train.to_csv(avg_dir / "train_predictions_avg.csv", index=False)
    avg_test.to_csv(avg_dir / "test_predictions_avg.csv", index=False)

    metrics = pd.DataFrame([
        m1_train,
        m1_test,
        m2_train,
        m2_test,
        {
            "view": "dualview_avg",
            "split": "train187",
            "best_epoch": -1,
            "best_val_mae": np.nan,
            "MAE": mae_np(avg_train_actual, avg_train_pred),
            "RMSE": rmse_np(avg_train_actual, avg_train_pred),
            "MedAE": float(np.median(np.abs(avg_train_actual - avg_train_pred))),
            "bias": float(np.mean(avg_train_pred - avg_train_actual)),
        },
        {
            "view": "dualview_avg",
            "split": "exact39_test",
            "best_epoch": -1,
            "best_val_mae": np.nan,
            "MAE": mae_np(avg_test_actual, avg_test_pred),
            "RMSE": rmse_np(avg_test_actual, avg_test_pred),
            "MedAE": float(np.median(np.abs(avg_test_actual - avg_test_pred))),
            "bias": float(np.mean(avg_test_pred - avg_test_actual)),
        },
    ])

    metrics.to_csv(OUT_DIR / f"seed{args.seed}" / "tl_metrics.csv", index=False)

    with open(OUT_DIR / f"seed{args.seed}" / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "=" * 100)
    print("[FINAL TL METRICS]")
    print(metrics.to_string(index=False))
    print("=" * 100)
    print("saved dir:", OUT_DIR / f"seed{args.seed}")


if __name__ == "__main__":
    main()

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, HuberRegressor
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mp.smrt_dataset import SMRTComplexDataset
from mp.complex import ComplexBatch
from net.topocellrt_cwn_replace import TopoCellRTCWNReplace


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def complex_collate_fn(batch):
    return ComplexBatch.from_complex_list(batch)


def metrics(y, p, prefix=""):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = np.abs(y - p)

    out = {
        "mae": float(e.mean()),
        "medae": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "gt80": int((e > 80).sum()),
        "gt100": int((e > 100).sum()),
        "gt200": int((e > 200).sum()),
        "bias": float(np.mean(p - y)),
        "n": int(len(e)),
    }

    if prefix:
        return {f"{prefix}_{k}": v for k, v in out.items()}
    return out


def parse_flag_value(x):
    if isinstance(x, (bool, np.bool_)):
        return float(x)
    if pd.isna(x):
        return 0.0

    s = str(x).strip().lower()
    if s in ["1", "true", "yes", "y"]:
        return 1.0
    if s in ["0", "false", "no", "n", "none", "nan", ""]:
        return 0.0

    try:
        return float(s)
    except Exception:
        return 0.0


def load_valid_meta(csv_path):
    df = pd.read_csv(csv_path, engine="python")
    df.columns = [str(c).lower().strip() for c in df.columns]

    if "smile" in df.columns and "smiles" not in df.columns:
        df.rename(columns={"smile": "smiles"}, inplace=True)

    if "smiles" not in df.columns or "rt" not in df.columns:
        raise ValueError(f"{csv_path} must contain smiles/rt columns")

    df["rt"] = df["rt"].astype(float)
    df = df[df["rt"] > 300.0].copy()

    rows = []
    for source_idx, row in df.iterrows():
        smi = str(row["smiles"])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        item = {
            "Source_Index": int(source_idx),
            "SMILES": smi,
            "Actual_RT": float(row["rt"]),
        }

        if "orig_smile" in df.columns:
            item["Orig_SMILES"] = str(row["orig_smile"])
        else:
            item["Orig_SMILES"] = smi

        for c in ["raw_changed", "real_changed", "formula_same", "heavy_same", "fallback", "reason"]:
            if c in df.columns:
                item[c] = row[c]

        rows.append(item)

    meta = pd.DataFrame(rows)

    if "real_changed" in meta.columns:
        meta["Taut_Changed"] = meta["real_changed"].apply(parse_flag_value).astype(float)
    elif "raw_changed" in meta.columns:
        meta["Taut_Changed"] = meta["raw_changed"].apply(parse_flag_value).astype(float)
    else:
        meta["Taut_Changed"] = 0.0

    return meta.reset_index(drop=True)


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()
    ys, ps = [], []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)

        if isinstance(pred, tuple):
            pred = pred[0]

        y = batch.y.view(-1).float()
        ys.append(y.detach().cpu())
        ps.append(pred.view(-1).detach().cpu())

    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return y, p


def train_one_epoch(model, loader, optimizer, device, huber_beta):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    steps = 0

    for batch in tqdm(loader, leave=False):
        batch = batch.to(device)
        target = batch.y.view(-1).float()

        optimizer.zero_grad()
        pred = model(batch)
        if isinstance(pred, tuple):
            pred = pred[0]

        loss = F.smooth_l1_loss(pred.view(-1), target, beta=huber_beta)
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        mae = F.l1_loss(pred.view(-1), target).item()
        total_loss += float(loss.item())
        total_mae += float(mae)
        steps += 1

    return total_loss / max(steps, 1), total_mae / max(steps, 1)


def make_model(args, device):
    model = TopoCellRTCWNReplace(
        emb_dim=256,
        cwn_layers=args.cwn_layers,
        cwn_hidden=args.cwn_hidden,
        max_dim=2,
        drop_ratio=0.0,
    ).to(device)
    return model


def train_fold_view(
    args,
    view_name,
    fold,
    dataset_train_full,
    dataset_test,
    train_idx,
    val_idx,
    out_dir,
    device,
):
    fold_dir = Path(out_dir) / f"fold_{fold}" / view_name
    mkdir(fold_dir)

    val_pred_path = fold_dir / "val_pred.npy"
    val_y_path = fold_dir / "val_y.npy"
    test_pred_path = fold_dir / "test_pred.npy"
    best_ckpt_path = fold_dir / "best_model.pth"
    log_path = fold_dir / "train_log.jsonl"

    if args.resume and val_pred_path.exists() and test_pred_path.exists() and val_y_path.exists():
        print(f"[RESUME] fold={fold} view={view_name}, load cached predictions")
        return (
            np.load(val_y_path),
            np.load(val_pred_path),
            np.load(test_pred_path),
        )

    set_seed(args.seed + fold * 100 + (0 if view_name == "origin" else 13))

    train_set = Subset(dataset_train_full, list(map(int, train_idx)))
    val_set = Subset(dataset_train_full, list(map(int, val_idx)))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=complex_collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=0,
    )

    model = make_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(args.epochs, 1),
    )

    best_val_mae = float("inf")
    best_epoch = -1
    bad = 0

    with open(log_path, "w", encoding="utf-8") as fw:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_mae = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args.huber_beta,
            )

            val_y, val_pred = predict_model(model, val_loader, device)
            val_m = metrics(val_y, val_pred, "val")
            scheduler.step()

            row = {
                "fold": fold,
                "view": view_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                **val_m,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }

            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
            fw.flush()

            print(
                f"[fold {fold}][{view_name}] epoch={epoch:03d} "
                f"train_mae={train_mae:.3f} val_mae={val_m['val_mae']:.4f} "
                f"best={best_val_mae:.4f}"
            )

            if val_m["val_mae"] < best_val_mae:
                best_val_mae = val_m["val_mae"]
                best_epoch = epoch
                bad = 0
                torch.save(model.state_dict(), best_ckpt_path)
            else:
                bad += 1

            if bad >= args.patience:
                print(f"[fold {fold}][{view_name}] early stop at epoch={epoch}, best_epoch={best_epoch}")
                break

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    val_y, val_pred = predict_model(model, val_loader, device)
    _, test_pred = predict_model(model, test_loader, device)

    np.save(val_y_path, val_y)
    np.save(val_pred_path, val_pred)
    np.save(test_pred_path, test_pred)

    with open(fold_dir / "best_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "fold": fold,
                "view": view_name,
                "best_epoch": best_epoch,
                "best_val_mae": best_val_mae,
                "val_metrics": metrics(val_y, val_pred),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return val_y, val_pred, test_pred


def build_stack_features(origin_pred, taut_pred, changed):
    origin_pred = np.asarray(origin_pred, dtype=np.float64)
    taut_pred = np.asarray(taut_pred, dtype=np.float64)
    changed = np.asarray(changed, dtype=np.float64)

    diff = np.abs(origin_pred - taut_pred)
    mean_pred = 0.5 * (origin_pred + taut_pred)
    min_pred = np.minimum(origin_pred, taut_pred)
    max_pred = np.maximum(origin_pred, taut_pred)

    x = np.vstack(
        [
            origin_pred,
            taut_pred,
            diff,
            mean_pred,
            min_pred,
            max_pred,
            changed,
            diff * changed,
            origin_pred * changed / 1000.0,
            taut_pred * changed / 1000.0,
        ]
    ).T

    return x


def disagreement_fusion(origin_pred, taut_pred, alpha, tau, temperature):
    origin_pred = np.asarray(origin_pred, dtype=np.float64)
    taut_pred = np.asarray(taut_pred, dtype=np.float64)

    diff = np.abs(origin_pred - taut_pred)
    soft_use = 1.0 / (1.0 + np.exp(-((diff - tau) / temperature)))

    mixed = alpha * origin_pred + (1.0 - alpha) * taut_pred
    final = (1.0 - soft_use) * origin_pred + soft_use * mixed
    return final


def fit_oof_stackers(oof_df, test_df, args):
    y = oof_df["Actual_RT"].values.astype(np.float64)

    o_oof = oof_df["Origin_OOF_Pred"].values
    t_oof = oof_df["Taut_OOF_Pred"].values
    c_oof = oof_df["Taut_Changed"].values

    o_test = test_df["Origin_Test_Pred"].values
    t_test = test_df["Taut_Test_Pred"].values
    c_test = test_df["Taut_Changed"].values

    candidates = {}

    candidates["origin_only"] = {
        "oof_pred": o_oof,
        "test_pred": o_test,
    }
    candidates["taut_only"] = {
        "oof_pred": t_oof,
        "test_pred": t_test,
    }
    candidates["mean_origin_taut"] = {
        "oof_pred": 0.5 * (o_oof + t_oof),
        "test_pred": 0.5 * (o_test + t_test),
    }

    # OOF-selected fixed disagreement gate. No test tuning.
    best = None
    for tau in args.tau_grid:
        for alpha in args.alpha_grid:
            p = disagreement_fusion(o_oof, t_oof, alpha=alpha, tau=tau, temperature=args.stack_temperature)
            m = metrics(y, p)
            if best is None or m["mae"] < best["mae"]:
                best = {
                    "alpha": float(alpha),
                    "tau": float(tau),
                    "mae": float(m["mae"]),
                    "oof_pred": p,
                    "test_pred": disagreement_fusion(
                        o_test,
                        t_test,
                        alpha=alpha,
                        tau=tau,
                        temperature=args.stack_temperature,
                    ),
                }

    candidates["oof_selected_fixed_gate"] = {
        "oof_pred": best["oof_pred"],
        "test_pred": best["test_pred"],
        "params": {
            "alpha": best["alpha"],
            "tau": best["tau"],
            "temperature": args.stack_temperature,
        },
    }

    x_oof = build_stack_features(o_oof, t_oof, c_oof)
    x_test = build_stack_features(o_test, t_test, c_test)

    ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.array([1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0])),
    )
    ridge.fit(x_oof, y)
    candidates["ridge_stack"] = {
        "oof_pred": ridge.predict(x_oof),
        "test_pred": ridge.predict(x_test),
        "params": {"model": "StandardScaler+RidgeCV"},
    }

    huber = make_pipeline(
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=args.huber_alpha, max_iter=1000),
    )
    huber.fit(x_oof, y)
    candidates["huber_stack"] = {
        "oof_pred": huber.predict(x_oof),
        "test_pred": huber.predict(x_test),
        "params": {"model": "StandardScaler+HuberRegressor", "alpha": args.huber_alpha},
    }

    summary = {}
    for name, item in candidates.items():
        summary[name] = {
            "oof": metrics(y, item["oof_pred"]),
            "params": item.get("params", {}),
        }

    best_name = min(summary.keys(), key=lambda k: summary[k]["oof"]["mae"])

    return best_name, candidates, summary


def make_stratified_bins(y, n_bins=10):
    y = np.asarray(y, dtype=np.float64)
    q = min(n_bins, len(np.unique(y)))
    bins = pd.qcut(y, q=q, labels=False, duplicates="drop")
    return np.asarray(bins, dtype=int)


def check_pairing(origin_meta, taut_meta, name):
    if len(origin_meta) != len(taut_meta):
        raise RuntimeError(f"{name}: origin/taut meta length mismatch: {len(origin_meta)} vs {len(taut_meta)}")

    rt_diff = np.abs(origin_meta["Actual_RT"].values - taut_meta["Actual_RT"].values)
    max_diff = float(rt_diff.max())

    print(f"[{name}] pair check rows={len(origin_meta)} max_rt_diff={max_diff}")

    if max_diff > 1e-6:
        bad = np.where(rt_diff > 1e-6)[0][:10]
        raise RuntimeError(f"{name}: RT mismatch examples: {bad.tolist()}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--origin_train_csv", default="data/SMRT_train.csv")
    parser.add_argument("--origin_test_csv", default="data/SMRT_test.csv")
    parser.add_argument("--taut_train_csv", default="data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv")
    parser.add_argument("--taut_test_csv", default="data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv")

    parser.add_argument("--origin_train_root", default="smrt_cwn_oof_origin_train")
    parser.add_argument("--origin_test_root", default="smrt_cwn_oof_origin_test")
    parser.add_argument("--taut_train_root", default="smrt_cwn_oof_taut_train")
    parser.add_argument("--taut_test_root", default="smrt_cwn_oof_taut_test")

    parser.add_argument("--out_dir", default="results_OOF_DualView_Stack_v1")

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--huber_beta", type=float, default=1.0)

    parser.add_argument("--max_ring_size", type=int, default=6)
    parser.add_argument("--cwn_layers", type=int, default=6)
    parser.add_argument("--cwn_hidden", type=int, default=256)

    parser.add_argument("--stack_temperature", type=float, default=5.0)
    parser.add_argument("--huber_alpha", type=float, default=1e-4)
    parser.add_argument("--resume", type=int, default=1)

    args = parser.parse_args()

    args.alpha_grid = np.linspace(0.0, 1.0, 101)
    args.tau_grid = np.array([0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0])

    out_dir = Path(args.out_dir)
    mkdir(out_dir)
    mkdir(out_dir / "folds")

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        clean_args = vars(args).copy()
        clean_args["alpha_grid"] = clean_args["alpha_grid"].tolist()
        clean_args["tau_grid"] = clean_args["tau_grid"].tolist()
        json.dump(clean_args, f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== OOF DualView Stack Config ===")
    print(json.dumps(clean_args, ensure_ascii=False, indent=2))
    print("device:", device)

    set_seed(args.seed)

    print("\n=== Loading metadata ===")
    origin_train_meta = load_valid_meta(args.origin_train_csv)
    taut_train_meta = load_valid_meta(args.taut_train_csv)
    origin_test_meta = load_valid_meta(args.origin_test_csv)
    taut_test_meta = load_valid_meta(args.taut_test_csv)

    check_pairing(origin_train_meta, taut_train_meta, "TRAIN")
    check_pairing(origin_test_meta, taut_test_meta, "TEST")

    print("\n=== Loading datasets ===")
    origin_train_dataset = SMRTComplexDataset(
        args.origin_train_root,
        args.origin_train_csv,
        args.max_ring_size,
        use_edge_features=True,
    )
    origin_test_dataset = SMRTComplexDataset(
        args.origin_test_root,
        args.origin_test_csv,
        args.max_ring_size,
        use_edge_features=True,
    )
    taut_train_dataset = SMRTComplexDataset(
        args.taut_train_root,
        args.taut_train_csv,
        args.max_ring_size,
        use_edge_features=True,
    )
    taut_test_dataset = SMRTComplexDataset(
        args.taut_test_root,
        args.taut_test_csv,
        args.max_ring_size,
        use_edge_features=True,
    )

    if len(origin_train_dataset) != len(origin_train_meta):
        raise RuntimeError(f"origin train dataset/meta mismatch: {len(origin_train_dataset)} vs {len(origin_train_meta)}")
    if len(taut_train_dataset) != len(taut_train_meta):
        raise RuntimeError(f"taut train dataset/meta mismatch: {len(taut_train_dataset)} vs {len(taut_train_meta)}")
    if len(origin_test_dataset) != len(origin_test_meta):
        raise RuntimeError(f"origin test dataset/meta mismatch: {len(origin_test_dataset)} vs {len(origin_test_meta)}")
    if len(taut_test_dataset) != len(taut_test_meta):
        raise RuntimeError(f"taut test dataset/meta mismatch: {len(taut_test_dataset)} vs {len(taut_test_meta)}")

    y_train = origin_train_meta["Actual_RT"].values.astype(np.float64)
    y_test = origin_test_meta["Actual_RT"].values.astype(np.float64)

    n_train = len(y_train)
    n_test = len(y_test)

    oof_origin = np.full(n_train, np.nan, dtype=np.float64)
    oof_taut = np.full(n_train, np.nan, dtype=np.float64)

    test_origin_folds = []
    test_taut_folds = []

    fold_assign = np.full(n_train, -1, dtype=int)

    bins = make_stratified_bins(y_train, n_bins=10)
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    print("\n=== Start OOF training ===")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_train), bins)):
        print("\n" + "=" * 80)
        print(f"FOLD {fold + 1}/{args.k}")
        print("train/val:", len(train_idx), len(val_idx))
        print("=" * 80)

        fold_assign[val_idx] = fold

        val_y_origin, val_pred_origin, test_pred_origin = train_fold_view(
            args=args,
            view_name="origin",
            fold=fold,
            dataset_train_full=origin_train_dataset,
            dataset_test=origin_test_dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            out_dir=out_dir / "folds",
            device=device,
        )

        val_y_taut, val_pred_taut, test_pred_taut = train_fold_view(
            args=args,
            view_name="taut",
            fold=fold,
            dataset_train_full=taut_train_dataset,
            dataset_test=taut_test_dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            out_dir=out_dir / "folds",
            device=device,
        )

        # safety check
        # Dataset stores RT as torch.float32, while CSV metadata is float64.
        # A tolerance of 1e-2 second is enough to catch real mismatch
        # but avoids false alarms from float32 rounding.
        origin_y_diff = float(np.max(np.abs(val_y_origin.astype(np.float64) - y_train[val_idx])))
        taut_y_diff = float(np.max(np.abs(val_y_taut.astype(np.float64) - y_train[val_idx])))

        print(f"[fold {fold}] origin_y_diff={origin_y_diff:.8f}, taut_y_diff={taut_y_diff:.8f}")

        if origin_y_diff > 1e-2:
            raise RuntimeError(f"fold {fold}: origin val y mismatch, max_diff={origin_y_diff}")
        if taut_y_diff > 1e-2:
            raise RuntimeError(f"fold {fold}: taut val y mismatch, max_diff={taut_y_diff}")

        oof_origin[val_idx] = val_pred_origin
        oof_taut[val_idx] = val_pred_taut

        test_origin_folds.append(test_pred_origin)
        test_taut_folds.append(test_pred_taut)

        fold_summary = {
            "fold": fold,
            "origin_val": metrics(y_train[val_idx], val_pred_origin),
            "taut_val": metrics(y_train[val_idx], val_pred_taut),
        }
        print("[fold summary]")
        print(json.dumps(fold_summary, ensure_ascii=False, indent=2))

    if np.isnan(oof_origin).any() or np.isnan(oof_taut).any():
        raise RuntimeError("OOF predictions contain NaN. Some fold did not fill predictions.")

    test_origin_mean = np.mean(np.vstack(test_origin_folds), axis=0)
    test_taut_mean = np.mean(np.vstack(test_taut_folds), axis=0)

    print("\n=== Build OOF / TEST prediction tables ===")

    oof_df = origin_train_meta.copy()
    oof_df["Fold"] = fold_assign
    oof_df["Origin_OOF_Pred"] = oof_origin
    oof_df["Taut_OOF_Pred"] = oof_taut
    oof_df["Origin_OOF_Abs_Error"] = np.abs(oof_df["Actual_RT"].values - oof_origin)
    oof_df["Taut_OOF_Abs_Error"] = np.abs(oof_df["Actual_RT"].values - oof_taut)
    oof_df["Taut_SMILES"] = taut_train_meta["SMILES"].values
    oof_df["Taut_Changed"] = taut_train_meta["Taut_Changed"].values

    test_df = origin_test_meta.copy()
    test_df["Origin_Test_Pred"] = test_origin_mean
    test_df["Taut_Test_Pred"] = test_taut_mean
    test_df["Origin_Test_Abs_Error"] = np.abs(test_df["Actual_RT"].values - test_origin_mean)
    test_df["Taut_Test_Abs_Error"] = np.abs(test_df["Actual_RT"].values - test_taut_mean)
    test_df["Taut_SMILES"] = taut_test_meta["SMILES"].values
    test_df["Taut_Changed"] = taut_test_meta["Taut_Changed"].values

    oof_df.to_csv(out_dir / "oof_base_predictions.csv", index=False)
    test_df.to_csv(out_dir / "test_base_predictions.csv", index=False)

    print("\n=== Fit OOF stacker ===")
    best_name, candidates, stack_summary = fit_oof_stackers(oof_df, test_df, args)

    print("best stacker selected by OOF MAE:", best_name)
    print(json.dumps(stack_summary, ensure_ascii=False, indent=2))

    final_oof_pred = candidates[best_name]["oof_pred"]
    final_test_pred = candidates[best_name]["test_pred"]

    oof_df["Final_OOF_Pred"] = final_oof_pred
    oof_df["Final_OOF_Abs_Error"] = np.abs(oof_df["Actual_RT"].values - final_oof_pred)

    test_df["Final_Pred"] = final_test_pred
    test_df["Final_Abs_Error"] = np.abs(test_df["Actual_RT"].values - final_test_pred)
    test_df["Gain_vs_Origin"] = test_df["Origin_Test_Abs_Error"] - test_df["Final_Abs_Error"]
    test_df["Gain_vs_Taut"] = test_df["Taut_Test_Abs_Error"] - test_df["Final_Abs_Error"]

    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)

    final_metrics = {
        "selected_stacker": best_name,
        "stacker_summary_oof": stack_summary,
        "oof_final": metrics(y_train, final_oof_pred),
        "oof_origin": metrics(y_train, oof_origin),
        "oof_taut": metrics(y_train, oof_taut),
        "test_final": metrics(y_test, final_test_pred),
        "test_origin_5fold_mean": metrics(y_test, test_origin_mean),
        "test_taut_5fold_mean": metrics(y_test, test_taut_mean),
        "paths": {
            "oof_predictions": str(out_dir / "oof_predictions.csv"),
            "test_predictions": str(out_dir / "test_predictions.csv"),
            "base_oof": str(out_dir / "oof_base_predictions.csv"),
            "base_test": str(out_dir / "test_base_predictions.csv"),
        },
    }

    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("FINAL OOF DUAL-VIEW STACK RESULT")
    print("=" * 80)
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    print("=" * 80)
    print("✅ Done:", out_dir)


if __name__ == "__main__":
    main()

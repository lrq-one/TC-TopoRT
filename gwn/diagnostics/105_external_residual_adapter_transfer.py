from pathlib import Path
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mp.smrt_dataset import SMRTComplexDataset
from mp.complex import ComplexBatch
from net.topocellrt_cwn_replace import TopoCellRTCWNReplace


RUNS = {
    "seed1":   {"dir": "results_OOF_DualView_Stack_v1",      "seed": 1},
    "seed79":  {"dir": "results_OOF_DualView_Stack_seed79",  "seed": 79},
    "seed123": {"dir": "results_OOF_DualView_Stack_seed123", "seed": 123},
    "seed256": {"dir": "results_OOF_DualView_Stack_seed256", "seed": 256},
    "seed5":   {"dir": "results_OOF_DualView_Stack_seed5",   "seed": 5},
}


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_csv(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path} shape={df.shape}")


def complex_collate_fn(batch):
    return ComplexBatch.from_complex_list(batch)


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


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
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


@torch.no_grad()
def extract_base_outputs(dataset, global_indices, run_dir, source_fold, view_name, batch_size, device):
    run_dir = Path(run_dir)
    ckpt = run_dir / "folds" / f"fold_{source_fold}" / view_name / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    cwn_layers = int(cfg.get("cwn_layers", 6))
    cwn_hidden = int(cfg.get("cwn_hidden", 256))

    model = make_model(cwn_layers, cwn_hidden, device)
    load_state_dict_safely(model, ckpt, device)
    model.eval()

    subset = Subset(dataset, list(map(int, global_indices)))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=0,
    )

    preds = []
    embs = []

    for batch in loader:
        batch = batch.to(device)
        out, aux = model(batch, include_partial=True)
        preds.append(out.view(-1).detach().cpu().numpy())
        embs.append(aux["mol_emb"].detach().cpu().numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(preds, axis=0), np.concatenate(embs, axis=0)


def make_view_features(emb, base_pred):
    base_pred = np.asarray(base_pred, dtype=np.float64).reshape(-1, 1)
    return np.concatenate([emb, base_pred], axis=1)


def make_dual_features(origin_emb, taut_emb, base_o, base_t):
    mean_emb = 0.5 * (origin_emb + taut_emb)
    absdiff = np.abs(origin_emb - taut_emb)

    base_o = np.asarray(base_o, dtype=np.float64).reshape(-1, 1)
    base_t = np.asarray(base_t, dtype=np.float64).reshape(-1, 1)
    base_mean = 0.5 * (base_o + base_t)
    base_diff = base_o - base_t
    base_absdiff = np.abs(base_diff)

    return np.concatenate(
        [mean_emb, absdiff, base_o, base_t, base_mean, base_diff, base_absdiff],
        axis=1
    ), base_mean.reshape(-1)


def residual_target(y, base, residual_mode):
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)

    if residual_mode == "raw":
        return y - base

    if residual_mode == "logratio":
        y_log = np.log1p(np.clip(y, a_min=0.0, a_max=None))
        b_log = np.log1p(np.clip(base, a_min=0.0, a_max=None))
        return y_log - b_log

    raise ValueError(residual_mode)


def combine_prediction(base, resid, residual_mode):
    base = np.asarray(base, dtype=np.float64)
    resid = np.asarray(resid, dtype=np.float64)

    if residual_mode == "raw":
        return base + resid

    if residual_mode == "logratio":
        b_log = np.log1p(np.clip(base, a_min=0.0, a_max=None))
        return np.expm1(b_log + resid)

    raise ValueError(residual_mode)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def train_residual_fold(
    X, y, base_pred, tr_idx, te_idx,
    residual_mode,
    seed,
    device,
    hidden_dim,
    dropout,
    lr,
    weight_decay,
    epochs,
    patience,
    batch_size,
    huber_beta,
):
    tr_idx = np.asarray(tr_idx, dtype=int)
    te_idx = np.asarray(te_idx, dtype=int)

    tr2, va = train_test_split(
        tr_idx,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
    )

    scaler = StandardScaler()
    scaler.fit(X[tr2])

    X_tr = scaler.transform(X[tr2]).astype(np.float32)
    X_va = scaler.transform(X[va]).astype(np.float32)
    X_te = scaler.transform(X[te_idx]).astype(np.float32)

    r_all = residual_target(y, base_pred, residual_mode)
    r_mean = float(np.mean(r_all[tr2]))
    r_std = float(np.std(r_all[tr2]))
    if r_std < 1e-8:
        r_std = 1.0

    z_tr = ((r_all[tr2] - r_mean) / r_std).astype(np.float32)
    z_va = ((r_all[va] - r_mean) / r_std).astype(np.float32)

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(z_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = ResidualMLP(
        input_dim=X_tr.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_mae = float("inf")
    best_epoch = -1
    bad = 0

    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, zb in train_loader:
            xb = xb.to(device)
            zb = zb.to(device)

            optimizer.zero_grad()
            pred_z = model(xb)
            loss = F.smooth_l1_loss(pred_z, zb, beta=huber_beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            va_z = model(X_va_t).detach().cpu().numpy()
            va_resid = va_z * r_std + r_mean
            va_pred = combine_prediction(base_pred[va], va_resid, residual_mode)
            val_mae = float(np.mean(np.abs(y[va] - va_pred)))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model.eval()
    with torch.no_grad():
        X_tr_all = scaler.transform(X[tr_idx]).astype(np.float32)
        pred_tr_z = model(torch.tensor(X_tr_all, dtype=torch.float32, device=device)).detach().cpu().numpy()
        pred_te_z = model(torch.tensor(X_te, dtype=torch.float32, device=device)).detach().cpu().numpy()

    pred_tr_resid = pred_tr_z * r_std + r_mean
    pred_te_resid = pred_te_z * r_std + r_mean

    pred_tr = combine_prediction(base_pred[tr_idx], pred_tr_resid, residual_mode)
    pred_te = combine_prediction(base_pred[te_idx], pred_te_resid, residual_mode)

    row = {
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val_mae),
        "train_mae": float(np.mean(np.abs(y[tr_idx] - pred_tr))),
        "test_mae": float(np.mean(np.abs(y[te_idx] - pred_te))),
        "r_mean": r_mean,
        "r_std": r_std,
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred_te, row


def cv_residual_predict(X, y, base_pred, residual_mode, args):
    y = np.asarray(y, dtype=np.float64)
    base_pred = np.asarray(base_pred, dtype=np.float64)

    pred_all = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows = []

    kf = KFold(n_splits=min(args.cv_folds, len(y)), shuffle=True, random_state=args.cv_seed)

    for fold, (tr, te) in enumerate(kf.split(np.zeros(len(y)))):
        pred_te, row = train_residual_fold(
            X=X,
            y=y,
            base_pred=base_pred,
            tr_idx=tr,
            te_idx=te,
            residual_mode=residual_mode,
            seed=args.cv_seed + fold * 17,
            device=args.device,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.head_batch_size,
            huber_beta=args.huber_beta,
        )

        pred_all[te] = pred_te
        row.update({
            "cv_fold": int(fold),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        })
        fold_rows.append(row)

        print(
            f"[fold={fold}] best_epoch={row['best_epoch']} "
            f"val_mae={row['best_val_mae']:.4f} "
            f"train_mae={row['train_mae']:.4f} "
            f"test_mae={row['test_mae']:.4f}"
        )

    return pred_all, pd.DataFrame(fold_rows)


def summarize_predictions(pred_df):
    rows = []
    for keys, sub in pred_df.groupby(["dataset_name", "run_key", "source_fold", "freeze_mode"]):
        dataset_name, run_key, source_fold, freeze_mode = keys
        y = sub["rt"].values.astype(float)

        pred_map = {
            "origin_tl": sub["origin_tl_pred"].values.astype(float),
            "taut_tl": sub["taut_tl_pred"].values.astype(float),
            "mean_tl": sub["mean_tl_pred"].values.astype(float),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stage4_meta_csv", default="paper_analysis_stage4_external/external_predret10_stage4_meta.csv")
    ap.add_argument("--origin_csv", default="paper_analysis_stage4_external/temp_external_predret10_origin.csv")
    ap.add_argument("--taut_csv", default="paper_analysis_stage4_external/temp_external_predret10_taut.csv")
    ap.add_argument("--origin_root", default="paper_analysis_stage4_external/cache/predret10_origin")
    ap.add_argument("--taut_root", default="paper_analysis_stage4_external/cache/predret10_taut")

    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--run_keys", nargs="+", default=["seed1"])
    ap.add_argument("--source_folds", nargs="+", type=int, default=[0])

    ap.add_argument("--residual_mode", choices=["raw", "logratio"], default="raw")
    ap.add_argument("--cv_folds", type=int, default=10)
    ap.add_argument("--cv_seed", type=int, default=1)

    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--huber_beta", type=float, default=1.0)
    ap.add_argument("--head_batch_size", type=int, default=64)

    ap.add_argument("--extract_batch_size", type=int, default=64)
    ap.add_argument("--max_ring_size", type=int, default=6)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    set_seed(args.cv_seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Stage 4AL residual-adapter transfer ===")
    print("out_dir:", out_dir)
    print("device:", args.device)
    print("datasets:", args.datasets)
    print("run_keys:", args.run_keys)
    print("source_folds:", args.source_folds)
    print("residual_mode:", args.residual_mode)

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

    all_pred_rows = []
    all_fold_rows = []

    for dataset_name in args.datasets:
        sub = meta[meta["dataset_name"].eq(dataset_name)].copy().reset_index(drop=True)
        if len(sub) == 0:
            print("[SKIP]", dataset_name)
            continue

        global_indices = sub["stage4_index"].values.astype(int)
        y = sub["rt"].values.astype(float)

        print("\n" + "=" * 100)
        print("DATASET:", dataset_name, "n=", len(sub))
        print("=" * 100)

        for run_key in args.run_keys:
            if run_key not in RUNS:
                raise ValueError(f"Unknown run_key={run_key}")

            run_dir = RUNS[run_key]["dir"]

            for source_fold in args.source_folds:
                print(f"\n--- run_key={run_key}, source_fold={source_fold} ---")

                print("[extract] origin base_pred + mol_emb")
                base_o, emb_o = extract_base_outputs(
                    dataset=origin_dataset,
                    global_indices=global_indices,
                    run_dir=run_dir,
                    source_fold=source_fold,
                    view_name="origin",
                    batch_size=args.extract_batch_size,
                    device=args.device,
                )

                print("[extract] taut base_pred + mol_emb")
                base_t, emb_t = extract_base_outputs(
                    dataset=taut_dataset,
                    global_indices=global_indices,
                    run_dir=run_dir,
                    source_fold=source_fold,
                    view_name="taut",
                    batch_size=args.extract_batch_size,
                    device=args.device,
                )

                base_mean = 0.5 * (base_o + base_t)

                print("[BASE frozen metrics]")
                print("origin:", metrics(y, base_o))
                print("taut  :", metrics(y, base_t))
                print("mean  :", metrics(y, base_mean))

                X_o = make_view_features(emb_o, base_o)
                X_t = make_view_features(emb_t, base_t)
                X_m, base_m = make_dual_features(emb_o, emb_t, base_o, base_t)

                freeze_mode = f"frozen_tcdv_residual_adapter_{args.residual_mode}"

                print("[train residual] origin_tl")
                pred_o, fold_o = cv_residual_predict(X_o, y, base_o, args.residual_mode, args)
                fold_o["method"] = "origin_tl"

                print("[train residual] taut_tl")
                pred_t, fold_t = cv_residual_predict(X_t, y, base_t, args.residual_mode, args)
                fold_t["method"] = "taut_tl"

                print("[train residual] mean_tl dual")
                pred_m, fold_m = cv_residual_predict(X_m, y, base_m, args.residual_mode, args)
                fold_m["method"] = "mean_tl"

                keep_cols = [
                    "stage4_index", "dataset_name", "record_id", "name",
                    "origin_smiles", "taut_smiles", "rt", "formula", "inchikey",
                    "canonical_smiles", "taut_changed", "smrt_exact_overlap",
                ]
                keep_cols = [c for c in keep_cols if c in sub.columns]

                pred_df = sub[keep_cols].copy()
                pred_df["run_key"] = run_key
                pred_df["run_dir"] = str(run_dir)
                pred_df["source_fold"] = int(source_fold)
                pred_df["freeze_mode"] = freeze_mode
                pred_df["origin_base_pred"] = base_o
                pred_df["taut_base_pred"] = base_t
                pred_df["mean_base_pred"] = base_mean
                pred_df["origin_tl_pred"] = pred_o
                pred_df["taut_tl_pred"] = pred_t
                pred_df["mean_tl_pred"] = pred_m
                pred_df["origin_tl_abs_error"] = np.abs(pred_df["rt"] - pred_df["origin_tl_pred"])
                pred_df["taut_tl_abs_error"] = np.abs(pred_df["rt"] - pred_df["taut_tl_pred"])
                pred_df["mean_tl_abs_error"] = np.abs(pred_df["rt"] - pred_df["mean_tl_pred"])

                for fdf in [fold_o, fold_t, fold_m]:
                    fdf["dataset_name"] = dataset_name
                    fdf["run_key"] = run_key
                    fdf["run_dir"] = str(run_dir)
                    fdf["source_fold"] = int(source_fold)
                    fdf["freeze_mode"] = freeze_mode
                    fdf["residual_mode"] = args.residual_mode

                all_pred_rows.append(pred_df)
                all_fold_rows.extend([fold_o, fold_t, fold_m])

                tmp = summarize_predictions(pred_df)
                print("\n[SUMMARY current]")
                print(tmp[["dataset_name", "run_key", "source_fold", "method", "mae", "rmse", "r2", "spearman", "bias"]].to_string(index=False))

    if not all_pred_rows:
        raise RuntimeError("No predictions produced")

    pred_all = pd.concat(all_pred_rows, ignore_index=True)
    fold_all = pd.concat(all_fold_rows, ignore_index=True)

    save_csv(pred_all, out_dir / "external_tl_predictions.csv")
    save_csv(fold_all, out_dir / "external_tl_fold_metrics.csv")

    metrics_all = summarize_predictions(pred_all)
    save_csv(metrics_all, out_dir / "external_tl_metrics_by_run.csv")

    print("\n=== Final metrics ===")
    print(metrics_all.sort_values(["dataset_name", "mae"])[
        ["dataset_name", "run_key", "source_fold", "freeze_mode", "method", "mae", "rmse", "r2", "spearman", "bias"]
    ].to_string(index=False))

    print("\n✅ Done:", out_dir)


if __name__ == "__main__":
    main()

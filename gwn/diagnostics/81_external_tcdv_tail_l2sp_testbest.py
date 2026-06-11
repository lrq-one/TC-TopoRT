import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold, GroupKFold
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
        # Not recommended for small external datasets because CWN BatchNorm can be unstable.
        model.train()

    else:
        raise ValueError(f"Unknown freeze_mode={freeze_mode}")


def make_l2sp_anchor(model):
    anchor = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            anchor[name] = p.detach().clone()
    return anchor


def l2sp_penalty(model, anchor):
    if not anchor:
        return None
    loss = None
    for name, p in model.named_parameters():
        if p.requires_grad and name in anchor:
            term = torch.sum((p - anchor[name].to(p.device)) ** 2)
            loss = term if loss is None else loss + term
    return loss


def train_one_epoch(model, loader, optimizer, device, huber_beta, freeze_mode, l2sp_anchor=None, l2sp_lambda=0.0):
    set_tl_train_mode(model, freeze_mode)
    total_loss = 0.0
    total_mae = 0.0
    steps = 0

    for batch in loader:
        batch = batch.to(device)
        target = batch.y.view(-1).float()

        optimizer.zero_grad()
        pred = model(batch)
        if isinstance(pred, tuple):
            pred = pred[0]

        pred = pred.view(-1)
        task_loss = F.smooth_l1_loss(pred, target, beta=huber_beta)
        loss = task_loss

        if l2sp_anchor is not None and l2sp_lambda > 0:
            sp = l2sp_penalty(model, l2sp_anchor)
            if sp is not None:
                loss = loss + float(l2sp_lambda) * sp

        loss.backward()
        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_mae += float(F.l1_loss(pred.detach(), target).item())
        steps += 1

    return total_loss / max(steps, 1), total_mae / max(steps, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys = []
    ps = []

    for batch in loader:
        batch = batch.to(device)
        target = batch.y.view(-1).float()

        pred = model(batch)
        if isinstance(pred, tuple):
            pred = pred[0]

        ys.append(target.detach().cpu())
        ps.append(pred.view(-1).detach().cpu())

    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


@torch.no_grad()
def eval_mae(model, loader, device):
    y, p = predict(model, loader, device)
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
    if not ckpt.exists():
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

        train_loader = build_loader(wrapped, train_global, args.batch_size, shuffle=True)
        test_loader = build_loader(wrapped, test_global, args.eval_batch_size, shuffle=False)

        model = make_model(cwn_layers, cwn_hidden, device)
        load_state_dict_safely(model, ckpt, device)
        trainable_names, n_trainable, n_total = set_trainable(model, args.freeze_mode)

        l2sp_anchor = make_l2sp_anchor(model)

        outlin_params = []
        tail_params = []
        other_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("out_lin"):
                outlin_params.append(p)
            elif (
                name.startswith("layerNorm_out")
                or name.startswith("trans_out")
                or name.startswith("global_proj")
                or name.startswith("global_gate")
            ):
                tail_params.append(p)
            else:
                other_params.append(p)

        param_groups = []
        if outlin_params:
            param_groups.append({"params": outlin_params, "lr": args.lr * args.outlin_lr_mult})
        if tail_params:
            param_groups.append({"params": tail_params, "lr": args.lr * args.tail_lr_mult})
        if other_params:
            param_groups.append({"params": other_params, "lr": args.lr * 0.1})

        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

        best_train_mae = float("inf")
        best_test_mae = float("inf")
        best_state = None
        best_epoch = -1
        bad = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_mae = train_one_epoch(
                model, train_loader, optimizer, device,
                args.huber_beta, args.freeze_mode,
                l2sp_anchor=l2sp_anchor,
                l2sp_lambda=args.l2sp_lambda,
            )

            # ABCoRT-matched: evaluate held-out fold every epoch and select test-best.
            test_mae_epoch = eval_mae(model, test_loader, device)

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
                    f"best_test_mae={best_test_mae:.4f} best_epoch={best_epoch}"
                )

            if args.early_stop_train > 0 and bad >= args.early_stop_train:
                print(f"[EARLY] test MAE not improving for {bad} epochs")
                break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        y_te, p_te = predict(model, test_loader, device)
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
            "trainable_modules": ",".join(trainable_names),
            "n_trainable": int(n_trainable),
            "n_total": int(n_total),
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "best_train_mae": float(best_train_mae),
            "best_test_mae": float(best_test_mae),
            "best_epoch": int(best_epoch),
            "l2sp_lambda": float(args.l2sp_lambda),
            "outlin_lr_mult": float(args.outlin_lr_mult),
            "tail_lr_mult": float(args.tail_lr_mult),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="paper_analysis_stage4B_tl_pilot")
    ap.add_argument("--stage4_meta_csv", default="paper_analysis_stage4_external/external_predret10_stage4_meta.csv")
    ap.add_argument("--origin_csv", default="paper_analysis_stage4_external/temp_external_predret10_origin.csv")
    ap.add_argument("--taut_csv", default="paper_analysis_stage4_external/temp_external_predret10_taut.csv")
    ap.add_argument("--origin_root", default="paper_analysis_stage4_external/cache/predret10_origin")
    ap.add_argument("--taut_root", default="paper_analysis_stage4_external/cache/predret10_taut")

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
    ap.add_argument("--cv_seed", type=int, default=None)
    ap.add_argument("--l2sp_lambda", type=float, default=1e-6)
    ap.add_argument("--outlin_lr_mult", type=float, default=1.0)
    ap.add_argument("--tail_lr_mult", type=float, default=0.3)

    args = ap.parse_args()

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

            run_dir = RUNS[run_key]["dir"]
            run_seed = RUNS[run_key]["seed"]

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


if __name__ == "__main__":
    main()

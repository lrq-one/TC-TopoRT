import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold

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


def reset_module(module):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def complex_pair_collate_fn(batch):
    origin_list, taut_list = zip(*batch)
    return ComplexBatch.from_complex_list(list(origin_list)), ComplexBatch.from_complex_list(list(taut_list))


class ExternalPairedDataset(Dataset):
    def __init__(self, origin_dataset, taut_dataset, targets):
        assert len(origin_dataset) == len(taut_dataset)
        self.origin_dataset = origin_dataset
        self.taut_dataset = taut_dataset
        self.targets = np.asarray(targets, dtype=np.float32)

    def __len__(self):
        return len(self.origin_dataset)

    def __getitem__(self, idx):
        idx = int(idx)
        o = self.origin_dataset[idx]
        t = self.taut_dataset[idx]
        y = torch.tensor([float(self.targets[idx])], dtype=torch.float32)
        o.y = y.clone()
        t.y = y.clone()
        return o, t


class DualViewFusionHead(nn.Module):
    def __init__(self, emb_dim=512, hidden=512, dropout=0.10):
        super().__init__()
        # origin emb, taut emb, abs diff emb, mean emb, origin pred, taut pred, mean pred, diff pred
        in_dim = emb_dim * 4 + 4
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, emb_o, emb_t, pred_o, pred_t):
        pred_o = pred_o.view(-1, 1)
        pred_t = pred_t.view(-1, 1)
        mean_pred = 0.5 * (pred_o + pred_t)
        diff_pred = torch.abs(pred_o - pred_t)

        feat = torch.cat([
            emb_o,
            emb_t,
            torch.abs(emb_o - emb_t),
            0.5 * (emb_o + emb_t),
            pred_o,
            pred_t,
            mean_pred,
            diff_pred,
        ], dim=-1)

        return self.net(feat).view(-1)


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


def make_model(cwn_layers, cwn_hidden, device):
    return TopoCellRTCWNReplace(
        emb_dim=256,
        cwn_layers=int(cwn_layers),
        cwn_hidden=int(cwn_hidden),
        max_dim=2,
        drop_ratio=0.0,
    ).to(device)


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


def set_trainable_rt_head(model):
    for p in model.parameters():
        p.requires_grad = False

    train_modules = [
        model.trans_graph,
        model.trans_add,
        model.layerNorm_out,
        model.trans_out,
        model.global_proj,
        model.global_gate,
        model.out_lin,
    ]

    for module in train_modules:
        for p in module.parameters():
            p.requires_grad = True


def set_dual_train_mode(model_o, model_t, fusion):
    # 保持 CWN encoder 的 BN / dropout 冻结
    model_o.eval()
    model_t.eval()

    for model in [model_o, model_t]:
        model.trans_graph.train()
        model.trans_add.train()
        model.layerNorm_out.train()
        model.trans_out.train()
        model.global_proj.train()
        model.global_gate.train()
        model.out_lin.train()

    fusion.train()


def build_loader(dataset, indices, batch_size, shuffle):
    return DataLoader(
        Subset(dataset, list(map(int, indices))),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=complex_pair_collate_fn,
        num_workers=0,
    )


def forward_dual(model_o, model_t, fusion, batch_o, batch_t):
    pred_o, part_o = model_o(batch_o, include_partial=True)
    pred_t, part_t = model_t(batch_t, include_partial=True)

    emb_o = part_o["mol_emb"]
    emb_t = part_t["mol_emb"]

    pred_o = pred_o.view(-1)
    pred_t = pred_t.view(-1)

    pred_fused = fusion(emb_o, emb_t, pred_o, pred_t)
    return pred_o, pred_t, pred_fused


def train_one_epoch(model_o, model_t, fusion, loader, optimizer, device, y_mean, y_std, huber_beta, aux_weight, cons_weight):
    set_dual_train_mode(model_o, model_t, fusion)

    total_loss = 0.0
    total_mae = 0.0
    steps = 0

    for batch_o, batch_t in loader:
        batch_o = batch_o.to(device)
        batch_t = batch_t.to(device)

        y_raw = batch_o.y.view(-1).float()
        y_z = (y_raw - y_mean) / y_std

        optimizer.zero_grad()

        pred_o, pred_t, pred_fused = forward_dual(model_o, model_t, fusion, batch_o, batch_t)

        loss_fused = F.smooth_l1_loss(pred_fused, y_z, beta=huber_beta)
        loss_o = F.smooth_l1_loss(pred_o, y_z, beta=huber_beta)
        loss_t = F.smooth_l1_loss(pred_t, y_z, beta=huber_beta)
        loss_cons = F.smooth_l1_loss(pred_o, pred_t.detach(), beta=huber_beta)

        loss = loss_fused + aux_weight * (loss_o + loss_t) + cons_weight * loss_cons

        loss.backward()

        params = (
            [p for p in model_o.parameters() if p.requires_grad] +
            [p for p in model_t.parameters() if p.requires_grad] +
            list(fusion.parameters())
        )
        clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        pred_raw = pred_fused.detach() * y_std + y_mean
        total_loss += float(loss.item())
        total_mae += float(F.l1_loss(pred_raw, y_raw).item())
        steps += 1

    return total_loss / max(steps, 1), total_mae / max(steps, 1)


@torch.no_grad()
def predict_dual(model_o, model_t, fusion, loader, device, y_mean, y_std):
    model_o.eval()
    model_t.eval()
    fusion.eval()

    ys = []
    po = []
    pt = []
    pf = []

    for batch_o, batch_t in loader:
        batch_o = batch_o.to(device)
        batch_t = batch_t.to(device)

        y_raw = batch_o.y.view(-1).float()

        pred_o, pred_t, pred_fused = forward_dual(model_o, model_t, fusion, batch_o, batch_t)

        ys.append(y_raw.detach().cpu().numpy())
        po.append((pred_o * y_std + y_mean).detach().cpu().numpy())
        pt.append((pred_t * y_std + y_mean).detach().cpu().numpy())
        pf.append((pred_fused * y_std + y_mean).detach().cpu().numpy())

    return (
        np.concatenate(ys),
        np.concatenate(po),
        np.concatenate(pt),
        np.concatenate(pf),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="paper_analysis_stage4E_dual_zfusion_lifeold")
    ap.add_argument("--stage4_meta_csv", default="paper_analysis_stage4_external/external_predret10_stage4_meta.csv")
    ap.add_argument("--origin_csv", default="paper_analysis_stage4_external/temp_external_predret10_origin.csv")
    ap.add_argument("--taut_csv", default="paper_analysis_stage4_external/temp_external_predret10_taut.csv")
    ap.add_argument("--origin_root", default="paper_analysis_stage4_external/cache/predret10_origin")
    ap.add_argument("--taut_root", default="paper_analysis_stage4_external/cache/predret10_taut")

    ap.add_argument("--dataset", default="LIFE_old_194")
    ap.add_argument("--run_key", default="seed1")
    ap.add_argument("--source_fold", type=int, default=0)
    ap.add_argument("--group_col", default="inchikey")
    ap.add_argument("--cv_folds", type=int, default=10)

    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--huber_beta", type=float, default=1.0)
    ap.add_argument("--aux_weight", type=float, default=0.2)
    ap.add_argument("--cons_weight", type=float, default=0.02)
    ap.add_argument("--reset_out_lin", type=int, default=1)
    ap.add_argument("--fusion_dropout", type=float, default=0.10)
    ap.add_argument("--max_ring_size", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=30)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== TCDV Dual-view ZFusion transfer learning ===")
    print("dataset:", args.dataset)
    print("run_key:", args.run_key)
    print("source_fold:", args.source_fold)
    print("device:", device)

    if args.run_key not in RUNS:
        raise ValueError(f"Unknown run_key={args.run_key}")

    run_dir = Path(RUNS[args.run_key]["dir"])

    meta = pd.read_csv(args.stage4_meta_csv).sort_values("stage4_index").reset_index(drop=True)
    sub = meta[meta["dataset_name"] == args.dataset].copy().reset_index(drop=True)

    global_indices = sub["stage4_index"].values.astype(int)
    y_all = sub["rt"].values.astype(np.float32)
    groups = sub[args.group_col].fillna(sub["stage4_index"].astype(str)).astype(str).values

    print("dataset rows:", len(sub), "unique groups:", len(np.unique(groups)))

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

    paired_dataset = ExternalPairedDataset(
        origin_dataset,
        taut_dataset,
        targets=meta.sort_values("stage4_index")["rt"].values,
    )

    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cwn_layers = int(cfg.get("cwn_layers", 6))
    cwn_hidden = int(cfg.get("cwn_hidden", 256))

    origin_ckpt = run_dir / "folds" / f"fold_{args.source_fold}" / "origin" / "best_model.pth"
    taut_ckpt = run_dir / "folds" / f"fold_{args.source_fold}" / "taut" / "best_model.pth"

    if not origin_ckpt.exists():
        raise FileNotFoundError(origin_ckpt)
    if not taut_ckpt.exists():
        raise FileNotFoundError(taut_ckpt)

    cv = GroupKFold(n_splits=min(args.cv_folds, len(np.unique(groups))))

    pred_origin_all = np.full(len(sub), np.nan, dtype=np.float64)
    pred_taut_all = np.full(len(sub), np.nan, dtype=np.float64)
    pred_fused_all = np.full(len(sub), np.nan, dtype=np.float64)

    fold_rows = []

    for fold, (tr_local, te_local) in enumerate(cv.split(np.zeros(len(sub)), y_all, groups)):
        seed = int(RUNS[args.run_key]["seed"]) + int(args.source_fold) * 1000 + fold * 17
        set_seed(seed)

        train_global = global_indices[tr_local]
        test_global = global_indices[te_local]

        y_train = y_all[tr_local]
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        if y_std < 1e-6:
            y_std = 1.0

        train_loader = build_loader(paired_dataset, train_global, args.batch_size, shuffle=True)
        test_loader = build_loader(paired_dataset, test_global, args.eval_batch_size, shuffle=False)

        model_o = make_model(cwn_layers, cwn_hidden, device)
        model_t = make_model(cwn_layers, cwn_hidden, device)
        load_state_dict_safely(model_o, origin_ckpt, device)
        load_state_dict_safely(model_t, taut_ckpt, device)

        if args.reset_out_lin:
            reset_module(model_o.out_lin)
            reset_module(model_t.out_lin)

        set_trainable_rt_head(model_o)
        set_trainable_rt_head(model_t)

        fusion = DualViewFusionHead(emb_dim=512, hidden=512, dropout=args.fusion_dropout).to(device)

        params = (
            [p for p in model_o.parameters() if p.requires_grad] +
            [p for p in model_t.parameters() if p.requires_grad] +
            list(fusion.parameters())
        )

        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            amsgrad=True,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(args.epochs, 1),
        )

        best_train_mae = float("inf")
        best_state = None
        best_epoch = -1

        for epoch in range(1, args.epochs + 1):
            loss, train_mae = train_one_epoch(
                model_o=model_o,
                model_t=model_t,
                fusion=fusion,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                y_mean=y_mean,
                y_std=y_std,
                huber_beta=args.huber_beta,
                aux_weight=args.aux_weight,
                cons_weight=args.cons_weight,
            )
            scheduler.step()

            if train_mae < best_train_mae:
                best_train_mae = train_mae
                best_epoch = epoch
                best_state = {
                    "model_o": {k: v.detach().cpu().clone() for k, v in model_o.state_dict().items()},
                    "model_t": {k: v.detach().cpu().clone() for k, v in model_t.state_dict().items()},
                    "fusion": {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()},
                }

            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                print(
                    f"[fold={fold}] epoch={epoch:03d} "
                    f"train_mae_fused={train_mae:.4f} best={best_train_mae:.4f} "
                    f"y_mean={y_mean:.2f} y_std={y_std:.2f}"
                )

        if best_state is not None:
            model_o.load_state_dict(best_state["model_o"], strict=True)
            model_t.load_state_dict(best_state["model_t"], strict=True)
            fusion.load_state_dict(best_state["fusion"], strict=True)

        y_te, p_o, p_t, p_f = predict_dual(
            model_o, model_t, fusion, test_loader, device, y_mean, y_std
        )

        pred_origin_all[te_local] = p_o
        pred_taut_all[te_local] = p_t
        pred_fused_all[te_local] = p_f

        fold_rows.append({
            "dataset_name": args.dataset,
            "fold": int(fold),
            "n_train": int(len(tr_local)),
            "n_test": int(len(te_local)),
            "y_mean": y_mean,
            "y_std": y_std,
            "best_epoch": int(best_epoch),
            "best_train_mae": float(best_train_mae),
            **{f"fused_test_{k}": v for k, v in metrics(y_te, p_f).items()},
            **{f"origin_test_{k}": v for k, v in metrics(y_te, p_o).items()},
            **{f"taut_test_{k}": v for k, v in metrics(y_te, p_t).items()},
        })

        del model_o, model_t, fusion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_df = sub.copy()
    pred_df["origin_pred"] = pred_origin_all
    pred_df["taut_pred"] = pred_taut_all
    pred_df["fused_pred"] = pred_fused_all
    pred_df["mean_pred"] = 0.5 * (pred_origin_all + pred_taut_all)

    rows = []
    for method, col in [
        ("origin_zaux", "origin_pred"),
        ("taut_zaux", "taut_pred"),
        ("mean_origin_taut", "mean_pred"),
        ("dualview_zfusion", "fused_pred"),
    ]:
        rows.append({
            "dataset_name": args.dataset,
            "run_key": args.run_key,
            "source_fold": int(args.source_fold),
            "method": method,
            **metrics(pred_df["rt"].values, pred_df[col].values),
        })

    metrics_df = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows)

    pred_df.to_csv(out_dir / "dual_zfusion_predictions.csv", index=False)
    fold_df.to_csv(out_dir / "dual_zfusion_fold_metrics.csv", index=False)
    metrics_df.to_csv(out_dir / "dual_zfusion_metrics.csv", index=False)

    print("\n=== FINAL METRICS ===")
    print(metrics_df.to_string(index=False))
    print("\n✅ Done:", out_dir)


if __name__ == "__main__":
    main()

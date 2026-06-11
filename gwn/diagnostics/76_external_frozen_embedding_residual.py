import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score

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


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def complex_collate_fn(batch):
    return ComplexBatch.from_complex_list(batch)


def build_loader(dataset, indices, batch_size):
    return DataLoader(
        Subset(dataset, list(map(int, indices))),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=0,
    )


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


@torch.no_grad()
def collect_pred_emb(model, dataset, global_indices, batch_size, device):
    model.eval()
    loader = build_loader(dataset, global_indices, batch_size=batch_size)

    preds = []
    embs = []

    for batch in loader:
        batch = batch.to(device)
        out, part = model(batch, include_partial=True)
        preds.append(out.view(-1).detach().cpu().numpy())
        embs.append(part["mol_emb"].detach().cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(embs, axis=0)


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "medae": float(median_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "pearson": float(pd.Series(y).corr(pd.Series(p), method="pearson")) if len(y) > 1 else np.nan,
        "spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")) if len(y) > 1 else np.nan,
        "bias": float(np.mean(p - y)),
    }


def fit_iso(xtr, ytr, xte):
    corr = np.corrcoef(xtr, ytr)[0, 1]
    inc = True if np.isfinite(corr) and corr >= 0 else False
    iso = IsotonicRegression(increasing=inc, out_of_bounds="clip")
    iso.fit(xtr, ytr)
    return iso.predict(xtr), iso.predict(xte), iso


def ridge_residual_predict(Xtr, rtr, Xte, alpha):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    model = Ridge(alpha=float(alpha))
    model.fit(Xtr_s, rtr)
    return model.predict(Xte_s), model


def make_low_features(base_o, base_t, emb_o=None, emb_t=None, mode="oemb"):
    mean_base = 0.5 * (base_o + base_t)
    diff_base = np.abs(base_o - base_t).reshape(-1, 1)
    low = np.column_stack([
        base_o,
        base_t,
        mean_base,
        np.abs(base_o - base_t),
    ])

    if mode == "low":
        return low

    if mode == "oemb":
        return np.concatenate([emb_o, low], axis=1)

    if mode == "temb":
        return np.concatenate([emb_t, low], axis=1)

    if mode == "pair_emb":
        return np.concatenate([emb_o, emb_t, np.abs(emb_o - emb_t), low], axis=1)

    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="paper_analysis_stage4G_lifeold_frozen_emb_residual")
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
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--max_ring_size", type=int, default=6)

    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0, 10.0, 100.0, 1000.0, 10000.0])

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Stage 4G frozen embedding residual transfer ===")
    print("out_dir:", out_dir)
    print("device:", device)
    print("dataset:", args.dataset)
    print("run_key:", args.run_key)
    print("source_fold:", args.source_fold)
    print("alphas:", args.alphas)

    if args.run_key not in RUNS:
        raise ValueError(f"Unknown run_key={args.run_key}")

    run_dir = Path(RUNS[args.run_key]["dir"])
    run_seed = int(RUNS[args.run_key]["seed"])
    set_seed(run_seed + int(args.source_fold) * 1000)

    meta = pd.read_csv(args.stage4_meta_csv).sort_values("stage4_index").reset_index(drop=True)
    sub = meta[meta["dataset_name"] == args.dataset].copy().reset_index(drop=True)

    global_indices = sub["stage4_index"].values.astype(int)
    y = sub["rt"].values.astype(float)
    groups = sub[args.group_col].fillna(sub["stage4_index"].astype(str)).astype(str).values

    print("rows:", len(sub), "unique groups:", len(np.unique(groups)))

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

    print("loading origin:", origin_ckpt)
    model_o = make_model(cwn_layers, cwn_hidden, device)
    load_state_dict_safely(model_o, origin_ckpt, device)

    print("collect origin pred/emb...")
    base_o, emb_o = collect_pred_emb(
        model_o, origin_dataset, global_indices, args.eval_batch_size, device
    )

    del model_o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("loading taut:", taut_ckpt)
    model_t = make_model(cwn_layers, cwn_hidden, device)
    load_state_dict_safely(model_t, taut_ckpt, device)

    print("collect taut pred/emb...")
    base_t, emb_t = collect_pred_emb(
        model_t, taut_dataset, global_indices, args.eval_batch_size, device
    )

    del model_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("base_o:", base_o.shape, "emb_o:", emb_o.shape)
    print("base_t:", base_t.shape, "emb_t:", emb_t.shape)

    cv = GroupKFold(n_splits=min(args.cv_folds, len(np.unique(groups))))

    methods = {}

    # raw / simple calibrated predictions
    for name in [
        "origin_raw", "taut_raw", "mean_raw",
        "origin_iso", "taut_iso", "mean_iso",
    ]:
        methods[name] = np.full(len(sub), np.nan)

    # residual methods for each alpha / feature mode
    feature_modes = ["low", "oemb", "temb", "pair_emb"]
    for alpha in args.alphas:
        for mode in feature_modes:
            methods[f"origin_iso_resid_{mode}_ridge_a{alpha:g}"] = np.full(len(sub), np.nan)

    fold_rows = []

    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(sub)), y, groups)):
        print(f"\n[fold {fold}] train={len(tr)} test={len(te)}")

        # raw
        methods["origin_raw"][te] = base_o[te]
        methods["taut_raw"][te] = base_t[te]
        methods["mean_raw"][te] = 0.5 * (base_o[te] + base_t[te])

        # isotonic calibration
        iso_o_tr, iso_o_te, _ = fit_iso(base_o[tr], y[tr], base_o[te])
        iso_t_tr, iso_t_te, _ = fit_iso(base_t[tr], y[tr], base_t[te])
        mean_tr_raw = 0.5 * (base_o[tr] + base_t[tr])
        mean_te_raw = 0.5 * (base_o[te] + base_t[te])
        iso_m_tr, iso_m_te, _ = fit_iso(mean_tr_raw, y[tr], mean_te_raw)

        methods["origin_iso"][te] = iso_o_te
        methods["taut_iso"][te] = iso_t_te
        methods["mean_iso"][te] = iso_m_te

        # residual target: y - origin isotonic
        rtr = y[tr] - iso_o_tr

        for alpha in args.alphas:
            for mode in feature_modes:
                Xtr = make_low_features(
                    base_o[tr], base_t[tr], emb_o=emb_o[tr], emb_t=emb_t[tr], mode=mode
                )
                Xte = make_low_features(
                    base_o[te], base_t[te], emb_o=emb_o[te], emb_t=emb_t[te], mode=mode
                )

                rpred_te, _ = ridge_residual_predict(Xtr, rtr, Xte, alpha=alpha)
                pred_te = iso_o_te + rpred_te
                methods[f"origin_iso_resid_{mode}_ridge_a{alpha:g}"][te] = pred_te

        fold_rows.append({
            "fold": int(fold),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "origin_iso_fold_mae": metrics(y[te], iso_o_te)["mae"],
            "taut_iso_fold_mae": metrics(y[te], iso_t_te)["mae"],
            "mean_iso_fold_mae": metrics(y[te], iso_m_te)["mae"],
        })

    rows = []
    for name, pred in methods.items():
        m = metrics(y, pred)
        rows.append({"method": name, **m})

    metric_df = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows)

    pred_df = sub.copy()
    pred_df["base_origin_pred"] = base_o
    pred_df["base_taut_pred"] = base_t
    for name, pred in methods.items():
        pred_df[name] = pred
        pred_df[f"{name}_abs_error"] = np.abs(y - pred)

    metric_path = out_dir / "frozen_emb_residual_metrics.csv"
    pred_path = out_dir / "frozen_emb_residual_predictions.csv"
    fold_path = out_dir / "frozen_emb_residual_fold_metrics.csv"

    metric_df.to_csv(metric_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    fold_df.to_csv(fold_path, index=False)

    print("\n=== FINAL METRICS top 30 ===")
    cols = ["method", "n", "mae", "medae", "rmse", "r2", "pearson", "spearman", "bias"]
    print(metric_df[cols].head(30).to_string(index=False))

    print("\n[SAVE]", metric_path)
    print("[SAVE]", pred_path)
    print("[SAVE]", fold_path)
    print("\n✅ Done:", out_dir)


if __name__ == "__main__":
    main()

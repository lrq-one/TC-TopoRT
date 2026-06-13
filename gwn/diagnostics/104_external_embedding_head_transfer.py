from pathlib import Path
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

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
def extract_embeddings_for_view(dataset, global_indices, run_dir, source_fold, view_name, batch_size, device):
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

    embs = []
    for batch in loader:
        batch = batch.to(device)
        _, aux = model(batch, include_partial=True)
        z = aux["mol_emb"].detach().cpu().numpy()
        embs.append(z)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(embs, axis=0)


def make_dual_features(origin_emb, taut_emb, feature_mode):
    if feature_mode == "origin":
        return origin_emb
    if feature_mode == "taut":
        return taut_emb
    if feature_mode == "mean":
        return 0.5 * (origin_emb + taut_emb)
    if feature_mode == "concat":
        return np.concatenate([origin_emb, taut_emb], axis=1)
    if feature_mode == "mean_absdiff":
        mean = 0.5 * (origin_emb + taut_emb)
        absdiff = np.abs(origin_emb - taut_emb)
        return np.concatenate([mean, absdiff], axis=1)
    if feature_mode == "mean_diff_absdiff":
        mean = 0.5 * (origin_emb + taut_emb)
        diff = origin_emb - taut_emb
        absdiff = np.abs(diff)
        return np.concatenate([mean, diff, absdiff], axis=1)
    raise ValueError(f"Unknown feature_mode={feature_mode}")


def transform_y(y, target_mode):
    y = np.asarray(y, dtype=np.float64)
    if target_mode == "raw":
        return y
    if target_mode == "log1p":
        return np.log1p(np.clip(y, a_min=0.0, a_max=None))
    raise ValueError(target_mode)


def inverse_y(z, target_mode):
    z = np.asarray(z, dtype=np.float64)
    if target_mode == "raw":
        return z
    if target_mode == "log1p":
        return np.expm1(z)
    raise ValueError(target_mode)


def choose_alpha_inner_cv(X, y_raw, train_idx, alphas, target_mode, inner_folds, seed):
    train_idx = np.asarray(train_idx, dtype=int)
    y_train_raw = y_raw[train_idx]
    y_train_t = transform_y(y_train_raw, target_mode)

    if len(train_idx) < inner_folds:
        inner_folds = max(2, min(len(train_idx), 3))

    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    alpha_rows = []

    for alpha in alphas:
        maes = []
        for inner_tr_local, inner_va_local in kf.split(np.zeros(len(train_idx))):
            tr = train_idx[inner_tr_local]
            va = train_idx[inner_va_local]

            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=float(alpha))
            )
            model.fit(X[tr], transform_y(y_raw[tr], target_mode))
            pred_va_t = model.predict(X[va])
            pred_va = inverse_y(pred_va_t, target_mode)
            maes.append(mean_absolute_error(y_raw[va], pred_va))

        alpha_rows.append({
            "alpha": float(alpha),
            "inner_mae": float(np.mean(maes)),
        })

    best = sorted(alpha_rows, key=lambda r: r["inner_mae"])[0]
    return best["alpha"], alpha_rows


def cv_predict_ridge_head(X, y_raw, cv_seed, alphas, target_mode, inner_folds):
    y_raw = np.asarray(y_raw, dtype=np.float64)
    pred = np.full(len(y_raw), np.nan, dtype=np.float64)
    fold_rows = []

    outer = KFold(n_splits=min(10, len(y_raw)), shuffle=True, random_state=cv_seed)

    for fold, (tr, te) in enumerate(outer.split(np.zeros(len(y_raw)))):
        alpha, alpha_rows = choose_alpha_inner_cv(
            X=X,
            y_raw=y_raw,
            train_idx=tr,
            alphas=alphas,
            target_mode=target_mode,
            inner_folds=inner_folds,
            seed=cv_seed + fold * 17,
        )

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(alpha))
        )
        model.fit(X[tr], transform_y(y_raw[tr], target_mode))
        pred_te_t = model.predict(X[te])
        pred_te = inverse_y(pred_te_t, target_mode)
        pred[te] = pred_te

        fold_rows.append({
            "cv_fold": int(fold),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "alpha": float(alpha),
            "train_mae": float(mean_absolute_error(y_raw[tr], inverse_y(model.predict(X[tr]), target_mode))),
            "test_mae": float(mean_absolute_error(y_raw[te], pred_te)),
            "target_mode": target_mode,
        })

        print(
            f"[fold={fold}] alpha={alpha:g} "
            f"train_mae={fold_rows[-1]['train_mae']:.4f} "
            f"test_mae={fold_rows[-1]['test_mae']:.4f}"
        )

    return pred, pd.DataFrame(fold_rows)


def summarize_predictions(pred_merged):
    rows = []
    for keys, sub in pred_merged.groupby(["dataset_name", "run_key", "source_fold", "freeze_mode"]):
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
                "source_fold": source_fold,
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

    ap.add_argument("--feature_mode", default="mean_absdiff",
                    choices=["origin", "taut", "mean", "concat", "mean_absdiff", "mean_diff_absdiff"])
    ap.add_argument("--target_mode", default="raw", choices=["raw", "log1p"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[1, 3, 10, 30, 100, 300, 1000, 3000])
    ap.add_argument("--inner_folds", type=int, default=5)

    ap.add_argument("--cv_seed", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_ring_size", type=int, default=6)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    set_seed(args.cv_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Stage 4AK embedding-head transfer ===")
    print("out_dir:", out_dir)
    print("device:", device)
    print("datasets:", args.datasets)
    print("run_keys:", args.run_keys)
    print("source_folds:", args.source_folds)
    print("feature_mode:", args.feature_mode)
    print("target_mode:", args.target_mode)
    print("alphas:", args.alphas)

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
        sub = meta[meta["dataset_name"].eq(dataset_name)].copy().reset_index(drop=True)
        if len(sub) == 0:
            print("[SKIP] no rows:", dataset_name)
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

                print("[extract] origin embeddings")
                origin_emb = extract_embeddings_for_view(
                    dataset=origin_dataset,
                    global_indices=global_indices,
                    run_dir=run_dir,
                    source_fold=source_fold,
                    view_name="origin",
                    batch_size=args.batch_size,
                    device=device,
                )

                print("[extract] taut embeddings")
                taut_emb = extract_embeddings_for_view(
                    dataset=taut_dataset,
                    global_indices=global_indices,
                    run_dir=run_dir,
                    source_fold=source_fold,
                    view_name="taut",
                    batch_size=args.batch_size,
                    device=device,
                )

                X_origin = origin_emb
                X_taut = taut_emb
                X_dual = make_dual_features(origin_emb, taut_emb, args.feature_mode)

                freeze_mode = f"embedding_head_ridge_{args.target_mode}_{args.feature_mode}"

                print("[train head] origin_tl")
                pred_origin, fold_origin = cv_predict_ridge_head(
                    X=X_origin,
                    y_raw=y,
                    cv_seed=args.cv_seed,
                    alphas=args.alphas,
                    target_mode=args.target_mode,
                    inner_folds=args.inner_folds,
                )
                fold_origin["method"] = "origin_tl"

                print("[train head] taut_tl")
                pred_taut, fold_taut = cv_predict_ridge_head(
                    X=X_taut,
                    y_raw=y,
                    cv_seed=args.cv_seed,
                    alphas=args.alphas,
                    target_mode=args.target_mode,
                    inner_folds=args.inner_folds,
                )
                fold_taut["method"] = "taut_tl"

                print("[train head] mean_tl / dual")
                pred_dual, fold_dual = cv_predict_ridge_head(
                    X=X_dual,
                    y_raw=y,
                    cv_seed=args.cv_seed,
                    alphas=args.alphas,
                    target_mode=args.target_mode,
                    inner_folds=args.inner_folds,
                )
                fold_dual["method"] = "mean_tl"

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
                pred_df["origin_tl_pred"] = pred_origin
                pred_df["taut_tl_pred"] = pred_taut
                pred_df["mean_tl_pred"] = pred_dual
                pred_df["origin_tl_abs_error"] = np.abs(pred_df["rt"] - pred_df["origin_tl_pred"])
                pred_df["taut_tl_abs_error"] = np.abs(pred_df["rt"] - pred_df["taut_tl_pred"])
                pred_df["mean_tl_abs_error"] = np.abs(pred_df["rt"] - pred_df["mean_tl_pred"])

                for fdf in [fold_origin, fold_taut, fold_dual]:
                    fdf["dataset_name"] = dataset_name
                    fdf["run_key"] = run_key
                    fdf["run_dir"] = str(run_dir)
                    fdf["source_fold"] = int(source_fold)
                    fdf["freeze_mode"] = freeze_mode
                    fdf["feature_mode"] = args.feature_mode

                all_pred_rows.append(pred_df)
                all_fold_rows.extend([fold_origin, fold_taut, fold_dual])

                tmp_metrics = summarize_predictions(pred_df)
                print("\n[SUMMARY current]")
                print(tmp_metrics[["dataset_name", "run_key", "source_fold", "method", "mae", "rmse", "r2", "spearman", "bias"]].to_string(index=False))

    if not all_pred_rows:
        raise RuntimeError("No predictions produced.")

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

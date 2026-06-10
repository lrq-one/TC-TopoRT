import argparse
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import GroupKFold, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

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


def complex_collate_fn(batch):
    return ComplexBatch.from_complex_list(batch)


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]

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


@torch.no_grad()
def extract_view_features(model, dataset, indices, batch_size, device):
    subset = Subset(dataset, list(map(int, indices)))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=0,
    )

    model.eval()
    preds = []
    embs = []
    scores = []

    for batch in loader:
        batch = batch.to(device)
        out, partial = model(batch, include_partial=True)

        preds.append(out.view(-1).detach().cpu().numpy())
        embs.append(partial["mol_emb"].detach().cpu().numpy())

        if "score" in partial and partial["score"] is not None:
            scores.append(partial["score"].detach().cpu().numpy())
        else:
            scores.append(np.zeros((out.view(-1).shape[0], 512), dtype=np.float32))

    pred = np.concatenate(preds, axis=0).astype(np.float32)
    emb = np.concatenate(embs, axis=0).astype(np.float32)
    score = np.concatenate(scores, axis=0).astype(np.float32)

    return pred, emb, score


def morgan_fp_array(smiles_list, n_bits=1024, radius=2):
    arrs = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        arr = np.zeros((n_bits,), dtype=np.float32)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            DataStructs.ConvertToNumpyArray(fp, arr)
        arrs.append(arr)
    return np.vstack(arrs).astype(np.float32)


def build_feature_matrices(meta_sub, origin_pred, taut_pred, origin_emb, taut_emb, origin_score, taut_score):
    changed = meta_sub["taut_changed"].values.astype(np.float32) if "taut_changed" in meta_sub.columns else np.zeros(len(meta_sub), dtype=np.float32)

    origin_pred = origin_pred.reshape(-1, 1).astype(np.float32)
    taut_pred = taut_pred.reshape(-1, 1).astype(np.float32)
    mean_pred = 0.5 * (origin_pred + taut_pred)
    diff_pred = np.abs(origin_pred - taut_pred)
    changed = changed.reshape(-1, 1)

    pred_feat = np.hstack([
        origin_pred,
        taut_pred,
        mean_pred,
        diff_pred,
        changed,
        origin_pred * changed / 1000.0,
        taut_pred * changed / 1000.0,
    ]).astype(np.float32)

    emb_diff = np.abs(origin_emb - taut_emb).astype(np.float32)
    emb_mean = (0.5 * (origin_emb + taut_emb)).astype(np.float32)

    smiles_col = "canonical_smiles" if "canonical_smiles" in meta_sub.columns else "origin_smiles"
    fp = morgan_fp_array(meta_sub[smiles_col].values, n_bits=1024, radius=2)

    feature_sets = {
        "pred_only": pred_feat,
        "emb_pred": np.hstack([origin_emb, taut_emb, pred_feat]).astype(np.float32),
        "emb_diff_pred": np.hstack([origin_emb, taut_emb, emb_diff, emb_mean, pred_feat]).astype(np.float32),
        "emb_score_pred": np.hstack([origin_emb, taut_emb, origin_score, taut_score, pred_feat]).astype(np.float32),
        "fp_pred": np.hstack([fp, pred_feat]).astype(np.float32),
        "emb_fp_pred": np.hstack([origin_emb, taut_emb, emb_diff, emb_mean, fp, pred_feat]).astype(np.float32),
    }

    for k in list(feature_sets.keys()):
        feature_sets[k] = np.nan_to_num(feature_sets[k], nan=0.0, posinf=0.0, neginf=0.0)

    return feature_sets


def get_pc_grid(n_train, n_feat):
    max_pc = max(2, min(int(n_train) - 2, int(n_feat), 96))
    raw = [4, 8, 16, 32, 64, 96]
    pcs = [x for x in raw if x <= max_pc]
    if not pcs:
        pcs = [max_pc]
    return sorted(set(pcs))


def make_inner_cv(groups_train, seed):
    groups_train = np.asarray(groups_train).astype(str)
    n_groups = len(np.unique(groups_train))

    if n_groups >= 5:
        return GroupKFold(n_splits=5), groups_train

    return KFold(n_splits=min(5, len(groups_train)), shuffle=True, random_state=int(seed)), None


def make_estimator(method, n_train, n_feat, seed):
    pcs = get_pc_grid(n_train, n_feat)

    if method == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-4, 4, 25))),
        ]), False

    if method == "huber":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000)),
        ]), False

    if method == "pca_ridge":
        return GridSearchCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 25))),
            ]),
            param_grid={
                "pca__n_components": pcs,
            },
            scoring="neg_mean_absolute_error",
            cv=5,
            n_jobs=-1,
            refit=True,
        ), True

    if method == "pca_svr":
        return GridSearchCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("model", SVR(kernel="rbf")),
            ]),
            param_grid={
                "pca__n_components": pcs,
                "model__C": [1.0, 3.0, 10.0, 30.0, 100.0],
                "model__gamma": ["scale", 0.001, 0.003, 0.01, 0.03],
                "model__epsilon": [0.05, 0.1, 0.3, 0.5],
            },
            scoring="neg_mean_absolute_error",
            cv=5,
            n_jobs=-1,
            refit=True,
        ), True

    if method == "pca_krr":
        return GridSearchCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("model", KernelRidge(kernel="rbf")),
            ]),
            param_grid={
                "pca__n_components": pcs,
                "model__alpha": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
                "model__gamma": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
            },
            scoring="neg_mean_absolute_error",
            cv=5,
            n_jobs=-1,
            refit=True,
        ), True

    if method == "extratrees":
        return GridSearchCV(
            ExtraTreesRegressor(
                n_estimators=500,
                random_state=int(seed),
                n_jobs=-1,
            ),
            param_grid={
                "max_features": ["sqrt", 0.25, 0.5],
                "min_samples_leaf": [1, 2, 3, 5],
            },
            scoring="neg_mean_absolute_error",
            cv=5,
            n_jobs=-1,
            refit=True,
        ), True

    raise ValueError(f"Unknown method={method}")


def run_outer_cv(X, y, groups, meta_sub, feature_set_name, method, seed):
    groups = np.asarray(groups).astype(str)
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float32)

    outer = GroupKFold(n_splits=min(10, len(np.unique(groups))))

    pred_all = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows = []

    for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        g_tr = groups[tr]

        y_mean = float(np.mean(y_tr))
        y_std = float(np.std(y_tr))
        if y_std < 1e-8:
            y_std = 1.0
        y_tr_s = (y_tr - y_mean) / y_std

        est, is_grid = make_estimator(method, len(tr), X.shape[1], seed + fold * 17)

        if is_grid:
            inner_cv, inner_groups = make_inner_cv(g_tr, seed + fold * 17)
            est.cv = inner_cv
            if inner_groups is not None:
                est.fit(X_tr, y_tr_s, groups=inner_groups)
            else:
                est.fit(X_tr, y_tr_s)
        else:
            est.fit(X_tr, y_tr_s)

        p_te_s = np.asarray(est.predict(X_te)).reshape(-1)
        p_te = p_te_s * y_std + y_mean
        pred_all[te] = p_te

        fm = metrics(y_te, p_te)
        best_params = getattr(est, "best_params_", None)

        fold_rows.append({
            "feature_set": feature_set_name,
            "method": method,
            "fold": int(fold),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_train_groups": int(len(np.unique(g_tr))),
            "n_test_groups": int(len(np.unique(groups[te]))),
            "best_params": json.dumps(best_params, ensure_ascii=False) if best_params is not None else "",
            **{f"test_{k}": v for k, v in fm.items()},
        })

    pred_df = meta_sub[["stage4_index", "dataset_name", "rt"]].copy()
    if "inchikey" in meta_sub.columns:
        pred_df["inchikey"] = meta_sub["inchikey"].values
    if "canonical_smiles" in meta_sub.columns:
        pred_df["canonical_smiles"] = meta_sub["canonical_smiles"].values

    pred_df["feature_set"] = feature_set_name
    pred_df["method"] = method
    pred_df["pred"] = pred_all
    pred_df["abs_error"] = np.abs(pred_df["rt"].values - pred_all)

    total_m = metrics(y, pred_all)
    metric_row = {
        "feature_set": feature_set_name,
        "method": method,
        **total_m,
    }

    return pred_df, pd.DataFrame(fold_rows), metric_row


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", default="paper_analysis_stage4C_embtr_lifeold")
    ap.add_argument("--stage4_meta_csv", default="paper_analysis_stage4_external/external_predret10_stage4_meta.csv")
    ap.add_argument("--origin_csv", default="paper_analysis_stage4_external/temp_external_predret10_origin.csv")
    ap.add_argument("--taut_csv", default="paper_analysis_stage4_external/temp_external_predret10_taut.csv")
    ap.add_argument("--origin_root", default="paper_analysis_stage4_external/cache/predret10_origin")
    ap.add_argument("--taut_root", default="paper_analysis_stage4_external/cache/predret10_taut")

    ap.add_argument("--dataset", default="LIFE_old_194")
    ap.add_argument("--run_key", default="seed1")
    ap.add_argument("--source_fold", type=int, default=0)

    ap.add_argument("--group_col", default="inchikey")
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--methods", nargs="+", default=["ridge", "pca_ridge", "pca_svr", "pca_krr", "huber"])
    ap.add_argument("--feature_sets", nargs="+", default=["pred_only", "emb_diff_pred", "emb_fp_pred"])

    ap.add_argument("--max_ring_size", type=int, default=6)

    args = ap.parse_args()

    set_seed(20260610)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run_key not in RUNS:
        raise ValueError(f"Unknown run_key={args.run_key}; available={list(RUNS.keys())}")

    run_dir = Path(RUNS[args.run_key]["dir"])

    print("=== TCDV frozen embedding transfer regressor ===")
    print("dataset:", args.dataset)
    print("run_key:", args.run_key)
    print("source_fold:", args.source_fold)
    print("device:", device)
    print("methods:", args.methods)
    print("feature_sets:", args.feature_sets)

    meta = pd.read_csv(args.stage4_meta_csv).sort_values("stage4_index").reset_index(drop=True)
    sub = meta[meta["dataset_name"] == args.dataset].copy().reset_index(drop=True)

    if len(sub) == 0:
        raise RuntimeError(f"No rows for dataset={args.dataset}")

    if args.group_col not in sub.columns:
        raise RuntimeError(f"group_col={args.group_col} not found")

    groups = sub[args.group_col].fillna(sub["stage4_index"].astype(str)).astype(str).values
    print(f"dataset rows={len(sub)} unique_groups={len(np.unique(groups))}")

    indices = sub["stage4_index"].values.astype(int)

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

    print("origin_ckpt:", origin_ckpt)
    print("taut_ckpt:", taut_ckpt)

    origin_model = make_model(cwn_layers, cwn_hidden, device)
    load_state_dict_safely(origin_model, origin_ckpt, device)

    taut_model = make_model(cwn_layers, cwn_hidden, device)
    load_state_dict_safely(taut_model, taut_ckpt, device)

    print("\n=== Extract origin features ===")
    origin_pred, origin_emb, origin_score = extract_view_features(
        origin_model, origin_dataset, indices, args.batch_size, device
    )

    print("origin_pred:", origin_pred.shape)
    print("origin_emb:", origin_emb.shape)
    print("origin_score:", origin_score.shape)

    print("\n=== Extract taut features ===")
    taut_pred, taut_emb, taut_score = extract_view_features(
        taut_model, taut_dataset, indices, args.batch_size, device
    )

    print("taut_pred:", taut_pred.shape)
    print("taut_emb:", taut_emb.shape)
    print("taut_score:", taut_score.shape)

    np.savez_compressed(
        out_dir / f"external_embedding_features_{args.dataset}_{args.run_key}_fold{args.source_fold}.npz",
        stage4_index=indices,
        rt=sub["rt"].values.astype(np.float32),
        origin_pred=origin_pred,
        taut_pred=taut_pred,
        origin_emb=origin_emb,
        taut_emb=taut_emb,
        origin_score=origin_score,
        taut_score=taut_score,
    )

    feature_mats = build_feature_matrices(
        sub,
        origin_pred,
        taut_pred,
        origin_emb,
        taut_emb,
        origin_score,
        taut_score,
    )

    y = sub["rt"].values.astype(np.float64)

    all_pred = []
    all_fold = []
    all_metrics = []

    for fs in args.feature_sets:
        if fs not in feature_mats:
            raise ValueError(f"Unknown feature_set={fs}; available={list(feature_mats.keys())}")

        X = feature_mats[fs]
        print(f"\n=== Feature set: {fs}, X={X.shape} ===")

        for method in args.methods:
            print(f"[RUN] feature_set={fs}, method={method}")
            pred_df, fold_df, metric_row = run_outer_cv(
                X=X,
                y=y,
                groups=groups,
                meta_sub=sub,
                feature_set_name=fs,
                method=method,
                seed=20260610,
            )

            all_pred.append(pred_df)
            all_fold.append(fold_df)
            all_metrics.append(metric_row)

            print(
                f"  -> MAE={metric_row['mae']:.4f}, RMSE={metric_row['rmse']:.4f}, "
                f"R2={metric_row['r2']:.4f}, Spearman={metric_row['spearman']:.4f}"
            )

    pred_all = pd.concat(all_pred, ignore_index=True)
    fold_all = pd.concat(all_fold, ignore_index=True)
    metrics_all = pd.DataFrame(all_metrics).sort_values("mae").reset_index(drop=True)

    pred_path = out_dir / "lifeold_embedding_transfer_predictions.csv"
    fold_path = out_dir / "lifeold_embedding_transfer_fold_metrics.csv"
    metrics_path = out_dir / "lifeold_embedding_transfer_metrics.csv"

    pred_all.to_csv(pred_path, index=False)
    fold_all.to_csv(fold_path, index=False)
    metrics_all.to_csv(metrics_path, index=False)

    print("\n=== FINAL METRICS ===")
    show_cols = ["feature_set", "method", "n", "mae", "medae", "rmse", "r2", "pearson", "spearman", "bias"]
    print(metrics_all[show_cols].to_string(index=False))

    print("\n[SAVE]", pred_path)
    print("[SAVE]", fold_path)
    print("[SAVE]", metrics_path)
    print("✅ Done:", out_dir)


if __name__ == "__main__":
    main()

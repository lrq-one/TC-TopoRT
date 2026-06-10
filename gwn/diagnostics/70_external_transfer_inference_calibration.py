import argparse
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, HuberRegressor, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mp.smrt_dataset import SMRTComplexDataset
from mp.complex import ComplexBatch
from net.topocellrt_cwn_replace import TopoCellRTCWNReplace


RUNS = [
    ("seed1",   "results_OOF_DualView_Stack_v1"),
    ("seed79",  "results_OOF_DualView_Stack_seed79"),
    ("seed123", "results_OOF_DualView_Stack_seed123"),
    ("seed256", "results_OOF_DualView_Stack_seed256"),
    ("seed5",   "results_OOF_DualView_Stack_seed5"),
]


TAUT = rdMolStandardize.TautomerEnumerator()
try:
    TAUT.SetMaxTautomers(128)
except Exception:
    pass
try:
    TAUT.SetMaxTransforms(128)
except Exception:
    pass


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_csv(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path} shape={df.shape}")


def complex_collate_fn(batch):
    return ComplexBatch.from_complex_list(batch)


def safe_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol
    except Exception:
        return None


def canon_smiles(mol):
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def formula(mol):
    if mol is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


def inchikey(mol):
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def strict_tautomer_view(smiles):
    original = str(smiles)
    mol = safe_mol(original)

    if mol is None:
        return {
            "taut_smiles": original,
            "orig_canon": original,
            "taut_canon": original,
            "raw_changed": 0,
            "real_changed": 0,
            "formula_same": 0,
            "heavy_same": 0,
            "fallback": 1,
            "reason": "parse_failed",
        }

    try:
        orig_canon = canon_smiles(mol)
        orig_formula = formula(mol)
        orig_heavy = mol.GetNumHeavyAtoms()

        taut_mol = TAUT.Canonicalize(mol)
        if taut_mol is None:
            raise RuntimeError("tautomer_none")

        taut_canon = canon_smiles(taut_mol)
        taut_formula = formula(taut_mol)
        taut_heavy = taut_mol.GetNumHeavyAtoms()

        formula_same = int(orig_formula == taut_formula)
        heavy_same = int(orig_heavy == taut_heavy)

        if not formula_same or not heavy_same:
            return {
                "taut_smiles": original,
                "orig_canon": orig_canon,
                "taut_canon": taut_canon,
                "raw_changed": int(taut_canon != original),
                "real_changed": 0,
                "formula_same": formula_same,
                "heavy_same": heavy_same,
                "fallback": 1,
                "reason": "formula_or_heavy_changed",
            }

        real_changed = int(taut_canon != orig_canon)
        taut_smiles = taut_canon if real_changed else original

        return {
            "taut_smiles": taut_smiles,
            "orig_canon": orig_canon,
            "taut_canon": taut_canon,
            "raw_changed": int(taut_canon != original),
            "real_changed": real_changed,
            "formula_same": formula_same,
            "heavy_same": heavy_same,
            "fallback": 0,
            "reason": "ok",
        }

    except Exception as e:
        return {
            "taut_smiles": original,
            "orig_canon": original,
            "taut_canon": original,
            "raw_changed": 0,
            "real_changed": 0,
            "formula_same": 0,
            "heavy_same": 0,
            "fallback": 1,
            "reason": f"exception:{type(e).__name__}",
        }


def load_smrt_overlap_sets(train_csv, test_csv):
    can_set = set()
    ik_set = set()

    for path in [train_csv, test_csv]:
        df = pd.read_csv(path, engine="python")
        df.columns = [str(c).lower().strip() for c in df.columns]
        if "smile" in df.columns and "smiles" not in df.columns:
            df = df.rename(columns={"smile": "smiles"})
        if "smiles" not in df.columns or "rt" not in df.columns:
            continue

        df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
        df = df[df["rt"] > 300.0].copy()

        for s in df["smiles"].astype(str).tolist():
            mol = safe_mol(s)
            if mol is None:
                continue
            c = canon_smiles(mol)
            k = inchikey(mol)
            if c:
                can_set.add(c)
            if k:
                ik_set.add(k)

    return can_set, ik_set


def prepare_external_predret10(clean_csv, out_dir, dummy_rt=999.0):
    df = pd.read_csv(clean_csv)
    df = df[(df["valid_rdkit"] == 1) & df["canonical_smiles"].notna() & df["rt"].notna()].copy()
    df = df.reset_index(drop=True)

    rows_meta = []
    origin_rows = []
    taut_rows = []
    taut_audit_rows = []

    for i, r in df.iterrows():
        origin_smi = str(r["canonical_smiles"])
        info = strict_tautomer_view(origin_smi)
        taut_smi = info["taut_smiles"]

        rows_meta.append({
            "stage4_index": i,
            "dataset_group": r.get("dataset_group", "predret10"),
            "dataset_name": r["dataset_name"],
            "source_file": r.get("source_file", ""),
            "source_row": r.get("source_row", i),
            "record_id": r.get("record_id", f"external_{i}"),
            "name": r.get("name", ""),
            "origin_smiles": origin_smi,
            "taut_smiles": taut_smi,
            "rt": float(r["rt"]),
            "formula": r.get("formula", ""),
            "inchikey": r.get("inchikey", ""),
            "canonical_smiles": r.get("canonical_smiles", origin_smi),
            "taut_changed": float(info["real_changed"]),
            "taut_fallback": int(info["fallback"]),
            "taut_reason": info["reason"],
        })

        origin_rows.append({"smile": origin_smi, "rt": dummy_rt})
        taut_rows.append({"smile": taut_smi, "rt": dummy_rt})

        taut_audit_rows.append({
            "stage4_index": i,
            "origin_smiles": origin_smi,
            **info,
        })

    meta = pd.DataFrame(rows_meta)
    origin_csv = Path(out_dir) / "temp_external_predret10_origin.csv"
    taut_csv = Path(out_dir) / "temp_external_predret10_taut.csv"
    audit_csv = Path(out_dir) / "external_predret10_tautomer_audit.csv"

    save_csv(pd.DataFrame(origin_rows), origin_csv)
    save_csv(pd.DataFrame(taut_rows), taut_csv)
    save_csv(pd.DataFrame(taut_audit_rows), audit_csv)

    return meta, origin_csv, taut_csv


def make_model(cwn_layers, cwn_hidden, device):
    model = TopoCellRTCWNReplace(
        emb_dim=256,
        cwn_layers=int(cwn_layers),
        cwn_hidden=int(cwn_hidden),
        max_dim=2,
        drop_ratio=0.0,
    ).to(device)
    return model


@torch.no_grad()
def predict_model(model, dataset, batch_size, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=0,
    )

    model.eval()
    preds = []

    for batch in loader:
        batch = batch.to(device)
        p = model(batch)
        if isinstance(p, tuple):
            p = p[0]
        preds.append(p.view(-1).detach().cpu())

    return torch.cat(preds).numpy()


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

    pearson = float(pd.Series(y).corr(pd.Series(p), method="pearson")) if len(y) > 1 else np.nan
    spearman = float(pd.Series(y).corr(pd.Series(p), method="spearman")) if len(y) > 1 else np.nan

    return {
        "n": int(len(y)),
        "mae": float(e.mean()),
        "medae": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan),
        "pearson": pearson,
        "spearman": spearman,
        "bias": float(np.mean(p - y)),
    }


def build_stack_features(origin_pred, taut_pred, changed):
    origin_pred = np.asarray(origin_pred, dtype=np.float64)
    taut_pred = np.asarray(taut_pred, dtype=np.float64)
    changed = np.asarray(changed, dtype=np.float64)

    diff = np.abs(origin_pred - taut_pred)
    mean_pred = 0.5 * (origin_pred + taut_pred)
    min_pred = np.minimum(origin_pred, taut_pred)
    max_pred = np.maximum(origin_pred, taut_pred)

    return np.vstack([
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
    ]).T


def disagreement_fusion(origin_pred, taut_pred, alpha, tau, temperature):
    origin_pred = np.asarray(origin_pred, dtype=np.float64)
    taut_pred = np.asarray(taut_pred, dtype=np.float64)
    diff = np.abs(origin_pred - taut_pred)
    soft_use = 1.0 / (1.0 + np.exp(-((diff - tau) / temperature)))
    mixed = alpha * origin_pred + (1.0 - alpha) * taut_pred
    final = (1.0 - soft_use) * origin_pred + soft_use * mixed
    return final


def fit_external_stackers(run_dir, ext_origin, ext_taut, ext_changed):
    run_dir = Path(run_dir)
    oof_path = run_dir / "oof_predictions.csv"
    config_path = run_dir / "config.json"
    final_metrics_path = run_dir / "final_metrics.json"

    oof = pd.read_csv(oof_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    selected_from_json = None
    if final_metrics_path.exists():
        with open(final_metrics_path, "r", encoding="utf-8") as f:
            fm = json.load(f)
        selected_from_json = fm.get("selected_stacker", None)

    y = oof["Actual_RT"].values.astype(np.float64)
    o_oof = oof["Origin_OOF_Pred"].values.astype(np.float64)
    t_oof = oof["Taut_OOF_Pred"].values.astype(np.float64)
    c_oof = oof["Taut_Changed"].values.astype(np.float64)

    x_oof = build_stack_features(o_oof, t_oof, c_oof)
    x_ext = build_stack_features(ext_origin, ext_taut, ext_changed)

    alpha_grid = np.asarray(cfg.get("alpha_grid", np.linspace(0, 1, 101)), dtype=np.float64)
    tau_grid = np.asarray(cfg.get("tau_grid", [0, 2, 5, 8, 10, 15, 20, 30, 50]), dtype=np.float64)
    temp = float(cfg.get("stack_temperature", 5.0))
    huber_alpha = float(cfg.get("huber_alpha", 1e-4))

    candidates = {}

    candidates["origin_only"] = {
        "oof": o_oof,
        "ext": ext_origin,
    }
    candidates["taut_only"] = {
        "oof": t_oof,
        "ext": ext_taut,
    }
    candidates["mean_origin_taut"] = {
        "oof": 0.5 * (o_oof + t_oof),
        "ext": 0.5 * (ext_origin + ext_taut),
    }

    best_gate = None
    for tau in tau_grid:
        for alpha in alpha_grid:
            p = disagreement_fusion(o_oof, t_oof, alpha=alpha, tau=tau, temperature=temp)
            m = metrics(y, p)
            if best_gate is None or m["mae"] < best_gate["mae"]:
                best_gate = {
                    "alpha": float(alpha),
                    "tau": float(tau),
                    "mae": float(m["mae"]),
                    "oof": p,
                    "ext": disagreement_fusion(ext_origin, ext_taut, alpha=alpha, tau=tau, temperature=temp),
                }

    candidates["oof_selected_fixed_gate"] = {
        "oof": best_gate["oof"],
        "ext": best_gate["ext"],
    }

    ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.array([1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0])),
    )
    ridge.fit(x_oof, y)
    candidates["ridge_stack"] = {
        "oof": ridge.predict(x_oof),
        "ext": ridge.predict(x_ext),
    }

    huber = make_pipeline(
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=huber_alpha, max_iter=1000),
    )
    huber.fit(x_oof, y)
    candidates["huber_stack"] = {
        "oof": huber.predict(x_oof),
        "ext": huber.predict(x_ext),
    }

    oof_summary = {name: metrics(y, item["oof"]) for name, item in candidates.items()}
    selected_by_recomputed_oof = min(oof_summary.keys(), key=lambda k: oof_summary[k]["mae"])

    selected = selected_from_json if selected_from_json in candidates else selected_by_recomputed_oof

    return {
        "origin_pred": candidates["origin_only"]["ext"],
        "taut_pred": candidates["taut_only"]["ext"],
        "mean_pred": candidates["mean_origin_taut"]["ext"],
        "huber_pred": candidates["huber_stack"]["ext"],
        "selected_final_pred": candidates[selected]["ext"],
        "selected_stacker": selected,
        "selected_by_recomputed_oof": selected_by_recomputed_oof,
        "oof_summary": oof_summary,
    }


def predict_all_runs(args, meta, origin_dataset, taut_dataset, device):
    all_rows = []
    stacker_rows = []

    for seed_name, run_dir in RUNS:
        run_dir = Path(run_dir)
        config_path = run_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        k = int(cfg.get("k", 5))
        cwn_layers = int(cfg.get("cwn_layers", 6))
        cwn_hidden = int(cfg.get("cwn_hidden", 256))
        eval_batch_size = int(cfg.get("eval_batch_size", args.eval_batch_size))

        print(f"\n=== External inference: {seed_name} | {run_dir} ===")

        origin_fold_preds = []
        taut_fold_preds = []

        for fold in range(k):
            for view_name, dataset, pred_list in [
                ("origin", origin_dataset, origin_fold_preds),
                ("taut", taut_dataset, taut_fold_preds),
            ]:
                ckpt = run_dir / "folds" / f"fold_{fold}" / view_name / "best_model.pth"
                if not ckpt.exists():
                    raise FileNotFoundError(ckpt)

                model = make_model(cwn_layers=cwn_layers, cwn_hidden=cwn_hidden, device=device)
                load_state_dict_safely(model, ckpt, device)

                pred = predict_model(model, dataset, batch_size=eval_batch_size, device=device)
                pred_list.append(pred)

                print(f"[{seed_name}] fold={fold} view={view_name} pred_mean={float(np.mean(pred)):.4f}")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        ext_origin = np.mean(np.vstack(origin_fold_preds), axis=0)
        ext_taut = np.mean(np.vstack(taut_fold_preds), axis=0)
        ext_changed = meta["taut_changed"].values.astype(np.float64)

        stack = fit_external_stackers(
            run_dir=run_dir,
            ext_origin=ext_origin,
            ext_taut=ext_taut,
            ext_changed=ext_changed,
        )

        df_seed = meta.copy()
        df_seed["seed"] = seed_name
        df_seed["run_dir"] = str(run_dir)
        df_seed["origin_pred"] = stack["origin_pred"]
        df_seed["taut_pred"] = stack["taut_pred"]
        df_seed["mean_pred"] = stack["mean_pred"]
        df_seed["huber_pred"] = stack["huber_pred"]
        df_seed["selected_final_pred"] = stack["selected_final_pred"]
        df_seed["selected_stacker"] = stack["selected_stacker"]
        df_seed["selected_by_recomputed_oof"] = stack["selected_by_recomputed_oof"]

        all_rows.append(df_seed)

        for name, m in stack["oof_summary"].items():
            stacker_rows.append({
                "seed": seed_name,
                "run_dir": str(run_dir),
                "stacker": name,
                "selected_stacker": stack["selected_stacker"],
                "selected_by_recomputed_oof": stack["selected_by_recomputed_oof"],
                **{f"oof_{k}": v for k, v in m.items()},
            })

    pred_all = pd.concat(all_rows, ignore_index=True)
    stacker_summary = pd.DataFrame(stacker_rows)

    return pred_all, stacker_summary


def add_five_seed_mean(pred_all):
    meta_cols = [
        "stage4_index", "dataset_group", "dataset_name", "source_file", "source_row",
        "record_id", "name", "origin_smiles", "taut_smiles", "rt", "formula",
        "inchikey", "canonical_smiles", "taut_changed", "taut_fallback", "taut_reason",
        "smrt_exact_overlap",
    ]

    pred_cols = ["origin_pred", "taut_pred", "mean_pred", "huber_pred", "selected_final_pred"]

    first = pred_all.groupby("stage4_index", as_index=False).first()
    mean_pred = pred_all.groupby("stage4_index")[pred_cols].mean().reset_index()

    out = first[meta_cols].merge(mean_pred, on="stage4_index", how="left")
    out["seed"] = "5seed_mean"
    out["run_dir"] = "mean_over_5_seeds"
    out["selected_stacker"] = "mean_over_seed_selected_final"
    out["selected_by_recomputed_oof"] = "mean_over_seed_selected_final"
    return out


def cross_validated_calibration_metrics(df, pred_col, dataset_name, scope, seed_label, method, min_n, n_splits, calib_seed):
    sub = df.copy()
    sub = sub[np.isfinite(sub[pred_col].values) & np.isfinite(sub["rt"].values)].copy()

    if len(sub) < min_n:
        return []

    y = sub["rt"].values.astype(np.float64)
    p = sub[pred_col].values.astype(np.float64)

    rows = []

    raw_m = metrics(y, p)
    rows.append({
        "seed": seed_label,
        "dataset_name": dataset_name,
        "scope": scope,
        "method": method,
        "eval_mode": "raw_uncalibrated",
        "calibration_model": "none",
        **raw_m,
    })

    k = min(int(n_splits), len(sub))
    if k < 2:
        return rows

    cv = KFold(n_splits=k, shuffle=True, random_state=calib_seed)
    pred_cal = np.full(len(sub), np.nan, dtype=np.float64)

    fold_rows = []
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(sub)))):
        x_tr = p[tr].reshape(-1, 1)
        y_tr = y[tr]
        x_te = p[te].reshape(-1, 1)

        model = LinearRegression()
        model.fit(x_tr, y_tr)
        pred_cal[te] = model.predict(x_te)

        fm = metrics(y[te], pred_cal[te])
        fold_rows.append({
            "seed": seed_label,
            "dataset_name": dataset_name,
            "scope": scope,
            "method": method,
            "eval_mode": "linear_calibrated_fold",
            "fold": fold,
            "calib_a": float(model.coef_[0]),
            "calib_b": float(model.intercept_),
            **fm,
        })

    cal_m = metrics(y, pred_cal)
    rows.append({
        "seed": seed_label,
        "dataset_name": dataset_name,
        "scope": scope,
        "method": method,
        "eval_mode": "linear_calibrated",
        "calibration_model": "external_5fold_linear",
        **cal_m,
    })

    return rows, fold_rows


def run_calibration(pred_df, args):
    method_cols = {
        "origin": "origin_pred",
        "taut": "taut_pred",
        "mean": "mean_pred",
        "huber": "huber_pred",
        "selected_final": "selected_final_pred",
    }

    rows = []
    fold_rows_all = []

    seed_labels = sorted(pred_df["seed"].unique().tolist())

    for seed_label in seed_labels:
        df_seed = pred_df[pred_df["seed"] == seed_label].copy()

        for dataset_name, sub0 in df_seed.groupby("dataset_name"):
            scopes = {
                "all": sub0.copy(),
                "no_smrt_exact_overlap": sub0[sub0["smrt_exact_overlap"] == 0].copy(),
            }

            for scope, sub in scopes.items():
                for method, pred_col in method_cols.items():
                    result = cross_validated_calibration_metrics(
                        sub,
                        pred_col=pred_col,
                        dataset_name=dataset_name,
                        scope=scope,
                        seed_label=seed_label,
                        method=method,
                        min_n=args.min_external_n,
                        n_splits=args.calib_folds,
                        calib_seed=args.calib_seed,
                    )

                    if not result:
                        continue

                    if isinstance(result, tuple):
                        r, fr = result
                        rows.extend(r)
                        fold_rows_all.extend(fr)
                    else:
                        rows.extend(result)

    return pd.DataFrame(rows), pd.DataFrame(fold_rows_all)


def summarize_across_seeds(metrics_df):
    # only summarize the five individual seeds, not 5seed_mean
    df = metrics_df[metrics_df["seed"] != "5seed_mean"].copy()

    metric_cols = ["n", "mae", "medae", "rmse", "r2", "pearson", "spearman", "bias"]

    rows = []
    group_cols = ["dataset_name", "scope", "method", "eval_mode"]

    for keys, sub in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["num_seeds"] = int(sub["seed"].nunique())

        for c in metric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
            row[f"{c}_min"] = float(vals.min())
            row[f"{c}_max"] = float(vals.max())

        rows.append(row)

    return pd.DataFrame(rows)


def write_readme(out_dir):
    text = """Stage 4 external transfer and calibration

Inputs:
- paper_analysis_external/external_predret10_clean.csv
- 5 seed OOF dual-view checkpoints under results_OOF_DualView_Stack_*/
- SMRT train/test CSVs for exact-overlap audit

Main outputs:
- external_predictions_per_seed.csv
- external_predictions_5seed_mean.csv
- external_predictions_all_with_5seed_mean.csv
- external_stacker_oof_summary.csv
- external_calibration_metrics.csv
- external_calibration_fold_metrics.csv
- external_calibration_summary_5seed.csv

Interpretation:
- raw_uncalibrated metrics compare SMRT-scale predicted RT to external RT and are not the main conclusion.
- linear_calibrated metrics evaluate whether SMRT-trained predictions transfer after dataset-specific linear calibration.
- no_smrt_exact_overlap removes exact canonical-SMILES/InChIKey overlap with SMRT train/test.
"""
    path = Path(out_dir) / "README_outputs.txt"
    path.write_text(text, encoding="utf-8")
    print(f"[SAVE] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="paper_analysis_stage4_external")
    ap.add_argument("--external_clean_csv", default="paper_analysis_external/external_predret10_clean.csv")
    ap.add_argument("--smrt_train_csv", default="data/SMRT_train.csv")
    ap.add_argument("--smrt_test_csv", default="data/SMRT_test.csv")
    ap.add_argument("--max_ring_size", type=int, default=6)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--force_reprocess", type=int, default=1)
    ap.add_argument("--calib_folds", type=int, default=5)
    ap.add_argument("--calib_seed", type=int, default=20260610)
    ap.add_argument("--min_external_n", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Stage 4 external transfer / calibration ===")
    print("out_dir:", out_dir)
    print("device:", device)

    print("\n=== Prepare external origin/tautomer CSVs ===")
    meta, origin_csv, taut_csv = prepare_external_predret10(args.external_clean_csv, out_dir)

    smrt_can, smrt_ik = load_smrt_overlap_sets(args.smrt_train_csv, args.smrt_test_csv)
    meta["smrt_exact_overlap"] = (
        meta["canonical_smiles"].isin(smrt_can) | meta["inchikey"].isin(smrt_ik)
    ).astype(int)

    save_csv(meta, out_dir / "external_predret10_stage4_meta.csv")

    print("\nExternal rows:", len(meta))
    print("Datasets:", meta["dataset_name"].nunique())
    print("SMRT exact overlap rows:", int(meta["smrt_exact_overlap"].sum()))

    cache_dir = out_dir / "cache"
    origin_root = cache_dir / "predret10_origin"
    taut_root = cache_dir / "predret10_taut"

    if args.force_reprocess:
        for p in [origin_root, taut_root]:
            if p.exists():
                print("[REMOVE CACHE]", p)
                shutil.rmtree(p)

    print("\n=== Build external Complex datasets ===")
    origin_dataset = SMRTComplexDataset(
        root=str(origin_root),
        csv_path=str(origin_csv),
        max_ring_size=args.max_ring_size,
        use_edge_features=True,
    )
    taut_dataset = SMRTComplexDataset(
        root=str(taut_root),
        csv_path=str(taut_csv),
        max_ring_size=args.max_ring_size,
        use_edge_features=True,
    )

    if len(origin_dataset) != len(meta):
        raise RuntimeError(f"origin external dataset length mismatch: {len(origin_dataset)} vs {len(meta)}")
    if len(taut_dataset) != len(meta):
        raise RuntimeError(f"taut external dataset length mismatch: {len(taut_dataset)} vs {len(meta)}")

    print("origin_dataset:", len(origin_dataset))
    print("taut_dataset:", len(taut_dataset))

    print("\n=== Predict external RT with all checkpoints ===")
    pred_all_seed, stacker_summary = predict_all_runs(
        args=args,
        meta=meta,
        origin_dataset=origin_dataset,
        taut_dataset=taut_dataset,
        device=device,
    )

    save_csv(pred_all_seed, out_dir / "external_predictions_per_seed.csv")
    save_csv(stacker_summary, out_dir / "external_stacker_oof_summary.csv")

    pred_5seed = add_five_seed_mean(pred_all_seed)
    save_csv(pred_5seed, out_dir / "external_predictions_5seed_mean.csv")

    pred_all = pd.concat([pred_all_seed, pred_5seed], ignore_index=True)
    save_csv(pred_all, out_dir / "external_predictions_all_with_5seed_mean.csv")

    print("\n=== External 5-fold linear calibration ===")
    calib_metrics, calib_fold_metrics = run_calibration(pred_all, args)
    save_csv(calib_metrics, out_dir / "external_calibration_metrics.csv")
    save_csv(calib_fold_metrics, out_dir / "external_calibration_fold_metrics.csv")

    calib_summary = summarize_across_seeds(calib_metrics)
    save_csv(calib_summary, out_dir / "external_calibration_summary_5seed.csv")

    print("\n=== Quick view: calibrated no-overlap summary ===")
    sel = calib_summary[
        (calib_summary["eval_mode"] == "linear_calibrated") &
        (calib_summary["scope"] == "no_smrt_exact_overlap") &
        (calib_summary["method"].isin(["origin", "taut", "mean", "huber", "selected_final"]))
    ].copy()

    cols = [
        "dataset_name", "method", "n_mean", "mae_mean", "mae_std",
        "rmse_mean", "r2_mean", "spearman_mean", "pearson_mean",
    ]
    cols = [c for c in cols if c in sel.columns]
    print(sel[cols].sort_values(["dataset_name", "mae_mean"]).to_string(index=False))

    print("\n=== Quick view: 5seed_mean calibrated no-overlap ===")
    q = calib_metrics[
        (calib_metrics["seed"] == "5seed_mean") &
        (calib_metrics["eval_mode"] == "linear_calibrated") &
        (calib_metrics["scope"] == "no_smrt_exact_overlap")
    ].copy()
    cols2 = ["dataset_name", "method", "n", "mae", "rmse", "r2", "spearman", "pearson", "bias"]
    cols2 = [c for c in cols2 if c in q.columns]
    print(q[cols2].sort_values(["dataset_name", "mae"]).to_string(index=False))

    write_readme(out_dir)

    print("\n✅ Done. Outputs are in:", out_dir)


if __name__ == "__main__":
    main()

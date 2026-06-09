import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn.exceptions import ConvergenceWarning

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


DEFAULT_RUNS = [
    ("seed1", "results_OOF_DualView_Stack_v1"),
    ("seed79", "results_OOF_DualView_Stack_seed79"),
    ("seed123", "results_OOF_DualView_Stack_seed123"),
    ("seed256", "results_OOF_DualView_Stack_seed256"),
    ("seed5", "results_OOF_DualView_Stack_seed5"),
]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_csv(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path} shape={df.shape}")


def metric_dict(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = np.abs(y - p)
    err = p - y

    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rel = e / (np.abs(y) + 1e-8) * 100.0

    return {
        "n": int(len(y)),
        "mae": float(np.mean(e)),
        "mre": float(np.mean(rel)),
        "medae": float(np.median(e)),
        "medre": float(np.median(rel)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2": float(r2),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "gt80": int((e > 80).sum()),
        "gt100": int((e > 100).sum()),
        "gt150": int((e > 150).sum()),
        "gt200": int((e > 200).sum()),
        "bias": float(np.mean(err)),
    }


def summarize_mean_std(df, group_cols, metric_cols):
    rows = []
    grouped = df.groupby(group_cols, dropna=False)

    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        if "run" in sub.columns:
            row["num_runs"] = int(sub["run"].nunique())
        else:
            row["num_rows"] = int(len(sub))

        for c in metric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def build_stack_features(origin_pred, taut_pred, changed):
    origin_pred = np.asarray(origin_pred, dtype=np.float64)
    taut_pred = np.asarray(taut_pred, dtype=np.float64)
    changed = np.asarray(changed, dtype=np.float64)

    diff = np.abs(origin_pred - taut_pred)
    mean_pred = 0.5 * (origin_pred + taut_pred)
    min_pred = np.minimum(origin_pred, taut_pred)
    max_pred = np.maximum(origin_pred, taut_pred)

    x = np.vstack([
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

    return x


def disagreement_fusion(origin_pred, taut_pred, alpha, tau, temperature):
    origin_pred = np.asarray(origin_pred, dtype=np.float64)
    taut_pred = np.asarray(taut_pred, dtype=np.float64)

    diff = np.abs(origin_pred - taut_pred)
    soft_use = 1.0 / (1.0 + np.exp(-((diff - tau) / temperature)))
    mixed = alpha * origin_pred + (1.0 - alpha) * taut_pred
    final = (1.0 - soft_use) * origin_pred + soft_use * mixed
    return final


def fit_candidate_stackers(oof_df, test_df, huber_alpha=1e-4, stack_temperature=5.0):
    y_oof = oof_df["Actual_RT"].values.astype(np.float64)

    o_oof = oof_df["Origin_OOF_Pred"].values.astype(np.float64)
    t_oof = oof_df["Taut_OOF_Pred"].values.astype(np.float64)
    c_oof = oof_df["Taut_Changed"].values.astype(np.float64)

    o_test = test_df["Origin_Test_Pred"].values.astype(np.float64)
    t_test = test_df["Taut_Test_Pred"].values.astype(np.float64)
    c_test = test_df["Taut_Changed"].values.astype(np.float64)

    candidates = {
        "origin_only": {
            "oof_pred": o_oof,
            "test_pred": o_test,
            "params": {},
        },
        "taut_only": {
            "oof_pred": t_oof,
            "test_pred": t_test,
            "params": {},
        },
        "mean_origin_taut": {
            "oof_pred": 0.5 * (o_oof + t_oof),
            "test_pred": 0.5 * (o_test + t_test),
            "params": {},
        },
    }

    alpha_grid = np.linspace(0.0, 1.0, 101)
    tau_grid = np.array([0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0])

    best = None
    for tau in tau_grid:
        for alpha in alpha_grid:
            p_oof = disagreement_fusion(
                o_oof,
                t_oof,
                alpha=alpha,
                tau=tau,
                temperature=stack_temperature,
            )
            m = metric_dict(y_oof, p_oof)
            if best is None or m["mae"] < best["mae"]:
                best = {
                    "mae": m["mae"],
                    "alpha": float(alpha),
                    "tau": float(tau),
                    "oof_pred": p_oof,
                    "test_pred": disagreement_fusion(
                        o_test,
                        t_test,
                        alpha=alpha,
                        tau=tau,
                        temperature=stack_temperature,
                    ),
                }

    candidates["oof_selected_fixed_gate"] = {
        "oof_pred": best["oof_pred"],
        "test_pred": best["test_pred"],
        "params": {
            "alpha": best["alpha"],
            "tau": best["tau"],
            "temperature": stack_temperature,
        },
    }

    x_oof = build_stack_features(o_oof, t_oof, c_oof)
    x_test = build_stack_features(o_test, t_test, c_test)

    ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.array([1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0])),
    )
    ridge.fit(x_oof, y_oof)
    candidates["ridge_stack"] = {
        "oof_pred": ridge.predict(x_oof),
        "test_pred": ridge.predict(x_test),
        "params": {"model": "StandardScaler+RidgeCV"},
    }

    huber = make_pipeline(
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=huber_alpha, max_iter=1000),
    )
    huber.fit(x_oof, y_oof)
    candidates["huber_stack"] = {
        "oof_pred": huber.predict(x_oof),
        "test_pred": huber.predict(x_test),
        "params": {"model": "StandardScaler+HuberRegressor", "alpha": huber_alpha},
    }

    return candidates


def required_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Existing columns: {df.columns.tolist()}")


def load_run(run_name, run_dir):
    run_dir = Path(run_dir)

    oof_path = run_dir / "oof_predictions.csv"
    test_path = run_dir / "test_predictions.csv"

    if not oof_path.exists():
        raise FileNotFoundError(oof_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    oof = pd.read_csv(oof_path)
    test = pd.read_csv(test_path)

    required_cols(
        oof,
        ["Actual_RT", "Origin_OOF_Pred", "Taut_OOF_Pred", "Taut_Changed"],
        f"{run_name}/oof_predictions.csv",
    )
    required_cols(
        test,
        ["Actual_RT", "Origin_Test_Pred", "Taut_Test_Pred", "Final_Pred", "Taut_Changed"],
        f"{run_name}/test_predictions.csv",
    )

    if len(oof) != 70182:
        print(f"[WARN] {run_name} oof rows = {len(oof)}, expected 70182")
    if len(test) != 7798:
        print(f"[WARN] {run_name} test rows = {len(test)}, expected 7798")

    return oof, test


def collect_ablation(run_specs, args):
    rows = []
    candidate_cache = {}

    for run_name, run_dir in run_specs:
        print(f"\n[ABLATION] {run_name} {run_dir}")
        oof, test = load_run(run_name, run_dir)
        candidates = fit_candidate_stackers(
            oof,
            test,
            huber_alpha=args.huber_alpha,
            stack_temperature=args.stack_temperature,
        )
        candidate_cache[run_name] = (oof, test, candidates)

        y_oof = oof["Actual_RT"].values
        y_test = test["Actual_RT"].values

        for method, item in candidates.items():
            row_oof = {
                "run": run_name,
                "split": "oof",
                "method": method,
                **metric_dict(y_oof, item["oof_pred"]),
            }
            row_test = {
                "run": run_name,
                "split": "test",
                "method": method,
                **metric_dict(y_test, item["test_pred"]),
            }

            params = item.get("params", {})
            row_oof["params_json"] = json.dumps(params, ensure_ascii=False)
            row_test["params_json"] = json.dumps(params, ensure_ascii=False)

            rows.append(row_oof)
            rows.append(row_test)

    return pd.DataFrame(rows), candidate_cache


def collect_subgroup(candidate_cache):
    rows = []
    methods = ["origin_only", "taut_only", "mean_origin_taut", "huber_stack"]

    for run_name, (oof, test, candidates) in candidate_cache.items():
        y = test["Actual_RT"].values.astype(np.float64)
        changed = test["Taut_Changed"].values.astype(np.float64)

        for group_value in [0.0, 1.0]:
            mask = changed == group_value
            group_name = f"Taut_Changed={int(group_value)}"

            for method in methods:
                p = candidates[method]["test_pred"]
                rows.append({
                    "run": run_name,
                    "group": group_name,
                    "method": method,
                    **metric_dict(y[mask], p[mask]),
                })

    subgroup = pd.DataFrame(rows)

    gain_rows = []
    for (run, group), sub in subgroup.groupby(["run", "group"]):
        def get_mae(method):
            return float(sub[sub["method"] == method]["mae"].iloc[0])

        origin_mae = get_mae("origin_only")
        taut_mae = get_mae("taut_only")
        mean_mae = get_mae("mean_origin_taut")
        final_mae = get_mae("huber_stack")
        n = int(sub[sub["method"] == "huber_stack"]["n"].iloc[0])

        gain_rows.append({
            "run": run,
            "group": group,
            "n": n,
            "origin_mae": origin_mae,
            "taut_mae": taut_mae,
            "mean_mae": mean_mae,
            "final_mae": final_mae,
            "gain_final_vs_origin": origin_mae - final_mae,
            "gain_final_vs_taut": taut_mae - final_mae,
            "gain_final_vs_mean": mean_mae - final_mae,
        })

    gains = pd.DataFrame(gain_rows)
    return subgroup, gains


def fit_huber_predict(oof_origin, oof_taut, oof_changed, y_oof, test_origin, test_taut, test_changed, huber_alpha):
    x_oof = build_stack_features(oof_origin, oof_taut, oof_changed)
    x_test = build_stack_features(test_origin, test_taut, test_changed)

    model = make_pipeline(
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=huber_alpha, max_iter=1000),
    )
    model.fit(x_oof, y_oof)
    return model.predict(x_test)


def collect_shuffle(candidate_cache, args):
    rows = []

    for run_i, (run_name, (oof, test, candidates)) in enumerate(candidate_cache.items()):
        print(f"\n[SHUFFLE] {run_name}")

        y_oof = oof["Actual_RT"].values.astype(np.float64)
        y_test = test["Actual_RT"].values.astype(np.float64)

        o_oof = oof["Origin_OOF_Pred"].values.astype(np.float64)
        t_oof = oof["Taut_OOF_Pred"].values.astype(np.float64)
        c_oof = oof["Taut_Changed"].values.astype(np.float64)

        o_test = test["Origin_Test_Pred"].values.astype(np.float64)
        t_test = test["Taut_Test_Pred"].values.astype(np.float64)
        c_test = test["Taut_Changed"].values.astype(np.float64)

        paired_pred = candidates["huber_stack"]["test_pred"]
        paired_m = metric_dict(y_test, paired_pred)

        rows.append({
            "run": run_name,
            "repeat": -1,
            "setting": "paired_huber_stack",
            "paired_mae_reference": paired_m["mae"],
            "delta_mae_vs_paired": 0.0,
            **paired_m,
        })

        for rep in range(args.shuffle_repeats):
            rng = np.random.default_rng(args.shuffle_seed + run_i * 10000 + rep)

            perm_oof = rng.permutation(len(oof))
            perm_test = rng.permutation(len(test))

            # Setting A: break tautomer prediction pairing, keep original Taut_Changed flag.
            pred_a = fit_huber_predict(
                oof_origin=o_oof,
                oof_taut=t_oof[perm_oof],
                oof_changed=c_oof,
                y_oof=y_oof,
                test_origin=o_test,
                test_taut=t_test[perm_test],
                test_changed=c_test,
                huber_alpha=args.huber_alpha,
            )
            m_a = metric_dict(y_test, pred_a)
            rows.append({
                "run": run_name,
                "repeat": rep,
                "setting": "shuffle_taut_pred_only",
                "paired_mae_reference": paired_m["mae"],
                "delta_mae_vs_paired": m_a["mae"] - paired_m["mae"],
                **m_a,
            })

            # Setting B: break tautomer prediction and Taut_Changed flag together.
            pred_b = fit_huber_predict(
                oof_origin=o_oof,
                oof_taut=t_oof[perm_oof],
                oof_changed=c_oof[perm_oof],
                y_oof=y_oof,
                test_origin=o_test,
                test_taut=t_test[perm_test],
                test_changed=c_test[perm_test],
                huber_alpha=args.huber_alpha,
            )
            m_b = metric_dict(y_test, pred_b)
            rows.append({
                "run": run_name,
                "repeat": rep,
                "setting": "shuffle_taut_pred_and_flag",
                "paired_mae_reference": paired_m["mae"],
                "delta_mae_vs_paired": m_b["mae"] - paired_m["mae"],
                **m_b,
            })

    return pd.DataFrame(rows)


def find_smiles_col(df):
    for c in ["SMILES", "Orig_SMILES", "smiles", "orig_smile", "smile"]:
        if c in df.columns:
            return c
    return None


SMARTS = {
    "amide": "C(=O)N",
    "sulfonamide": "S(=O)(=O)N",
    "urea": "NC(=O)N",
    "piperazine": "N1CCNCC1",
    "morpholine": "O1CCNCC1",
    "imidazole_like": "c1ncc[nH]1",
}


def safe_mol(smiles):
    try:
        m = Chem.MolFromSmiles(str(smiles))
        return m
    except Exception:
        return None


def mol_features(smiles):
    m = safe_mol(smiles)
    if m is None:
        return {
            "rdkit_valid": 0,
            "molwt": np.nan,
            "logp": np.nan,
            "tpsa": np.nan,
            "hba": np.nan,
            "hbd": np.nan,
            "rot_bonds": np.nan,
            "heavy_atoms": np.nan,
            "hetero_atoms": np.nan,
            "hetero_ratio": np.nan,
            "halogen_atoms": np.nan,
            "ring_count": np.nan,
            "aromatic_ring_count": np.nan,
            "hetero_aromatic_ring_count": np.nan,
            **{f"flag_{k}": np.nan for k in SMARTS},
        }

    atoms = list(m.GetAtoms())
    heavy = m.GetNumHeavyAtoms()
    hetero = sum(1 for a in atoms if a.GetAtomicNum() not in [1, 6])
    halogen = sum(1 for a in atoms if a.GetAtomicNum() in [9, 17, 35, 53])

    ri = m.GetRingInfo()
    atom_rings = list(ri.AtomRings())
    aromatic_ring_count = 0
    hetero_aromatic_ring_count = 0

    for ring in atom_rings:
        ring_atoms = [m.GetAtomWithIdx(i) for i in ring]
        is_arom = all(a.GetIsAromatic() for a in ring_atoms)
        if is_arom:
            aromatic_ring_count += 1
            if any(a.GetAtomicNum() not in [6, 1] for a in ring_atoms):
                hetero_aromatic_ring_count += 1

    out = {
        "rdkit_valid": 1,
        "molwt": float(Descriptors.MolWt(m)),
        "logp": float(Crippen.MolLogP(m)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(m)),
        "hba": float(rdMolDescriptors.CalcNumHBA(m)),
        "hbd": float(rdMolDescriptors.CalcNumHBD(m)),
        "rot_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(m)),
        "heavy_atoms": float(heavy),
        "hetero_atoms": float(hetero),
        "hetero_ratio": float(hetero / max(heavy, 1)),
        "halogen_atoms": float(halogen),
        "ring_count": float(ri.NumRings()),
        "aromatic_ring_count": float(aromatic_ring_count),
        "hetero_aromatic_ring_count": float(hetero_aromatic_ring_count),
    }

    for name, smarts in SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        out[f"flag_{name}"] = int(m.HasSubstructMatch(patt)) if patt is not None else 0

    return out


def collect_hard_molecules(candidate_cache, args):
    run_items = list(candidate_cache.items())
    first_run, (_, first_test, _) = run_items[0]

    base = first_test.copy()
    smiles_col = find_smiles_col(base)

    if smiles_col is None:
        print("[WARN] no SMILES column found in test_predictions.csv. Hard molecule feature analysis will be limited.")
        base["SMILES_FOR_FEATURES"] = ""
    else:
        base["SMILES_FOR_FEATURES"] = base[smiles_col].astype(str)

    origin_preds = []
    taut_preds = []
    final_preds = []

    for run_name, (oof, test, candidates) in run_items:
        if len(test) != len(base):
            raise RuntimeError(f"{run_name}: test row count mismatch")
        if np.max(np.abs(test["Actual_RT"].values - base["Actual_RT"].values)) > 1e-6:
            raise RuntimeError(f"{run_name}: Actual_RT order mismatch")

        origin_preds.append(candidates["origin_only"]["test_pred"])
        taut_preds.append(candidates["taut_only"]["test_pred"])
        final_preds.append(candidates["huber_stack"]["test_pred"])

    y = base["Actual_RT"].values.astype(np.float64)

    base["Mean_Origin_Pred_5seed"] = np.mean(np.vstack(origin_preds), axis=0)
    base["Mean_Taut_Pred_5seed"] = np.mean(np.vstack(taut_preds), axis=0)
    base["Mean_Final_Pred_5seed"] = np.mean(np.vstack(final_preds), axis=0)

    base["AbsErr_Origin"] = np.abs(y - base["Mean_Origin_Pred_5seed"].values)
    base["AbsErr_Taut"] = np.abs(y - base["Mean_Taut_Pred_5seed"].values)
    base["AbsErr_Final"] = np.abs(y - base["Mean_Final_Pred_5seed"].values)

    base["Improvement_vs_Origin"] = base["AbsErr_Origin"] - base["AbsErr_Final"]
    base["BestSingle_AbsErr"] = np.minimum(base["AbsErr_Origin"], base["AbsErr_Taut"])
    base["Harm_vs_BestSingle"] = base["AbsErr_Final"] - base["BestSingle_AbsErr"]

    feature_rows = []
    for smi in base["SMILES_FOR_FEATURES"].tolist():
        feature_rows.append(mol_features(smi))

    feat_df = pd.DataFrame(feature_rows)
    full = pd.concat([base.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    top_k = args.top_k

    worst = full.sort_values("AbsErr_Final", ascending=False).head(top_k).copy()
    worst.insert(0, "rank", np.arange(1, len(worst) + 1))
    worst.insert(0, "list_type", "worst_final")

    improved = full.sort_values("Improvement_vs_Origin", ascending=False).head(top_k).copy()
    improved.insert(0, "rank", np.arange(1, len(improved) + 1))
    improved.insert(0, "list_type", "improved_vs_origin")

    harmed = full.sort_values("Harm_vs_BestSingle", ascending=False).head(top_k).copy()
    harmed.insert(0, "rank", np.arange(1, len(harmed) + 1))
    harmed.insert(0, "list_type", "harmed_vs_best_single")

    combined = pd.concat([worst, improved, harmed], ignore_index=True)

    feature_cols = [
        "Taut_Changed",
        "AbsErr_Origin",
        "AbsErr_Taut",
        "AbsErr_Final",
        "Improvement_vs_Origin",
        "Harm_vs_BestSingle",
        "molwt",
        "logp",
        "tpsa",
        "hba",
        "hbd",
        "rot_bonds",
        "heavy_atoms",
        "hetero_atoms",
        "hetero_ratio",
        "halogen_atoms",
        "ring_count",
        "aromatic_ring_count",
        "hetero_aromatic_ring_count",
        "flag_amide",
        "flag_sulfonamide",
        "flag_urea",
        "flag_piperazine",
        "flag_morpholine",
        "flag_imidazole_like",
    ]

    summary_rows = []
    for list_type, sub in combined.groupby("list_type"):
        row = {
            "list_type": list_type,
            "n": int(len(sub)),
        }
        for c in feature_cols:
            if c in sub.columns:
                row[f"{c}_mean"] = float(pd.to_numeric(sub[c], errors="coerce").mean())
                row[f"{c}_median"] = float(pd.to_numeric(sub[c], errors="coerce").median())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    return combined, summary, full


def write_output_readme(out_dir):
    text = """TCDV-TopoRT paper analysis outputs

Main CSV files:
- dualview_ablation_5seed.csv
- dualview_ablation_5seed_summary.csv
- taut_changed_subgroup_5seed.csv
- taut_changed_subgroup_5seed_summary.csv
- taut_changed_gain_5seed.csv
- taut_changed_gain_5seed_summary.csv
- shuffle_tautomer_ablation.csv
- shuffle_tautomer_ablation_summary.csv
- tail_error_summary_5seed.csv
- top_hard_molecules_5seed_mean.csv
- hard_molecule_feature_summary.csv
- test_predictions_5seed_mean_with_features.csv

Notes:
- No GNN retraining is performed.
- Stacker candidates are refit from OOF predictions only.
- Shuffle ablation breaks the sample-wise tautomer pairing and refits the Huber stacker on shuffled OOF features.
- Hard molecule analysis uses mean predictions across the five fixed seeds.
"""
    path = Path(out_dir) / "README_outputs.txt"
    path.write_text(text, encoding="utf-8")
    print(f"[SAVE] {path}")


def parse_runs(s):
    if s is None or s.strip() == "":
        return DEFAULT_RUNS

    runs = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Custom runs must be formatted as name:path,name:path")
        name, path = item.split(":", 1)
        runs.append((name.strip(), path.strip()))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="paper_analysis")
    ap.add_argument("--shuffle_repeats", type=int, default=20)
    ap.add_argument("--shuffle_seed", type=int, default=20260609)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--huber_alpha", type=float, default=1e-4)
    ap.add_argument("--stack_temperature", type=float, default=5.0)
    ap.add_argument(
        "--runs",
        default="",
        help="Optional custom run list formatted as name:path,name:path. Default uses the five final runs.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    run_specs = parse_runs(args.runs)

    print("=== TCDV-TopoRT paper analysis ===")
    print("out_dir:", out_dir)
    print("shuffle_repeats:", args.shuffle_repeats)
    print("top_k:", args.top_k)
    print("runs:")
    for name, path in run_specs:
        print(" ", name, "->", path)

    ablation, candidate_cache = collect_ablation(run_specs, args)

    metric_cols = [
        "mae", "mre", "medae", "medre", "rmse", "r2",
        "p90", "p95", "p99", "gt80", "gt100", "gt150", "gt200", "bias",
    ]

    save_csv(ablation, out_dir / "dualview_ablation_5seed.csv")

    ablation_summary = summarize_mean_std(
        ablation[ablation["split"] == "test"],
        group_cols=["method"],
        metric_cols=metric_cols,
    )
    save_csv(ablation_summary, out_dir / "dualview_ablation_5seed_summary.csv")

    tail_methods = ["origin_only", "taut_only", "mean_origin_taut", "huber_stack"]
    tail = ablation[
        (ablation["split"] == "test") &
        (ablation["method"].isin(tail_methods))
    ].copy()
    tail_summary = summarize_mean_std(
        tail,
        group_cols=["method"],
        metric_cols=["p90", "p95", "p99", "gt80", "gt100", "gt150", "gt200", "mae", "rmse", "bias"],
    )
    save_csv(tail, out_dir / "tail_error_summary_5seed.csv")
    save_csv(tail_summary, out_dir / "tail_error_summary_5seed_mean_std.csv")

    subgroup, subgroup_gains = collect_subgroup(candidate_cache)
    save_csv(subgroup, out_dir / "taut_changed_subgroup_5seed.csv")

    subgroup_summary = summarize_mean_std(
        subgroup,
        group_cols=["group", "method"],
        metric_cols=metric_cols,
    )
    save_csv(subgroup_summary, out_dir / "taut_changed_subgroup_5seed_summary.csv")

    save_csv(subgroup_gains, out_dir / "taut_changed_gain_5seed.csv")

    gain_summary = summarize_mean_std(
        subgroup_gains,
        group_cols=["group"],
        metric_cols=["origin_mae", "taut_mae", "mean_mae", "final_mae", "gain_final_vs_origin", "gain_final_vs_taut", "gain_final_vs_mean"],
    )
    save_csv(gain_summary, out_dir / "taut_changed_gain_5seed_summary.csv")

    shuffle_df = collect_shuffle(candidate_cache, args)
    save_csv(shuffle_df, out_dir / "shuffle_tautomer_ablation.csv")

    shuffle_summary = summarize_mean_std(
        shuffle_df,
        group_cols=["setting"],
        metric_cols=["mae", "mre", "medae", "rmse", "r2", "p95", "p99", "gt100", "gt200", "delta_mae_vs_paired"],
    )
    save_csv(shuffle_summary, out_dir / "shuffle_tautomer_ablation_summary.csv")

    hard_combined, hard_summary, full_test_features = collect_hard_molecules(candidate_cache, args)
    save_csv(hard_combined, out_dir / "top_hard_molecules_5seed_mean.csv")
    save_csv(hard_summary, out_dir / "hard_molecule_feature_summary.csv")
    save_csv(full_test_features, out_dir / "test_predictions_5seed_mean_with_features.csv")

    write_output_readme(out_dir)

    print("\n=== Quick check: final ablation summary ===")
    show_cols = ["method", "mae_mean", "mae_std", "medae_mean", "rmse_mean", "r2_mean", "p95_mean", "p99_mean", "gt100_mean", "gt200_mean"]
    print(ablation_summary[show_cols].to_string(index=False))

    print("\n=== Quick check: tautomer changed gains ===")
    print(gain_summary.to_string(index=False))

    print("\n=== Quick check: shuffle summary ===")
    show_cols = ["setting", "mae_mean", "mae_std", "delta_mae_vs_paired_mean", "delta_mae_vs_paired_std"]
    print(shuffle_summary[show_cols].to_string(index=False))

    print("\n✅ Done. All analysis files are in:", out_dir)


if __name__ == "__main__":
    main()

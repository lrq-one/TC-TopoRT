from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr


OUT = Path("paper_analysis_stage4Z_oof_chem_residual_calibration")
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "Eawag_XBridgeC18_364": {
        "target": 45.30,
        "base_candidate": "deep_cwn_last2",
        "candidates": {
            "old_zscore": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_Eawag_XBridgeC18_364_src0",
            "deep_cwn_last1": "paper_analysis_stage4N_Eawag_deep_cwn_last1_lr5e5_src0",
            "deep_cwn_last2": "paper_analysis_stage4N_Eawag_deep_cwn_last2_lr3e5_src0",
            "tail_weighted": "paper_analysis_stage4X_Eawag_cwnlast2_tailw15_lr5e5_cwn03_wd5e4_src0",
        },
    },
    "FEM_long_412": {
        "target": 87.16,
        "base_candidate": "zscore_rtfull",
        "candidates": {
            "zscore_rtfull": "paper_analysis_stage4I_tcdv_tl_zscore_testbest_FEM_long_412_src0",
            "deep_cwn_last1": "paper_analysis_stage4N_FEMlong_deep_cwn_last1_lr5e5_src0",
            "deep_cwn_last2": "paper_analysis_stage4N_FEMlong_deep_cwn_last2_lr3e5_src0",
        },
    },
}

ELEMENTS = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
ALPHAS = np.logspace(-2, 4, 13)


def safe_corr(fn, y, p):
    try:
        v = fn(y, p)
        if hasattr(v, "correlation"):
            v = v.correlation
        elif isinstance(v, tuple):
            v = v[0]
        if pd.isna(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def metric_row(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "spearman": safe_corr(spearmanr, y, p),
        "pearson": safe_corr(pearsonr, y, p),
        "bias": float(np.mean(p - y)),
    }


def parse_formula(formula):
    formula = "" if pd.isna(formula) else str(formula)
    out = {e: 0 for e in ELEMENTS}
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if elem in out:
            out[elem] += int(num) if num else 1
    return out


def rdkit_desc(smiles):
    vals = {
        "MolWt": np.nan,
        "MolLogP": np.nan,
        "TPSA": np.nan,
        "HBD": np.nan,
        "HBA": np.nan,
        "RotB": np.nan,
        "RingCount": np.nan,
        "AromaticRings": np.nan,
        "AliphaticRings": np.nan,
        "HeavyAtomCount": np.nan,
        "FractionCSP3": np.nan,
        "HalogenCount": np.nan,
    }
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

        mol = Chem.MolFromSmiles("" if pd.isna(smiles) else str(smiles))
        if mol is None:
            return vals

        vals["MolWt"] = Descriptors.MolWt(mol)
        vals["MolLogP"] = Crippen.MolLogP(mol)
        vals["TPSA"] = rdMolDescriptors.CalcTPSA(mol)
        vals["HBD"] = rdMolDescriptors.CalcNumHBD(mol)
        vals["HBA"] = rdMolDescriptors.CalcNumHBA(mol)
        vals["RotB"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        vals["RingCount"] = rdMolDescriptors.CalcNumRings(mol)
        vals["AromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        vals["AliphaticRings"] = rdMolDescriptors.CalcNumAliphaticRings(mol)
        vals["HeavyAtomCount"] = mol.GetNumHeavyAtoms()
        vals["FractionCSP3"] = rdMolDescriptors.CalcFractionCSP3(mol)
        vals["HalogenCount"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53])
        return vals
    except Exception:
        return vals


def add_chem_features(df):
    out = df.copy()

    smiles_col = None
    for c in ["origin_smiles", "smiles", "canonical_smiles"]:
        if c in out.columns:
            smiles_col = c
            break

    if smiles_col is None:
        smiles = pd.Series([""] * len(out))
    else:
        smiles = out[smiles_col].fillna("").astype(str)

    desc = pd.DataFrame([rdkit_desc(s) for s in smiles])
    for c in desc.columns:
        out[f"desc_{c}"] = desc[c].values

    if "formula" in out.columns:
        frows = [parse_formula(x) for x in out["formula"]]
    else:
        frows = [parse_formula("") for _ in range(len(out))]

    fdf = pd.DataFrame(frows)
    for e in ELEMENTS:
        out[f"n_{e}"] = fdf[e].values

    out["heavy_formula"] = out[[f"n_{e}" for e in ELEMENTS if e != "H"]].sum(axis=1)
    out["hetero_formula"] = out[[f"n_{e}" for e in ["N", "O", "S", "P", "F", "Cl", "Br", "I"]]].sum(axis=1)
    out["O_over_C"] = out["n_O"] / np.maximum(out["n_C"], 1)
    out["N_over_C"] = out["n_N"] / np.maximum(out["n_C"], 1)
    out["hetero_over_C"] = out["hetero_formula"] / np.maximum(out["n_C"], 1)

    return out


def load_candidate(dataset_name, cand_name, cand_dir):
    p = Path(cand_dir) / "external_tl_predictions.csv"
    if not p.exists():
        print("[SKIP missing]", dataset_name, cand_name, p)
        return None

    df = pd.read_csv(p)
    if "dataset_name" in df.columns:
        df = df[df["dataset_name"] == dataset_name].copy()

    if len(df) == 0:
        print("[SKIP empty]", dataset_name, cand_name)
        return None

    need = ["stage4_index", "rt", "origin_tl_pred", "taut_tl_pred", "mean_tl_pred"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print("[SKIP bad cols]", dataset_name, cand_name, miss)
        return None

    keep_meta = [
        "dataset_name", "stage4_index", "record_id", "name", "formula",
        "origin_smiles", "taut_smiles", "inchikey", "rt"
    ]
    keep_meta = [c for c in keep_meta if c in df.columns]

    small = df[keep_meta + ["origin_tl_pred", "taut_tl_pred", "mean_tl_pred"]].copy()
    small = small.rename(columns={
        "origin_tl_pred": f"{cand_name}__origin",
        "taut_tl_pred": f"{cand_name}__taut",
        "mean_tl_pred": f"{cand_name}__mean",
    })
    small[f"{cand_name}__gap"] = np.abs(small[f"{cand_name}__origin"] - small[f"{cand_name}__taut"])
    small[f"{cand_name}__minview"] = np.minimum(small[f"{cand_name}__origin"], small[f"{cand_name}__taut"])
    small[f"{cand_name}__maxview"] = np.maximum(small[f"{cand_name}__origin"], small[f"{cand_name}__taut"])
    return small


def build_table(dataset_name, cfg):
    base = None
    loaded = []

    for cand_name, cand_dir in cfg["candidates"].items():
        small = load_candidate(dataset_name, cand_name, cand_dir)
        if small is None:
            continue

        loaded.append(cand_name)

        if base is None:
            meta_cols = [
                "dataset_name", "stage4_index", "record_id", "name", "formula",
                "origin_smiles", "taut_smiles", "inchikey", "rt"
            ]
            meta_cols = [c for c in meta_cols if c in small.columns]
            pred_cols = [c for c in small.columns if c.startswith(cand_name + "__")]
            base = small[meta_cols + pred_cols].copy()
        else:
            pred_cols = ["stage4_index"] + [c for c in small.columns if c.startswith(cand_name + "__")]
            base = base.merge(small[pred_cols], on="stage4_index", how="left")

    if base is None:
        raise RuntimeError(f"No predictions found for {dataset_name}")

    base = add_chem_features(base)
    return base, loaded


def oof_residual_predict(df, base_col, feature_cols):
    y = df["rt"].values.astype(float)
    base_pred = df[base_col].values.astype(float)
    X = df[feature_cols].values.astype(float)

    pred = np.full(len(df), np.nan, dtype=float)
    kf = KFold(n_splits=10, shuffle=True, random_state=1)

    for tr, te in kf.split(X):
        residual = y[tr] - base_pred[tr]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ])

        model.fit(X[tr], residual)
        pred[te] = base_pred[te] + model.predict(X[te])

    return pred


def main():
    rows = []
    pred_outputs = []

    for dataset_name, cfg in DATASETS.items():
        df, loaded = build_table(dataset_name, cfg)
        target = float(cfg["target"])
        base_cand = cfg["base_candidate"]
        base_col = f"{base_cand}__mean"

        if base_col not in df.columns:
            raise RuntimeError(f"{dataset_name}: base column missing: {base_col}. loaded={loaded}")

        y = df["rt"].values.astype(float)

        pred_feature_cols = [
            c for c in df.columns
            if "__origin" in c or "__taut" in c or "__mean" in c or "__gap" in c or "__minview" in c or "__maxview" in c
        ]
        chem_feature_cols = [
            c for c in df.columns
            if c.startswith("desc_") or c.startswith("n_") or c in [
                "heavy_formula", "hetero_formula", "O_over_C", "N_over_C", "hetero_over_C"
            ]
        ]
        feature_cols = pred_feature_cols + chem_feature_cols

        # baselines for every available candidate mean
        for cand in loaded:
            col = f"{cand}__mean"
            if col in df.columns:
                m = metric_row(y, df[col].values)
                rows.append({
                    "dataset_name": dataset_name,
                    "method": f"{cand}__mean_none",
                    "target_abcort": target,
                    "n": len(df),
                    "n_features": 0,
                    **m,
                    "improvement_vs_abcort": target - m["mae"],
                })

        cal_pred = oof_residual_predict(df, base_col, feature_cols)
        m = metric_row(y, cal_pred)
        rows.append({
            "dataset_name": dataset_name,
            "method": "chem_ridge_residual_allcand",
            "target_abcort": target,
            "n": len(df),
            "n_features": len(feature_cols),
            **m,
            "improvement_vs_abcort": target - m["mae"],
        })

        out = df[["stage4_index", "rt"]].copy()
        for c in ["dataset_name", "record_id", "name", "formula", "origin_smiles"]:
            if c in df.columns:
                out[c] = df[c]
        out["base_col"] = base_col
        out["base_pred"] = df[base_col].values
        out["chem_resid_pred"] = cal_pred
        out["base_abs_err"] = np.abs(df[base_col].values - y)
        out["chem_resid_abs_err"] = np.abs(cal_pred - y)
        pred_outputs.append(out)

    res = pd.DataFrame(rows).sort_values(["dataset_name", "mae"]).reset_index(drop=True)
    res.to_csv(OUT / "external_oof_chem_residual_metrics.csv", index=False)

    pd.concat(pred_outputs, ignore_index=True).to_csv(
        OUT / "external_oof_chem_residual_predictions.csv",
        index=False,
    )

    print("\n=== CHEM RESIDUAL RESULTS ===")
    for ds, sub in res.groupby("dataset_name"):
        print("\n" + ds)
        print(sub.to_string(index=False))

    print("\n[SAVE]", OUT / "external_oof_chem_residual_metrics.csv")
    print("[SAVE]", OUT / "external_oof_chem_residual_predictions.csv")


if __name__ == "__main__":
    main()

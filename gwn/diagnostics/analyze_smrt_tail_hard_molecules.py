from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

RUN_DIRS = [
    "results_OOF_DualView_Stack_v1",
    "results_OOF_DualView_Stack_seed5",
    "results_OOF_DualView_Stack_seed79",
    "results_OOF_DualView_Stack_seed123",
    "results_OOF_DualView_Stack_seed256",
]

OUT = Path("final_smrt_results")
OUT.mkdir(exist_ok=True)

THRESHOLDS = [25, 50, 75, 80, 100, 150, 200, 300, 500]

def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = np.abs(y - p)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    out = {
        "n": int(len(y)),
        "mae": float(np.mean(e)),
        "medae": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "bias": float(np.mean(p - y)),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "max_abs_err": float(np.max(e)),
    }
    for t in THRESHOLDS:
        out[f"err_gt_{t}"] = int((e > t).sum())
    return out

def mol_descriptors(smiles):
    d = {
        "mol_valid": 0,
        "MolWt": np.nan,
        "MolLogP": np.nan,
        "TPSA": np.nan,
        "HeavyAtomCount": np.nan,
        "NumRotatableBonds": np.nan,
        "RingCount": np.nan,
        "NumAromaticRings": np.nan,
        "NumHAcceptors": np.nan,
        "NumHDonors": np.nan,
        "FractionCSP3": np.nan,
    }
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return d

    d["mol_valid"] = 1
    d["MolWt"] = float(Descriptors.MolWt(mol))
    d["MolLogP"] = float(Crippen.MolLogP(mol))
    d["TPSA"] = float(rdMolDescriptors.CalcTPSA(mol))
    d["HeavyAtomCount"] = int(mol.GetNumHeavyAtoms())
    d["NumRotatableBonds"] = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    d["RingCount"] = int(rdMolDescriptors.CalcNumRings(mol))
    d["NumAromaticRings"] = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    d["NumHAcceptors"] = int(rdMolDescriptors.CalcNumHBA(mol))
    d["NumHDonors"] = int(rdMolDescriptors.CalcNumHBD(mol))
    d["FractionCSP3"] = float(rdMolDescriptors.CalcFractionCSP3(mol))
    return d

all_pred_rows = []
tail_metric_rows = []

for run_dir in RUN_DIRS:
    p = Path(run_dir) / "test_predictions.csv"
    print("\n" + "=" * 100)
    print("[RUN]", run_dir)
    print("[FILE]", p, "exists=", p.exists())

    if not p.exists():
        continue

    df = pd.read_csv(p).copy()
    df["run_dir"] = run_dir
    df["row_id"] = np.arange(len(df))

    required = ["Actual_RT", "Origin_Test_Pred", "Taut_Test_Pred"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"{p} missing column: {c}")

    y = df["Actual_RT"].values.astype(np.float64)
    origin = df["Origin_Test_Pred"].values.astype(np.float64)
    taut = df["Taut_Test_Pred"].values.astype(np.float64)
    mean_pred = 0.5 * (origin + taut)

    if "Final_Pred" in df.columns:
        final = df["Final_Pred"].values.astype(np.float64)
    else:
        raise RuntimeError(f"{p} missing Final_Pred")

    pred_map = {
        "Origin only": origin,
        "Tautomer only": taut,
        "Mean fusion": mean_pred,
        "Final stack": final,
    }

    for method, pred in pred_map.items():
        m = metrics(y, pred)
        tail_metric_rows.append({
            "run_dir": run_dir,
            "method": method,
            **m,
        })

    keep_cols = [
        "run_dir", "row_id", "Actual_RT",
        "Origin_Test_Pred", "Taut_Test_Pred", "Final_Pred",
    ]
    for c in ["Source_Index", "SMILES", "Orig_SMILES", "Taut_SMILES", "Taut_Changed"]:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    sub = df[keep_cols].copy()
    sub["Mean_Fusion_Pred"] = mean_pred

    sub["Origin_AbsErr"] = np.abs(sub["Actual_RT"].values - sub["Origin_Test_Pred"].values)
    sub["Taut_AbsErr"] = np.abs(sub["Actual_RT"].values - sub["Taut_Test_Pred"].values)
    sub["Mean_AbsErr"] = np.abs(sub["Actual_RT"].values - sub["Mean_Fusion_Pred"].values)
    sub["Final_AbsErr"] = np.abs(sub["Actual_RT"].values - sub["Final_Pred"].values)

    sub["Gain_Final_vs_Origin"] = sub["Origin_AbsErr"] - sub["Final_AbsErr"]
    sub["Gain_Final_vs_Taut"] = sub["Taut_AbsErr"] - sub["Final_AbsErr"]
    sub["Gain_Final_vs_Mean"] = sub["Mean_AbsErr"] - sub["Final_AbsErr"]

    all_pred_rows.append(sub)

pred_all = pd.concat(all_pred_rows, ignore_index=True)
tail_by_run = pd.DataFrame(tail_metric_rows)

pred_all.to_csv(OUT / "smrt_tail_all_seed_predictions_long.csv", index=False)
tail_by_run.to_csv(OUT / "smrt_tail_metrics_by_run.csv", index=False)

# Summary of tail metrics by method across seeds.
summary_rows = []
for method, sub in tail_by_run.groupby("method"):
    row = {
        "method": method,
        "runs": len(sub),
    }
    metric_cols = ["mae", "medae", "rmse", "r2", "bias", "p90", "p95", "p99", "max_abs_err"]
    metric_cols += [f"err_gt_{t}" for t in THRESHOLDS]
    for c in metric_cols:
        vals = pd.to_numeric(sub[c], errors="coerce").dropna()
        row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
        row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    summary_rows.append(row)

tail_summary = pd.DataFrame(summary_rows).sort_values("mae_mean")
tail_summary.to_csv(OUT / "smrt_tail_metrics_summary_mean_std.csv", index=False)

# Consensus molecule-level table across five seeds.
id_cols = ["row_id"]
for c in ["Source_Index", "SMILES", "Orig_SMILES", "Taut_SMILES", "Taut_Changed"]:
    if c in pred_all.columns:
        id_cols.append(c)

agg_map = {
    "Actual_RT": "mean",
    "Origin_Test_Pred": ["mean", "std"],
    "Taut_Test_Pred": ["mean", "std"],
    "Mean_Fusion_Pred": ["mean", "std"],
    "Final_Pred": ["mean", "std"],
    "Origin_AbsErr": ["mean", "std"],
    "Taut_AbsErr": ["mean", "std"],
    "Mean_AbsErr": ["mean", "std"],
    "Final_AbsErr": ["mean", "std", "max"],
    "Gain_Final_vs_Origin": ["mean", "std"],
    "Gain_Final_vs_Taut": ["mean", "std"],
    "Gain_Final_vs_Mean": ["mean", "std"],
}

cons = pred_all.groupby(id_cols, dropna=False).agg(agg_map).reset_index()
cons.columns = [
    "_".join([str(x) for x in col if str(x) != ""]) if isinstance(col, tuple) else str(col)
    for col in cons.columns
]

# Rename common columns.
rename = {
    "Actual_RT_mean": "Actual_RT",
    "Origin_Test_Pred_mean": "Origin_Pred_mean",
    "Origin_Test_Pred_std": "Origin_Pred_std",
    "Taut_Test_Pred_mean": "Taut_Pred_mean",
    "Taut_Test_Pred_std": "Taut_Pred_std",
    "Mean_Fusion_Pred_mean": "Mean_Pred_mean",
    "Mean_Fusion_Pred_std": "Mean_Pred_std",
    "Final_Pred_mean": "Final_Pred_mean",
    "Final_Pred_std": "Final_Pred_std",
}
cons = cons.rename(columns=rename)

# Tail frequency across seeds.
tmp = pred_all.copy()
for t in THRESHOLDS:
    tmp[f"Final_gt_{t}"] = (tmp["Final_AbsErr"] > t).astype(int)
    tmp[f"Origin_gt_{t}"] = (tmp["Origin_AbsErr"] > t).astype(int)

freq = tmp.groupby("row_id").agg({
    **{f"Final_gt_{t}": "sum" for t in THRESHOLDS},
    **{f"Origin_gt_{t}": "sum" for t in THRESHOLDS},
}).reset_index()

cons = cons.merge(freq, on="row_id", how="left")

# Add RDKit descriptors once.
smiles_col = "SMILES" if "SMILES" in cons.columns else None
if smiles_col is not None:
    desc_rows = []
    for smi in cons[smiles_col].astype(str).values:
        desc_rows.append(mol_descriptors(smi))
    desc = pd.DataFrame(desc_rows)
    cons = pd.concat([cons.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)

cons = cons.sort_values("Final_AbsErr_mean", ascending=False)
cons.to_csv(OUT / "smrt_hard_molecule_consensus_all.csv", index=False)

hard_top = cons.head(200).copy()
hard_top.to_csv(OUT / "smrt_hard_molecule_top200_by_final_error.csv", index=False)

improved = cons.sort_values("Gain_Final_vs_Origin_mean", ascending=False).head(200).copy()
improved.to_csv(OUT / "smrt_top200_improved_vs_origin.csv", index=False)

harmed = cons.sort_values("Gain_Final_vs_Origin_mean", ascending=True).head(200).copy()
harmed.to_csv(OUT / "smrt_top200_harmed_vs_origin.csv", index=False)

# Descriptor summary: hard final gt100 in all/most seeds.
desc_cols = [
    "MolWt", "MolLogP", "TPSA", "HeavyAtomCount",
    "NumRotatableBonds", "RingCount", "NumAromaticRings",
    "NumHAcceptors", "NumHDonors", "FractionCSP3",
]

group_rows = []
if all(c in cons.columns for c in desc_cols):
    groups = {
        "all_molecules": np.ones(len(cons), dtype=bool),
        "final_gt100_at_least_1_seed": cons["Final_gt_100"] >= 1,
        "final_gt100_all_5_seeds": cons["Final_gt_100"] >= 5,
        "final_gt200_at_least_1_seed": cons["Final_gt_200"] >= 1,
        "top200_final_error": cons.index.isin(hard_top.index),
        "top200_improved_vs_origin": cons.index.isin(improved.index),
        "top200_harmed_vs_origin": cons.index.isin(harmed.index),
    }

    for gname, mask in groups.items():
        sub = cons.loc[mask].copy()
        row = {
            "group": gname,
            "n": int(mask.sum()),
            "Final_AbsErr_mean": float(sub["Final_AbsErr_mean"].mean()) if len(sub) else np.nan,
            "Origin_AbsErr_mean": float(sub["Origin_AbsErr_mean"].mean()) if len(sub) else np.nan,
            "Gain_Final_vs_Origin_mean": float(sub["Gain_Final_vs_Origin_mean"].mean()) if len(sub) else np.nan,
        }
        for c in desc_cols:
            row[f"{c}_mean"] = float(sub[c].mean()) if len(sub) else np.nan
            row[f"{c}_median"] = float(sub[c].median()) if len(sub) else np.nan
        if "Taut_Changed" in sub.columns:
            row["Taut_Changed_rate"] = float(pd.to_numeric(sub["Taut_Changed"], errors="coerce").mean())
        group_rows.append(row)

desc_summary = pd.DataFrame(group_rows)
desc_summary.to_csv(OUT / "smrt_hard_molecule_descriptor_group_summary.csv", index=False)

print("\n=== Tail metrics summary ===")
show_cols = ["method", "runs", "mae_mean", "mae_std", "p95_mean", "p99_mean"]
show_cols += [f"err_gt_{t}_mean" for t in [50, 80, 100, 150, 200, 300, 500]]
print(tail_summary[show_cols].to_string(index=False))

print("\n=== Hard molecule descriptor group summary ===")
if len(desc_summary):
    print(desc_summary.to_string(index=False))
else:
    print("[WARN] descriptor summary empty")

print("\n=== Top 30 hard molecules by final mean abs error ===")
cols = [
    "row_id", "Actual_RT", "Final_AbsErr_mean", "Origin_AbsErr_mean",
    "Taut_AbsErr_mean", "Mean_AbsErr_mean", "Gain_Final_vs_Origin_mean",
    "Final_gt_100", "Final_gt_200",
]
for c in ["SMILES", "Taut_Changed", "MolWt", "TPSA", "RingCount", "NumAromaticRings", "NumRotatableBonds"]:
    if c in cons.columns:
        cols.append(c)
print(cons[cols].head(30).to_string(index=False))

print("\n=== Top 20 improved vs origin ===")
print(improved[cols].head(20).to_string(index=False))

print("\n=== Top 20 harmed vs origin ===")
print(harmed[cols].head(20).to_string(index=False))

print("\n[SAVE]", OUT / "smrt_tail_metrics_by_run.csv")
print("[SAVE]", OUT / "smrt_tail_metrics_summary_mean_std.csv")
print("[SAVE]", OUT / "smrt_hard_molecule_consensus_all.csv")
print("[SAVE]", OUT / "smrt_hard_molecule_top200_by_final_error.csv")
print("[SAVE]", OUT / "smrt_top200_improved_vs_origin.csv")
print("[SAVE]", OUT / "smrt_top200_harmed_vs_origin.csv")
print("[SAVE]", OUT / "smrt_hard_molecule_descriptor_group_summary.csv")

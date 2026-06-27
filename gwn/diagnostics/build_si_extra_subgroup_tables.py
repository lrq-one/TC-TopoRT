from pathlib import Path
import re
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except Exception as e:
    raise SystemExit(f"[ERROR] RDKit import failed: {e}")

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = ROOT / "gwn/final_paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED_DIRS = [
    ROOT / "gwn/results_OOF_DualView_Stack_v1",
    ROOT / "gwn/results_OOF_DualView_Stack_seed5",
    ROOT / "gwn/results_OOF_DualView_Stack_seed79",
    ROOT / "gwn/results_OOF_DualView_Stack_seed123",
    ROOT / "gwn/results_OOF_DualView_Stack_seed256",
]

FULL_SEED5 = ROOT / "gwn/results_OOF_DualView_Stack_seed5/test_predictions.csv"
NO2CELL_SEED5 = ROOT / "ablations/gwn_cwn_structural_ablation/results_Ablation_No2Cell_DualView_Stack_seed5/test_predictions.csv"


def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())


def find_col(df, candidates=None, contains_all=None, contains_any=None, exclude_any=None):
    candidates = candidates or []
    contains_all = contains_all or []
    contains_any = contains_any or []
    exclude_any = exclude_any or []

    norm_map = {norm_col(c): c for c in df.columns}

    for cand in candidates:
        key = norm_col(cand)
        if key in norm_map:
            return norm_map[key]

    for c in df.columns:
        k = norm_col(c)
        if contains_all and not all(x in k for x in contains_all):
            continue
        if contains_any and not any(x in k for x in contains_any):
            continue
        if exclude_any and any(x in k for x in exclude_any):
            continue
        return c

    return None


def detect_prediction_columns(df):
    y_col = find_col(
        df,
        candidates=["rt", "RT", "true_rt", "label", "y", "target", "experimental_rt", "rt_sec"],
        contains_any=["rt"],
        exclude_any=["pred", "err", "abs", "delta"],
    )
    changed_col = find_col(
        df,
        candidates=["taut_changed", "is_taut_changed", "tautomer_changed", "changed"],
        contains_all=["taut", "chang"],
    )

    origin_pred_col = find_col(
        df,
        candidates=["pred_origin", "origin_pred", "origin_prediction", "y_origin_pred", "ori_pred"],
        contains_all=["origin", "pred"],
        exclude_any=["err", "abs", "delta"],
    )
    if origin_pred_col is None:
        origin_pred_col = find_col(
            df,
            contains_all=["ori", "pred"],
            exclude_any=["err", "abs", "delta"],
        )

    taut_pred_col = find_col(
        df,
        candidates=["pred_taut", "taut_pred", "tautomer_pred", "taut_prediction", "y_taut_pred"],
        contains_all=["taut", "pred"],
        exclude_any=["err", "abs", "delta", "chang"],
    )

    fusion_pred_col = find_col(
        df,
        candidates=[
            "pred_stack", "stack_pred", "huber_pred", "pred_huber",
            "final_pred", "pred_final", "y_pred", "stacked_pred", "fusion_pred",
        ],
        contains_any=["stack", "fusion", "final"],
        exclude_any=["err", "abs", "delta"],
    )

    if fusion_pred_col is None:
        # fallback: use the last numeric prediction-like column, excluding base views and errors
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        pred_like = []
        for c in numeric_cols:
            k = norm_col(c)
            if any(x in k for x in ["err", "abs", "delta", "rt", "label", "fold", "changed"]):
                continue
            if c not in {origin_pred_col, taut_pred_col}:
                pred_like.append(c)
        if pred_like:
            fusion_pred_col = pred_like[-1]

    missing = {
        "rt": y_col,
        "taut_changed": changed_col,
        "origin_pred": origin_pred_col,
        "tautomer_pred": taut_pred_col,
        "fusion_pred": fusion_pred_col,
    }
    bad = [k for k, v in missing.items() if v is None]
    if bad:
        print("[DEBUG] columns:", df.columns.tolist())
        raise ValueError(f"Could not detect required columns: {bad}")

    return y_col, changed_col, origin_pred_col, taut_pred_col, fusion_pred_col


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))))


def make_mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


def fmt_mean_std(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def build_table_s18():
    rows_numeric = []

    for seed_dir in SEED_DIRS:
        p = seed_dir / "test_predictions.csv"
        if not p.exists():
            raise FileNotFoundError(p)

        df = pd.read_csv(p)
        y_col, changed_col, origin_col, taut_col, fusion_col = detect_prediction_columns(df)

        changed = pd.to_numeric(df[changed_col], errors="coerce").fillna(0).astype(int)

        groups = [
            ("tautomer-changed molecules", changed == 1),
            ("unchanged molecules", changed == 0),
            ("all molecules", np.ones(len(df), dtype=bool)),
        ]

        for group, mask in groups:
            sub = df.loc[mask]
            rows_numeric.append({
                "seed_dir": seed_dir.name,
                "group": group,
                "N": int(len(sub)),
                "Origin MAE (s)": mae(sub[y_col], sub[origin_col]),
                "Tautomer MAE (s)": mae(sub[y_col], sub[taut_col]),
                "Fusion MAE (s)": mae(sub[y_col], sub[fusion_col]),
            })

    detail = pd.DataFrame(rows_numeric)

    compact_rows = []
    for group, g in detail.groupby("group", sort=False):
        n_values = sorted(g["N"].unique().tolist())
        n_display = n_values[0] if len(n_values) == 1 else ";".join(map(str, n_values))

        origin_m, origin_s = make_mean_std(g["Origin MAE (s)"])
        taut_m, taut_s = make_mean_std(g["Tautomer MAE (s)"])
        fusion_m, fusion_s = make_mean_std(g["Fusion MAE (s)"])

        compact_rows.append({
            "Group": group,
            "N": n_display,
            "Origin MAE (s)": fmt_mean_std(origin_m, origin_s),
            "Tautomer MAE (s)": fmt_mean_std(taut_m, taut_s),
            "Fusion MAE (s)": fmt_mean_std(fusion_m, fusion_s),
            "Origin MAE mean": origin_m,
            "Origin MAE std": origin_s,
            "Tautomer MAE mean": taut_m,
            "Tautomer MAE std": taut_s,
            "Fusion MAE mean": fusion_m,
            "Fusion MAE std": fusion_s,
        })

    compact = pd.DataFrame(compact_rows)

    detail_path = OUT_DIR / "Table_S18_tautomer_changed_vs_unchanged_subgroup_detail.csv"
    compact_path = OUT_DIR / "Table_S18_tautomer_changed_vs_unchanged_subgroup.csv"

    detail.to_csv(detail_path, index=False)
    compact.to_csv(compact_path, index=False)

    print("\n===== Table S18 compact =====")
    print(compact[["Group", "N", "Origin MAE (s)", "Tautomer MAE (s)", "Fusion MAE (s)"]].to_string(index=False))
    print("[WROTE]", compact_path)
    print("[WROTE]", detail_path)


def find_smiles_col(df):
    for c in ["smiles", "SMILES", "origin_smiles", "original_smiles", "mol_smiles"]:
        if c in df.columns:
            return c
    smiles_cols = [c for c in df.columns if "smiles" in c.lower()]
    if not smiles_cols:
        raise ValueError(f"No SMILES column found: {df.columns.tolist()}")

    # prefer non-tautomer column
    for c in smiles_cols:
        if "taut" not in c.lower():
            return c
    return smiles_cols[0]


def detect_y_and_fusion(df):
    y_col = find_col(
        df,
        candidates=["rt", "RT", "true_rt", "label", "y", "target", "experimental_rt", "rt_sec"],
        contains_any=["rt"],
        exclude_any=["pred", "err", "abs", "delta"],
    )

    fusion_col = find_col(
        df,
        candidates=[
            "pred_stack", "stack_pred", "huber_pred", "pred_huber",
            "final_pred", "pred_final", "y_pred", "stacked_pred", "fusion_pred",
        ],
        contains_any=["stack", "fusion", "final"],
        exclude_any=["err", "abs", "delta"],
    )

    if fusion_col is None:
        # choose last prediction-like numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        pred_like = []
        for c in numeric_cols:
            k = norm_col(c)
            if any(x in k for x in ["err", "abs", "delta", "fold", "changed"]):
                continue
            if c != y_col:
                pred_like.append(c)
        if pred_like:
            fusion_col = pred_like[-1]

    if y_col is None or fusion_col is None:
        print("[DEBUG] columns:", df.columns.tolist())
        raise ValueError("Could not detect y/fusion prediction columns.")

    return y_col, fusion_col


def ring_flags(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {
            "valid": False,
            "acyclic molecules": False,
            "ring-containing molecules": False,
            "aromatic-ring molecules": False,
            "heterocycle-containing molecules": False,
            "multi-ring molecules": False,
        }

    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())
    n_rings = len(atom_rings)

    aromatic_ring = False
    heterocycle = False
    for ring in atom_rings:
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if all(a.GetIsAromatic() for a in atoms):
            aromatic_ring = True
        if any(a.GetAtomicNum() != 6 for a in atoms):
            heterocycle = True

    return {
        "valid": True,
        "acyclic molecules": n_rings == 0,
        "ring-containing molecules": n_rings > 0,
        "aromatic-ring molecules": aromatic_ring,
        "heterocycle-containing molecules": heterocycle,
        "multi-ring molecules": n_rings >= 2,
    }


def build_table_s19():
    if not FULL_SEED5.exists():
        raise FileNotFoundError(FULL_SEED5)
    if not NO2CELL_SEED5.exists():
        raise FileNotFoundError(NO2CELL_SEED5)

    full_df = pd.read_csv(FULL_SEED5)
    no2_df = pd.read_csv(NO2CELL_SEED5)

    full_y, full_pred = detect_y_and_fusion(full_df)
    no2_y, no2_pred = detect_y_and_fusion(no2_df)
    smiles_col = find_smiles_col(full_df)

    if len(full_df) != len(no2_df):
        raise ValueError(f"Row count mismatch: full={len(full_df)}, no2cell={len(no2_df)}")

    # Check RT alignment.
    rt_diff = np.nanmax(np.abs(pd.to_numeric(full_df[full_y]) - pd.to_numeric(no2_df[no2_y])))
    if rt_diff > 1e-6:
        raise ValueError(f"RT mismatch between full and no2cell predictions, max diff={rt_diff}")

    flags = [ring_flags(s) for s in full_df[smiles_col].tolist()]
    flag_df = pd.DataFrame(flags)

    groups = [
        "acyclic molecules",
        "ring-containing molecules",
        "aromatic-ring molecules",
        "heterocycle-containing molecules",
        "multi-ring molecules",
    ]

    rows = []
    y = pd.to_numeric(full_df[full_y], errors="coerce")
    full_p = pd.to_numeric(full_df[full_pred], errors="coerce")
    no2_p = pd.to_numeric(no2_df[no2_pred], errors="coerce")

    for group in groups:
        mask = flag_df[group].to_numpy(dtype=bool)
        n = int(mask.sum())
        full_mae = mae(y[mask], full_p[mask]) if n else np.nan
        no2_mae = mae(y[mask], no2_p[mask]) if n else np.nan
        rows.append({
            "Group": group,
            "N": n,
            "Full MAE (s)": full_mae,
            "w/o ring 2-cells MAE (s)": no2_mae,
            "Delta MAE (s)": no2_mae - full_mae,
        })

    table = pd.DataFrame(rows)

    compact = table.copy()
    for c in ["Full MAE (s)", "w/o ring 2-cells MAE (s)", "Delta MAE (s)"]:
        compact[c] = compact[c].map(lambda x: f"{x:.3f}")

    path = OUT_DIR / "Table_S19_ring_context_subgroup.csv"
    numeric_path = OUT_DIR / "Table_S19_ring_context_subgroup_numeric.csv"

    compact.to_csv(path, index=False)
    table.to_csv(numeric_path, index=False)

    print("\n===== Table S19 compact =====")
    print(compact.to_string(index=False))
    print("[WROTE]", path)
    print("[WROTE]", numeric_path)


def main():
    build_table_s18()
    build_table_s19()


if __name__ == "__main__":
    main()

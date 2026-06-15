from pathlib import Path
import json
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

OUT = Path("final_smrt_results")
OUT.mkdir(exist_ok=True)

RUN_DIRS = [
    "results_OOF_DualView_Stack_v1",
    "results_OOF_DualView_Stack_seed5",
    "results_OOF_DualView_Stack_seed79",
    "results_OOF_DualView_Stack_seed123",
    "results_OOF_DualView_Stack_seed256",
]

DATA_PATHS = {
    "origin_train": "data/SMRT_train.csv",
    "origin_test": "data/SMRT_test.csv",
    "taut_train": "data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv",
    "taut_test": "data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv",
}

EXTERNAL_MANIFEST = Path("configs/external_table2_final_manifest.csv")
EXTERNAL_FINAL_TABLE = Path("final_external_results/table2_final_from_manifest.csv")

checks = []

def add_check(section, name, status, observed="", expected="", note=""):
    checks.append({
        "section": section,
        "check": name,
        "status": status,
        "observed": str(observed),
        "expected": str(expected),
        "note": str(note),
    })

def parse_flag_value(x):
    if isinstance(x, (bool, np.bool_)):
        return float(x)
    if pd.isna(x):
        return 0.0
    s = str(x).strip().lower()
    if s in ["1", "true", "yes", "y"]:
        return 1.0
    if s in ["0", "false", "no", "n", "none", "nan", ""]:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

def safe_inchikey(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return ""
        return Chem.MolToInchiKey(mol)
    except Exception:
        return ""

def load_valid_meta(csv_path):
    p = Path(csv_path)
    if not p.exists():
        add_check("data", f"{csv_path} exists", "FAIL", False, True, "missing csv")
        return pd.DataFrame()

    df = pd.read_csv(p, engine="python")
    df.columns = [str(c).lower().strip() for c in df.columns]

    if "smile" in df.columns and "smiles" not in df.columns:
        df.rename(columns={"smile": "smiles"}, inplace=True)

    if "smiles" not in df.columns or "rt" not in df.columns:
        add_check("data", f"{csv_path} required columns", "FAIL", list(df.columns), "smiles/rt", "")
        return pd.DataFrame()

    df["rt"] = df["rt"].astype(float)
    df = df[df["rt"] > 300.0].copy()

    rows = []
    invalid = 0
    for source_idx, row in df.iterrows():
        smi = str(row["smiles"])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid += 1
            continue

        item = {
            "Source_Index": int(source_idx),
            "SMILES": smi,
            "Actual_RT": float(row["rt"]),
            "InChIKey": safe_inchikey(smi),
        }

        if "orig_smile" in df.columns:
            item["Orig_SMILES"] = str(row["orig_smile"])
        elif "orig_smiles" in df.columns:
            item["Orig_SMILES"] = str(row["orig_smiles"])
        else:
            item["Orig_SMILES"] = smi

        for c in ["raw_changed", "real_changed", "formula_same", "heavy_same", "fallback", "reason"]:
            if c in df.columns:
                item[c] = row[c]

        rows.append(item)

    meta = pd.DataFrame(rows)

    if len(meta) > 0:
        if "real_changed" in meta.columns:
            meta["Taut_Changed"] = meta["real_changed"].apply(parse_flag_value).astype(float)
        elif "raw_changed" in meta.columns:
            meta["Taut_Changed"] = meta["raw_changed"].apply(parse_flag_value).astype(float)
        else:
            meta["Taut_Changed"] = 0.0

    add_check("data", f"{csv_path} valid rows", "PASS", len(meta), "nonzero", f"invalid_mol={invalid}")
    return meta.reset_index(drop=True)

def max_abs_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return np.inf
    return float(np.max(np.abs(a - b)))

def finite_check(df, cols):
    bad_cols = []
    for c in cols:
        if c not in df.columns:
            bad_cols.append(f"{c}:missing")
            continue
        arr = pd.to_numeric(df[c], errors="coerce").values
        if not np.isfinite(arr).all():
            bad_cols.append(f"{c}:nan_or_inf")
    return bad_cols

def audit_pairing():
    metas = {k: load_valid_meta(v) for k, v in DATA_PATHS.items()}

    ot = metas["origin_train"]
    tt = metas["taut_train"]
    oe = metas["origin_test"]
    te = metas["taut_test"]

    for split, a, b in [
        ("train", ot, tt),
        ("test", oe, te),
    ]:
        add_check(
            "pairing",
            f"{split} origin/taut row count",
            "PASS" if len(a) == len(b) and len(a) > 0 else "FAIL",
            f"origin={len(a)}, taut={len(b)}",
            "equal and nonzero",
            "",
        )

        rt_diff = max_abs_diff(a.get("Actual_RT", []), b.get("Actual_RT", []))
        add_check(
            "pairing",
            f"{split} origin/taut RT max diff",
            "PASS" if rt_diff <= 1e-6 else "FAIL",
            rt_diff,
            "<=1e-6",
            "",
        )

        if "Source_Index" in a.columns and "Source_Index" in b.columns and len(a) == len(b):
            same_source = bool((a["Source_Index"].values == b["Source_Index"].values).all())
            add_check(
                "pairing",
                f"{split} Source_Index order",
                "PASS" if same_source else "FAIL",
                same_source,
                True,
                "",
            )

        if "Orig_SMILES" in b.columns and len(a) == len(b):
            match_orig = int((a["SMILES"].astype(str).values == b["Orig_SMILES"].astype(str).values).sum())
            add_check(
                "pairing",
                f"{split} taut Orig_SMILES matches origin SMILES",
                "PASS" if match_orig == len(a) else "WARN",
                f"{match_orig}/{len(a)}",
                "all rows",
                "WARN only because taut file may not store Orig_SMILES exactly",
            )

        if "Taut_Changed" in b.columns:
            changed_n = int((b["Taut_Changed"].values >= 0.5).sum())
            add_check(
                "pairing",
                f"{split} taut changed count",
                "PASS",
                changed_n,
                "recorded",
                f"rate={changed_n / max(len(b), 1):.4f}",
            )

    # Train/test split overlap audit.
    if len(ot) and len(oe):
        train_smiles = set(ot["SMILES"].astype(str))
        test_smiles = set(oe["SMILES"].astype(str))
        overlap_smiles = train_smiles.intersection(test_smiles)

        add_check(
            "split",
            "origin train/test exact SMILES overlap",
            "PASS" if len(overlap_smiles) == 0 else "WARN",
            len(overlap_smiles),
            0,
            "WARN if duplicates exist in official split",
        )

        train_keys = set([x for x in ot["InChIKey"].astype(str) if x])
        test_keys = set([x for x in oe["InChIKey"].astype(str) if x])
        overlap_keys = train_keys.intersection(test_keys)

        add_check(
            "split",
            "origin train/test InChIKey overlap",
            "PASS" if len(overlap_keys) == 0 else "WARN",
            len(overlap_keys),
            0,
            "WARN if stereochemistry/salt-normalized duplicates exist",
        )

    # Save loaded meta stats.
    meta_stats = []
    for name, df in metas.items():
        meta_stats.append({
            "name": name,
            "rows": len(df),
            "rt_min": float(df["Actual_RT"].min()) if len(df) else np.nan,
            "rt_max": float(df["Actual_RT"].max()) if len(df) else np.nan,
            "taut_changed_count": int((df["Taut_Changed"] >= 0.5).sum()) if "Taut_Changed" in df.columns and len(df) else 0,
            "taut_changed_rate": float((df["Taut_Changed"] >= 0.5).mean()) if "Taut_Changed" in df.columns and len(df) else 0.0,
        })
    pd.DataFrame(meta_stats).to_csv(OUT / "pairing_audit_meta_stats.csv", index=False)

    return metas

def audit_training_source_code():
    src = Path("train_oof_dualview_stack.py")
    if not src.exists():
        add_check("source_code", "train_oof_dualview_stack.py exists", "FAIL", False, True, "")
        return

    s = src.read_text(encoding="utf-8")

    patterns = {
        "pair_check_function_exists": "def check_pairing" in s,
        "train_pair_check_called": 'check_pairing(origin_train_meta, taut_train_meta, "TRAIN")' in s,
        "test_pair_check_called": 'check_pairing(origin_test_meta, taut_test_meta, "TEST")' in s,
        "stratified_oof_split": "StratifiedKFold" in s and "skf.split" in s,
        "selected_stacker_by_oof_mae": 'best_name = min(summary.keys(), key=lambda k: summary[k]["oof"]["mae"])' in s,
        "test_pred_after_oof_selection": 'final_test_pred = candidates[best_name]["test_pred"]' in s,
        "final_metrics_include_test_final": '"test_final": metrics(y_test, final_test_pred)' in s,
    }

    for name, ok in patterns.items():
        add_check(
            "source_code",
            name,
            "PASS" if ok else "FAIL",
            ok,
            True,
            "",
        )

def audit_seed_outputs(metas):
    train_meta = metas["origin_train"]
    test_meta = metas["origin_test"]

    for run_dir in RUN_DIRS:
        root = Path(run_dir)
        section = f"seed_output::{run_dir}"

        files = {
            "config": root / "config.json",
            "final_metrics": root / "final_metrics.json",
            "oof_base": root / "oof_base_predictions.csv",
            "oof_final": root / "oof_predictions.csv",
            "test_base": root / "test_base_predictions.csv",
            "test_final": root / "test_predictions.csv",
        }

        for name, p in files.items():
            add_check(section, f"{name} exists", "PASS" if p.exists() else "FAIL", p.exists(), True, str(p))

        if not files["oof_base"].exists() or not files["test_base"].exists() or not files["test_final"].exists():
            continue

        oof_base = pd.read_csv(files["oof_base"])
        oof_final = pd.read_csv(files["oof_final"]) if files["oof_final"].exists() else pd.DataFrame()
        test_base = pd.read_csv(files["test_base"])
        test_final = pd.read_csv(files["test_final"])

        add_check(section, "oof_base row count", "PASS" if len(oof_base) == len(train_meta) else "FAIL", len(oof_base), len(train_meta), "")
        add_check(section, "test_base row count", "PASS" if len(test_base) == len(test_meta) else "FAIL", len(test_base), len(test_meta), "")
        add_check(section, "test_final row count", "PASS" if len(test_final) == len(test_meta) else "FAIL", len(test_final), len(test_meta), "")

        if len(oof_base) == len(train_meta):
            rt_diff = max_abs_diff(oof_base["Actual_RT"].values, train_meta["Actual_RT"].values)
            add_check(section, "oof Actual_RT matches train meta", "PASS" if rt_diff <= 1e-2 else "FAIL", rt_diff, "<=1e-2", "float32 tolerance")

        if len(test_base) == len(test_meta):
            rt_diff = max_abs_diff(test_base["Actual_RT"].values, test_meta["Actual_RT"].values)
            add_check(section, "test_base Actual_RT matches test meta", "PASS" if rt_diff <= 1e-2 else "FAIL", rt_diff, "<=1e-2", "float32 tolerance")

        if len(test_final) == len(test_meta):
            rt_diff = max_abs_diff(test_final["Actual_RT"].values, test_meta["Actual_RT"].values)
            add_check(section, "test_final Actual_RT matches test meta", "PASS" if rt_diff <= 1e-2 else "FAIL", rt_diff, "<=1e-2", "float32 tolerance")

        if "Fold" in oof_base.columns:
            fold_vals = sorted(pd.unique(oof_base["Fold"]).tolist())
            no_missing_fold = bool((pd.to_numeric(oof_base["Fold"], errors="coerce") >= 0).all())
            add_check(section, "OOF fold assignment exists for every train row", "PASS" if no_missing_fold else "FAIL", no_missing_fold, True, f"folds={fold_vals}")

            fold_counts = oof_base["Fold"].value_counts().sort_index().to_dict()
            add_check(section, "OOF fold count", "PASS" if len(fold_counts) == 5 else "WARN", fold_counts, "5 folds", "")

        pred_cols_base = ["Origin_OOF_Pred", "Taut_OOF_Pred"]
        bad = finite_check(oof_base, pred_cols_base)
        add_check(section, "OOF base predictions finite", "PASS" if not bad else "FAIL", bad if bad else "all finite", "all finite", "")

        pred_cols_test = ["Origin_Test_Pred", "Taut_Test_Pred"]
        bad = finite_check(test_base, pred_cols_test)
        add_check(section, "test base predictions finite", "PASS" if not bad else "FAIL", bad if bad else "all finite", "all finite", "")

        bad = finite_check(test_final, ["Final_Pred"])
        add_check(section, "test Final_Pred finite", "PASS" if not bad else "FAIL", bad if bad else "all finite", "all finite", "")

        if len(oof_final):
            bad = finite_check(oof_final, ["Final_OOF_Pred"])
            add_check(section, "Final_OOF_Pred finite", "PASS" if not bad else "FAIL", bad if bad else "all finite", "all finite", "")

        if files["final_metrics"].exists():
            try:
                obj = json.loads(files["final_metrics"].read_text(encoding="utf-8"))
                selected = obj.get("selected_stacker", "")
                summary = obj.get("stacker_summary_oof", {})

                if isinstance(summary, dict) and len(summary):
                    maes = {}
                    for k, v in summary.items():
                        try:
                            maes[k] = float(v["oof"]["mae"])
                        except Exception:
                            pass

                    if len(maes):
                        best = min(maes, key=maes.get)
                        add_check(
                            section,
                            "selected_stacker equals min OOF MAE",
                            "PASS" if selected == best else "FAIL",
                            f"selected={selected}; best={best}; maes={maes}",
                            "selected == best_oof",
                            "",
                        )
                    else:
                        add_check(section, "stacker_summary_oof parse", "FAIL", "no maes", "maes available", "")
                else:
                    add_check(section, "stacker_summary_oof exists", "FAIL", type(summary), "dict", "")
            except Exception as e:
                add_check(section, "final_metrics parse", "FAIL", repr(e), "valid json", "")

def audit_external_table2():
    section = "external_table2"

    add_check(section, "external manifest exists", "PASS" if EXTERNAL_MANIFEST.exists() else "FAIL", EXTERNAL_MANIFEST.exists(), True, str(EXTERNAL_MANIFEST))
    add_check(section, "external final table exists", "PASS" if EXTERNAL_FINAL_TABLE.exists() else "FAIL", EXTERNAL_FINAL_TABLE.exists(), True, str(EXTERNAL_FINAL_TABLE))

    if EXTERNAL_FINAL_TABLE.exists():
        df = pd.read_csv(EXTERNAL_FINAL_TABLE)
        if "delta_vs_ABCoRT" in df.columns:
            max_delta = float(pd.to_numeric(df["delta_vs_ABCoRT"], errors="coerce").max())
            add_check(section, "all final external deltas below ABCoRT", "PASS" if max_delta < 0 else "WARN", max_delta, "<0", "WARN if any dataset not below reported ABCoRT")
        add_check(section, "external final table rows", "PASS" if len(df) == 6 else "WARN", len(df), 6, "")

    if not EXTERNAL_MANIFEST.exists():
        return

    mf = pd.read_csv(EXTERNAL_MANIFEST)
    add_check(section, "manifest rows", "PASS" if len(mf) == 6 else "WARN", len(mf), 6, "")

    external_rows = []
    for _, r in mf.iterrows():
        dataset = str(r.get("dataset", ""))
        metric_csv = Path(str(r.get("metric_csv", "")))
        method = str(r.get("select_method", ""))
        top_n = r.get("select_top_n", "")

        row_status = {
            "dataset": dataset,
            "metric_csv": str(metric_csv),
            "method": method,
            "top_n": top_n,
            "metric_csv_exists": metric_csv.exists(),
            "selected_row_found": False,
            "noleak_folds_exists": False,
            "noleak_fold_count": "",
        }

        add_check(
            section,
            f"{dataset} metric csv exists",
            "PASS" if metric_csv.exists() else "FAIL",
            metric_csv.exists(),
            True,
            str(metric_csv),
        )

        if metric_csv.exists():
            try:
                df = pd.read_csv(metric_csv)
                sub = df.copy()

                dcol = "dataset" if "dataset" in sub.columns else ("dataset_name" if "dataset_name" in sub.columns else None)
                if dcol:
                    sub = sub[sub[dcol].astype(str).eq(dataset)]
                if "method" in sub.columns and method:
                    sub = sub[sub["method"].astype(str).eq(method)]
                if "top_n" in sub.columns and pd.notna(top_n) and str(top_n).strip() != "":
                    sub = sub[sub["top_n"].astype(int).eq(int(float(top_n)))]

                row_status["selected_row_found"] = len(sub) > 0
                add_check(
                    section,
                    f"{dataset} selected metric row found",
                    "PASS" if len(sub) > 0 else "FAIL",
                    len(sub),
                    ">=1",
                    f"method={method}; top_n={top_n}",
                )

                if len(sub) > 0 and "mae" in sub.columns:
                    row_status["selected_mae"] = float(sub.sort_values("mae").iloc[0]["mae"])

            except Exception as e:
                add_check(section, f"{dataset} metric csv parse", "FAIL", repr(e), "readable", "")

        # If this is a no-leak stacking result, audit folds file.
        folds_csv = metric_csv.parent / "noleak_stacking_folds.csv"
        if folds_csv.exists():
            row_status["noleak_folds_exists"] = True
            try:
                fdf = pd.read_csv(folds_csv)
                if "fold" in fdf.columns:
                    fold_count = int(fdf["fold"].nunique())
                    row_status["noleak_fold_count"] = fold_count
                    add_check(
                        section,
                        f"{dataset} noleak fold count",
                        "PASS" if fold_count >= 5 else "WARN",
                        fold_count,
                        ">=5",
                        str(folds_csv),
                    )
                else:
                    add_check(section, f"{dataset} noleak folds column", "WARN", list(fdf.columns), "contains fold", str(folds_csv))
            except Exception as e:
                add_check(section, f"{dataset} noleak folds parse", "FAIL", repr(e), "readable", str(folds_csv))
        elif "noleak" in str(metric_csv.parent).lower():
            add_check(section, f"{dataset} noleak folds file exists", "WARN", False, True, str(folds_csv))

        external_rows.append(row_status)

    pd.DataFrame(external_rows).to_csv(OUT / "external_table2_manifest_audit.csv", index=False)

def main():
    metas = audit_pairing()
    audit_training_source_code()
    audit_seed_outputs(metas)
    audit_external_table2()

    df = pd.DataFrame(checks)
    df.to_csv(OUT / "pairing_noleak_audit_checks.csv", index=False)

    status_counts = df["status"].value_counts().to_dict()
    print("\n=== Pairing / no-leakage audit status counts ===")
    print(status_counts)

    print("\n=== FAIL checks ===")
    fail = df[df["status"].eq("FAIL")]
    if len(fail):
        print(fail.to_string(index=False))
    else:
        print("No FAIL checks.")

    print("\n=== WARN checks ===")
    warn = df[df["status"].eq("WARN")]
    if len(warn):
        print(warn.to_string(index=False))
    else:
        print("No WARN checks.")

    print("\n=== Key PASS checks sample ===")
    print(df[df["status"].eq("PASS")].head(40).to_string(index=False))

    print("\n[SAVE]", OUT / "pairing_noleak_audit_checks.csv")
    print("[SAVE]", OUT / "pairing_audit_meta_stats.csv")
    print("[SAVE]", OUT / "external_table2_manifest_audit.csv")

    if len(fail):
        raise SystemExit("[AUDIT FAILED] See final_smrt_results/pairing_noleak_audit_checks.csv")
    else:
        print("\n[AUDIT PASS] No critical leakage/pairing failures detected.")

if __name__ == "__main__":
    main()

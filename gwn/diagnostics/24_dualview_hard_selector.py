import argparse
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def norm_pred(path, pred_name):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    smi = cols.get("smiles", "SMILES")
    y = cols.get("actual_rt", None) or cols.get("y", None) or cols.get("y_true", None)
    p = cols.get("predicted_rt", None) or cols.get("y_pred", None) or cols.get("pred", None)

    if y is None or p is None:
        raise ValueError(f"bad columns in {path}: {df.columns.tolist()}")

    out = df[[smi, y, p]].copy()
    out.columns = ["SMILES", "y", pred_name]
    out["SMILES"] = out["SMILES"].astype(str)
    out["y"] = out["y"].astype(float)
    out[pred_name] = out[pred_name].astype(float)
    return out


def merge_views(base_path, aux_path, aux_name):
    b = norm_pred(base_path, "base_pred")
    a = norm_pred(aux_path, aux_name)

    df = b.merge(a[["SMILES", aux_name]], on="SMILES", how="inner")
    df["abs_err"] = (df["y"] - df["base_pred"]).abs()

    df["view_diff"] = (df["base_pred"] - df[aux_name]).abs()
    df["view_mean"] = 0.5 * (df["base_pred"] + df[aux_name])
    df["view_min"] = np.minimum(df["base_pred"], df[aux_name])
    df["view_max"] = np.maximum(df["base_pred"], df[aux_name])
    df["base_high"] = (df["base_pred"] > 1050).astype(float)
    df["base_low"] = (df["base_pred"] < 650).astype(float)

    # 简单 SMILES 风险特征
    s = df["SMILES"].astype(str)
    df["imidic_NC_O"] = s.str.contains("N=C\\(O\\)|C\\(O\\)=N|C\\(=N\\)O", regex=True).astype(float)
    df["sulfone"] = s.str.contains("S\\(=O\\)\\(=O\\)", regex=True).astype(float)
    df["cf3"] = s.str.contains("C\\(F\\)\\(F\\)F", regex=False).astype(float)
    df["halogen"] = s.str.contains("Cl|Br|F|I", regex=True).astype(float)
    df["lower_hetero_arom_count"] = s.apply(lambda x: x.count("n") + x.count("o") + x.count("s")).astype(float)
    df["smiles_len"] = s.str.len().astype(float)

    return df


def report_topk(df, score_col, name, hard_thr):
    print(f"\n=== {name} top-K by {score_col} ===")
    d = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    rows = []
    for k in [100, 200, 400, 600, 800, 1000]:
        if k > len(d):
            continue
        sub = d.head(k)
        rows.append({
            "K": k,
            "mean_abs_err": sub["abs_err"].mean(),
            "median_abs_err": sub["abs_err"].median(),
            "hard_count": int((sub["abs_err"] >= hard_thr).sum()),
            "precision": float((sub["abs_err"] >= hard_thr).mean()),
            ">100": int((sub["abs_err"] > 100).sum()),
            ">200": int((sub["abs_err"] > 200).sum()),
            "mean_view_diff": sub["view_diff"].mean(),
        })
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    return rep


def eval_score(df, score_col, hard_thr, name):
    y = (df["abs_err"].values >= hard_thr).astype(int)
    s = df[score_col].values
    print(
        f"{name} {score_col}: AUC={roc_auc_score(y, s):.4f}, "
        f"AP={average_precision_score(y, s):.4f}, hard_rate={y.mean():.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_train", required=True)
    ap.add_argument("--base_val", required=True)
    ap.add_argument("--base_test", required=True)
    ap.add_argument("--aux_train", required=True)
    ap.add_argument("--aux_val", required=True)
    ap.add_argument("--aux_test", required=True)
    ap.add_argument("--aux_name", default="aux_pred")
    ap.add_argument("--existing_3d_npz", default="data/unimol_012_hard_k2n1_np1compat.npz")
    ap.add_argument("--out_dir", default="results_TopoCellRT_CWNReplace_orig/dualview_hard_selector")
    ap.add_argument("--hard_thr", type=float, default=80.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tr = merge_views(args.base_train, args.aux_train, args.aux_name)
    va = merge_views(args.base_val, args.aux_val, args.aux_name)
    te = merge_views(args.base_test, args.aux_test, args.aux_name)

    print("rows train/val/test:", len(tr), len(va), len(te))
    print("hard counts:", int((tr.abs_err>=args.hard_thr).sum()), int((va.abs_err>=args.hard_thr).sum()), int((te.abs_err>=args.hard_thr).sum()))

    # 先看不用学习，直接 view_diff 排序是否能选 hard
    for name, df in [("VAL", va), ("TEST diagnostic", te)]:
        eval_score(df, "view_diff", args.hard_thr, name)
        report_topk(df, "view_diff", name, args.hard_thr)

    feat_cols = [
        "base_pred", args.aux_name, "view_diff", "view_mean", "view_min", "view_max",
        "base_high", "base_low",
        "imidic_NC_O", "sulfone", "cf3", "halogen",
        "lower_hetero_arom_count", "smiles_len",
    ]

    xtr = tr[feat_cols].values.astype(np.float32)
    xva = va[feat_cols].values.astype(np.float32)
    xte = te[feat_cols].values.astype(np.float32)

    ytr = (tr["abs_err"].values >= args.hard_thr).astype(int)

    clf = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(xtr, ytr)

    tr["risk_score"] = clf.predict_proba(xtr)[:, 1]
    va["risk_score"] = clf.predict_proba(xva)[:, 1]
    te["risk_score"] = clf.predict_proba(xte)[:, 1]

    for name, df in [("VAL", va), ("TEST diagnostic", te)]:
        eval_score(df, "risk_score", args.hard_thr, name)
        report_topk(df, "risk_score", name, args.hard_thr)

    # 和旧 3D oracle hard set overlap
    try:
        z = np.load(args.existing_3d_npz, allow_pickle=False)
        s3d = set(z["smiles"].astype(str).tolist())
        for k in [100, 200, 400, 600, 800, 1000]:
            sel = set(te.sort_values("risk_score", ascending=False).head(k)["SMILES"].astype(str))
            print(f"test top{k} risk overlap existing_3d:", len(sel & s3d), "/", k)
    except Exception as e:
        print("overlap skipped:", e)

    tr.to_csv(f"{args.out_dir}/train_dualview_hard_scores.csv", index=False)
    va.to_csv(f"{args.out_dir}/val_dualview_hard_scores.csv", index=False)
    te.to_csv(f"{args.out_dir}/test_dualview_hard_scores.csv", index=False)

    selected = te.sort_values("risk_score", ascending=False).head(600).copy()
    selected["split"] = "test"
    selected.to_csv(f"{args.out_dir}/test_dualview_selector_top600_for_unimol.csv", index=False)

    imp = pd.DataFrame({"feature": feat_cols, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
    imp.to_csv(f"{args.out_dir}/feature_importance.csv", index=False)
    print("\nTop feature importance:")
    print(imp.to_string(index=False))
    print("\nsaved:", args.out_dir)


if __name__ == "__main__":
    main()

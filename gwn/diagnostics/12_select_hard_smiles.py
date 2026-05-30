import argparse
import re
import pandas as pd


MOTIFS = {
    "amide_imidic": [
        r"N=C\(O\)",
        r"C\(O\)=N",
        r"NC\(=O\)",
        r"C\(=O\)N",
        r"C\(=N\)O",
        r"C\(O\)=N",
    ],
    "sulfonamide_sulfone": [
        r"S\(=O\)\(=O\)N",
        r"NS\(=O\)\(=O\)",
        r"S\(=O\)\(=O\)",
    ],
    "cf3_halogen": [
        r"C\(F\)\(F\)F",
        r"Cl",
        r"Br",
        r"I",
    ],
    "piperazine_morpholine": [
        r"N1CCN",
        r"N1CCOCC1",
        r"OCCN",
        r"CCN",
    ],
    "multi_hetero_ring": [
        r"n",
        r"o",
        r"s",
        r"nn",
        r"nc",
        r"no",
        r"ncn",
    ],
}


def norm_cols(df):
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "smiles":
            rename[c] = "SMILES"
        elif lc in ["actual_rt", "y", "y_true", "rt"]:
            rename[c] = "Actual_RT"
        elif lc in ["predicted_rt", "y_pred", "pred"]:
            rename[c] = "Predicted_RT"
        elif lc in ["abs_error", "ae", "error"]:
            rename[c] = "Abs_Error"
    return df.rename(columns=rename)


def motif_tags(smiles):
    tags = []
    for name, pats in MOTIFS.items():
        for p in pats:
            if re.search(p, smiles):
                tags.append(name)
                break
    return "|".join(tags)


def load_pred(path, split):
    df = pd.read_csv(path)
    df = norm_cols(df)

    need = ["SMILES", "Actual_RT", "Predicted_RT"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{path} missing column {c}; columns={df.columns.tolist()}")

    if "Abs_Error" not in df.columns:
        df["Abs_Error"] = (df["Actual_RT"] - df["Predicted_RT"]).abs()

    df["split"] = split
    df["motif_tags"] = df["SMILES"].astype(str).map(motif_tags)
    df["motif_risk"] = df["motif_tags"].astype(str).str.len() > 0
    return df[["split", "SMILES", "Actual_RT", "Predicted_RT", "Abs_Error", "motif_tags", "motif_risk"]]


def take_hard(df, top_k, err_thr, use_error=True):
    parts = []

    if use_error:
        a = df[df["Abs_Error"] >= err_thr].copy()
        b = df.sort_values("Abs_Error", ascending=False).head(top_k).copy()
        parts += [a, b]

    c = df[df["motif_risk"]].copy()
    parts.append(c)

    out = pd.concat(parts, axis=0).drop_duplicates("SMILES")
    out = out.sort_values(["motif_risk", "Abs_Error"], ascending=[False, False])
    if top_k > 0:
        out = out.head(top_k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pred", default="results_TopoCellRT_CWNReplace_orig/base_train_predictions.csv")
    ap.add_argument("--val_pred", default="results_TopoCellRT_CWNReplace_orig/base_val_predictions.csv")
    ap.add_argument("--test_pred", default="results_TopoCellRT_CWNReplace_orig/base_test_predictions.csv")
    ap.add_argument("--out", default="data/hard_smiles_for_unimol.csv")
    ap.add_argument("--train_top", type=int, default=1500)
    ap.add_argument("--val_top", type=int, default=600)
    ap.add_argument("--test_top", type=int, default=600)
    ap.add_argument("--err_thr", type=float, default=80.0)
    ap.add_argument("--diagnostic_use_test_error", action="store_true")
    args = ap.parse_args()

    train = load_pred(args.train_pred, "train")
    val = load_pred(args.val_pred, "val")
    test = load_pred(args.test_pred, "test")

    # train/val 可以用真实误差挖 hard，因为这是训练/调参数据
    train_h = take_hard(train, args.train_top, args.err_thr, use_error=True)
    val_h = take_hard(val, args.val_top, args.err_thr, use_error=True)

    # test 正式实验不能用 Abs_Error 选，否则泄漏；
    # 默认只用 motif_risk 选 test。想诊断 worst 能不能救，再加 --diagnostic_use_test_error。
    test_h = take_hard(
        test,
        args.test_top,
        args.err_thr,
        use_error=bool(args.diagnostic_use_test_error),
    )

    out = pd.concat([train_h, val_h, test_h], axis=0).drop_duplicates(["split", "SMILES"])
    out.to_csv(args.out, index=False)

    print("saved:", args.out)
    print("counts:")
    print(out["split"].value_counts())
    print("total:", len(out))
    print("motif risk:", int(out["motif_risk"].sum()))
    print("top examples:")
    print(out.head(10)[["split", "SMILES", "Abs_Error", "motif_tags"]].to_string(index=False))


if __name__ == "__main__":
    main()

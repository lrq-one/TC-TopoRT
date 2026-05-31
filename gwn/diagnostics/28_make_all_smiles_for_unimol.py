import argparse
import pandas as pd


def load_pred(path, split):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    smi = cols.get("smiles", "SMILES")
    out = df[[smi]].copy()
    out.columns = ["SMILES"]
    out["SMILES"] = out["SMILES"].astype(str)
    out["split"] = split
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pred", required=True)
    ap.add_argument("--val_pred", required=True)
    ap.add_argument("--test_pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train = load_pred(args.train_pred, "train")
    val = load_pred(args.val_pred, "val")
    test = load_pred(args.test_pred, "test")

    all_df = pd.concat([train, val, test], axis=0)
    all_df = all_df.drop_duplicates("SMILES")
    all_df.to_csv(args.out, index=False)

    print("train:", len(train))
    print("val:", len(val))
    print("test:", len(test))
    print("unique:", len(all_df))
    print("saved:", args.out)


if __name__ == "__main__":
    main()

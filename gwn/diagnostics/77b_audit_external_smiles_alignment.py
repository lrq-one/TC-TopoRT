import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except Exception:
    Chem = None


def canon_smiles(s):
    if pd.isna(s):
        return None
    s = str(s)
    if Chem is None:
        return s
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def mismatch_report(name, a, b, max_show=5):
    a = pd.Series(a).astype("object")
    b = pd.Series(b).astype("object")
    mism = (a.astype(str).values != b.astype(str).values)
    n_bad = int(mism.sum())
    print(f"{name}: mismatch = {n_bad} / {len(a)}")

    if n_bad > 0:
        bad_idx = np.where(mism)[0][:max_show]
        print(f"  first bad indices for {name}:")
        for i in bad_idx:
            print(f"    idx={i}")
            print(f"      A: {a.iloc[i]}")
            print(f"      B: {b.iloc[i]}")
    return n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", default="paper_analysis_stage4_external/external_predret10_stage4_meta.csv")
    ap.add_argument("--origin_csv", default="paper_analysis_stage4_external/temp_external_predret10_origin.csv")
    ap.add_argument("--taut_csv", default="paper_analysis_stage4_external/temp_external_predret10_taut.csv")
    ap.add_argument("--origin_root", default="paper_analysis_stage4_external/cache/predret10_origin")
    ap.add_argument("--taut_root", default="paper_analysis_stage4_external/cache/predret10_taut")
    ap.add_argument("--dataset", default="LIFE_old_194")
    ap.add_argument("--max_ring_size", type=int, default=6)
    ap.add_argument("--check_dataset_len", type=int, default=1)
    args = ap.parse_args()

    meta = pd.read_csv(args.meta_csv).sort_values("stage4_index").reset_index(drop=True)
    ori = pd.read_csv(args.origin_csv).reset_index(drop=True)
    tau = pd.read_csv(args.taut_csv).reset_index(drop=True)

    print("=== basic length ===")
    print("meta:", len(meta))
    print("origin_csv:", len(ori))
    print("taut_csv:", len(tau))

    if len(meta) != len(ori) or len(meta) != len(tau):
        print("[FATAL] length mismatch before graph dataset!")
    else:
        print("[OK] raw csv lengths match")

    print("\n=== dummy RT check ===")
    print("origin rt unique head:", sorted(pd.Series(ori["rt"]).unique())[:10] if "rt" in ori.columns else "no rt")
    print("taut rt unique head:", sorted(pd.Series(tau["rt"]).unique())[:10] if "rt" in tau.columns else "no rt")
    print("NOTE: rt=999 in temp csv is expected dummy RT for SMRTComplexDataset filtering.")

    print("\n=== full-table raw SMILES alignment ===")
    if "origin_smiles" in meta.columns:
        mismatch_report("origin raw: meta.origin_smiles vs origin.smile",
                        meta["origin_smiles"], ori["smile"])
    else:
        print("[WARN] meta has no origin_smiles")

    if "taut_smiles" in meta.columns:
        mismatch_report("taut raw: meta.taut_smiles vs taut.smile",
                        meta["taut_smiles"], tau["smile"])
    else:
        print("[WARN] meta has no taut_smiles")

    print("\n=== full-table RDKit canonical alignment ===")
    if Chem is None:
        print("[WARN] RDKit not available, skipped canonical check")
    else:
        meta_origin_can = [canon_smiles(x) for x in meta["origin_smiles"]]
        ori_can = [canon_smiles(x) for x in ori["smile"]]
        meta_taut_can = [canon_smiles(x) for x in meta["taut_smiles"]]
        tau_can = [canon_smiles(x) for x in tau["smile"]]

        mismatch_report("origin canonical: canon(meta.origin_smiles) vs canon(origin.smile)",
                        meta_origin_can, ori_can)
        mismatch_report("taut canonical: canon(meta.taut_smiles) vs canon(taut.smile)",
                        meta_taut_can, tau_can)

    sub = meta[meta["dataset_name"] == args.dataset].copy()
    idx = sub["stage4_index"].values.astype(int)

    print("\n=== subset check ===")
    print("dataset:", args.dataset)
    print("n:", len(sub))
    print("stage4_index min/max:", int(idx.min()), int(idx.max()))
    print("unique stage4_index:", len(np.unique(idx)))

    print("\n=== subset raw SMILES alignment ===")
    if "origin_smiles" in meta.columns:
        mismatch_report(f"{args.dataset} origin raw",
                        meta.iloc[idx]["origin_smiles"].values,
                        ori.iloc[idx]["smile"].values)
    if "taut_smiles" in meta.columns:
        mismatch_report(f"{args.dataset} taut raw",
                        meta.iloc[idx]["taut_smiles"].values,
                        tau.iloc[idx]["smile"].values)

    print("\n=== subset RDKit canonical alignment ===")
    if Chem is not None:
        mismatch_report(f"{args.dataset} origin canonical",
                        [canon_smiles(x) for x in meta.iloc[idx]["origin_smiles"]],
                        [canon_smiles(x) for x in ori.iloc[idx]["smile"]])
        mismatch_report(f"{args.dataset} taut canonical",
                        [canon_smiles(x) for x in meta.iloc[idx]["taut_smiles"]],
                        [canon_smiles(x) for x in tau.iloc[idx]["smile"]])

    if args.check_dataset_len:
        print("\n=== processed ComplexDataset length check ===")
        from mp.smrt_dataset import SMRTComplexDataset

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

        print("origin_dataset len:", len(origin_dataset), "meta len:", len(meta))
        print("taut_dataset len:", len(taut_dataset), "meta len:", len(meta))

        if len(origin_dataset) == len(meta) and len(taut_dataset) == len(meta):
            print("[OK] processed graph dataset lengths match meta")
        else:
            print("[FATAL] processed graph dataset length mismatch!")

    print("\n✅ alignment audit done")


if __name__ == "__main__":
    main()

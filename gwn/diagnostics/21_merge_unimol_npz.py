import argparse
import numpy as np


def load_npz(path):
    z = np.load(path, allow_pickle=False)

    if "smiles" not in z.files:
        raise KeyError(f"{path} has no 'smiles'. keys={z.files}")
    if "diff_token_feat" not in z.files:
        raise KeyError(f"{path} has no 'diff_token_feat'. keys={z.files}")

    smiles = z["smiles"].astype(str)
    feat = z["diff_token_feat"].astype(np.float32)

    if feat.ndim != 3:
        raise ValueError(f"{path} diff_token_feat should be [N,3,1537], got {feat.shape}")
    if len(smiles) != feat.shape[0]:
        raise ValueError(f"{path} smiles length {len(smiles)} != feat N {feat.shape[0]}")

    return smiles, feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    smi1, x1 = load_npz(args.base)
    smi2, x2 = load_npz(args.extra)

    if x1.shape[1:] != x2.shape[1:]:
        raise ValueError(f"feature shape mismatch: base {x1.shape}, extra {x2.shape}")

    mp = {}

    for s, x in zip(smi1, x1):
        mp[str(s)] = x

    n_overlap = 0
    for s, x in zip(smi2, x2):
        s = str(s)
        if s in mp:
            n_overlap += 1
        mp[s] = x

    smiles = np.asarray(list(mp.keys()), dtype="<U1024")
    feat = np.stack([mp[s] for s in smiles], axis=0).astype(np.float32)
    split = np.asarray(["merged"] * len(smiles), dtype="<U32")
    failed = np.asarray([], dtype="<U1024")

    np.savez_compressed(
        args.out,
        diff_token_feat=feat,
        smiles=smiles,
        split=split,
        failed=failed,
    )

    print("base:", len(smi1), x1.shape)
    print("extra:", len(smi2), x2.shape)
    print("overlap:", n_overlap)
    print("merged unique:", len(smiles))
    print("saved:", args.out)
    print("feat:", feat.shape, feat.dtype)
    print("finite:", np.isfinite(feat).all())


if __name__ == "__main__":
    main()

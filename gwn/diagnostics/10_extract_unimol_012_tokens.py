import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

from unimol_tools import UniMolRepr


def read_smrt_csv(path, split_name):
    df = pd.read_csv(path, engine="python")
    df.columns = [str(c).lower().strip() for c in df.columns]

    if "smile" in df.columns and "smiles" not in df.columns:
        df.rename(columns={"smile": "smiles"}, inplace=True)

    if "smiles" not in df.columns or "rt" not in df.columns:
        df = pd.read_csv(path, sep=r"\s+", names=["smiles", "rt"], header=0, engine="python")

    df = df[df["rt"] > 300.0].copy()
    df["split"] = split_name
    df["rt"] = df["rt"].astype(float)
    df["smiles"] = df["smiles"].astype(str)
    return df[["split", "smiles", "rt"]]


def generate_conformers(smiles, k=5, seed=17, max_attempts=1000):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    # Some RDKit builds do not expose maxAttempts on ETKDGv3 EmbedParameters.
    # EmbedMultipleConfs will still work without this field.
    try:
        params.maxAttempts = int(max_attempts)
    except AttributeError:
        pass

    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=int(k), params=params))

    if len(conf_ids) == 0:
        return None, None, None

    try:
        props = AllChem.MMFFGetMoleculeProperties(mol_h)
        energies = []
        for cid in conf_ids:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, mmffVariant="MMFF94s", confId=int(cid), maxIters=100)
                ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=int(cid))
                energies.append(float(ff.CalcEnergy()))
            except Exception:
                energies.append(1e9)
        order = np.argsort(energies)
        conf_ids = [conf_ids[i] for i in order]
    except Exception:
        pass

    heavy_ids = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() > 1]
    atoms = [mol_h.GetAtomWithIdx(i).GetSymbol() for i in heavy_ids]

    coords_list = []
    for cid in conf_ids[:k]:
        conf = mol_h.GetConformer(int(cid))
        coords = []
        for i in heavy_ids:
            p = conf.GetAtomPosition(i)
            coords.append([p.x, p.y, p.z])
        coords_list.append(np.asarray(coords, dtype=np.float32))

    return mol, atoms, coords_list


def atom_to_012_tokens(mol, atom_repr):
    """
    atom_repr: [num_heavy_atoms, D]
    return: [3, D]
    """
    atom_repr = np.asarray(atom_repr, dtype=np.float32)
    d = atom_repr.shape[1]

    # 0-cell token: atom-level geometry
    atom_token = atom_repr.mean(axis=0)

    # 1-cell token: bond-level geometry, endpoints average
    bond_vecs = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        if i < atom_repr.shape[0] and j < atom_repr.shape[0]:
            bond_vecs.append((atom_repr[i] + atom_repr[j]) * 0.5)

    if len(bond_vecs) > 0:
        bond_token = np.stack(bond_vecs, axis=0).mean(axis=0)
    else:
        bond_token = np.zeros(d, dtype=np.float32)

    # 2-cell token: ring-level geometry
    ring_vecs = []
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        idx = [i for i in ring if i < atom_repr.shape[0]]
        if len(idx) > 0:
            ring_vecs.append(atom_repr[idx].mean(axis=0))

    if len(ring_vecs) > 0:
        ring_token = np.stack(ring_vecs, axis=0).mean(axis=0)
    else:
        ring_token = np.zeros(d, dtype=np.float32)

    return np.stack([atom_token, bond_token, ring_token], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="../SMRT_data/data/SMRT_train.csv")
    parser.add_argument("--test_csv", default="../SMRT_data/data/SMRT_test.csv")
    parser.add_argument("--out", default="data/unimol_012_noise_features.npz")
    parser.add_argument("--k_confs", type=int, default=5)
    parser.add_argument("--n_noise", type=int, default=3)
    parser.add_argument("--noise_sigma", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_name", default="unimolv2")
    parser.add_argument("--model_size", default="84m")
    parser.add_argument("--max_mols", type=int, default=-1)
    parser.add_argument("--select_csv", default="", help="optional CSV with SMILES column to restrict extraction")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df_train = read_smrt_csv(args.train_csv, "train")
    df_test = read_smrt_csv(args.test_csv, "test")
    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    if args.select_csv:
        sel_df = pd.read_csv(args.select_csv)
        sel_cols = {c.lower(): c for c in sel_df.columns}
        if "smiles" not in sel_cols:
            raise ValueError(f"select_csv must have SMILES/smiles column, got {sel_df.columns.tolist()}")
        sel = set(sel_df[sel_cols["smiles"]].astype(str).tolist())
        before = len(df)
        df = df[df["smiles"].astype(str).isin(sel)].copy()
        print(f"selected molecules by {args.select_csv}: {before} -> {len(df)}")

    if args.max_mols > 0:
        df = df.iloc[:args.max_mols].copy()

    print("total molecules:", len(df))
    print("k_confs:", args.k_confs, "n_noise:", args.n_noise, "noise_sigma:", args.noise_sigma)

    repr_model = UniMolRepr(
        data_type="molecule",
        batch_size=args.batch_size,
        remove_hs=False,
        model_name=args.model_name,
        model_size=args.model_size,
        use_ddp=False,
        use_cuda=True,
    )

    all_smiles = []
    all_split = []
    all_y = []
    all_feat = []
    failed = []

    rng = np.random.default_rng(args.seed)

    # UniMolRepr 的 custom atoms/coordinates 模式在不同 unimol_tools 版本里很挑格式。
    # 如果 custom 模式失败，就自动切到官方稳定的 SMILES get_repr 模式，避免全部 failed。
    custom_failed_once = False

    for row_idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row["smiles"]
        rt = float(row["rt"])

        mol, atoms, coords_list = generate_conformers(
            smiles,
            k=args.k_confs,
            seed=args.seed + row_idx,
        )

        if mol is None or atoms is None or len(coords_list) == 0:
            failed.append(smiles)
            continue

        atoms_batch = []
        coords_batch = []

        for coords in coords_list:
            coords = np.asarray(coords, dtype=np.float32)

            # custom dict 模式最好传纯 Python list，避免 numpy array 触发
            # ValueError('cannot extract desired type from sequence')
            atoms_batch.append(list(atoms))
            coords_batch.append(coords.astype(float).tolist())

            for _ in range(args.n_noise):
                noisy = coords + rng.normal(
                    0.0,
                    args.noise_sigma,
                    size=coords.shape,
                ).astype(np.float32)

                atoms_batch.append(list(atoms))
                coords_batch.append(noisy.astype(float).tolist())

        try:
            data = {
                "atoms": atoms_batch,
                "coordinates": coords_batch,
            }

            try:
                # 优先用 custom coordinates 模式：这条路才包含 multi-conformer + noise
                reps = repr_model.get_repr(data, return_atomic_reprs=True)
                atomic_reprs = reps["atomic_reprs"]

            except Exception as e_custom:
                # 如果 custom 模式在当前 unimol_tools 版本不兼容，
                # 退回官方最稳定的 SMILES 模式，至少先拿到 Uni-Mol2 atomic representation。
                if not custom_failed_once:
                    print("\n[WARN] custom atoms/coordinates get_repr failed once:")
                    print(repr(e_custom))
                    print("[WARN] fallback to official SMILES get_repr mode. Noise ensemble disabled for fallback.\n")
                    custom_failed_once = True

                reps = repr_model.get_repr([smiles], return_atomic_reprs=True)
                atomic_reprs = reps["atomic_reprs"]

            token_samples = []
            for atom_repr in atomic_reprs:
                token_samples.append(atom_to_012_tokens(mol, atom_repr))

            token_samples = np.stack(token_samples, axis=0)  # [AUG, 3, D]
            token_mean = token_samples.mean(axis=0)
            token_std = token_samples.std(axis=0)

            # 每个 token 一个 uncertainty 标量，反映该 token 在 conformer/noise 下是否稳定
            token_unc = np.linalg.norm(token_std, axis=1, keepdims=True)  # [3, 1]

            token_feat = np.concatenate([token_mean, token_std, token_unc], axis=1).astype(np.float32)

            all_smiles.append(smiles)
            all_split.append(row["split"])
            all_y.append(rt)
            all_feat.append(token_feat)

        except Exception as e:
            failed.append(smiles)
            print("failed:", smiles, repr(e))
            continue

    if len(all_feat) == 0:
        raise RuntimeError(
            "No Uni-Mol features were extracted. "
            "Both custom coordinates mode and SMILES fallback failed."
        )

    all_feat = np.stack(all_feat, axis=0)  # [N, 3, 2D+1]

    np.savez_compressed(
        args.out,
        smiles=np.asarray(all_smiles, dtype=object),
        split=np.asarray(all_split, dtype=object),
        y=np.asarray(all_y, dtype=np.float32),
        diff_token_feat=all_feat,
        failed=np.asarray(failed, dtype=object),
    )

    print("saved:", args.out)
    print("features:", all_feat.shape)
    print("failed:", len(failed))


if __name__ == "__main__":
    main()

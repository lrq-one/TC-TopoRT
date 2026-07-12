from __future__ import annotations

"""
Conventional atom-bond GNN baseline used in the TC-TopoRT study.

This baseline uses ordinary GINE atom-bond message passing with paired
dataset-provided and strict tautomer-standardized molecular views.

The baseline follows the same SMRT split, training schedule, stratified
five-fold OOF prediction generation, and OOF-only Huber fusion principle
as TC-TopoRT. It uses its own conventional graph featurization:

    atom features:   38 dimensions
    bond features:   12 dimensions
    global context:  24 dimensions

It should therefore be interpreted as a conventional atom-bond GNN
backbone comparison, not as an identical-featurization topology-only
ablation.

Generated caches, predictions, metrics, and configurations are local
artifacts and should not be committed to Git.
"""

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool


ROOT = Path(__file__).resolve().parents[2]


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# CSV helpers
# -------------------------
def pick_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    low = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]
    raise RuntimeError(f"Cannot find any of columns {candidates}; available={cols}")


# -------------------------
# Featurization
# -------------------------
ATOM_NUMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
DEGREES = [0, 1, 2, 3, 4, 5, 6]
HYBRIDS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def one_hot_unknown(x, choices):
    return [1.0 if x == c else 0.0 for c in choices] + [0.0 if x in choices else 1.0]


def atom_features(atom: Chem.Atom):
    feats = []
    feats += one_hot_unknown(atom.GetAtomicNum(), ATOM_NUMS)
    feats += one_hot_unknown(atom.GetTotalDegree(), DEGREES)
    feats += one_hot_unknown(atom.GetHybridization(), HYBRIDS)
    feats += one_hot_unknown(atom.GetChiralTag(), CHIRAL)
    feats += [
        float(atom.GetFormalCharge()) / 5.0,
        float(atom.GetTotalNumHs()) / 4.0,
        float(atom.GetTotalValence()) / 8.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetMass()) / 200.0,
    ]
    return feats


def bond_features(bond: Chem.Bond):
    feats = []
    feats += one_hot_unknown(bond.GetBondType(), BOND_TYPES)
    feats += one_hot_unknown(bond.GetStereo(), STEREO_TYPES)
    feats += [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]
    return feats


def safe_float(x, scale=1.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v / scale
    except Exception:
        return 0.0


def safe_call(fn, mol, default=0.0):
    try:
        return fn(mol)
    except Exception:
        return default


def safe_module_call(module, name, mol, default=0.0):
    try:
        fn = getattr(module, name, None)
        if fn is None:
            return default
        return fn(mol)
    except Exception:
        return default


def calc_nhoh_count_manual(mol):
    # Approximate RDKit NHOH count: total hydrogens attached to N or O atoms.
    try:
        return sum(
            int(a.GetTotalNumHs())
            for a in mol.GetAtoms()
            if a.GetAtomicNum() in {7, 8}
        )
    except Exception:
        return 0.0


def calc_no_count_manual(mol):
    # Approximate RDKit NO count: number of N/O atoms.
    try:
        return sum(
            1
            for a in mol.GetAtoms()
            if a.GetAtomicNum() in {7, 8}
        )
    except Exception:
        return 0.0


def global_descriptors(mol: Chem.Mol):
    # 24D global descriptor vector, scaled to moderate numeric ranges.
    # Written with safe getattr-style calls to tolerate RDKit version differences.
    vals = [
        safe_float(safe_call(Descriptors.MolWt, mol), 1000.0),
        safe_float(safe_call(Descriptors.MolLogP, mol), 10.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcTPSA", mol), 300.0),
        safe_float(mol.GetNumHeavyAtoms(), 100.0),
        safe_float(safe_call(Lipinski.NumHAcceptors, mol), 20.0),
        safe_float(safe_call(Lipinski.NumHDonors, mol), 10.0),
        safe_float(safe_call(Lipinski.NumRotatableBonds, mol), 30.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcNumRings", mol), 10.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcNumAromaticRings", mol), 10.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcNumAliphaticRings", mol), 10.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcFractionCSP3", mol), 1.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcNumHeteroatoms", mol), 50.0),
        safe_float(calc_nhoh_count_manual(mol), 20.0),
        safe_float(calc_no_count_manual(mol), 20.0),
        safe_float(safe_call(Descriptors.NumValenceElectrons, mol), 500.0),
        safe_float(safe_call(Descriptors.MolMR, mol), 300.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcNumSaturatedRings", mol), 10.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcNumAromaticHeterocycles", mol), 10.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcNumAromaticCarbocycles", mol), 10.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcNumAliphaticHeterocycles", mol), 10.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcNumAliphaticCarbocycles", mol), 10.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcNumSaturatedHeterocycles", mol), 10.0),
        safe_float(safe_module_call(rdMolDescriptors, "CalcNumSaturatedCarbocycles", mol), 10.0),
        safe_float(sum(a.GetFormalCharge() for a in mol.GetAtoms()), 5.0),
    ]
    return vals


def smiles_to_data(smiles: str, y: float, source_index: int):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index = []
    edge_attr = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        bond_dim = len(bond_features_dummy())
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_dim), dtype=torch.float)

    g = torch.tensor(global_descriptors(mol), dtype=torch.float).view(1, -1)
    y = torch.tensor([float(y)], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_feat=g,
        y=y,
        source_index=torch.tensor([int(source_index)], dtype=torch.long),
    )


def bond_features_dummy():
    feats = []
    feats += one_hot_unknown(Chem.rdchem.BondType.SINGLE, BOND_TYPES)
    feats += one_hot_unknown(Chem.rdchem.BondStereo.STEREONONE, STEREO_TYPES)
    feats += [0.0, 0.0]
    return feats


def build_or_load_dataset(csv_path: Path, cache_path: Path):
    if cache_path.exists():
        return torch.load(cache_path)

    df = pd.read_csv(csv_path)
    smiles_col = pick_col(df, ["smile", "SMILES", "smiles", "Orig_SMILES", "Taut_SMILES"])
    y_col = pick_col(df, ["rt", "RT", "Actual_RT", "retention_time"])
    id_col = None
    for c in ["Source_Index", "source_index", "Unnamed: 0", "ID", "id"]:
        if c in df.columns:
            id_col = c
            break

    data_list = []
    bad = 0
    first_errors = []

    for i, r in df.iterrows():
        try:
            source_index = int(r[id_col]) if id_col is not None else int(i)
            data_list.append(smiles_to_data(r[smiles_col], r[y_col], source_index))
        except Exception as e:
            bad += 1
            if len(first_errors) < 5:
                first_errors.append((i, str(r.get(smiles_col, "")), repr(e)))

    if bad:
        print(f"[WARN] {csv_path} skipped molecules: {bad}")
        print("[WARN] first errors:")
        for i, smi, err in first_errors:
            print(f"  row={i} smiles={smi} error={err}")

    if len(data_list) == 0:
        raise RuntimeError(
            f"No valid molecules were built from {csv_path}. "
            f"Check SMILES column={smiles_col}, RT column={y_col}, first_errors={first_errors}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_list, cache_path)
    return data_list


# -------------------------
# Model
# -------------------------
class AtomBondGNN(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        global_dim: int,
        hidden: int = 256,
        layers: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout

        self.atom_encoder = nn.Linear(atom_dim, hidden)
        self.edge_encoder = nn.Linear(bond_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=hidden))
            self.norms.append(nn.BatchNorm1d(hidden))

        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        x = self.atom_encoder(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, batch.edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h

        pooled = global_mean_pool(x, batch.batch)

        gf = batch.global_feat
        if gf.dim() == 1:
            gf = gf.view(pooled.size(0), -1)
        gf = self.global_encoder(gf)

        out = self.head(torch.cat([pooled, gf], dim=-1)).view(-1)
        return out


# -------------------------
# Training
# -------------------------
def make_bins(y, n_bins=10):
    y = np.asarray(y, dtype=float)
    try:
        return pd.qcut(y, q=n_bins, duplicates="drop", labels=False)
    except Exception:
        return pd.cut(y, bins=n_bins, labels=False)


def train_one_fold(
    train_data,
    val_data,
    test_data,
    args,
    seed,
    view_name,
    fold_id,
    device,
):
    train_y = np.array([d.y.item() for d in train_data], dtype=float)
    y_mean = float(train_y.mean())
    y_std = float(train_y.std() + 1e-8)

    atom_dim = train_data[0].x.size(-1)
    bond_dim = train_data[0].edge_attr.size(-1)
    global_dim = train_data[0].global_feat.size(-1)

    model = AtomBondGNN(
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        global_dim=global_dim,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.scheduler_t0,
        T_mult=1,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    best_mae = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred_norm = model(batch)
            target_norm = (batch.y.view(-1) - y_mean) / y_std

            loss = F.smooth_l1_loss(pred_norm, target_norm, beta=args.huber_beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            losses.append(float(loss.item()))

        scheduler.step(epoch)

        val_pred = predict(model, val_loader, device, y_mean, y_std)
        val_y = np.array([d.y.item() for d in val_data], dtype=float)
        val_mae = mean_absolute_error(val_y, val_pred)

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"[{view_name}] fold={fold_id} epoch={epoch:03d} "
                f"loss={np.mean(losses):.5f} val_mae={val_mae:.4f} best={best_mae:.4f}"
            )

        if bad >= args.patience:
            print(f"[{view_name}] fold={fold_id} early stop at epoch {epoch}, best val MAE={best_mae:.4f}")
            break

    model.load_state_dict(best_state)

    val_loader = DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    val_pred = predict(model, val_loader, device, y_mean, y_std)
    test_pred = predict(model, test_loader, device, y_mean, y_std)

    return val_pred, test_pred, best_mae


@torch.no_grad()
def predict(model, loader, device, y_mean, y_std):
    model.eval()
    preds = []
    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch)
        pred = pred_norm.detach().cpu().numpy() * y_std + y_mean
        preds.append(pred)
    return np.concatenate(preds, axis=0)


def run_view_oof(train_data, test_data, y, args, seed, view_name, device):
    bins = make_bins(y, args.strata_bins)
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=seed)

    oof = np.zeros(len(train_data), dtype=float)
    test_fold_preds = []
    fold_rows = []

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), bins), 1):
        tr = [train_data[i] for i in tr_idx]
        va = [train_data[i] for i in va_idx]

        print(f"\n===== {view_name} fold {fold_id}/{args.k}: train={len(tr)} val={len(va)} =====")

        val_pred, test_pred, best_mae = train_one_fold(
            tr, va, test_data, args, seed, view_name, fold_id, device
        )

        oof[va_idx] = val_pred
        test_fold_preds.append(test_pred)

        fold_rows.append({
            "view": view_name,
            "fold": fold_id,
            "val_mae": float(best_mae),
            "n_train": len(tr),
            "n_val": len(va),
        })

    test_pred = np.mean(np.vstack(test_fold_preds), axis=0)
    return oof, test_pred, pd.DataFrame(fold_rows)


# -------------------------
# Metrics / output
# -------------------------
def metrics(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "MedAE": float(np.median(np.abs(pred - y))),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the conventional atom-bond GNN dual-view OOF baseline "
            "used in the TC-TopoRT structural comparison."
        )
    )

    parser.add_argument("--origin_train_csv", default="gwn/data/SMRT_train.csv")
    parser.add_argument("--origin_test_csv", default="gwn/data/SMRT_test.csv")
    parser.add_argument("--taut_train_csv", default="gwn/data_taut_strict_origin_order/SMRT_train_tautomer_strict.csv")
    parser.add_argument("--taut_test_csv", default="gwn/data_taut_strict_origin_order/SMRT_test_tautomer_strict.csv")

    parser.add_argument(
        "--out_dir",
        default=None,
        help=(
            "Output directory. Default: "
            "artifacts/results/atom_bond_gnn/seed<seed>"
        ),
    )
    parser.add_argument(
        "--cache_dir",
        default="artifacts/cache/atom_bond_gnn",
        help="Directory for generated PyTorch graph caches.",
    )

    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--huber_beta", type=float, default=1.0)

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--huber_alpha", type=float, default=1e-4)
    parser.add_argument("--scheduler_t0", type=int, default=20)
    parser.add_argument("--strata_bins", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = (
            f"artifacts/results/atom_bond_gnn/seed{args.seed}"
        )

    seed_everything(args.seed)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Atom-bond GNN dual-view OOF baseline config ===")
    print(json.dumps(vars(args), indent=2))
    print("device:", device)

    # Build / load datasets.
    origin_train = build_or_load_dataset(
        ROOT / args.origin_train_csv,
        cache_dir / "origin_train.pt",
    )
    origin_test = build_or_load_dataset(
        ROOT / args.origin_test_csv,
        cache_dir / "origin_test.pt",
    )
    taut_train = build_or_load_dataset(
        ROOT / args.taut_train_csv,
        cache_dir / "taut_train.pt",
    )
    taut_test = build_or_load_dataset(
        ROOT / args.taut_test_csv,
        cache_dir / "taut_test.pt",
    )

    y_train = np.array([d.y.item() for d in origin_train], dtype=float)
    y_test = np.array([d.y.item() for d in origin_test], dtype=float)

    if len(origin_train) != len(taut_train):
        raise RuntimeError(
            "Original/tautomer training-set length mismatch: "
            f"{len(origin_train)} != {len(taut_train)}"
        )

    if len(origin_test) != len(taut_test):
        raise RuntimeError(
            "Original/tautomer test-set length mismatch: "
            f"{len(origin_test)} != {len(taut_test)}"
        )

    origin_train_ids = np.array(
        [int(d.source_index.item()) for d in origin_train],
        dtype=np.int64,
    )
    taut_train_ids = np.array(
        [int(d.source_index.item()) for d in taut_train],
        dtype=np.int64,
    )
    origin_test_ids = np.array(
        [int(d.source_index.item()) for d in origin_test],
        dtype=np.int64,
    )
    taut_test_ids = np.array(
        [int(d.source_index.item()) for d in taut_test],
        dtype=np.int64,
    )

    if not np.array_equal(origin_train_ids, taut_train_ids):
        mismatch = np.flatnonzero(origin_train_ids != taut_train_ids)
        raise RuntimeError(
            "Original/tautomer training rows are not aligned. "
            f"First mismatched positions: {mismatch[:10].tolist()}"
        )

    if not np.array_equal(origin_test_ids, taut_test_ids):
        mismatch = np.flatnonzero(origin_test_ids != taut_test_ids)
        raise RuntimeError(
            "Original/tautomer test rows are not aligned. "
            f"First mismatched positions: {mismatch[:10].tolist()}"
        )

    taut_y_train = np.array(
        [d.y.item() for d in taut_train],
        dtype=float,
    )
    taut_y_test = np.array(
        [d.y.item() for d in taut_test],
        dtype=float,
    )

    if not np.allclose(y_train, taut_y_train, rtol=0.0, atol=1e-6):
        raise RuntimeError(
            "Original/tautomer training RT labels are inconsistent."
        )

    if not np.allclose(y_test, taut_y_test, rtol=0.0, atol=1e-6):
        raise RuntimeError(
            "Original/tautomer test RT labels are inconsistent."
        )

    print(
        "Pairing audit: PASS "
        f"(train={len(origin_train)}, test={len(origin_test)})"
    )

    origin_oof, origin_test_pred, origin_folds = run_view_oof(
        origin_train, origin_test, y_train, args, args.seed, "origin", device
    )
    taut_oof, taut_test_pred, taut_folds = run_view_oof(
        taut_train, taut_test, y_train, args, args.seed, "tautomer", device
    )

    mean_oof = 0.5 * (origin_oof + taut_oof)
    mean_test = 0.5 * (origin_test_pred + taut_test_pred)

    # Huber stacker on training OOF predictions only.
    X_oof = np.column_stack([
        origin_oof,
        taut_oof,
        mean_oof,
        np.abs(origin_oof - taut_oof),
    ])
    X_test = np.column_stack([
        origin_test_pred,
        taut_test_pred,
        mean_test,
        np.abs(origin_test_pred - taut_test_pred),
    ])

    x_scaler = StandardScaler()
    X_oof_s = x_scaler.fit_transform(X_oof)
    X_test_s = x_scaler.transform(X_test)

    stacker = HuberRegressor(alpha=args.huber_alpha, max_iter=1000)
    stacker.fit(X_oof_s, y_train)

    final_test = stacker.predict(X_test_s)
    final_oof = stacker.predict(X_oof_s)

    # Save predictions.
    test_source_index = np.array([int(d.source_index.item()) for d in origin_test])

    test_df = pd.DataFrame({
        "Source_Index": test_source_index,
        "Actual_RT": y_test,
        "Origin_Test_Pred": origin_test_pred,
        "Taut_Test_Pred": taut_test_pred,
        "Mean_Pred": mean_test,
        "Final_Pred": final_test,
    })
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)

    train_source_index = np.array([int(d.source_index.item()) for d in origin_train])
    train_df = pd.DataFrame({
        "Source_Index": train_source_index,
        "Actual_RT": y_train,
        "Origin_OOF_Pred": origin_oof,
        "Taut_OOF_Pred": taut_oof,
        "Mean_OOF_Pred": mean_oof,
        "Final_OOF_Pred": final_oof,
    })
    train_df.to_csv(out_dir / "oof_train_predictions.csv", index=False)

    folds = pd.concat([origin_folds, taut_folds], ignore_index=True)
    folds.to_csv(out_dir / "fold_metrics.csv", index=False)

    metric_rows = []
    for name, pred in [
        ("AtomBondGNN original view", origin_test_pred),
        ("AtomBondGNN tautomer view", taut_test_pred),
        ("AtomBondGNN paired mean fusion", mean_test),
        ("AtomBondGNN OOF Huber stack", final_test),
    ]:
        row = {"Method": name, **metrics(y_test, pred)}
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n===== Atom-bond GNN baseline test metrics =====")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n===== wrote =====")
    for p in [
        out_dir / "test_predictions.csv",
        out_dir / "oof_train_predictions.csv",
        out_dir / "metrics.csv",
        out_dir / "fold_metrics.csv",
        out_dir / "config.json",
    ]:
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()

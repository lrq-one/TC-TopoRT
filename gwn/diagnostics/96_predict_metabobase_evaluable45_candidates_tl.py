#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import inspect
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mp.smrt_dataset import SMRTComplexDataset
from mp.complex import ComplexBatch
from net.topocellrt_cwn_replace import TopoCellRTCWNReplace


SPLIT_DIR = Path("experiments_candidate_filtering/metabobase_tl_evaluable45/seed42")
TRAIN_DIR = Path("experiments_candidate_filtering/metabobase_tl_evaluable45_training/seed42")
OUT_DIR = Path("experiments_candidate_filtering/metabobase_evaluable45_predictions_tl_seed42")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAND_VALID = Path("experiments_candidate_filtering/parsed_candidates/msfinder_candidates_valid.csv")

TARGET_SHIFT = 300.0
DUMMY_RT = 999.0


def canon_smiles(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return ""
        return Chem.MolToSmiles(m, isomericSmiles=True)
    except Exception:
        return ""


def tautomer_strict_smiles(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return ""
        enum = rdMolStandardize.TautomerEnumerator()
        tm = enum.Canonicalize(m)
        return Chem.MolToSmiles(tm, isomericSmiles=True)
    except Exception:
        return ""


def complex_collate(batch):
    sig = inspect.signature(ComplexBatch.from_complex_list)
    kwargs = {}
    if "follow_batch" in sig.parameters:
        kwargs["follow_batch"] = []
    if "max_dim" in sig.parameters:
        kwargs["max_dim"] = 2
    return ComplexBatch.from_complex_list(batch, **kwargs)


def make_dataset(root, csv_path):
    return SMRTComplexDataset(
        root=str(root),
        csv_path=str(csv_path),
        max_ring_size=6,
        use_edge_features=True,
        n_jobs=4,
        init_method="sum",
        include_down_adj=True,
    )


def make_model(device):
    model = TopoCellRTCWNReplace(
        emb_dim=256,
        cwn_layers=6,
        cwn_hidden=256,
        max_dim=2,
        drop_ratio=0.0,
    )
    return model.to(device)


def move_batch(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def forward_pred(model, batch):
    out = model(batch)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.view(-1).float()


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        batch = move_batch(batch, device)
        p = forward_pred(model, batch)
        preds.append(p.detach().cpu().numpy())
    return np.concatenate(preds).reshape(-1)


def load_model(ckpt, device):
    model = make_model(device)
    sd = torch.load(ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("loaded:", ckpt)
    print("missing:", len(missing), "unexpected:", len(unexpected))
    if missing:
        print("missing sample:", missing[:10])
    if unexpected:
        print("unexpected sample:", unexpected[:10])
    return model


def to_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("device:", device)

    meta = pd.read_csv(SPLIT_DIR / "metabobase_test45_evaluable45_metadata.csv")
    meta = meta.reset_index(drop=True)
    meta["s10_row"] = np.arange(len(meta))
    query_ids = set(meta["query_id"].astype(str))
    qid_to_row = dict(zip(meta["query_id"].astype(str), meta["s10_row"]))

    cand = pd.read_csv(CAND_VALID, dtype=str, low_memory=False).fillna("")
    cand = cand[cand["query_id"].astype(str).isin(query_ids)].copy()

    cand["s10_row"] = cand["query_id"].astype(str).map(qid_to_row).astype(int)
    cand["candidate_rank"] = pd.to_numeric(cand["candidate_rank"], errors="coerce")
    cand["candidate_score"] = pd.to_numeric(cand["candidate_score"], errors="coerce").fillna(0.0)
    cand["rt_sec"] = pd.to_numeric(cand["rt_sec"], errors="coerce")
    cand["is_true"] = to_bool_series(cand["is_true"])

    before_rows = len(cand)
    cand["candidate_smiles_canon"] = cand["candidate_smiles"].map(canon_smiles)
    cand = cand[cand["candidate_smiles_canon"].astype(str).str.len() > 0].copy()
    cand = cand.sort_values(["s10_row", "candidate_rank", "candidate_score"], ascending=[True, True, False]).reset_index(drop=True)

    unique = cand[["candidate_smiles_canon"]].drop_duplicates().reset_index(drop=True)
    unique["candidate_uid"] = ["cand_%06d" % i for i in range(len(unique))]
    smi_to_uid = dict(zip(unique["candidate_smiles_canon"], unique["candidate_uid"]))
    cand["candidate_uid"] = cand["candidate_smiles_canon"].map(smi_to_uid)

    origin_csv = OUT_DIR / "evaluable45_unique_candidate_origin_for_prediction.csv"
    taut_csv = OUT_DIR / "evaluable45_unique_candidate_taut_for_prediction.csv"
    unique_csv = OUT_DIR / "evaluable45_unique_candidate_smiles.csv"
    row_map_csv = OUT_DIR / "evaluable45_candidate_row_map.csv"

    origin_df = pd.DataFrame({
        "name": unique["candidate_uid"],
        "smiles": unique["candidate_smiles_canon"],
        "rt": DUMMY_RT,
    })

    taut_df = origin_df.copy()
    taut_df["smiles"] = taut_df["smiles"].map(tautomer_strict_smiles)
    taut_df = taut_df[taut_df["smiles"].astype(str).str.len() > 0].copy()

    if len(taut_df) != len(origin_df):
        raise RuntimeError(f"tautomer conversion dropped molecules: origin={len(origin_df)} taut={len(taut_df)}")

    origin_df.to_csv(origin_csv, index=False)
    taut_df.to_csv(taut_csv, index=False)
    unique.to_csv(unique_csv, index=False)
    cand.to_csv(row_map_csv, index=False)

    print("=" * 100)
    print("[candidate input]")
    print("evaluable45 queries:", len(meta))
    print("candidate valid rows before canon:", before_rows)
    print("candidate rows after canon:", len(cand))
    print("unique candidates:", len(unique))
    print("queries covered:", cand["query_id"].nunique())
    print("true candidates covered:", int(cand.groupby("query_id")["is_true"].any().sum()), "/", len(meta))
    print("=" * 100)

    ds_origin = make_dataset(OUT_DIR / "datasets" / "origin_candidates", origin_csv)
    ds_taut = make_dataset(OUT_DIR / "datasets" / "taut_candidates", taut_csv)

    if len(ds_origin) != len(unique) or len(ds_taut) != len(unique):
        raise RuntimeError(f"dataset length mismatch: origin={len(ds_origin)} taut={len(ds_taut)} unique={len(unique)}")

    loader_origin = DataLoader(
        ds_origin,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=complex_collate,
    )
    loader_taut = DataLoader(
        ds_taut,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=complex_collate,
    )

    origin_ckpt = TRAIN_DIR / "origin" / "best_model.pth"
    taut_ckpt = TRAIN_DIR / "taut" / "best_model.pth"

    model_origin = load_model(origin_ckpt, device)
    pred_origin_shifted = predict_loader(model_origin, loader_origin, device)
    del model_origin
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_taut = load_model(taut_ckpt, device)
    pred_taut_shifted = predict_loader(model_taut, loader_taut, device)
    del model_taut
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pred_origin = pred_origin_shifted - TARGET_SHIFT
    pred_taut = pred_taut_shifted - TARGET_SHIFT
    pred_avg = 0.5 * (pred_origin + pred_taut)

    unique_out = unique.copy()
    unique_out["tl_origin_pred_shifted"] = pred_origin_shifted
    unique_out["tl_taut_pred_shifted"] = pred_taut_shifted
    unique_out["tl_origin_pred_rt"] = pred_origin
    unique_out["tl_taut_pred_rt"] = pred_taut
    unique_out["candidate_pred_rt"] = pred_avg
    unique_out["tl_origin_taut_absdiff"] = np.abs(pred_origin - pred_taut)

    final = cand.merge(
        unique_out[
            [
                "candidate_uid",
                "tl_origin_pred_shifted",
                "tl_taut_pred_shifted",
                "tl_origin_pred_rt",
                "tl_taut_pred_rt",
                "candidate_pred_rt",
                "tl_origin_taut_absdiff",
            ]
        ],
        on="candidate_uid",
        how="left",
    )

    final["abs_rt_delta"] = (final["candidate_pred_rt"].astype(float) - final["rt_sec"].astype(float)).abs()

    unique_out.to_csv(OUT_DIR / "evaluable45_unique_candidate_predictions_tl_seed42.csv", index=False)
    final.to_csv(OUT_DIR / "evaluable45_candidate_predictions_tl_seed42.csv", index=False)

    qstat = final.groupby("query_id").agg(
        n_rows=("candidate_uid", "size"),
        true_in_candidates=("is_true", "any"),
        min_true_rank=("candidate_rank", lambda x: np.nan),
    ).reset_index()

    # compute true rank separately
    true_rank_rows = []
    for qid, sub in final.groupby("query_id"):
        tr = sub[sub["is_true"]].sort_values("candidate_rank")
        true_rank_rows.append({
            "query_id": qid,
            "true_candidate_rank": int(tr.iloc[0]["candidate_rank"]) if len(tr) else np.nan,
            "n_candidates": len(sub),
            "true_in_candidates": bool(len(tr)),
        })
    qstat = pd.DataFrame(true_rank_rows)
    qstat.to_csv(OUT_DIR / "evaluable45_query_candidate_stats.csv", index=False)

    print("=" * 100)
    print("[saved]")
    print(OUT_DIR / "evaluable45_candidate_predictions_tl_seed42.csv")
    print(OUT_DIR / "evaluable45_unique_candidate_predictions_tl_seed42.csv")
    print(OUT_DIR / "evaluable45_query_candidate_stats.csv")
    print("=" * 100)
    print("final shape:", final.shape)
    print("queries:", final["query_id"].nunique())
    print("prediction NaN:", int(final["candidate_pred_rt"].isna().sum()))
    print("true queries with candidate:", int(qstat["true_in_candidates"].sum()), "/", len(qstat))
    print("\ncandidate_pred_rt summary:")
    print(final["candidate_pred_rt"].describe().to_string())
    print("\nabs_rt_delta summary:")
    print(final["abs_rt_delta"].describe().to_string())
    print("\nquery stats:")
    print(qstat.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import inspect
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mp.smrt_dataset import SMRTComplexDataset
from mp.complex import ComplexBatch
from net.topocellrt_cwn_replace import TopoCellRTCWNReplace


IN_DIR = Path("experiments_candidate_filtering/metabobase_s10_prediction_inputs")
OUT_DIR = Path("experiments_candidate_filtering/metabobase_s10_predictions_tl_exact39")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SHIFT = 300.0


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


def load_tl_model(ckpt, device):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("device:", device)

    origin_csv = IN_DIR / "metabobase_s10_candidate_origin.csv"
    taut_csv = IN_DIR / "metabobase_s10_candidate_taut_strict.csv"
    unique_csv = IN_DIR / "metabobase_s10_unique_candidate_smiles.csv"
    row_map_csv = IN_DIR / "metabobase_s10_candidate_row_map.csv"

    unique = pd.read_csv(unique_csv)
    row_map = pd.read_csv(row_map_csv)

    ds_origin = make_dataset(
        OUT_DIR / "datasets" / "origin_candidates",
        origin_csv,
    )
    ds_taut = make_dataset(
        OUT_DIR / "datasets" / "taut_candidates",
        taut_csv,
    )

    print("unique candidates:", len(unique))
    print("row candidates:", len(row_map))
    print("origin dataset:", len(ds_origin))
    print("taut dataset:", len(ds_taut))

    if len(ds_origin) != len(unique) or len(ds_taut) != len(unique):
        raise RuntimeError(
            f"dataset length mismatch: origin={len(ds_origin)} taut={len(ds_taut)} unique={len(unique)}"
        )

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

    tl_dir = Path("experiments_candidate_filtering/metabobase_tl_exact39_training") / f"seed{args.seed}"
    origin_ckpt = tl_dir / "origin" / "best_model.pth"
    taut_ckpt = tl_dir / "taut" / "best_model.pth"

    model_origin = load_tl_model(origin_ckpt, device)
    pred_origin_shifted = predict_loader(model_origin, loader_origin, device)
    del model_origin
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_taut = load_tl_model(taut_ckpt, device)
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

    final = row_map.merge(
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

    unique_out.to_csv(OUT_DIR / f"metabobase_s10_unique_candidate_predictions_tl_seed{args.seed}.csv", index=False)
    final.to_csv(OUT_DIR / f"metabobase_s10_candidate_predictions_tl_seed{args.seed}.csv", index=False)

    print("=" * 100)
    print("[saved]")
    print(OUT_DIR / f"metabobase_s10_unique_candidate_predictions_tl_seed{args.seed}.csv")
    print(OUT_DIR / f"metabobase_s10_candidate_predictions_tl_seed{args.seed}.csv")
    print("=" * 100)
    print("final shape:", final.shape)
    print("queries:", final["s10_row"].nunique())
    print("prediction NaN:", int(final["candidate_pred_rt"].isna().sum()))
    print("candidate_pred_rt summary:")
    print(final["candidate_pred_rt"].describe().to_string())
    print("abs_rt_delta summary:")
    print(final["abs_rt_delta"].describe().to_string())
    print("\nexamples:")
    cols = ["s10_row", "true_name", "rt_sec", "candidate_rank", "candidate_name", "candidate_pred_rt", "abs_rt_delta", "is_true"]
    cols = [c for c in cols if c in final.columns]
    print(final[cols].head(20).to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()

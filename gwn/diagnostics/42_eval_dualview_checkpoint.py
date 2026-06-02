import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from mp.smrt_dataset import SMRTComplexDataset
from net.topocellrt_cwn_dualview import TopoCellRTCWNDualView
from train_topocellrt_cwn_dualview import (
    Config,
    PairedComplexDataset,
    complex_collate_fn,
    load_valid_meta,
    evaluate,
    export_predictions,
)


def add_taut_meta(origin_meta, taut_meta, indices=None):
    if indices is None:
        out = origin_meta.reset_index(drop=True).copy()
        taut_part = taut_meta.reset_index(drop=True)
    else:
        out = origin_meta.iloc[indices].reset_index(drop=True).copy()
        taut_part = taut_meta.iloc[indices].reset_index(drop=True)

    out["Taut_SMILES"] = taut_part["SMILES"].values

    for c in ["raw_changed", "real_changed", "formula_same", "heavy_same", "fallback", "reason"]:
        if c in taut_part.columns:
            out[c] = taut_part[c].values

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Eval DualView Checkpoint ===")
    print("ckpt:", args.ckpt)
    print("out_dir:", args.out_dir)
    print("device:", Config.DEVICE)

    print("\n=== Loading datasets ===")
    origin_train_full = SMRTComplexDataset(
        Config.ORIGIN_TRAIN_ROOT,
        Config.ORIGIN_TRAIN_CSV,
        Config.MAX_RING_SIZE,
        use_edge_features=Config.USE_EDGE_ATTR,
    )
    taut_train_full = SMRTComplexDataset(
        Config.TAUT_TRAIN_ROOT,
        Config.TAUT_TRAIN_CSV,
        Config.MAX_RING_SIZE,
        use_edge_features=Config.USE_EDGE_ATTR,
    )
    origin_test = SMRTComplexDataset(
        Config.ORIGIN_TEST_ROOT,
        Config.ORIGIN_TEST_CSV,
        Config.MAX_RING_SIZE,
        use_edge_features=Config.USE_EDGE_ATTR,
    )
    taut_test = SMRTComplexDataset(
        Config.TAUT_TEST_ROOT,
        Config.TAUT_TEST_CSV,
        Config.MAX_RING_SIZE,
        use_edge_features=Config.USE_EDGE_ATTR,
    )

    paired_train_full = PairedComplexDataset(origin_train_full, taut_train_full, full_check=False)
    paired_test = PairedComplexDataset(origin_test, taut_test, full_check=False)

    total_len = len(paired_train_full)
    train_len = int(0.9 * total_len)
    val_len = total_len - train_len

    train_set, val_set = random_split(
        paired_train_full,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(Config.SPLIT_SEED),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        paired_test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=args.num_workers,
    )

    print("N train_full/val/test:", total_len, len(val_set), len(paired_test))

    print("\n=== Loading model ===")
    model = TopoCellRTCWNDualView(
        emb_dim=256,
        cwn_layers=Config.CWN_LAYERS,
        cwn_hidden=Config.CWN_HIDDEN,
        max_dim=2,
        drop_ratio=0.0,
        share_encoder=Config.SHARE_ENCODER,
        init_tau=Config.INIT_TAU,
        temperature=Config.TEMPERATURE,
        gate_prior_alpha=Config.GATE_PRIOR_ALPHA,
    ).to(Config.DEVICE)

    state = torch.load(args.ckpt, map_location=Config.DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)

    print("\n=== Evaluate ===")
    val_metrics = evaluate(model, val_loader, epoch=-1, prefix="val")
    test_metrics = evaluate(model, test_loader, epoch=-1, prefix="test")

    print("\n=== VAL ===")
    print(json.dumps(val_metrics, ensure_ascii=False, indent=2))
    print("\n=== TEST ===")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

    with open(os.path.join(args.out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, ensure_ascii=False, indent=2)

    print("\n=== Export predictions ===")
    origin_meta_train = load_valid_meta(Config.ORIGIN_TRAIN_CSV)
    taut_meta_train = load_valid_meta(Config.TAUT_TRAIN_CSV)
    origin_meta_test = load_valid_meta(Config.ORIGIN_TEST_CSV)
    taut_meta_test = load_valid_meta(Config.TAUT_TEST_CSV)

    val_indices = list(val_set.indices)
    val_meta = add_taut_meta(origin_meta_train, taut_meta_train, val_indices)
    test_meta = add_taut_meta(origin_meta_test, taut_meta_test, None)

    export_predictions(
        model,
        val_loader,
        val_meta,
        os.path.join(args.out_dir, "dualview_val_predictions.csv"),
        "val",
    )
    export_predictions(
        model,
        test_loader,
        test_meta,
        os.path.join(args.out_dir, "dualview_test_predictions.csv"),
        "test",
    )

    print("\n✅ eval done:", args.out_dir)


if __name__ == "__main__":
    main()

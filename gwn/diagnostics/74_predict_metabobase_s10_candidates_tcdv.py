#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import inspect
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mp.smrt_dataset import SMRTComplexDataset
from mp.complex import ComplexBatch
import train_oof_dualview_stack as t
from net.topocellrt_cwn_replace import TopoCellRTCWNReplace



def complex_collate(batch):
    """
    Collate mp.complex.Complex objects into a ComplexBatch.
    This is required because vanilla PyTorch default_collate cannot batch Complex objects.
    """
    sig = inspect.signature(ComplexBatch.from_complex_list)
    kwargs = {}
    if "follow_batch" in sig.parameters:
        kwargs["follow_batch"] = []
    if "max_dim" in sig.parameters:
        kwargs["max_dim"] = 2
    return ComplexBatch.from_complex_list(batch, **kwargs)


RUNS = [
    "results_OOF_DualView_Stack_v1",
    "results_OOF_DualView_Stack_seed79",
    "results_OOF_DualView_Stack_seed123",
    "results_OOF_DualView_Stack_seed256",
    "results_OOF_DualView_Stack_seed5",
]


def load_model(ckpt_path, device):
    model = TopoCellRTCWNReplace(
        emb_dim=256,
        cwn_layers=6,
        cwn_hidden=256,
        max_dim=2,
        drop_ratio=0.0,
    ).to(device)

    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def predict_one_checkpoint(ckpt_path, loader, device, expected_n=None):
    model = load_model(ckpt_path, device)
    with torch.no_grad():
        out = t.predict_model(model, loader, device)

    # train_oof_dualview_stack.predict_model returns a tuple/list.
    # In this repo, the first array can be dummy labels, e.g. all 999.0.
    # We must select the non-dummy prediction array.
    if isinstance(out, (tuple, list)):
        arrays = []
        for i, x in enumerate(out):
            try:
                a = np.asarray(x, dtype=float).reshape(-1)
                arrays.append((i, a))
            except Exception:
                pass

        print("[predict_model output arrays]")
        for i, a in arrays:
            print(
                "  index=", i,
                "shape=", a.shape,
                "mean=", float(np.mean(a)),
                "std=", float(np.std(a)),
                "min=", float(np.min(a)),
                "max=", float(np.max(a)),
            )

        candidates = []
        for i, a in arrays:
            if expected_n is not None and len(a) != expected_n:
                continue
            # dummy label array is exactly/near 999.0 with almost zero std
            is_dummy_999 = np.allclose(a, 999.0, atol=1e-6)
            if not is_dummy_999:
                candidates.append((i, a))

        if candidates:
            pred = candidates[0][1]
            print("[selected prediction array index]", candidates[0][0])
        else:
            # fallback: choose last array with expected length
            valid = [(i, a) for i, a in arrays if expected_n is None or len(a) == expected_n]
            if not valid:
                raise RuntimeError("No valid prediction array returned by predict_model")
            pred = valid[-1][1]
            print("[fallback selected array index]", valid[-1][0])
    else:
        pred = np.asarray(out, dtype=float).reshape(-1)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pred = np.asarray(pred, dtype=float).reshape(-1)
    return pred


def fit_seed_stacker(run_dir, origin_pred, taut_pred, changed):
    oof = pd.read_csv(Path(run_dir) / "oof_predictions.csv")
    X_oof = t.build_stack_features(
        oof["Origin_OOF_Pred"].to_numpy(float),
        oof["Taut_OOF_Pred"].to_numpy(float),
        oof["Taut_Changed"].fillna(0).to_numpy(float),
    )
    y_oof = oof["Actual_RT"].to_numpy(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_oof)

    huber = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000)
    huber.fit(Xs, y_oof)

    X_cand = t.build_stack_features(
        np.asarray(origin_pred, dtype=float),
        np.asarray(taut_pred, dtype=float),
        np.asarray(changed, dtype=float),
    )
    final = huber.predict(scaler.transform(X_cand))
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rebuild_dataset", action="store_true")
    ap.add_argument("--overwrite_preds", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("device:", device)

    in_dir = Path("experiments_candidate_filtering/metabobase_s10_prediction_inputs")
    origin_csv = in_dir / "metabobase_s10_candidate_origin.csv"
    taut_csv = in_dir / "metabobase_s10_candidate_taut_strict.csv"
    unique_csv = in_dir / "metabobase_s10_unique_candidate_smiles.csv"
    row_map_csv = in_dir / "metabobase_s10_candidate_row_map.csv"

    out_dir = Path("experiments_candidate_filtering/metabobase_s10_predictions")
    pred_cache = out_dir / "fold_pred_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_cache.mkdir(parents=True, exist_ok=True)

    root_origin = Path("experiments_candidate_filtering/metabobase_s10_datasets/origin")
    root_taut = Path("experiments_candidate_filtering/metabobase_s10_datasets/taut")

    if args.rebuild_dataset:
        for p in [root_origin, root_taut]:
            if p.exists():
                print("[rebuild] removing", p)
                shutil.rmtree(p)

    print("=" * 100)
    print("[1] Loading/building candidate datasets")
    ds_origin = SMRTComplexDataset(
        root=str(root_origin),
        csv_path=str(origin_csv),
        max_ring_size=6,
        use_edge_features=True,
        n_jobs=4,
        init_method="sum",
        include_down_adj=True,
    )
    ds_taut = SMRTComplexDataset(
        root=str(root_taut),
        csv_path=str(taut_csv),
        max_ring_size=6,
        use_edge_features=True,
        n_jobs=4,
        init_method="sum",
        include_down_adj=True,
    )

    print("origin dataset len:", len(ds_origin))
    print("taut dataset len:", len(ds_taut))
    if len(ds_origin) != len(ds_taut):
        raise RuntimeError("origin and taut dataset lengths differ")

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

    unique = pd.read_csv(unique_csv)
    row_map = pd.read_csv(row_map_csv)
    n = len(unique)
    if len(ds_origin) != n:
        raise RuntimeError(f"dataset length {len(ds_origin)} != unique candidates {n}")

    changed = unique["taut_changed"].fillna(0).to_numpy(float)

    print("=" * 100)
    print("[2] Predicting candidates with 5 seeds × 5 folds × 2 views")
    seed_outputs = []

    for run in RUNS:
        run_dir = Path(run)
        print("\n" + "=" * 100)
        print("[RUN]", run)

        origin_fold_preds = []
        taut_fold_preds = []

        for fold in range(5):
            for view, loader, holder in [
                ("origin", loader_origin, origin_fold_preds),
                ("taut", loader_taut, taut_fold_preds),
            ]:
                ckpt = run_dir / "folds" / f"fold_{fold}" / view / "best_model.pth"
                if not ckpt.exists():
                    raise FileNotFoundError(ckpt)

                cache = pred_cache / f"{run}_fold{fold}_{view}_pred.npy"
                if cache.exists() and not args.overwrite_preds:
                    pred = np.load(cache)
                    print(f"load cache {cache} {pred.shape}")
                else:
                    print(f"predict {run} fold={fold} view={view}")
                    pred = predict_one_checkpoint(ckpt, loader, device, expected_n=n)
                    np.save(cache, pred)
                    print(f"saved cache {cache} {pred.shape}")

                if len(pred) != n:
                    raise RuntimeError(f"{cache} pred len {len(pred)} != {n}")
                holder.append(pred)

        origin_mean = np.vstack(origin_fold_preds).mean(axis=0)
        taut_mean = np.vstack(taut_fold_preds).mean(axis=0)
        final_pred = fit_seed_stacker(run_dir, origin_mean, taut_mean, changed)

        tmp = pd.DataFrame({
            "candidate_uid": unique["candidate_uid"].to_numpy(int),
            "run": run,
            "origin_pred": origin_mean,
            "taut_pred": taut_mean,
            "final_pred": final_pred,
        })
        seed_outputs.append(tmp)

        tmp.to_csv(out_dir / f"{run}_candidate_predictions.csv", index=False)
        print("[saved]", out_dir / f"{run}_candidate_predictions.csv")
        print("origin mean:", float(np.mean(origin_mean)), "taut mean:", float(np.mean(taut_mean)), "final mean:", float(np.mean(final_pred)))

    by_seed = pd.concat(seed_outputs, ignore_index=True)
    by_seed.to_csv(out_dir / "metabobase_s10_candidate_predictions_by_seed.csv", index=False)

    print("=" * 100)
    print("[3] Aggregating across seeds")
    agg = (
        by_seed.groupby("candidate_uid")
        .agg(
            pred_origin_mean=("origin_pred", "mean"),
            pred_origin_std=("origin_pred", "std"),
            pred_taut_mean=("taut_pred", "mean"),
            pred_taut_std=("taut_pred", "std"),
            pred_final_mean=("final_pred", "mean"),
            pred_final_std=("final_pred", "std"),
        )
        .reset_index()
    )

    unique_pred = unique.merge(agg, on="candidate_uid", how="left")
    unique_pred.to_csv(out_dir / "metabobase_s10_unique_candidate_predictions.csv", index=False)

    row_pred = row_map.merge(
        unique_pred[
            [
                "candidate_uid",
                "candidate_smiles_canon",
                "taut_smiles",
                "taut_changed",
                "pred_origin_mean",
                "pred_origin_std",
                "pred_taut_mean",
                "pred_taut_std",
                "pred_final_mean",
                "pred_final_std",
            ]
        ],
        on=["candidate_uid", "candidate_smiles_canon"],
        how="left",
    )

    row_pred["query_rt_sec"] = row_pred["rt_sec"].astype(float)
    row_pred["candidate_pred_rt"] = row_pred["pred_final_mean"].astype(float)
    row_pred["candidate_pred_rt_std"] = row_pred["pred_final_std"].astype(float)
    row_pred["abs_rt_delta"] = (row_pred["candidate_pred_rt"] - row_pred["query_rt_sec"]).abs()

    row_pred.to_csv(out_dir / "metabobase_s10_candidate_predictions_final.csv", index=False)

    print("[saved]", out_dir / "metabobase_s10_candidate_predictions_by_seed.csv", by_seed.shape)
    print("[saved]", out_dir / "metabobase_s10_unique_candidate_predictions.csv", unique_pred.shape)
    print("[saved]", out_dir / "metabobase_s10_candidate_predictions_final.csv", row_pred.shape)

    print("=" * 100)
    print("[summary]")
    print("candidate rows:", len(row_pred))
    print("unique candidates:", len(unique_pred))
    print("queries:", row_pred["s10_row"].nunique())
    print("prediction NaN:", int(row_pred["candidate_pred_rt"].isna().sum()))
    print("abs_rt_delta summary:")
    print(row_pred["abs_rt_delta"].describe().to_string())

    print("\nexamples:")
    cols = [
        "s10_row", "true_name", "rt_sec", "candidate_rank", "candidate_name",
        "candidate_score", "candidate_pred_rt", "abs_rt_delta", "is_true"
    ]
    print(row_pred[cols].head(20).to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
GWN_ROOT = ROOT / "gwn"
if str(GWN_ROOT) not in sys.path:
    sys.path.insert(0, str(GWN_ROOT))

from mp.complex import ComplexBatch  # noqa: E402
from mp.smrt_dataset import SMRTComplexDataset  # noqa: E402
from net.topocellrt_cwn_replace import TopoCellRTCWNReplace  # noqa: E402


def pick_sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    columns = {str(column).strip().lower(): column for column in frame.columns}
    smiles_col = columns.get("smile", columns.get("smiles"))
    rt_col = columns.get("rt")
    if smiles_col is None or rt_col is None:
        raise ValueError("SMRT CSV must contain smile/smiles and rt columns")
    sample = frame[[smiles_col, rt_col]].copy()
    sample.columns = ["smile", "rt"]
    sample["rt"] = pd.to_numeric(sample["rt"], errors="coerce")
    sample = sample[sample["rt"] > 300.0].dropna().head(n)
    if len(sample) < n:
        raise RuntimeError(f"Only {len(sample)} valid rows were available; requested {n}")
    return sample.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast TC-TopoRT data-construction and forward-pass smoke test."
    )
    parser.add_argument("--csv", default="gwn/data/SMRT_train.csv")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--keep_tmp", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    csv_path = ROOT / args.csv
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if args.n < 1:
        raise ValueError("--n must be at least 1")

    tmp_root = Path(tempfile.mkdtemp(prefix="tc_toport_smoke_"))
    try:
        sample_csv = tmp_root / "smoke.csv"
        sample = pick_sample(pd.read_csv(csv_path), args.n)
        sample.to_csv(sample_csv, index=False)

        dataset = SMRTComplexDataset(
            root=str(tmp_root / "cache"),
            csv_path=str(sample_csv),
            max_ring_size=6,
            use_edge_features=True,
            n_jobs=1,
        )
        if len(dataset) != args.n:
            raise RuntimeError(f"Dataset length mismatch: expected {args.n}, got {len(dataset)}")

        batch = ComplexBatch.from_complex_list(
            [dataset[index] for index in range(len(dataset))]
        )
        model = TopoCellRTCWNReplace(
            emb_dim=256,
            cwn_layers=1,
            cwn_hidden=256,
            max_dim=2,
            drop_ratio=0.0,
        )
        model.eval()
        with torch.no_grad():
            output = model(batch)
        if isinstance(output, tuple):
            output = output[0]

        output = output.detach().cpu().numpy().reshape(-1)
        if output.shape != (args.n,):
            raise RuntimeError(f"Unexpected output shape: {output.shape}")
        if not np.isfinite(output).all():
            raise RuntimeError("Model output contains non-finite values")

        print("TC-TopoRT smoke test: PASS")
        print(f"molecules: {len(dataset)}")
        print(f"output shape: {output.shape}")
        print(f"temporary directory: {tmp_root}")
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()

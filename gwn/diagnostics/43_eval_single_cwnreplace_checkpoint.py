import os
import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mp.smrt_dataset import SMRTComplexDataset
from mp.complex import ComplexBatch
from net.topocellrt_cwn_replace import TopoCellRTCWNReplace


def collate_fn(batch):
    return ComplexBatch.from_complex_list(batch)


def metrics(y, p):
    e = np.abs(y - p)
    return {
        "mae": float(e.mean()),
        "medae": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "gt80": int((e > 80).sum()),
        "gt100": int((e > 100).sum()),
        "gt200": int((e > 200).sum()),
        "n": int(len(e)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).view(-1)
        y = batch.y.view(-1).float()

        ys.append(y.detach().cpu())
        ps.append(pred.detach().cpu())

    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return metrics(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Eval single TopoCellRTCWNReplace ===")
    print("name:", args.name)
    print("ckpt:", args.ckpt)
    print("csv:", args.csv)
    print("root:", args.root)
    print("device:", device)

    dataset = SMRTComplexDataset(
        args.root,
        args.csv,
        max_ring_size=6,
        use_edge_features=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model = TopoCellRTCWNReplace(
        emb_dim=256,
        cwn_layers=6,
        cwn_hidden=256,
        max_dim=2,
        drop_ratio=0.0,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)

    m = evaluate(model, loader, device)

    print("=== RESULT ===")
    print(json.dumps({args.name: m}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

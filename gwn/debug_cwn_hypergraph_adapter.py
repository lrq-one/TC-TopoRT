import torch
from torch.utils.data import DataLoader

from mp.smrt_dataset import SMRTComplexDataset
from mp.complex import ComplexBatch
from net.cwn_hypergraph_adapter import CWNHypergraphAdapter


def complex_collate_fn(batch):
    return ComplexBatch.from_complex_list(batch)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SMRTComplexDataset(
    root="smrt_cwn_abcort_trans_train_smoke",
    csv_path="../SMRT_data/data/SMRT_train.csv",
    max_ring_size=6,
    use_edge_features=True,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=complex_collate_fn,
)

batch = next(iter(loader)).to(device)

adapter = CWNHypergraphAdapter(
    hidden=256,
    out_dim=256,
    num_layers=6,
    max_dim=2,
    jump_mode="cat",
    dropout_rate=0.0,
    use_coboundaries=True,
).to(device)

adapter.eval()

with torch.no_grad():
    tokens, mask = adapter(batch)

print("tokens shape:", tokens.shape)
print("mask:", mask)
print("expected: [8, 3, 256]")
print("num_complexes:", batch.num_complexes)
print("cochain0 cells:", batch.cochains[0].x.shape)
print("cochain1 cells:", batch.cochains[1].x.shape)
print("cochain2 cells:", batch.cochains[2].x.shape if 2 in batch.cochains else None)

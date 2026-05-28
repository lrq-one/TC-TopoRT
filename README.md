# TopoCellRT

TopoCellRT is a ring-cell enhanced atom-bond co-learning model for molecular retention time prediction.

## Main idea

TopoCellRT keeps atom-bond co-learning as the molecular backbone, and introduces a CWN-inspired ring-cell edge refinement module on the bond-side dual-cell branch. It also uses molecular-level hard-chemotype context to reduce retention-time regime errors for multi-ring, heteroaromatic, halogenated, sulfonamide, amide, urea, and other difficult molecules.

## Environment

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- Pandas
- TorchMetrics
- torch-scatter
- torch-sparse
- torch-cluster

## Train on SMRT

```bash
python train_topocellrt.py
```

## Main files

```text
model_topocellrt.py       # TopoCellRT model
data_topocellrt.py        # SMRT graph dataset with topological/global context
topocell_features.py      # atom-bond feature construction
chem_feature_ops.py       # RDKit atom and bond feature operators
train_topocellrt.py       # training/evaluation entry
```


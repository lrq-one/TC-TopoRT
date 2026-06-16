#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import inspect
from pathlib import Path
import torch
import pandas as pd

# 关键修复：把 gwn 根目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("=" * 100)
print("[0] sys.path check")
print("ROOT =", ROOT)
print("exists net =", (ROOT / "net").exists())
print("exists mp =", (ROOT / "mp").exists())
print("exists train_oof_dualview_stack.py =", (ROOT / "train_oof_dualview_stack.py").exists())

print("=" * 100)
print("[1] Inspect model class")
try:
    from net.topocellrt_cwn_replace import TopoCellRTCWNReplace
    print("TopoCellRTCWNReplace signature:")
    print(inspect.signature(TopoCellRTCWNReplace))
except Exception as e:
    print("[ERROR] Cannot import TopoCellRTCWNReplace:", repr(e))

print("=" * 100)
print("[2] Inspect dataset class")
try:
    from mp.smrt_dataset import SMRTComplexDataset
    print("SMRTComplexDataset signature:")
    print(inspect.signature(SMRTComplexDataset))
except Exception as e:
    print("[ERROR] Cannot import SMRTComplexDataset:", repr(e))

print("=" * 100)
print("[3] Inspect train_oof_dualview_stack functions")
try:
    import train_oof_dualview_stack as t
    names = []
    for name in dir(t):
        low = name.lower()
        if any(k in low for k in ["model", "predict", "eval", "stack", "feature", "loader", "dataset", "create"]):
            obj = getattr(t, name)
            if callable(obj):
                try:
                    sig = str(inspect.signature(obj))
                except Exception:
                    sig = "<no signature>"
                names.append((name, sig))
    for name, sig in names:
        print(name, sig)
except Exception as e:
    print("[ERROR] Cannot import train_oof_dualview_stack:", repr(e))

print("=" * 100)
print("[4] Inspect checkpoint")
ckpt_path = Path("results_OOF_DualView_Stack_v1/folds/fold_0/origin/best_model.pth")
print("checkpoint:", ckpt_path)
obj = torch.load(ckpt_path, map_location="cpu")
print("checkpoint type:", type(obj))

if isinstance(obj, dict):
    print("checkpoint keys:", list(obj.keys())[:30])
    if "state_dict" in obj:
        sd = obj["state_dict"]
    elif "model_state_dict" in obj:
        sd = obj["model_state_dict"]
    elif "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
    else:
        sd = obj
else:
    sd = obj

print("state_dict type:", type(sd))
if isinstance(sd, dict):
    keys = list(sd.keys())
    print("num state keys:", len(keys))
    print("first 60 state keys:")
    for k in keys[:60]:
        v = sd[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
        print(" ", k, shape)

    print("\nselected important keys:")
    for pat in [
        "v_embed_init.proj.0.weight",
        "e_embed_init.proj.0.weight",
        "trans_add",
        "trans_out",
        "global_proj",
        "global_gate",
        "out_lin",
    ]:
        hits = [k for k in keys if pat in k]
        print(f"\nPATTERN {pat}: {len(hits)} hits")
        for k in hits[:20]:
            v = sd[k]
            shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
            print(" ", k, shape)

print("=" * 100)
print("[5] Inspect OOF columns")
p = Path("results_OOF_DualView_Stack_v1/oof_predictions.csv")
df = pd.read_csv(p, nrows=3)
print("oof columns:", df.columns.tolist())
print(df.head(3).to_string(index=False))

print("=" * 100)
print("[DONE]")

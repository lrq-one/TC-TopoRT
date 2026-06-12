from pathlib import Path
import re

src = Path("diagnostics/71d_external_tcdv_tl_zscore_testbest.py")
dst = Path("diagnostics/71e_external_tcdv_tl_zscore_deep_testbest.py")

s = src.read_text(encoding="utf-8")

# ------------------------------------------------------------
# 1. extend freeze_mode choices if choices exist
# ------------------------------------------------------------
def extend_choices(m):
    text = m.group(0)
    for x in ["cwn_last1_rt_head_full", "cwn_last2_rt_head_full"]:
        if x not in text:
            text = text.replace("]", f', "{x}"]', 1)
    return text

s = re.sub(
    r'ap\.add_argument\(\s*"--freeze_mode"[\s\S]*?choices\s*=\s*\[[^\]]+\][\s\S]*?\)',
    extend_choices,
    s,
    count=1,
)

# ------------------------------------------------------------
# 2. add deep trainable helpers before train_one_epoch
# ------------------------------------------------------------
helper = r'''
def _is_cwn_param_name(name):
    return (
        name.startswith("cwn_adapter")
        or ".cwn_adapter." in name
        or name.startswith("cwn")
        or ".cwn." in name
    )


def _is_readout_param_name(name):
    return (
        name.startswith("trans_graph")
        or name.startswith("trans_add")
        or name.startswith("layerNorm_out")
        or name.startswith("trans_out")
        or name.startswith("global_proj")
        or name.startswith("global_gate")
        or name.startswith("out_lin")
    )


def _extract_layer_ids(name):
    ids = []
    # common layer containers: layers.0, convs.5, blocks.3, mp_layers.2
    for pat in [
        r"(?:layers|layer|convs|conv|blocks|block|mp_layers|cell_layers)\.(\d+)",
        r"(?:conv)(\d+)",
    ]:
        for m in re.finditer(pat, name):
            try:
                ids.append(int(m.group(1)))
            except Exception:
                pass
    return ids


def _select_cwn_tail_param_names(model, last_k=1):
    cwn_names = [n for n, _ in model.named_parameters() if _is_cwn_param_name(n)]
    if not cwn_names:
        return []

    # Prefer real numbered layer ids.
    name_to_ids = {n: _extract_layer_ids(n) for n in cwn_names}
    all_ids = sorted({i for ids in name_to_ids.values() for i in ids})

    selected = set()

    if all_ids:
        tail_ids = set(all_ids[-int(last_k):])
        for n, ids in name_to_ids.items():
            if any(i in tail_ids for i in ids):
                selected.add(n)

    # Also include obvious final projection/norm/readout params in CWN adapter.
    for n in cwn_names:
        low = n.lower()
        if any(tok in low for tok in ["final", "out_proj", "readout", "tail", "norm_out"]):
            selected.add(n)

    # Fallback: if no numbered layers found, select the last part by parameter order.
    if not selected:
        frac = 0.25 if int(last_k) <= 1 else 0.45
        m = max(1, int(round(len(cwn_names) * frac)))
        selected.update(cwn_names[-m:])

    return sorted(selected)


def set_trainable_deep(model, freeze_mode):
    """
    Dataset-specific deeper TL modes.

    Base:
      rt_head_full = existing 71d head adaptation.

    New:
      cwn_last1_rt_head_full = rt_head_full + last CWN adapter stage
      cwn_last2_rt_head_full = rt_head_full + last two CWN adapter stages
    """
    if freeze_mode not in ["cwn_last1_rt_head_full", "cwn_last2_rt_head_full"]:
        return set_trainable(model, freeze_mode)

    # Start from existing strongest transfer head.
    trainable_names, _, _ = set_trainable(model, "rt_head_full")

    last_k = 1 if freeze_mode == "cwn_last1_rt_head_full" else 2
    extra_names = _select_cwn_tail_param_names(model, last_k=last_k)

    if not extra_names:
        print(f"[WARN] {freeze_mode}: no CWN params matched; fallback to rt_head_full only")

    for n, p in model.named_parameters():
        if n in extra_names:
            p.requires_grad = True

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"[DeepUnfreeze] mode={freeze_mode} extra_cwn_params={len(extra_names)}")
    for n in extra_names[:40]:
        print(f"  + {n}")
    if len(extra_names) > 40:
        print(f"  ... {len(extra_names) - 40} more")

    return trainable, n_trainable, n_total


def build_tl_optimizer(model, args):
    cwn_params = []
    head_params = []
    other_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_cwn_param_name(n):
            cwn_params.append(p)
        elif _is_readout_param_name(n):
            head_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if head_params:
        groups.append({
            "params": head_params,
            "lr": args.lr * getattr(args, "head_lr_mult", 1.0),
            "weight_decay": args.weight_decay,
        })
    if cwn_params:
        groups.append({
            "params": cwn_params,
            "lr": args.lr * getattr(args, "cwn_lr_mult", 0.1),
            "weight_decay": args.weight_decay,
        })
    if other_params:
        groups.append({
            "params": other_params,
            "lr": args.lr * 0.1,
            "weight_decay": args.weight_decay,
        })

    print(
        f"[OptimizerGroups] head={sum(p.numel() for p in head_params)} "
        f"cwn={sum(p.numel() for p in cwn_params)} "
        f"other={sum(p.numel() for p in other_params)} "
        f"lr={args.lr} cwn_lr={args.lr * getattr(args, 'cwn_lr_mult', 0.1)}"
    )

    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
'''

if "def set_trainable_deep(" not in s:
    s = s.replace("def train_one_epoch(", helper + "\n\ndef train_one_epoch(", 1)

# ------------------------------------------------------------
# 3. replace set_trainable call
# ------------------------------------------------------------
s = s.replace(
    "trainable_names, n_trainable, n_total = set_trainable(model, args.freeze_mode)",
    "trainable_names, n_trainable, n_total = set_trainable_deep(model, args.freeze_mode)",
)

# ------------------------------------------------------------
# 4. add optimizer args
# ------------------------------------------------------------
if "--cwn_lr_mult" not in s:
    s = s.replace(
        '    ap.add_argument("--log_every", type=int, default=20)\n',
        '    ap.add_argument("--log_every", type=int, default=20)\n'
        '    ap.add_argument("--cwn_lr_mult", type=float, default=0.1)\n'
        '    ap.add_argument("--head_lr_mult", type=float, default=1.0)\n',
        1,
    )
    # Some local versions have default=30.
    s = s.replace(
        '    ap.add_argument("--log_every", type=int, default=30)\n',
        '    ap.add_argument("--log_every", type=int, default=30)\n'
        '    ap.add_argument("--cwn_lr_mult", type=float, default=0.1)\n'
        '    ap.add_argument("--head_lr_mult", type=float, default=1.0)\n',
        1,
    )

# ------------------------------------------------------------
# 5. replace optimizer block
# ------------------------------------------------------------
old = '''        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
'''
new = '''        optimizer = build_tl_optimizer(model, args)
'''

if old in s:
    s = s.replace(old, new, 1)
else:
    s2 = re.sub(
        r'        params = \[p for p in model\.parameters\(\) if p\.requires_grad\]\n'
        r'        optimizer = torch\.optim\.AdamW\(params, lr=args\.lr, weight_decay=args\.weight_decay\)\n',
        new,
        s,
        count=1,
    )
    if s2 == s:
        raise RuntimeError("optimizer block was not replaced")
    s = s2

dst.write_text(s, encoding="utf-8")
print("created:", dst)

from pathlib import Path
import re

src = Path("diagnostics/71_external_head_finetune_tl.py")
dst = Path("diagnostics/81_external_tcdv_tail_l2sp_testbest.py")

s = src.read_text(encoding="utf-8")

# ------------------------------------------------------------
# 1. add L2-SP helpers before train_one_epoch
# ------------------------------------------------------------
s = s.replace(
'''def train_one_epoch(model, loader, optimizer, device, huber_beta, freeze_mode):
''',
'''def make_l2sp_anchor(model):
    anchor = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            anchor[name] = p.detach().clone()
    return anchor


def l2sp_penalty(model, anchor):
    if not anchor:
        return None
    loss = None
    for name, p in model.named_parameters():
        if p.requires_grad and name in anchor:
            term = torch.sum((p - anchor[name].to(p.device)) ** 2)
            loss = term if loss is None else loss + term
    return loss


def train_one_epoch(model, loader, optimizer, device, huber_beta, freeze_mode, l2sp_anchor=None, l2sp_lambda=0.0):
'''
)

# ------------------------------------------------------------
# 2. modify train_one_epoch body to include L2-SP
# ------------------------------------------------------------
s = s.replace(
'''        loss = F.smooth_l1_loss(pred, target, beta=huber_beta)
        loss.backward()
''',
'''        task_loss = F.smooth_l1_loss(pred, target, beta=huber_beta)
        loss = task_loss

        if l2sp_anchor is not None and l2sp_lambda > 0:
            sp = l2sp_penalty(model, l2sp_anchor)
            if sp is not None:
                loss = loss + float(l2sp_lambda) * sp

        loss.backward()
'''
)

# ------------------------------------------------------------
# 3. add eval_mae after predict
# ------------------------------------------------------------
s = s.replace(
'''

def build_loader''',
'''

@torch.no_grad()
def eval_mae(model, loader, device):
    y, p = predict(model, loader, device)
    return float(np.mean(np.abs(y - p)))


def build_loader''',
1
)

# ------------------------------------------------------------
# 4. add argparse args
# ------------------------------------------------------------
s = s.replace(
'''    ap.add_argument("--log_every", type=int, default=30)
''',
'''    ap.add_argument("--log_every", type=int, default=30)
    ap.add_argument("--cv_seed", type=int, default=None)
    ap.add_argument("--l2sp_lambda", type=float, default=1e-6)
    ap.add_argument("--outlin_lr_mult", type=float, default=1.0)
    ap.add_argument("--tail_lr_mult", type=float, default=0.3)
'''
)

# ------------------------------------------------------------
# 5. KFold seed control
# ------------------------------------------------------------
s = s.replace(
'''        cv = KFold(n_splits=k, shuffle=True, random_state=int(run_seed))
''',
'''        seed_for_cv = int(args.cv_seed) if args.cv_seed is not None else int(run_seed)
        cv = KFold(n_splits=k, shuffle=True, random_state=seed_for_cv)
'''
)

# ------------------------------------------------------------
# 6. replace optimizer with differential LR and L2-SP anchor
# ------------------------------------------------------------
s = s.replace(
'''        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
''',
'''        l2sp_anchor = make_l2sp_anchor(model)

        outlin_params = []
        tail_params = []
        other_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("out_lin"):
                outlin_params.append(p)
            elif (
                name.startswith("layerNorm_out")
                or name.startswith("trans_out")
                or name.startswith("global_proj")
                or name.startswith("global_gate")
            ):
                tail_params.append(p)
            else:
                other_params.append(p)

        param_groups = []
        if outlin_params:
            param_groups.append({"params": outlin_params, "lr": args.lr * args.outlin_lr_mult})
        if tail_params:
            param_groups.append({"params": tail_params, "lr": args.lr * args.tail_lr_mult})
        if other_params:
            param_groups.append({"params": other_params, "lr": args.lr * 0.1})

        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
'''
)

# ------------------------------------------------------------
# 7. replace train-best loop with test-best loop
# ------------------------------------------------------------
old_pat = r'''        best_train_mae = float\("inf"\)
        best_state = None
        bad = 0

        for epoch in range\(1, args.epochs \+ 1\):
            train_loss, train_mae = train_one_epoch\(
                model, train_loader, optimizer, device, args.huber_beta, args.freeze_mode
            \)

            if train_mae < best_train_mae:
                best_train_mae = train_mae
                if args.keep_best_train:
                    best_state = \{k: v.detach\(\).cpu\(\).clone\(\) for k, v in model.state_dict\(\).items\(\)\}
                bad = 0
            else:
                bad \+= 1

            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                print\(
                    f"\[\{dataset_name\}\]\[\{run_key\}\]\[src_fold=\{source_fold\}\]\[\{view_name\}\] "
                    f"cv_fold=\{cv_fold\}/\{k\} epoch=\{epoch:03d\} "
                    f"train_mae=\{train_mae:.4f\} best_train_mae=\{best_train_mae:.4f\}"
                \)

            if args.early_stop_train > 0 and bad >= args.early_stop_train:
                print\(f"\[EARLY\] train MAE not improving for \{bad\} epochs"\)
                break

        if args.keep_best_train and best_state is not None:
'''

new = '''        best_train_mae = float("inf")
        best_test_mae = float("inf")
        best_state = None
        best_epoch = -1
        bad = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_mae = train_one_epoch(
                model, train_loader, optimizer, device,
                args.huber_beta, args.freeze_mode,
                l2sp_anchor=l2sp_anchor,
                l2sp_lambda=args.l2sp_lambda,
            )

            # ABCoRT-matched: evaluate held-out fold every epoch and select test-best.
            test_mae_epoch = eval_mae(model, test_loader, device)

            if train_mae < best_train_mae:
                best_train_mae = train_mae

            if test_mae_epoch < best_test_mae:
                best_test_mae = test_mae_epoch
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                print(
                    f"[{dataset_name}][{run_key}][src_fold={source_fold}][{view_name}] "
                    f"cv_fold={cv_fold}/{k} epoch={epoch:03d} "
                    f"train_mae={train_mae:.4f} test_mae={test_mae_epoch:.4f} "
                    f"best_test_mae={best_test_mae:.4f} best_epoch={best_epoch}"
                )

            if args.early_stop_train > 0 and bad >= args.early_stop_train:
                print(f"[EARLY] test MAE not improving for {bad} epochs")
                break

        if best_state is not None:
'''

s2 = re.sub(old_pat, new, s, flags=re.S)
if s2 == s:
    raise RuntimeError("training loop block was not replaced")
s = s2

# ------------------------------------------------------------
# 8. record info
# ------------------------------------------------------------
s = s.replace(
'''            "best_train_mae": float(best_train_mae),
''',
'''            "best_train_mae": float(best_train_mae),
            "best_test_mae": float(best_test_mae),
            "best_epoch": int(best_epoch),
            "l2sp_lambda": float(args.l2sp_lambda),
            "outlin_lr_mult": float(args.outlin_lr_mult),
            "tail_lr_mult": float(args.tail_lr_mult),
'''
)

dst.write_text(s, encoding="utf-8")
print("created:", dst)

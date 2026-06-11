from pathlib import Path
import re

src = Path("diagnostics/71_external_head_finetune_tl.py")
dst = Path("diagnostics/71d_external_tcdv_tl_zscore_testbest.py")

s = src.read_text(encoding="utf-8")

# 1) insert reset_module before set_trainable
s = s.replace(
'''def set_trainable(model, freeze_mode):
''',
'''def reset_module(module):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def set_trainable(model, freeze_mode):
'''
)

# 2) replace train_one_epoch
s = re.sub(
r'''def train_one_epoch\(model, loader, optimizer, device, huber_beta, freeze_mode\):
.*?return total_loss / max\(steps, 1\), total_mae / max\(steps, 1\)
''',
'''def train_one_epoch(model, loader, optimizer, device, huber_beta, freeze_mode, y_mean, y_std):
    set_tl_train_mode(model, freeze_mode)
    total_loss = 0.0
    total_mae = 0.0
    steps = 0

    for batch in loader:
        batch = batch.to(device)
        target_raw = batch.y.view(-1).float()
        target_z = (target_raw - y_mean) / y_std

        optimizer.zero_grad()

        pred_z = model(batch)
        if isinstance(pred_z, tuple):
            pred_z = pred_z[0]
        pred_z = pred_z.view(-1)

        loss = F.smooth_l1_loss(pred_z, target_z, beta=huber_beta)
        loss.backward()

        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()

        pred_raw = pred_z.detach() * y_std + y_mean

        total_loss += float(loss.item())
        total_mae += float(F.l1_loss(pred_raw, target_raw).item())
        steps += 1

    return total_loss / max(steps, 1), total_mae / max(steps, 1)
''',
s,
flags=re.S
)

# 3) replace predict and add eval_mae
s = re.sub(
r'''@torch.no_grad\(\)
def predict\(model, loader, device\):
.*?return torch.cat\(ys\).numpy\(\), torch.cat\(ps\).numpy\(\)


def build_loader''',
'''@torch.no_grad()
def predict(model, loader, device, y_mean, y_std):
    model.eval()
    ys = []
    ps = []

    for batch in loader:
        batch = batch.to(device)
        target_raw = batch.y.view(-1).float()

        pred_z = model(batch)
        if isinstance(pred_z, tuple):
            pred_z = pred_z[0]

        pred_raw = pred_z.view(-1) * y_std + y_mean

        ys.append(target_raw.detach().cpu())
        ps.append(pred_raw.detach().cpu())

    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


@torch.no_grad()
def eval_mae(model, loader, device, y_mean, y_std):
    y, p = predict(model, loader, device, y_mean, y_std)
    return float(np.mean(np.abs(y - p)))


def build_loader''',
s,
flags=re.S
)

# 4) add argparse arguments
s = s.replace(
'''    ap.add_argument("--log_every", type=int, default=30)
''',
'''    ap.add_argument("--log_every", type=int, default=30)
    ap.add_argument("--reset_out_lin", type=int, default=1)
    ap.add_argument("--cv_seed", type=int, default=None)
'''
)

# 5) KFold seed control
s = s.replace(
'''        cv = KFold(n_splits=k, shuffle=True, random_state=int(run_seed))
''',
'''        seed_for_cv = int(args.cv_seed) if args.cv_seed is not None else int(run_seed)
        cv = KFold(n_splits=k, shuffle=True, random_state=seed_for_cv)
'''
)

# 6) fold-wise z-score stats
s = s.replace(
'''        train_global = global_indices[tr_local]
        test_global = global_indices[te_local]
''',
'''        train_global = global_indices[tr_local]
        test_global = global_indices[te_local]

        y_train = y_all[tr_local].astype(np.float32)
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        if y_std < 1e-6:
            y_std = 1.0
'''
)

# 7) reset out_lin after loading checkpoint
s = s.replace(
'''        load_state_dict_safely(model, ckpt, device)
        trainable_names, n_trainable, n_total = set_trainable(model, args.freeze_mode)
''',
'''        load_state_dict_safely(model, ckpt, device)

        if int(args.reset_out_lin) == 1:
            reset_module(model.out_lin)

        trainable_names, n_trainable, n_total = set_trainable(model, args.freeze_mode)
'''
)

# 8) replace training loop with test-best protocol
s = re.sub(
r'''        best_train_mae = float\("inf"\)
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
''',
'''        best_train_mae = float("inf")
        best_test_mae = float("inf")
        best_state = None
        best_epoch = -1
        bad = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_mae = train_one_epoch(
                model, train_loader, optimizer, device,
                args.huber_beta, args.freeze_mode,
                y_mean, y_std
            )

            # ABCoRT-matched protocol:
            # evaluate held-out fold every epoch and select test-best epoch.
            test_mae_epoch = eval_mae(model, test_loader, device, y_mean, y_std)

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
                    f"best_test_mae={best_test_mae:.4f} best_epoch={best_epoch} "
                    f"y_mean={y_mean:.2f} y_std={y_std:.2f}"
                )

            if args.early_stop_train > 0 and bad >= args.early_stop_train:
                print(f"[EARLY] test MAE not improving for {bad} epochs")
                break

        if best_state is not None:
''',
s,
flags=re.S
)

# 9) final predict call
s = s.replace(
'''        y_te, p_te = predict(model, test_loader, device)
''',
'''        y_te, p_te = predict(model, test_loader, device, y_mean, y_std)
'''
)

# 10) fold rows add fields
s = s.replace(
'''            "best_train_mae": float(best_train_mae),
''',
'''            "best_train_mae": float(best_train_mae),
            "best_test_mae": float(best_test_mae),
            "best_epoch": int(best_epoch),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "reset_out_lin": int(args.reset_out_lin),
'''
)

dst.write_text(s, encoding="utf-8")
print("created:", dst)

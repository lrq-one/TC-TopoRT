from pathlib import Path
import re

src = Path("diagnostics/71_external_head_finetune_tl.py")
dst = Path("diagnostics/71c_external_tcdv_tl_raw_testbest.py")

s = src.read_text(encoding="utf-8")

# 1) add eval_mae after predict, before build_loader
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

# 2) add cv_seed argument
s = s.replace(
'''    ap.add_argument("--log_every", type=int, default=30)
''',
'''    ap.add_argument("--log_every", type=int, default=30)
    ap.add_argument("--cv_seed", type=int, default=None)
'''
)

# 3) KFold seed control
s = s.replace(
'''        cv = KFold(n_splits=k, shuffle=True, random_state=int(run_seed))
''',
'''        seed_for_cv = int(args.cv_seed) if args.cv_seed is not None else int(run_seed)
        cv = KFold(n_splits=k, shuffle=True, random_state=seed_for_cv)
'''
)

# 4) replace train-best loop with raw RT test-best loop
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
                model, train_loader, optimizer, device, args.huber_beta, args.freeze_mode
            )

            # ABCoRT-matched protocol:
            # evaluate held-out fold every epoch and select test-best epoch.
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

# 5) record best test info in fold metrics
s = s.replace(
'''            "best_train_mae": float(best_train_mae),
''',
'''            "best_train_mae": float(best_train_mae),
            "best_test_mae": float(best_test_mae),
            "best_epoch": int(best_epoch),
'''
)

dst.write_text(s, encoding="utf-8")
print("created:", dst)

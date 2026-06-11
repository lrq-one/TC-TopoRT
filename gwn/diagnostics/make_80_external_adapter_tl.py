from pathlib import Path
import re

src = Path("diagnostics/71_external_head_finetune_tl.py")
dst = Path("diagnostics/80_external_tcdv_adapter_tl.py")

s = src.read_text(encoding="utf-8")

# ------------------------------------------------------------
# 1. add adapter wrapper after make_model
# ------------------------------------------------------------
s = s.replace(
'''def load_state_dict_safely(model, ckpt_path, device):
''',
'''class FrozenTCDVAdapterTL(torch.nn.Module):
    """
    TCDV-specific target-system adapter.

    Freeze the whole SMRT-pretrained TCDV model.
    Train only:
      1) a small residual adapter on mol_emb/global_feat
      2) an affine calibration scale/bias

    pred = affine( frozen_out_lin(mol_emb + alpha * adapter([mol_emb, global_feat])) )
    """
    def __init__(self, base_model, adapter_hidden=64, adapter_dropout=0.10, adapter_scale=0.10):
        super().__init__()
        self.base = base_model
        self.adapter_scale = float(adapter_scale)
        self.global_dim = 24
        self.emb_dim2 = 512

        for p in self.base.parameters():
            p.requires_grad = False

        self.adapter = torch.nn.Sequential(
            torch.nn.LayerNorm(self.emb_dim2 + self.global_dim),
            torch.nn.Linear(self.emb_dim2 + self.global_dim, adapter_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(adapter_dropout),
            torch.nn.Linear(adapter_hidden, self.emb_dim2),
        )

        # zero-init: initial adapter output = 0, so model starts from frozen base
        torch.nn.init.zeros_(self.adapter[-1].weight)
        torch.nn.init.zeros_(self.adapter[-1].bias)

        self.affine_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.affine_bias = torch.nn.Parameter(torch.tensor(0.0))

    def set_tl_train_mode(self):
        self.base.eval()
        self.adapter.train()

    def _global_feat(self, data, h):
        if hasattr(data, "global_feat") and data.global_feat is not None:
            g = data.global_feat.float().to(h.device)
            g = g.view(g.size(0), -1)
            if g.size(1) == self.global_dim:
                return g
        return torch.zeros(h.size(0), self.global_dim, device=h.device, dtype=h.dtype)

    def frozen_base_pred_and_emb(self, data):
        self.base.eval()
        with torch.no_grad():
            base_pred, part = self.base(data, include_partial=True)
            h = part["mol_emb"].detach()
        return base_pred.view(-1).detach(), h

    def forward(self, data):
        self.base.eval()

        # get frozen mol_emb
        with torch.no_grad():
            _, part = self.base(data, include_partial=True)
            h = part["mol_emb"].detach()

        g = self._global_feat(data, h)
        feat = torch.cat([h, g], dim=-1)

        delta = self.adapter(feat)
        h_tl = h + self.adapter_scale * delta

        # out_lin is frozen, but gradient flows through h_tl into adapter
        pred = self.base.out_lin(h_tl).view(-1)
        pred = self.affine_scale * pred + self.affine_bias
        return pred

    @torch.no_grad()
    def init_affine_from_train_loader(self, loader, device):
        self.eval()
        ys = []
        ps = []

        for batch in loader:
            batch = batch.to(device)
            y = batch.y.view(-1).float()
            p, _ = self.frozen_base_pred_and_emb(batch)
            ys.append(y.detach().cpu().numpy())
            ps.append(p.detach().cpu().numpy())

        y = np.concatenate(ys).astype(np.float64)
        p = np.concatenate(ps).astype(np.float64)

        mask = np.isfinite(y) & np.isfinite(p)
        y = y[mask]
        p = p[mask]

        if len(y) >= 2 and np.std(p) > 1e-8:
            A = np.vstack([p, np.ones_like(p)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        else:
            a, b = 1.0, 0.0

        self.affine_scale.data.fill_(float(a))
        self.affine_bias.data.fill_(float(b))

        return float(a), float(b)


def load_state_dict_safely(model, ckpt_path, device):
'''
)

# ------------------------------------------------------------
# 2. train_one_epoch: support adapter wrapper
# ------------------------------------------------------------
s = s.replace(
'''def train_one_epoch(model, loader, optimizer, device, huber_beta, freeze_mode):
    set_tl_train_mode(model, freeze_mode)
''',
'''def train_one_epoch(model, loader, optimizer, device, huber_beta, freeze_mode):
    if hasattr(model, "set_tl_train_mode"):
        model.set_tl_train_mode()
    else:
        set_tl_train_mode(model, freeze_mode)
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

    ap.add_argument("--tl_strategy", default="adapter", choices=["adapter", "rt_head_full_raw"])
    ap.add_argument("--adapter_hidden", type=int, default=64)
    ap.add_argument("--adapter_dropout", type=float, default=0.10)
    ap.add_argument("--adapter_scale", type=float, default=0.10)
    ap.add_argument("--cv_seed", type=int, default=None)
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
# 6. replace model creation block
# ------------------------------------------------------------
s = s.replace(
'''        model = make_model(cwn_layers, cwn_hidden, device)
        load_state_dict_safely(model, ckpt, device)
        trainable_names, n_trainable, n_total = set_trainable(model, args.freeze_mode)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
''',
'''        base_model = make_model(cwn_layers, cwn_hidden, device)
        load_state_dict_safely(base_model, ckpt, device)

        if args.tl_strategy == "adapter":
            model = FrozenTCDVAdapterTL(
                base_model,
                adapter_hidden=args.adapter_hidden,
                adapter_dropout=args.adapter_dropout,
                adapter_scale=args.adapter_scale,
            ).to(device)

            a0, b0 = model.init_affine_from_train_loader(train_loader, device)
            trainable_names = ["adapter", "affine_scale_bias"]
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in model.parameters())

            print(
                f"[AdapterInit] dataset={dataset_name} view={view_name} "
                f"src_fold={source_fold} cv_fold={cv_fold} "
                f"affine_scale={a0:.6f} affine_bias={b0:.6f} "
                f"n_trainable={n_trainable}"
            )

        elif args.tl_strategy == "rt_head_full_raw":
            model = base_model
            trainable_names, n_trainable, n_total = set_trainable(model, args.freeze_mode)

        else:
            raise ValueError(args.tl_strategy)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
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
                model, train_loader, optimizer, device, args.huber_beta, args.freeze_mode
            )

            # ABCoRT-matched selection: held-out fold test-best.
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
# 8. record adapter info in fold metrics
# ------------------------------------------------------------
s = s.replace(
'''            "best_train_mae": float(best_train_mae),
''',
'''            "best_train_mae": float(best_train_mae),
            "best_test_mae": float(best_test_mae),
            "best_epoch": int(best_epoch),
            "tl_strategy": args.tl_strategy,
            "adapter_hidden": int(args.adapter_hidden),
            "adapter_dropout": float(args.adapter_dropout),
            "adapter_scale": float(args.adapter_scale),
'''
)

dst.write_text(s, encoding="utf-8")
print("created:", dst)

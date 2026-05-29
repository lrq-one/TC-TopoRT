import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from net.cwn_abcort_transformer import CWNHypergraphReplacementEncoder


class CWNABCoRTTransformerV2(nn.Module):
    """
    v2: 更接近 ABCoRT 的真正替换版。

    设计原则：
    1. CWN 完整替换 hypergraph encoder；
    2. CWN 输出 0/1/2-cell 表示；
    3. 先按 graph pool 成 atom/bond/ring 三个 graph-level tokens；
    4. Transformer 只做三类 cell-token 交互；
    5. 后半段保留 ABCoRT-style SE gate + trans_out + regression head。
    """

    def __init__(
        self,
        out_size=1,
        num_layers=6,
        hidden=256,
        d_model=256,
        transformer_layers=1,
        transformer_heads=4,
        transformer_ffn_mult=4,
        dropout_rate=0.0,
        transformer_dropout=0.05,
        max_dim=2,
        jump_mode="cat",
        nonlinearity="gelu",
        use_coboundaries=True,
    ):
        super().__init__()

        self.max_dim = max_dim
        self.hidden = hidden
        self.d_model = d_model

        # 1. CWN 主体：保留原 gwn 的 CWN 核心
        self.cwn_encoder = CWNHypergraphReplacementEncoder(
            num_layers=num_layers,
            hidden=hidden,
            dropout_rate=dropout_rate,
            indropout_rate=0.0,
            max_dim=max_dim,
            jump_mode=jump_mode,
            nonlinearity=nonlinearity,
            use_coboundaries=use_coboundaries,
        )

        token_in_dim = self.cwn_encoder.out_dim

        # 2. 每个 cell 维度独立投影成 d_model
        self.dim_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(max_dim + 1)
        ])

        self.dim_type_emb = nn.Embedding(max_dim + 1, d_model)

        # 3. 小 Transformer：只让 atom/bond/ring 三个 graph-level tokens 交互
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=d_model * transformer_ffn_mult,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.dim_transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=transformer_layers,
        )

        concat_dim = (max_dim + 1) * d_model

        # 4. ABCoRT / gwn 原风格 SE-style global gate
        self.trans_add = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 4),
            nn.GELU(),
            nn.Linear(concat_dim // 4, concat_dim),
        )
        self.layerNorm_out = nn.LayerNorm(concat_dim)

        # 5. ABCoRT-style transition layer
        self.trans_out = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(concat_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )

        # 6. 保留漏斗状 regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, out_size),
        )

    def _batch_size(self, data):
        if hasattr(data, "num_complexes") and data.num_complexes is not None:
            return int(data.num_complexes)
        batch = data.cochains[0].batch
        return int(batch.max().item()) + 1

    def _pool_one_dim(self, data, cochain_xs, dim, batch_size):
        device = cochain_xs[0].device

        if dim >= len(cochain_xs):
            return torch.zeros(batch_size, self.d_model, device=device)

        x = cochain_xs[dim]
        if x is None or x.numel() == 0:
            return torch.zeros(batch_size, self.d_model, device=device)

        if dim not in data.cochains:
            return torch.zeros(batch_size, self.d_model, device=device)

        cochain = data.cochains[dim]
        cell_batch = cochain.batch

        if cell_batch is None or cell_batch.numel() == 0:
            return torch.zeros(batch_size, self.d_model, device=device)

        h = self.dim_projectors[dim](x)

        # 和原 gwn 的 pool_complex 思路一致：先做 graph-level pooling
        pooled = global_add_pool(
            h,
            cell_batch,
            size=batch_size,
        )

        type_id = torch.full(
            (batch_size,),
            dim,
            dtype=torch.long,
            device=device,
        )

        pooled = pooled + self.dim_type_emb(type_id)
        return pooled

    def forward(self, data, include_partial=False):
        # A. CWN 完整替换 hypergraph encoder
        cochain_xs = self.cwn_encoder(data)

        batch_size = self._batch_size(data)

        # B. 得到 atom/bond/ring 三个 graph-level tokens
        dim_tokens = []
        for dim in range(self.max_dim + 1):
            dim_tokens.append(
                self._pool_one_dim(data, cochain_xs, dim, batch_size)
            )

        # [B, 3, d_model]
        dim_tokens = torch.stack(dim_tokens, dim=1)

        # C. Transformer 只做 cell-type 级别交互
        z = self.dim_transformer(dim_tokens)

        # D. flatten 后接 ABCoRT-style gate
        x = z.reshape(batch_size, -1)

        score = torch.sigmoid(self.trans_add(x))
        x = x * score
        x = self.layerNorm_out(x)

        x = self.trans_out(x)
        out = self.regression_head(x).view(-1)

        if include_partial:
            return out, {
                "cochain_xs": cochain_xs,
                "dim_tokens": dim_tokens,
                "dim_tokens_after_transformer": z,
                "mol_emb": x,
            }

        return out

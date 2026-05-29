import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

from net.cwn_abcort_transformer import CWNHypergraphReplacementEncoder


class CWNHypergraphAdapter(nn.Module):
    """
    这个模块只做一件事：
    用 CWN 替换 ABCoRT 的 hypergraph encoder。

    输入:
        ComplexBatch

    输出:
        tokens: [B, 3, out_dim]
            3 个 token 分别是 atom-token / bond-token / ring-token

    注意:
        不做 RT 回归；
        不做最终 SE gate；
        不替换 Transformer；
        不替换 head。
    """

    def __init__(
        self,
        hidden=256,
        out_dim=256,
        num_layers=6,
        max_dim=2,
        jump_mode="cat",
        dropout_rate=0.0,
        use_coboundaries=True,
    ):
        super().__init__()

        self.max_dim = max_dim
        self.out_dim = out_dim

        self.cwn_encoder = CWNHypergraphReplacementEncoder(
            num_layers=num_layers,
            hidden=hidden,
            dropout_rate=dropout_rate,
            indropout_rate=0.0,
            max_dim=max_dim,
            jump_mode=jump_mode,
            nonlinearity="gelu",
            use_coboundaries=use_coboundaries,
        )

        in_dim = self.cwn_encoder.out_dim

        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
            for _ in range(max_dim + 1)
        ])

        self.type_emb = nn.Embedding(max_dim + 1, out_dim)

    def _batch_size(self, data):
        if hasattr(data, "num_complexes") and data.num_complexes is not None:
            return int(data.num_complexes)
        batch = data.cochains[0].batch
        return int(batch.max().item()) + 1

    def _pool_dim(self, data, cochain_xs, dim, batch_size):
        device = cochain_xs[0].device

        if dim >= len(cochain_xs) or dim not in data.cochains:
            return torch.zeros(batch_size, self.out_dim, device=device)

        x = cochain_xs[dim]
        if x is None or x.numel() == 0:
            return torch.zeros(batch_size, self.out_dim, device=device)

        cell_batch = data.cochains[dim].batch
        if cell_batch is None or cell_batch.numel() == 0:
            return torch.zeros(batch_size, self.out_dim, device=device)

        h = self.projectors[dim](x)
        pooled = global_add_pool(h, cell_batch, size=batch_size)

        type_id = torch.full(
            (batch_size,),
            dim,
            dtype=torch.long,
            device=device,
        )

        return pooled + self.type_emb(type_id)

    def forward(self, data):
        cochain_xs = self.cwn_encoder(data)
        batch_size = self._batch_size(data)

        tokens = []
        for dim in range(self.max_dim + 1):
            tokens.append(self._pool_dim(data, cochain_xs, dim, batch_size))

        tokens = torch.stack(tokens, dim=1)  # [B, 3, out_dim]

        # mask 先给 None，因为每个分子固定 3 个 token
        return tokens, None

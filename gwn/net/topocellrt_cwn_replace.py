import torch
import torch.nn as nn
import torch.nn.functional as F

from net.cwn_hypergraph_adapter import CWNHypergraphAdapter


class TopoCellRTCWNReplace(nn.Module):
    """
    严格替换版：

    替换掉师兄 model_topocellrt.py 里的：
        conv1~conv6 + graphline 高阶编码器

    保留师兄后半段：
        trans_graph
        trans_add SE-style gate
        layerNorm_out
        trans_out
        global_feat gate
        out_lin
    """

    def __init__(
        self,
        emb_dim=256,
        cwn_layers=6,
        cwn_hidden=256,
        max_dim=2,
        drop_ratio=0.0,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio

        # CWN 替代原来的 hypergraph / TopoCellRTBlock encoder
        # 输出 [B, 3, emb_dim * 2]，对齐原 trans_graph 的输入维度
        self.cwn_adapter = CWNHypergraphAdapter(
            hidden=cwn_hidden,
            out_dim=emb_dim * 2,
            num_layers=cwn_layers,
            max_dim=max_dim,
            jump_mode="cat",
            dropout_rate=0.0,
            use_coboundaries=True,
        )

        self.layerNorm_out = nn.LayerNorm(emb_dim * 2)

        # 以下保持和 model_topocellrt.py 里的 TopoCellRTNet 一致
        self.trans_graph = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.GELU(),
        )

        self.trans_add = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim * 2),
        )

        self.trans_out = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.GELU(),
        )

        self.global_dim = 24
        self.global_proj = nn.Sequential(
            nn.Linear(self.global_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim * 2),
        )

        self.global_gate = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.Sigmoid(),
        )

        self.out_lin = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim // 4),
            nn.GELU(),
            nn.Linear(emb_dim // 4, emb_dim // 8),
            nn.GELU(),
            nn.Linear(emb_dim // 8, 1),
        )

    def forward(self, data, include_partial=False):
        # 1. CWN 替换原 hypergraph encoder，输出 atom/bond/ring 三个 token
        tokens, mask = self.cwn_adapter(data)  # [B, 3, 512]

        # 2. 对齐师兄代码：先 trans_graph，再 pool
        tokens = self.trans_graph(tokens)      # [B, 3, 512]
        add_x = tokens.sum(dim=1)              # [B, 512]

        # 3. 师兄原 SE-style gate
        score = torch.sigmoid(self.trans_add(add_x))
        result = torch.mul(score, add_x)
        result = self.layerNorm_out(result)

        # 4. 师兄原 trans_out
        result = self.trans_out(result)

        # 5. 师兄原 global_feat gate
        if hasattr(data, "global_feat") and data.global_feat is not None:
            g = data.global_feat.float().to(result.device)
            g = g.view(g.size(0), -1)

            if g.size(1) == self.global_dim:
                g = self.global_proj(g)
                gate = self.global_gate(torch.cat([result, g], dim=-1))
                result = result + 0.1 * gate * g

        out = self.out_lin(result).view(-1)

        if include_partial:
            return out, {
                "tokens": tokens,
                "mol_emb": result,
                "score": score,
            }

        return out

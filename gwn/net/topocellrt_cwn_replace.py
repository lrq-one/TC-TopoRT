import torch
import torch.nn as nn
import torch.nn.functional as F

from net.cwn_hypergraph_adapter import CWNHypergraphAdapter


class TopoCellRTCWNReplace(nn.Module):
    
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

        tokens, mask = self.cwn_adapter(data)  # [B, 3, 512]

        tokens = self.trans_graph(tokens)      # [B, 3, 512]
        add_x = tokens.sum(dim=1)              # [B, 512]

        score = torch.sigmoid(self.trans_add(add_x))
        result = torch.mul(score, add_x)
        result = self.layerNorm_out(result)

        result = self.trans_out(result)

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

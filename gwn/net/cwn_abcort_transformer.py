import torch
import torch.nn as nn
import torch.nn.functional as F

from mp.layers import InitReduceConv, OGBEmbedVEWithReduce, CINppConv
from mp.nn import get_nonlinearity, get_graph_norm
from net.cwn import ContinuousAtomEncoder, ContinuousBondEncoder


class CWNHypergraphReplacementEncoder(nn.Module):

    def __init__(
        self,
        num_layers=6,
        hidden=256,
        dropout_rate=0.0,
        indropout_rate=0.0,
        max_dim=2,
        jump_mode="cat",
        nonlinearity="gelu",
        train_eps=False,
        init_reduce="sum",
        use_coboundaries=True,
        graph_norm="bn",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden = hidden
        self.dropout_rate = dropout_rate
        self.indropout_rate = indropout_rate
        self.max_dim = max_dim
        self.jump_mode = jump_mode
        self.nonlinearity = nonlinearity

        embed_dim = hidden

        self.v_embed_init = ContinuousAtomEncoder(in_dim=55, emb_dim=embed_dim)
        self.e_embed_init = ContinuousBondEncoder(in_dim=21, emb_dim=embed_dim)

        self.reduce_init = InitReduceConv(reduce=init_reduce)
        self.init_conv = OGBEmbedVEWithReduce(
            self.v_embed_init,
            self.e_embed_init,
            self.reduce_init,
        )

        act_module = get_nonlinearity(nonlinearity, return_module=True)
        self.graph_norm = get_graph_norm(graph_norm)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden

            conv = CINppConv(
                up_msg_size=layer_dim,
                down_msg_size=layer_dim,
                boundary_msg_size=layer_dim,
                passed_msg_boundaries_nn=None,
                passed_msg_up_nn=None,
                passed_msg_down_nn=None,
                passed_update_up_nn=None,
                passed_update_down_nn=None,
                passed_update_boundaries_nn=None,
                train_eps=train_eps,
                max_dim=max_dim,
                hidden=hidden,
                act_module=act_module,
                layer_dim=layer_dim,
                graph_norm=self.graph_norm,
                use_coboundaries=use_coboundaries,
            )

            self.convs.append(conv)

        if jump_mode == "cat":
            self.out_dim = hidden * (num_layers + 1)
        elif jump_mode == "last":
            self.out_dim = hidden
        else:
            self.out_dim = hidden

    def forward(self, data):
        params = data.get_all_cochain_params(
            max_dim=self.max_dim,
            include_down_features=True,
            include_boundary_features=True,
        )

        xs = list(self.init_conv(*params))

        for i in range(len(xs)):
            xs[i] = F.dropout(
                xs[i],
                p=self.indropout_rate,
                training=self.training,
            )

        jk_list = [[x.clone() for x in xs]]
        data.set_xs(xs)

        for conv in self.convs:
            params = data.get_all_cochain_params(
                max_dim=self.max_dim,
                include_down_features=True,
                include_boundary_features=True,
            )

            xs = conv(*params)

            for i in range(len(xs)):
                xs[i] = F.dropout(
                    xs[i],
                    p=self.dropout_rate,
                    training=self.training,
                )

            data.set_xs(xs)
            jk_list.append([x.clone() for x in xs])

        final_cochain_xs = []

        for dim in range(len(jk_list[0])):
            layer_features = [jk_list[i][dim] for i in range(len(jk_list))]

            if self.jump_mode == "cat":
                out = torch.cat(layer_features, dim=-1)
            elif self.jump_mode == "last":
                out = layer_features[-1]
            else:
                out = torch.stack(layer_features, dim=0).sum(dim=0)

            final_cochain_xs.append(out)

        return final_cochain_xs


class CWNABCoRTTransformer(nn.Module):

    def __init__(
        self,
        out_size=1,
        num_layers=6,
        hidden=256,
        d_model=256,
        transformer_layers=2,
        transformer_heads=8,
        transformer_ffn_mult=4,
        dropout_rate=0.0,
        transformer_dropout=0.10,
        max_dim=2,
        jump_mode="cat",
        nonlinearity="gelu",
        use_coboundaries=True,
    ):
        super().__init__()

        self.max_dim = max_dim
        self.hidden = hidden
        self.d_model = d_model

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

        self.cell_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(max_dim + 1)
        ])

        self.cell_type_emb = nn.Embedding(max_dim + 1, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=d_model * transformer_ffn_mult,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        self.post_norm = nn.LayerNorm(d_model)

        self.mol_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, out_size),
        )

        nn.init.normal_(self.cls_token, std=0.02)

    def _batch_size(self, data):
        if hasattr(data, "num_complexes") and data.num_complexes is not None:
            return int(data.num_complexes)

        batch = data.cochains[0].batch
        return int(batch.max().item()) + 1

    def _build_transformer_tokens(self, data, cochain_xs):

        device = cochain_xs[0].device
        batch_size = self._batch_size(data)

        per_graph_tokens = [[] for _ in range(batch_size)]

        for dim in range(min(len(cochain_xs), self.max_dim + 1)):
            x = cochain_xs[dim]

            if x is None or x.numel() == 0:
                continue

            if dim not in data.cochains:
                continue

            cochain = data.cochains[dim]
            cell_batch = cochain.batch

            if cell_batch is None or cell_batch.numel() == 0:
                continue

            h = self.cell_projectors[dim](x)

            type_id = torch.full(
                (h.size(0),),
                fill_value=dim,
                dtype=torch.long,
                device=device,
            )
            h = h + self.cell_type_emb(type_id)

            for graph_id in range(batch_size):
                mask = cell_batch == graph_id
                if mask.any():
                    per_graph_tokens[graph_id].append(h[mask])

        seqs = []
        cls = self.cls_token.expand(batch_size, 1, self.d_model)

        for graph_id in range(batch_size):
            if len(per_graph_tokens[graph_id]) > 0:
                tokens = torch.cat(per_graph_tokens[graph_id], dim=0)
            else:
                tokens = torch.zeros(1, self.d_model, device=device)

            seq = torch.cat([cls[graph_id], tokens], dim=0)
            seqs.append(seq)

        max_len = max(seq.size(0) for seq in seqs)

        padded = torch.zeros(
            batch_size,
            max_len,
            self.d_model,
            device=device,
        )

        padding_mask = torch.ones(
            batch_size,
            max_len,
            dtype=torch.bool,
            device=device,
        )

        for i, seq in enumerate(seqs):
            length = seq.size(0)
            padded[i, :length] = seq
            padding_mask[i, :length] = False

        return padded, padding_mask

    def forward(self, data, include_partial=False):
        cochain_xs = self.cwn_encoder(data)

        tokens, padding_mask = self._build_transformer_tokens(data, cochain_xs)

        z = self.transformer(
            tokens,
            src_key_padding_mask=padding_mask,
        )

        cls_h = z[:, 0, :]
        cls_h = self.post_norm(cls_h)

        cls_h = cls_h * self.mol_gate(cls_h)

        out = self.regression_head(cls_h).view(-1)

        if include_partial:
            return out, {
                "mol_emb": cls_h,
                "tokens": tokens,
                "padding_mask": padding_mask,
                "cochain_xs": cochain_xs,
            }

        return out

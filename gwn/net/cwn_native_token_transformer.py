import torch
import torch.nn as nn
import torch.nn.functional as F

from net.cwn import OGBEmbedCINpp
from mp.nn import pool_complex, get_nonlinearity


class OGBEmbedCINppWithTokenTransformer(OGBEmbedCINpp):
    """
    安全版 CWN + Transformer。

    它不是重写 CWN。
    它继承原生 OGBEmbedCINpp，保留原来的：
        ContinuousAtomEncoder
        ContinuousBondEncoder
        OGBEmbedVEWithReduce
        CINppConv
        pool_complex
        trans_add SE-style gate
        trans_out
        regression_head

    只在 pool_complex 之后、flatten 之前，对 0/1/2-cell graph-level tokens
    加一个 zero-init residual Transformer。

    初始状态：
        token_res_scale = 0
        所以模型一开始严格退化为原生 OGBEmbedCINpp。
    """

    def __init__(
        self,
        *args,
        token_d_model=256,
        token_heads=4,
        token_layers=1,
        token_dropout=0.05,
        residual_init=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.token_down = nn.Sequential(
            nn.Linear(self.total_jk_dim, token_d_model),
            nn.LayerNorm(token_d_model),
            nn.GELU(),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_d_model,
            nhead=token_heads,
            dim_feedforward=token_d_model * 4,
            dropout=token_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.cell_token_transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=token_layers,
        )

        self.token_up = nn.Sequential(
            nn.Linear(token_d_model, self.total_jk_dim),
        )

        self.token_res_scale = nn.Parameter(
            torch.tensor(float(residual_init))
        )

    def _cell_token_residual_transformer(self, tokens):
        """
        tokens: [B, 3, total_jk_dim]
        """
        h = self.token_down(tokens)
        h = self.cell_token_transformer(h)
        delta = self.token_up(h)

        # tanh(0)=0，初始化等价于原生 CWN
        return tokens + torch.tanh(self.token_res_scale) * delta

    def forward(self, data, include_partial=False):
        act = get_nonlinearity(self.nonlinearity, return_module=False)
        res = {}

        # A. 原生 CWN 初始嵌入
        params = data.get_all_cochain_params(
            max_dim=self.max_dim,
            include_down_features=True,
        )

        xs = list(self.init_conv(*params))

        for i in range(len(xs)):
            xs[i] = F.dropout(
                xs[i],
                p=self.in_dropout_rate,
                training=self.training,
            )

        jk_list = [[x.clone() for x in xs]]
        data.set_xs(xs)

        # B. 原生 CINppConv 多层 cell message passing
        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params(
                max_dim=self.max_dim,
                include_down_features=True,
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

        # C. 原生 Jumping Knowledge 拼接
        num_layers_total = len(jk_list)
        final_cochain_xs = []

        for d in range(self.max_dim + 1):
            layer_features = [jk_list[i][d] for i in range(num_layers_total)]
            final_cochain_xs.append(torch.cat(layer_features, dim=-1))

        # D. 原生 pool_complex / attention pool
        if self.use_attention_pool:
            batch_size = data.cochains[0].batch.max() + 1
            pooled_xs = []

            for i in range(len(final_cochain_xs)):
                if data.cochains[i].batch is None or final_cochain_xs[i].size(0) == 0:
                    pooled_xs.append(
                        torch.zeros(
                            batch_size,
                            self.total_jk_dim,
                            device=final_cochain_xs[i].device,
                        )
                    )
                else:
                    pooled_xs.append(
                        self.att_poolers[i](
                            final_cochain_xs[i],
                            data.cochains[i].batch,
                            dim_size=batch_size,
                        )
                    )

            x = torch.stack(pooled_xs, dim=0)  # [3, B, total_jk_dim]
        else:
            x = pool_complex(
                final_cochain_xs,
                data,
                self.max_dim,
                self.readout,
            )  # [3, B, total_jk_dim]

        # E. 这里是唯一新增点：
        # 原生 CWN 是 x.transpose(0,1).flatten(start_dim=1)
        # 我们先得到 [B,3,D]，加 zero-init residual Transformer
        tokens = x.transpose(0, 1)  # [B, 3, total_jk_dim]
        tokens = self._cell_token_residual_transformer(tokens)

        x = tokens.flatten(start_dim=1)

        # F. 完全保留原生 ABCoRT-style gate/head
        score = torch.sigmoid(self.trans_add(x))
        x = x * score

        x = self.layerNorm_out(x)
        x = self.trans_out(x)

        x = self.regression_head(x)

        if include_partial:
            res["out"] = x
            res["tokens"] = tokens
            res["token_res_scale"] = self.token_res_scale.detach()
            return x, res

        return x

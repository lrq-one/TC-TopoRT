import math
import torch
import torch.nn as nn

from net.topocellrt_cwn_replace import TopoCellRTCWNReplace


class TopoCellRTCWNDualView(nn.Module):
    """
    Frozen-branch dual-view gate model.

    origin_data -> origin encoder -> origin_pred, origin_emb
    taut_data   -> taut encoder   -> taut_pred, taut_emb

    alpha_origin close to 1 means trust origin.
    alpha_origin close to 0 means trust taut.

    Compared with V1:
    1. Gate is initialized around post-hoc alpha=0.704.
    2. Gate outputs residual logit around this prior, not random sigmoid collapse.
    3. tau exists but should be fixed by training script.
    """

    def __init__(
        self,
        emb_dim=256,
        cwn_layers=6,
        cwn_hidden=256,
        max_dim=2,
        drop_ratio=0.0,
        share_encoder=False,
        init_tau=5.0,
        temperature=5.0,
        gate_prior_alpha=0.704,
    ):
        super().__init__()

        self.share_encoder = bool(share_encoder)
        self.temperature = float(temperature)
        self.gate_prior_alpha = float(gate_prior_alpha)

        self.origin_encoder = TopoCellRTCWNReplace(
            emb_dim=emb_dim,
            cwn_layers=cwn_layers,
            cwn_hidden=cwn_hidden,
            max_dim=max_dim,
            drop_ratio=drop_ratio,
        )

        if self.share_encoder:
            self.taut_encoder = self.origin_encoder
        else:
            self.taut_encoder = TopoCellRTCWNReplace(
                emb_dim=emb_dim,
                cwn_layers=cwn_layers,
                cwn_hidden=cwn_hidden,
                max_dim=max_dim,
                drop_ratio=drop_ratio,
            )

        hdim = emb_dim * 2

        # emb_o, emb_t, |emb_o-emb_t| = 3 * 512
        # pred_o, pred_t, |pred_o-pred_t| = 3
        gate_in_dim = hdim * 3 + 3

        self.gate_delta = nn.Sequential(
            nn.Linear(gate_in_dim, hdim),
            nn.LayerNorm(hdim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hdim, hdim // 2),
            nn.GELU(),
            nn.Linear(hdim // 2, 1),
        )

        # Start from current best post-hoc alpha=0.704.
        prior = min(max(float(gate_prior_alpha), 1e-4), 1.0 - 1e-4)
        logit_prior = math.log(prior / (1.0 - prior))
        self.register_buffer("gate_logit_prior", torch.tensor(float(logit_prior)))

        # Keep initial delta small, but allow sample-specific learning.
        last_linear = self.gate_delta[-1]
        nn.init.normal_(last_linear.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(last_linear.bias)

        # Tau should be fixed by the training script in V2/V3.
        self.tau = nn.Parameter(torch.tensor(float(init_tau)), requires_grad=False)

    def forward(self, origin_data, taut_data, return_aux=False):
        pred_o, aux_o = self.origin_encoder(origin_data, include_partial=True)
        pred_t, aux_t = self.taut_encoder(taut_data, include_partial=True)

        emb_o = aux_o["mol_emb"]
        emb_t = aux_t["mol_emb"]

        pred_o_col = pred_o.view(-1, 1)
        pred_t_col = pred_t.view(-1, 1)
        diff_pred_col = torch.abs(pred_o_col - pred_t_col)

        gate_input = torch.cat(
            [
                emb_o,
                emb_t,
                torch.abs(emb_o - emb_t),
                pred_o_col / 1000.0,
                pred_t_col / 1000.0,
                diff_pred_col / 100.0,
            ],
            dim=-1,
        )

        delta_logit = self.gate_delta(gate_input).view(-1)
        alpha_origin = torch.sigmoid(self.gate_logit_prior + delta_logit)

        mixed_pred = alpha_origin * pred_o + (1.0 - alpha_origin) * pred_t

        # Conservative disagreement mask.
        soft_use = torch.sigmoid(
            (torch.abs(pred_o - pred_t) - self.tau) / self.temperature
        )

        final_pred = (1.0 - soft_use) * pred_o + soft_use * mixed_pred

        if return_aux:
            return final_pred, {
                "origin_pred": pred_o,
                "taut_pred": pred_t,
                "alpha_origin": alpha_origin,
                "soft_use": soft_use,
                "tau": self.tau,
                "origin_emb": emb_o,
                "taut_emb": emb_t,
            }

        return final_pred

    def gate_prior_loss(self, aux):
        alpha = aux["alpha_origin"]
        target = torch.full_like(alpha, self.gate_prior_alpha)
        return torch.mean((alpha - target) ** 2)

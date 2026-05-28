from pathlib import Path

src = Path("model_topocellrt.py")
dst = Path("model_topocellrt_regime.py")

s = src.read_text(encoding="utf-8")

s = s.replace(
"""        self.out_lin = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim // 4),
            nn.GELU(),
            nn.Linear(emb_dim // 4, emb_dim // 8),
            nn.GELU(),
            nn.Linear(emb_dim // 8, 1),
        )""",
"""        self.rt_bin_head = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(emb_dim, 6),
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
        )"""
)

s = s.replace(
"    def forward(self, data):",
"    def forward(self, data, return_aux=False):"
)

s = s.replace(
"""        out = self.out_lin(result)

        return out.squeeze()
""",
"""        out = self.out_lin(result).squeeze(-1)
        rt_bin_logits = self.rt_bin_head(result)

        if return_aux:
            return out, rt_bin_logits

        return out
"""
)

dst.write_text(s, encoding="utf-8")
print("wrote", dst)

"""Virtual-token image/text fusion and survival risk prediction."""

import torch
import torch.nn as nn


class VirtualNodeSurvivalMLP(nn.Module):
    """Fuse image and text tokens through a virtual token and predict risk."""

    def __init__(
        self,
        image_dim,
        text_dim,
        hidden_dim=256,
        dropout_rate=0.2,
        num_layers=2,
        num_heads=8,
    ):
        super().__init__()
        node_dim = hidden_dim
        hidden_mid = max(32, hidden_dim // 2)

        self.img_proj = nn.Sequential(
            nn.Linear(image_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(text_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
        )

        if node_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.virtual_node = nn.Parameter(torch.zeros(1, 1, node_dim))
        nn.init.trunc_normal_(self.virtual_node, std=0.02)

        self.fusion_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=node_dim,
                    nhead=num_heads,
                    dim_feedforward=node_dim * 2,
                    dropout=dropout_rate,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )

        self.risk_head = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, hidden_mid),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_mid, 1),
        )

    def forward(self, img_emb, text_emb):
        img_node = self.img_proj(img_emb)
        txt_node = self.txt_proj(text_emb).unsqueeze(1)
        v_node = self.virtual_node.expand(img_emb.size(0), -1, -1)

        nodes = torch.cat([v_node, img_node, txt_node], dim=1)
        for layer in self.fusion_layers:
            nodes = layer(nodes)

        v_out = nodes[:, 0, :]
        risk_score = self.risk_head(v_out)
        return risk_score, v_out

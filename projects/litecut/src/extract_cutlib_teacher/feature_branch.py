from __future__ import annotations

import torch
from torch import nn


WL_BUCKETS = 4096
GROUP1_DIM = 8
GROUP2_DIM = 6
GROUP3_DIM = 5
FEATURE_INPUT_DIM = GROUP1_DIM + GROUP2_DIM + GROUP3_DIM


class ResidualGroupEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.skip = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class WLGroupEncoder(nn.Module):
    def __init__(self, num_buckets: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_buckets, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, wl_hash: torch.Tensor) -> torch.Tensor:
        wl_bucket = torch.remainder(wl_hash.long(), self.embedding.num_embeddings)
        return self.proj(self.embedding(wl_bucket))


class AttentionScore(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.score(z)


class ResidualFusion(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.skip = nn.Linear(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class FeatureTeacherBranch(nn.Module):
    def __init__(self, group_hidden_dim: int, fusion_hidden_dim: int) -> None:
        super().__init__()
        self.group1_encoder = ResidualGroupEncoder(GROUP1_DIM, hidden_dim=group_hidden_dim)
        self.group2_encoder = ResidualGroupEncoder(GROUP2_DIM, hidden_dim=group_hidden_dim)
        self.group3_encoder = ResidualGroupEncoder(GROUP3_DIM, hidden_dim=group_hidden_dim)
        self.group4_encoder = WLGroupEncoder(num_buckets=WL_BUCKETS, hidden_dim=group_hidden_dim)
        self.attention = AttentionScore(hidden_dim=group_hidden_dim)
        self.fusion = ResidualFusion(in_dim=5 * group_hidden_dim, hidden_dim=fusion_hidden_dim)

    def forward(self, x_scaled: torch.Tensor, wl_hash: torch.Tensor) -> torch.Tensor:
        g1 = x_scaled[:, :GROUP1_DIM]
        g2 = x_scaled[:, GROUP1_DIM:GROUP1_DIM + GROUP2_DIM]
        g3 = x_scaled[:, GROUP1_DIM + GROUP2_DIM:]

        z1 = self.group1_encoder(g1)
        z2 = self.group2_encoder(g2)
        z3 = self.group3_encoder(g3)
        z4 = self.group4_encoder(wl_hash)

        z_groups = [z1, z2, z3, z4]
        z_stack = torch.stack(z_groups, dim=1)
        att_logits = torch.cat([self.attention(z) for z in z_groups], dim=1)
        att_weight = torch.softmax(att_logits, dim=1)
        z_att = torch.sum(z_stack * att_weight.unsqueeze(-1), dim=1)
        z_fused = torch.cat([z_att, z1, z2, z3, z4], dim=1)
        return self.fusion(z_fused)

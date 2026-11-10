from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .config import ExtractCutLibConfig
from .feature_branch import FeatureTeacherBranch
from .graph_branch import GraphTensorSample, RelationAwareGraphBranch


class TeacherTwoBranchModel(nn.Module):
    def __init__(self, cfg: ExtractCutLibConfig) -> None:
        super().__init__()
        branch_dim = cfg.feature_fusion_hidden_dim
        self.branch_dim = branch_dim
        self.feature_branch = FeatureTeacherBranch(
            group_hidden_dim=cfg.feature_group_hidden_dim,
            fusion_hidden_dim=branch_dim,
        )
        self.graph_branch = RelationAwareGraphBranch(
            hidden_dim=cfg.gnn_hidden_dim,
            branch_out_dim=branch_dim,
            num_layers=cfg.gnn_num_layers,
            readout_pos_gate_hidden_dim=cfg.readout_pos_gate_hidden_dim,
            readout_content_hidden_dim=cfg.readout_content_hidden_dim,
            p_inv_init=cfg.p_inv_init,
            p_inv_init_std=cfg.p_inv_init_std,
        )
        self.cross_gate = nn.Sequential(
            nn.Linear(2 * branch_dim, branch_dim),
            nn.Sigmoid(),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(3 * branch_dim, cfg.regression_hidden_dim1),
            nn.LayerNorm(cfg.regression_hidden_dim1),
            nn.GELU(),
            nn.Dropout(cfg.regression_dropout),
            nn.Linear(cfg.regression_hidden_dim1, cfg.regression_hidden_dim2),
            nn.GELU(),
            nn.Linear(cfg.regression_hidden_dim2, 1),
        )

    def forward_feature(self, x_scaled: torch.Tensor, wl_hash: torch.Tensor) -> torch.Tensor:
        return self.feature_branch(x_scaled, wl_hash)

    def forward_graph(self, graph_batch: Sequence[GraphTensorSample]) -> torch.Tensor:
        return self.graph_branch(graph_batch)

    def fuse_branches(self, z_feat: torch.Tensor, z_graph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.cross_gate(torch.cat([z_feat, z_graph], dim=-1))
        z_cross = gate * z_feat + (1.0 - gate) * z_graph
        fusion_input = torch.cat([z_feat, z_graph, z_cross], dim=-1)
        y_pred = self.regression_head(fusion_input)
        return y_pred, z_cross

    def forward(
        self,
        x_scaled: torch.Tensor,
        wl_hash: torch.Tensor,
        graph_batch: Sequence[GraphTensorSample] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_feat = self.forward_feature(x_scaled, wl_hash)
        if graph_batch is None:
            z_graph = torch.zeros(
                z_feat.shape[0],
                self.branch_dim,
                dtype=z_feat.dtype,
                device=z_feat.device,
            )
        else:
            z_graph = self.forward_graph(graph_batch)
        y_pred, _ = self.fuse_branches(z_feat, z_graph)
        return y_pred, z_feat, z_graph

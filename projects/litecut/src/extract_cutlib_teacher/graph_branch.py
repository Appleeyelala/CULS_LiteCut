from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


NODE_FEAT_DIM = 12


@dataclass
class GraphTensorSample:
    node_feats: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    pos_vec: torch.Tensor
    root_index: int
    leaf_mask: torch.Tensor


@dataclass
class GraphTensorBatch:
    node_feats: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    pos_vec: torch.Tensor
    batch_index: torch.Tensor
    root_index: torch.Tensor
    leaf_mask: torch.Tensor
    num_graphs: int


class ResidualUpdateMLP(nn.Module):
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


class RelationAwareMessagePassingLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        p_inv_init: str = "zero",
        p_inv_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.w_norm = nn.Linear(hidden_dim, hidden_dim)
        self.w_inv = nn.Linear(hidden_dim, hidden_dim)
        self.p_inv = nn.Parameter(self._init_p_inv(hidden_dim, p_inv_init, p_inv_init_std))
        self.update = ResidualUpdateMLP(in_dim=2 * hidden_dim, hidden_dim=hidden_dim)

    @staticmethod
    def _init_p_inv(hidden_dim: int, mode: str, std: float) -> torch.Tensor:
        if mode == "zero":
            return torch.zeros(hidden_dim)
        if mode == "gaussian":
            return torch.randn(hidden_dim) * std
        raise ValueError(f"Unsupported p_inv_init mode: {mode}. Expected 'zero' or 'gaussian'.")

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        msg_norm = torch.zeros_like(h)
        msg_inv = torch.zeros_like(h)
        if edge_index.numel() > 0:
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            src_h = h[src_idx]

            norm_mask = edge_type == 0
            if torch.any(norm_mask):
                norm_msg = self.w_norm(src_h[norm_mask])
                msg_norm.index_add_(0, dst_idx[norm_mask], norm_msg)

            inv_mask = edge_type == 1
            if torch.any(inv_mask):
                inv_gate = torch.tanh(self.p_inv)
                inv_msg = self.w_inv(src_h[inv_mask]) * inv_gate
                msg_inv.index_add_(0, dst_idx[inv_mask], inv_msg)

        msg = msg_norm + msg_inv
        return self.update(torch.cat([h, msg], dim=-1))


class RoleAwareReadout(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        branch_out_dim: int,
        pos_gate_hidden_dim: int,
        content_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.content_score = nn.Sequential(
            nn.Linear(hidden_dim, content_hidden_dim),
            nn.Tanh(),
            nn.Linear(content_hidden_dim, 1),
        )
        self.pos_gate = nn.Sequential(
            nn.Linear(5, pos_gate_hidden_dim),
            nn.GELU(),
            nn.Linear(pos_gate_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, branch_out_dim),
            nn.LayerNorm(branch_out_dim),
            nn.GELU(),
            nn.Linear(branch_out_dim, branch_out_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        pos_vec: torch.Tensor,
        root_index: torch.Tensor,
        leaf_mask: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        s_content = self.content_score(h)
        g_pos = self.pos_gate(pos_vec)
        s_total = (s_content * g_pos).squeeze(-1)

        pooled = []
        for gid in range(num_graphs):
            node_mask = batch_index == gid
            h_g = h[node_mask]
            h_root = h[int(root_index[gid].item())]

            leaf_mask_g = leaf_mask[node_mask]
            if torch.any(leaf_mask_g):
                h_leaf = h_g[leaf_mask_g].mean(dim=0)
            else:
                h_leaf = h_g.mean(dim=0)

            att_weight = torch.softmax(s_total[node_mask], dim=0)
            h_global = torch.sum(h_g * att_weight.unsqueeze(-1), dim=0)
            pooled.append(self.fusion(torch.cat([h_root, h_leaf, h_global], dim=-1)))

        return torch.stack(pooled, dim=0)


class RelationAwareGraphBranch(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = 64,
        branch_out_dim: int = 128,
        num_layers: int = 3,
        readout_pos_gate_hidden_dim: int = 16,
        readout_content_hidden_dim: int = 32,
        p_inv_init: str = "zero",
        p_inv_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                RelationAwareMessagePassingLayer(
                    hidden_dim=hidden_dim,
                    p_inv_init=p_inv_init,
                    p_inv_init_std=p_inv_init_std,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = RoleAwareReadout(
            hidden_dim=hidden_dim,
            branch_out_dim=branch_out_dim,
            pos_gate_hidden_dim=readout_pos_gate_hidden_dim,
            content_hidden_dim=readout_content_hidden_dim,
        )

    @staticmethod
    def collate_graph_batch(graph_batch: Sequence[GraphTensorSample], device: torch.device) -> GraphTensorBatch:
        node_feats_list = []
        edge_index_list = []
        edge_type_list = []
        pos_vec_list = []
        batch_index_list = []
        root_index_list = []
        leaf_mask_list = []
        node_offset = 0

        for gid, sample in enumerate(graph_batch):
            num_nodes = sample.node_feats.shape[0]
            node_feats_list.append(sample.node_feats)
            pos_vec_list.append(sample.pos_vec)
            leaf_mask_list.append(sample.leaf_mask)
            batch_index_list.append(torch.full((num_nodes,), gid, dtype=torch.long))
            root_index_list.append(sample.root_index + node_offset)

            if sample.edge_index.numel() > 0:
                edge_index_list.append(sample.edge_index + node_offset)
                edge_type_list.append(sample.edge_type)

            node_offset += num_nodes

        edge_index = (
            torch.cat(edge_index_list, dim=1)
            if edge_index_list
            else torch.zeros((2, 0), dtype=torch.long)
        )
        edge_type = (
            torch.cat(edge_type_list, dim=0)
            if edge_type_list
            else torch.zeros((0,), dtype=torch.long)
        )

        return GraphTensorBatch(
            node_feats=torch.cat(node_feats_list, dim=0).to(device),
            edge_index=edge_index.to(device),
            edge_type=edge_type.to(device),
            pos_vec=torch.cat(pos_vec_list, dim=0).to(device),
            batch_index=torch.cat(batch_index_list, dim=0).to(device),
            root_index=torch.tensor(root_index_list, dtype=torch.long, device=device),
            leaf_mask=torch.cat(leaf_mask_list, dim=0).to(device),
            num_graphs=len(graph_batch),
        )

    def forward(self, graph_batch: Sequence[GraphTensorSample]) -> torch.Tensor:
        device = self.node_encoder[0].weight.device
        batch = self.collate_graph_batch(graph_batch, device)
        h = self.node_encoder(batch.node_feats)
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_type)
        return self.readout(
            h,
            batch.pos_vec,
            batch.root_index,
            batch.leaf_mask,
            batch.batch_index,
            batch.num_graphs,
        )

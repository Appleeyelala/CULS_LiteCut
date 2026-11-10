from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from extract_cutlib_teacher.feature_branch import FEATURE_INPUT_DIM, GROUP1_DIM, GROUP2_DIM, GROUP3_DIM


@dataclass
class StudentForwardOutput:
    prediction: torch.Tensor
    fused_hidden_raw: torch.Tensor
    fused_hidden: torch.Tensor
    align_hidden: torch.Tensor
    group_logits: torch.Tensor | None
    group_probs: torch.Tensor | None
    group_outputs: torch.Tensor
    selected_group: torch.Tensor
    expert_logits: torch.Tensor | None
    expert_probs: torch.Tensor | None
    expert_outputs: torch.Tensor
    selected_expert: torch.Tensor | None


@dataclass
class StudentDistillLosses:
    total: torch.Tensor
    hard: torch.Tensor
    soft: torch.Tensor
    align: torch.Tensor
    rank: torch.Tensor


class ResidualGroupEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.skip = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class GroupAttentionScore(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.score(z)


class SingleRegressionHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        regression_hidden_dim = max(8, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, regression_hidden_dim),
            nn.ReLU(),
            nn.Linear(regression_hidden_dim, 1),
        )

    def forward(self, fused_hidden: torch.Tensor) -> torch.Tensor:
        return self.head(fused_hidden)


class StudentModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 19,
        shared_hidden_dim: int = 18,
        with_context8: bool = True,
        with_moe: bool = True,
        num_experts: int = 4,
        gate_temperature: float = 1.0,
        teacher_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if gate_temperature <= 0:
            raise ValueError(f"gate_temperature must be positive, got {gate_temperature}")
        if shared_hidden_dim <= 0:
            raise ValueError(f"shared_hidden_dim must be positive, got {shared_hidden_dim}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")

        if with_context8:
            expected_input_dim = FEATURE_INPUT_DIM
            group_dims = (GROUP1_DIM, GROUP2_DIM, GROUP3_DIM)
        else:
            expected_input_dim = FEATURE_INPUT_DIM - GROUP1_DIM
            group_dims = (expected_input_dim,)

        if input_dim != expected_input_dim:
            raise ValueError(
                f"Student model expects input_dim={expected_input_dim} for with_context8={with_context8}, got {input_dim}"
            )

        self.input_dim = input_dim
        self.shared_hidden_dim = shared_hidden_dim
        self.gate_temperature = gate_temperature
        self.with_context8 = with_context8
        self.with_moe = with_moe
        self.num_experts = num_experts if with_moe else 1
        self.group_dims = group_dims

        self.group_encoders = nn.ModuleList(
            [ResidualGroupEncoder(dim, hidden_dim=shared_hidden_dim) for dim in self.group_dims]
        )
        self.group_attention = GroupAttentionScore(hidden_dim=shared_hidden_dim)
        self.post_fusion = nn.Sequential(
            nn.LayerNorm(shared_hidden_dim),
            nn.ReLU(),
        )
        self.expert_gate = nn.Linear(shared_hidden_dim, self.num_experts) if with_moe else None
        self.align_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, shared_hidden_dim),
            nn.LayerNorm(shared_hidden_dim),
        )
        self.experts = (
            nn.ModuleList(SingleRegressionHead(shared_hidden_dim) for _ in range(self.num_experts))
            if with_moe
            else None
        )
        self.single_head = None if with_moe else SingleRegressionHead(shared_hidden_dim)

        align_in_dim = teacher_hidden_dim if teacher_hidden_dim is not None else shared_hidden_dim
        self.teacher_align_adapter = (
            nn.Identity()
            if align_in_dim == shared_hidden_dim
            else nn.Linear(align_in_dim, shared_hidden_dim)
        )

        self.register_buffer("input_mu", torch.zeros(input_dim), persistent=True)
        self.register_buffer("input_inv_sigma", torch.ones(input_dim), persistent=True)
        self._has_input_stats = False

    def set_input_stats(self, mu: torch.Tensor, inv_sigma: torch.Tensor) -> None:
        if mu.shape[-1] != self.input_dim or inv_sigma.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input norm stats must have last dim {self.input_dim}, got {mu.shape} and {inv_sigma.shape}"
            )
        self.input_mu.copy_(mu.detach().to(self.input_mu.device, dtype=self.input_mu.dtype).view(-1))
        self.input_inv_sigma.copy_(
            inv_sigma.detach().to(self.input_inv_sigma.device, dtype=self.input_inv_sigma.dtype).view(-1)
        )
        self._has_input_stats = True

    def normalize_input(self, x_raw: torch.Tensor) -> torch.Tensor:
        if not self._has_input_stats:
            raise RuntimeError("Input normalization stats are not set. Call set_input_stats() first.")
        return (x_raw - self.input_mu) * self.input_inv_sigma

    def _split_groups(self, x_scaled: torch.Tensor) -> tuple[torch.Tensor, ...]:
        splits = torch.split(x_scaled, self.group_dims, dim=-1)
        return tuple(splits)

    def forward(self, x_scaled: torch.Tensor) -> StudentForwardOutput:
        if x_scaled.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x_scaled.shape[-1]}")

        groups = self._split_groups(x_scaled)
        encoded_groups = [encoder(group) for encoder, group in zip(self.group_encoders, groups)]
        group_outputs = torch.stack(encoded_groups, dim=1)
        if len(encoded_groups) == 1:
            group_logits = None
            group_probs = None
            selected_group = torch.zeros((x_scaled.shape[0],), dtype=torch.long, device=x_scaled.device)
            fused_hidden_raw = encoded_groups[0]
        else:
            group_logits = torch.cat([self.group_attention(z) for z in encoded_groups], dim=1)
            group_probs = torch.softmax(group_logits / self.gate_temperature, dim=-1)
            selected_group = torch.argmax(group_probs, dim=-1)
            fused_hidden_raw = torch.sum(group_outputs * group_probs.unsqueeze(-1), dim=1)
        fused_hidden = self.post_fusion(fused_hidden_raw)
        if self.with_moe:
            assert self.expert_gate is not None
            assert self.experts is not None
            expert_logits = self.expert_gate(fused_hidden)
            expert_probs = torch.softmax(expert_logits / self.gate_temperature, dim=-1)
            selected_expert = torch.argmax(expert_probs, dim=-1)
            expert_outputs = torch.cat([expert(fused_hidden) for expert in self.experts], dim=-1)
            prediction = torch.sum(expert_outputs * expert_probs, dim=-1, keepdim=True)
        else:
            assert self.single_head is not None
            prediction = self.single_head(fused_hidden)
            expert_logits = None
            expert_probs = None
            expert_outputs = prediction
            selected_expert = None
        align_hidden = self.align_head(fused_hidden_raw)

        return StudentForwardOutput(
            prediction=prediction,
            fused_hidden_raw=fused_hidden_raw,
            fused_hidden=fused_hidden,
            align_hidden=align_hidden,
            group_logits=group_logits,
            group_probs=group_probs,
            group_outputs=group_outputs,
            selected_group=selected_group,
            expert_logits=expert_logits,
            expert_probs=expert_probs,
            expert_outputs=expert_outputs,
            selected_expert=selected_expert,
        )

    def forward_raw(self, x_raw: torch.Tensor) -> StudentForwardOutput:
        return self.forward(self.normalize_input(x_raw))

    def adapt_teacher_hidden(self, teacher_hidden: torch.Tensor) -> torch.Tensor:
        return self.teacher_align_adapter(teacher_hidden)

    def compute_distill_losses(
        self,
        output: StudentForwardOutput,
        y_true: torch.Tensor,
        teacher_prediction: torch.Tensor | None,
        teacher_hidden: torch.Tensor | None,
        lambda_hard: float = 1.0,
        lambda_soft: float = 1.0,
        lambda_align: float = 1.0,
        rank_loss_weight: float = 0.0,
        huber_delta: float = 1.0,
    ) -> StudentDistillLosses:
        hard_loss = F.huber_loss(output.prediction, y_true, delta=huber_delta)
        if teacher_prediction is not None and lambda_soft > 0.0:
            soft_loss = F.mse_loss(output.prediction, teacher_prediction)
        else:
            soft_loss = hard_loss.new_zeros(())

        if teacher_hidden is not None and lambda_align > 0.0:
            teacher_hidden_aligned = self.adapt_teacher_hidden(teacher_hidden)
            align_mse = F.mse_loss(output.align_hidden, teacher_hidden_aligned)
            cosine_sim = F.cosine_similarity(output.align_hidden, teacher_hidden_aligned, dim=-1, eps=1e-8)
            align_cosine = 1.0 - cosine_sim.mean()
            align_loss = align_mse + 0.3 * align_cosine
        else:
            align_loss = hard_loss.new_zeros(())
        rank_loss = self.compute_rank_loss(y_true, output.prediction)
        total_loss = (
            lambda_hard * hard_loss
            + lambda_soft * soft_loss
            + lambda_align * align_loss
            + rank_loss_weight * rank_loss
        )
        return StudentDistillLosses(
            total=total_loss,
            hard=hard_loss,
            soft=soft_loss,
            align=align_loss,
            rank=rank_loss,
        )

    @staticmethod
    def compute_rank_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true_vec = y_true.reshape(-1)
        y_pred_vec = y_pred.reshape(-1)
        if y_true_vec.numel() < 2:
            return y_pred_vec.new_zeros(())

        true_diff = y_true_vec.unsqueeze(1) - y_true_vec.unsqueeze(0)
        sign = torch.sign(true_diff)
        pair_mask = torch.triu(sign != 0, diagonal=1)
        if not torch.any(pair_mask):
            return y_pred_vec.new_zeros(())

        pred_diff = y_pred_vec.unsqueeze(1) - y_pred_vec.unsqueeze(0)
        pair_sign = sign[pair_mask]
        pair_pred_diff = pred_diff[pair_mask]
        return F.softplus(-pair_sign * pair_pred_diff).mean()


StudentMoEModel = StudentModel

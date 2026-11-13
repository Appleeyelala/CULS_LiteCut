"""Trust-region joint proposals with one timing adjoint per feedback round.

The inner objective is a local quadratic approximation of the existing scalar
surrogate, NOT a new measured timing label. CPU and CUDA use the same float64
operations and CPU-generated starting noise. True replay/CEC/mapping stay in
pipeline.Experiment; no partial rewrite is promoted to success here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
from torch import nn

from .arrival import reference_arrivals, smooth_arrivals
from .surrogate import scalar_objective


def project_neighborhood(catalogue, anchor, scores, max_changes):
    """Greedy compatible edits, preserving all unedited incumbent decisions.

    Distance counts candidate substitutions and activation/deactivation alike.
    This is a conservative projection, not an exact constrained optimizer.
    """
    catalogue.validate(anchor)
    if type(max_changes) is not int or max_changes < 0:
        raise ValueError('max_changes must be a nonnegative integer.')
    if len(scores) != len(anchor):
        raise ValueError('Neighborhood score shape mismatch.')
    proposals = []
    for i, (row, decision) in enumerate(zip(scores, catalogue.decisions)):
        if len(row) != decision['candidate_count'] + 1 or not all(math.isfinite(x) for x in row):
            raise ValueError('Non-finite or malformed neighborhood scores.')
        choice = max(range(len(row)), key=lambda c: (row[c], -c))
        if choice != anchor[i] and row[choice] > row[anchor[i]]:
            proposals.append((row[choice] - row[anchor[i]], i, choice))
    neighbors = [set() for _ in anchor]
    for left, right in catalogue.conflicts:
        neighbors[left].add(right)
        neighbors[right].add(left)
    result = list(anchor)
    active = {i for i, choice in enumerate(result) if choice}
    changes = 0
    for _, index, choice in sorted(proposals, key=lambda r: (-r[0], r[1])):
        if changes == max_changes:
            break
        if choice and (neighbors[index] & (active - {index})):
            continue
        result[index] = choice
        active.discard(index)
        if choice:
            active.add(index)
        changes += 1
    catalogue.validate(result)
    return result


class FrozenCriticalityObjective(nn.Module):
    """Match the full surrogate's value/first derivative at one incumbent.

    Freeze the outer PO/area and cost-exp sensitivities. Retain the exact
    bilinear adjacent-region feature products inside this local model:
      Q(v) = J(v0) + sum_i L_i (e_i-e_i0)
             + sum_(r,s) C_rs (F_r*F_s-F_r0*F_s0).
    The DAG reverse pass is performed once on CPU, not each inner Adam step.
    """

    def __init__(self, model, anchor, mode='area_constraint', area_limit_ratio=1.05,
                 area_weight=0.5):
        super().__init__()
        model.catalogue.validate(anchor)
        self.catalogue = model.catalogue
        self.counts = model.counts
        self.anchor = list(anchor)
        # Construction is explicitly CPU: existing arrival plans are CPU-only.
        if model.features.device.type != 'cpu':
            raise ValueError('Build frozen criticality from the existing CPU surrogate.')
        with torch.no_grad():
            features = model.features.detach().clone()
            selected = (model.one_hot(anchor).unsqueeze(-1) * features).sum(1)
            region = selected.new_zeros((model.n_regions, 4)).index_add(0, model.decision_region, selected)
            local = torch.sparse.mm(model.port_incidence, selected)
            upstream = region[model.edge_sources.clamp_min(0)] * (model.edge_sources >= 0)[:, None]
            interaction = region[model.edge_targets] * upstream
            logs = (local * model.local_weights).sum(-1) + (interaction * model.interaction_weights).sum(-1)
            costs0 = model.edge_depths * model.depth_to_ps * torch.exp(2 * torch.tanh(logs / 2))
            summary = (selected * model.sizes[:, None]).sum(0) / max(model.partition.aig.and_count, 1)
            area_log = (summary * model.area_weights).sum()
            area0 = model.area_baseline * torch.exp(2 * torch.tanh(area_log / 2))
        costs = costs0.unsqueeze(0).requires_grad_()
        area = area0.detach().requires_grad_()
        propagate = reference_arrivals if model.propagation_backend == 'reference' else smooth_arrivals
        arrivals = propagate(costs, model.arrival_layers, model.n_regions, model.temperature)
        raw_po = arrivals[:, model.output_regions.clamp_min(0)] * (model.output_regions >= 0)[None, :]
        po = torch.clamp(model.baseline_po + raw_po[0] - model.baseline_dp, min=0)
        objective = scalar_objective(po, area, model.delay_reference, model.area_reference,
                                     mode, area_limit_ratio, area_weight=area_weight)
        cost_grad, area_grad = torch.autograd.grad(objective, (costs, area))
        with torch.no_grad():
            log_adjoint = cost_grad[0] * costs0 * (1 - torch.tanh(logs / 2).square())
            local_coeff = torch.sparse.mm(model.port_incidence.transpose(0, 1),
                                         log_adjoint[:, None] * model.local_weights[None, :])
            area_coeff = area_grad * area0 * (1 - torch.tanh(area_log / 2).square()) * model.area_weights
            linear = local_coeff + model.sizes[:, None] / max(model.partition.aig.and_count, 1) * area_coeff
            active_regions, inverse = torch.unique(model.decision_region, sorted=True, return_inverse=True)
            compressed = {rid: i for i, rid in enumerate(active_regions.tolist())}
            pairs = {}
            for edge, (source, target) in enumerate(zip(model.edge_sources.tolist(), model.edge_targets.tolist())):
                if source in compressed and target in compressed:
                    pair = (compressed[target], compressed[source])
                    value = log_adjoint[edge] * model.interaction_weights.detach()
                    pairs[pair] = pairs.get(pair, torch.zeros_like(value)) + value
            # Zero interaction weights are common before calibration. Remove
            # exactly zero terms; CPU and GPU receive identical compressed data.
            pairs = {pair: value for pair, value in pairs.items() if torch.count_nonzero(value)}
            keys = sorted(pairs)
            self.n_active_regions = len(active_regions)
            self.register_buffer('features', features)
            self.register_buffer('linear_candidates', (features * linear[:, None, :]).sum(-1))
            self.register_buffer('valid', model.valid.detach().clone())
            self.register_buffer('decision_region', inverse)
            self.register_buffer('pair_target', torch.tensor([p[0] for p in keys], dtype=torch.long))
            self.register_buffer('pair_source', torch.tensor([p[1] for p in keys], dtype=torch.long))
            self.register_buffer('pair_weights', torch.stack([pairs[p] for p in keys]) if keys else torch.empty((0, 4), dtype=torch.float64))
            self.register_buffer('conflict_left', model.conflict_left.detach().clone())
            self.register_buffer('conflict_right', model.conflict_right.detach().clone())
            anchor_hot = model.one_hot(anchor)
            self.register_buffer('anchor_hot', anchor_hot)
            self.register_buffer('offset', objective.detach() - self._unshifted(anchor_hot)[0])
            self.anchor_objective = float(objective.detach())

    def _unshifted(self, values):
        batch = values.unsqueeze(0) if values.ndim == 2 else values
        linear = (batch * self.linear_candidates).sum(dim=(1, 2))
        if self.pair_target.numel() == 0:
            return linear
        expected = (batch.unsqueeze(-1) * self.features).sum(2)
        region = expected.new_zeros((batch.shape[0], self.n_active_regions, 4)).index_add(1, self.decision_region, expected)
        coupled = region[:, self.pair_target] * region[:, self.pair_source] * self.pair_weights
        return linear + coupled.sum(dim=(1, 2))

    def forward(self, values):
        result = self.offset + self._unshifted(values)
        return result[0] if values.ndim == 2 else result


@dataclass(frozen=True)
class JointProposalConfig:
    device: str = 'cpu'
    trajectories: int = 32
    max_changes: int = 32
    trust_penalty: float = 0.05
    conflict_penalty: float = 0.1
    learning_rate: float = 0.15

    def validate(self):
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('Joint proposal device must be cpu or cuda.')
        if min(self.trajectories, self.max_changes) < 1:
            raise ValueError('Positive trajectory count and edit limit required.')
        if any(not math.isfinite(x) or x <= 0 for x in (self.trust_penalty, self.conflict_penalty, self.learning_rate)):
            raise ValueError('Positive finite penalties and learning rate required.')


def joint_proposals(model, seen, seed, incumbent=None, steps=40, restarts=2,
                    mode='area_constraint', area_limit_ratio=1.05, area_weight=0.5,
                    config=JointProposalConfig()):
    """Batched soft-relaxation Adam, then CPU legal neighborhood projection.

    ``restarts`` is accepted for the pipeline callback interface; the actual
    independently initialized trajectories are explicitly configured/logged.
    No per-step CPU projection or device synchronization is used. The full
    surrogate predicts each actually requested point later in the pipeline.
    """
    config.validate()
    if steps <= 0 or restarts <= 0:
        raise ValueError('Positive optimization steps/restarts required.')
    if model.n_decisions == 0:
        return [], [{'reason': 'No executable decisions.'}]
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing a silent CPU fallback.')
    started = time.perf_counter()
    anchor = incumbent if incumbent is not None else [0] * model.n_decisions
    model.requires_grad_(False)
    objective = FrozenCriticalityObjective(model, anchor, mode, area_limit_ratio, area_weight)
    anchor_value = model.one_hot(anchor).requires_grad_()
    signal_norm = float(torch.autograd.grad(objective(anchor_value), anchor_value)[0].norm())
    prepared = time.perf_counter()
    device = torch.device(config.device)
    objective = objective.to(device)
    count, width = model.features.shape[:2]
    limit = min(config.max_changes, count)
    budget_grid = sorted({min(limit, k) for k in (1, 2, 4, 8, 16, 32, limit)})
    budgets_cpu = torch.tensor([budget_grid[b % len(budget_grid)] for b in range(config.trajectories)], dtype=torch.float64)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn((config.trajectories, count, width), dtype=torch.float64, generator=generator) * 0.5
    # Start near the same actual incumbent; no additional mapping labels.
    bias = torch.log((count * max(width - 1, 1)) / budgets_cpu) + 2.0
    starting = noise + bias[:, None, None] * objective.anchor_hot.detach().cpu()[None, :, :]
    logits = nn.Parameter(starting.to(device))
    budgets = budgets_cpu.to(device)
    optimizer = torch.optim.Adam([logits], lr=config.learning_rate, eps=1e-12)
    trace_tensors = []
    for step in range(steps):
        temperature = max(0.3, 1 - 0.7 * step / max(steps - 1, 1))
        probabilities = torch.softmax(logits.masked_fill(~objective.valid, -torch.inf) / temperature, dim=-1)
        value = objective(probabilities)
        changes = (1 - (probabilities * objective.anchor_hot).sum(-1)).sum(-1)
        trust = torch.relu(changes / budgets - 1).square()
        active = 1 - probabilities[..., 0]
        if objective.conflict_left.numel():
            conflict = (active[:, objective.conflict_left] * active[:, objective.conflict_right]).mean(-1)
        else:
            conflict = torch.zeros_like(value)
        loss_rows = value + config.trust_penalty * trust + config.conflict_penalty * conflict
        optimizer.zero_grad()
        # Sum keeps each trajectory's Adam and clipping independent of batch size.
        loss_rows.sum().backward()
        norms = logits.grad.flatten(1).norm(dim=-1)
        logits.grad.mul_((5 / norms.clamp_min(1e-30)).clamp_max(1)[:, None, None])
        if step in (0, steps - 1) or step % 10 == 0:
            trace_tensors.append(torch.stack((value.detach(), trust.detach(), conflict.detach(), norms.detach(), changes.detach()), dim=-1))
        optimizer.step()
    if not torch.isfinite(logits).all() or not torch.isfinite(logits.grad).all():
        raise ValueError('Non-finite joint optimization state.')
    # One transfer after all inner steps. Final legal projection is conservative
    # and includes every unchanged incumbent decision when checking conflicts.
    rows = logits.detach().cpu().tolist()
    proposals = {}
    for trajectory, scores in enumerate(rows):
        choice = project_neighborhood(model.catalogue, anchor,
                                      [r[:n] for r, n in zip(scores, model.counts)], int(budgets_cpu[trajectory]))
        key = tuple(choice)
        if key not in seen:
            proposals.setdefault(key, {'choices': choice, 'trajectory': trajectory,
                                       'max_changes': int(budgets_cpu[trajectory]),
                                       'actual_changes': sum(a != b for a, b in zip(anchor, choice))})
    # Use CPU coefficients for reproducible final ranking and do not claim the
    # frozen objective to be a path/global delay prediction in picoseconds.
    cpu_objective = objective.cpu()
    with torch.no_grad():
        for proposal in proposals.values():
            proposal['predicted_objective'] = float(cpu_objective(model.one_hot(proposal['choices'])))
            proposal['objective_kind'] = 'frozen_criticality_quadratic'
    trace_values = torch.stack(trace_tensors).cpu().tolist()
    elapsed = time.perf_counter() - started
    details = {'trace_schema': 'joint_criticality_v1',
               'device': config.device, 'dtype': 'float64', 'trajectories': config.trajectories,
               'steps': steps, 'seed': seed, 'anchor': list(anchor),
               'anchor_objective': objective.anchor_objective,
               'anchor_objective_gradient_norm': signal_norm,
               'retained_interaction_pairs': int(cpu_objective.pair_target.numel()),
               'prepare_adjoint_s': prepared - started, 'total_proposal_s': elapsed,
               'sampled_values': trace_values, 'trace_columns': ['local_objective', 'trust_penalty', 'conflict_penalty', 'total_gradient_norm', 'soft_edit_count'],
               'scope': 'Local surrogate approximation; soft inner optimization, hard legal final projection; no gradient through EDA or LiteCut.'}
    return sorted(proposals.values(), key=lambda p: (p['predicted_objective'], p['choices'])), [details]

"""Calibrated, port-aware Region DAG surrogate; never a replacement for STA.

All baseline and feedback labels use the same ABC Liberty timing manager.
Soft max propagates every structural predecessor, rather than summing all
regions. The baseline anchor is exact by construction, not accuracy evidence.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class TimingSurrogate(nn.Module):
    def __init__(self, partition, catalogue, scores, baseline, temperature_ratio=0.005):
        super().__init__()
        self.partition = partition
        self.catalogue = catalogue
        self.counts = [d['candidate_count'] + 1 for d in catalogue.decisions]
        self.width = max(self.counts, default=1)
        self.n_decisions = len(self.counts)
        self.n_regions = len(partition.regions)
        self.delay_reference = max(float(baseline['delay_ps']), 1e-6)
        self.area_reference = max(float(baseline['area']), 1e-6)
        if temperature_ratio <= 0 or not math.isfinite(temperature_ratio):
            raise ValueError('Timing smoothing temperature must be positive.')
        self.temperature = max(self.delay_reference * temperature_ratio, 1e-8)
        baseline_po = torch.tensor(baseline['po_arrivals_ps'], dtype=torch.float64)
        if len(baseline_po) != partition.aig.po_count or not torch.isfinite(baseline_po).all():
            raise ValueError('Baseline PO identity/count mismatch.')
        self.register_buffer('baseline_po', baseline_po)
        features = torch.zeros((self.n_decisions, self.width, 4), dtype=torch.float64)
        valid = torch.zeros((self.n_decisions, self.width), dtype=torch.bool)
        groups = {(row['decision_index'], row['choice_index']): row for row in scores}
        sizes = []
        for i, count in enumerate(self.counts):
            valid[i, :count] = True
            original = groups[i, 0]
            local0 = original['predicted_local_abc_delay']
            raw0 = original['continuous_features_19']
            sizes.append(max(raw0[11], 1.0))
            for c in range(1, count):
                row = groups[i, c]
                raw = row['continuous_features_19']
                features[i, c] = torch.tensor([
                    (row['predicted_local_abc_delay'] - local0) / max(abs(local0), 1.0),
                    (raw[8] - raw0[8]) / max(abs(raw0[8]), 1.0),
                    (raw[11] - raw0[11]) / max(abs(raw0[11]), 1.0), 1.0], dtype=torch.float64)
        # Bound extrapolation from extreme local proxy ratios; actual metrics
        # are never clipped. This is an explicit surrogate feature transform.
        features = features.clamp(-5.0, 5.0)
        if not torch.isfinite(features).all():
            raise ValueError('Non-finite surrogate features.')
        self.register_buffer('features', features)
        self.register_buffer('valid', valid)
        self.register_buffer('sizes', torch.tensor(sizes, dtype=torch.float64))
        incidence = torch.zeros((self.n_regions, self.n_decisions), dtype=torch.float64)
        for rid, indices in catalogue.region_decisions.items():
            for i in indices:
                incidence[rid, i] = 1.0
        self.register_buffer('incidence', incidence)
        self.local_weights = nn.Parameter(torch.tensor([0.08, 0.15, 0.0, 0.0], dtype=torch.float64))
        self.interaction_weights = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        self.area_weights = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64))
        self.register_buffer('prior_local', self.local_weights.detach().clone())
        self.register_buffer('prior_area', self.area_weights.detach().clone())
        # Find which original boundary ports can reach each decision root.
        ports_at = {}
        for node in sorted(partition.aig.and_nodes):
            rid = partition.owner[node]
            ports = set()
            for literal in partition.aig.fanin_literals[node]:
                source = literal >> 1
                ports.update(ports_at[source] if partition.owner.get(source) == rid else {source})
            ports_at[node] = ports
        self.port_decisions = {}
        for rid, transfers in enumerate(partition.transfers):
            for port in transfers:
                self.port_decisions[rid, port] = [i for i in catalogue.region_decisions[rid]
                                                if port in ports_at[catalogue.decisions[i]['root']]]
        structural_arrivals = {}
        for rid in partition.order:
            structural_arrivals[rid] = max((structural_arrivals[partition.owner[p]] if p in partition.owner else 0) + d
                                            for p, d in partition.transfers[rid].items())
        max_depth = max((structural_arrivals[partition.owner[lit >> 1]] if (lit >> 1) in partition.owner else 0
                         for lit in partition.aig.po_literals), default=1)
        self.depth_to_ps = self.delay_reference / max(max_depth, 1)
        with torch.no_grad():
            zero = torch.zeros((1, self.n_decisions, 4), dtype=torch.float64)
            baseline_dp, _ = self._propagate(zero)
        self.register_buffer('baseline_dp', baseline_dp[0])

    def one_hot(self, choices):
        self.catalogue.validate(choices)
        result = torch.zeros((self.n_decisions, self.width), dtype=self.features.dtype, device=self.features.device)
        for i, choice in enumerate(choices):
            result[i, choice] = 1
        return result

    def _propagate(self, expected):
        batch = expected.shape[0]
        zero = expected.new_zeros(batch)
        region_features = torch.einsum('rd,bdf->brf', self.incidence, expected)
        arrivals = {}
        for rid in self.partition.order:
            inputs = []
            for port, depth in self.partition.transfers[rid].items():
                indices = self.port_decisions[rid, port]
                local = expected[:, indices, :].sum(dim=1) if indices else expected.new_zeros((batch, 4))
                if port in self.partition.owner:
                    pred = self.partition.owner[port]
                    upstream = arrivals[pred]
                    interaction = region_features[:, rid, :] * region_features[:, pred, :]
                else:
                    upstream = zero
                    interaction = expected.new_zeros((batch, 4))
                log_scale = (local * self.local_weights).sum(-1) + (interaction * self.interaction_weights).sum(-1)
                cost = depth * self.depth_to_ps * torch.exp(2.0 * torch.tanh(log_scale / 2.0))
                inputs.append(upstream + cost)
            arrivals[rid] = self.temperature * torch.logsumexp(torch.stack(inputs, dim=-1) / self.temperature, dim=-1)
        endpoints = [arrivals[self.partition.owner[lit >> 1]] if (lit >> 1) in self.partition.owner else zero
                     for lit in self.partition.aig.po_literals]
        po = torch.stack(endpoints, dim=-1) if endpoints else expected.new_zeros((batch, 0))
        return po, arrivals

    def forward(self, probabilities):
        single = probabilities.ndim == 2
        values = probabilities.unsqueeze(0) if single else probabilities
        if values.shape[1:] != self.features.shape[:2]:
            raise ValueError('Selection tensor shape mismatch.')
        expected = (values.unsqueeze(-1) * self.features.unsqueeze(0)).sum(dim=2)
        dp, region_arrivals = self._propagate(expected)
        po = torch.clamp(self.baseline_po + dp - self.baseline_dp, min=0.0)
        # Gate-count change is only a predictor of mapped area and is fitted
        # against real mapped areas, never reported as measured area.
        summary = (expected * self.sizes[None, :, None]).sum(dim=1) / max(self.partition.aig.and_count, 1)
        area = self.area_reference * torch.exp(2.0 * torch.tanh((summary * self.area_weights).sum(-1) / 2.0))
        if single:
            return po[0], area[0]
        return po, area

    def fit_observations(self, observations, steps=120, learning_rate=0.025):
        usable = [row for row in observations if row.get('status') == 'PASS']
        if not usable:
            return {'samples': 0, 'losses': []}
        x = torch.stack([self.one_hot(row['choices']) for row in usable])
        target_po = torch.tensor([row['metrics']['po_arrivals_ps'] for row in usable], dtype=torch.float64)
        target_area = torch.tensor([row['metrics']['area'] for row in usable], dtype=torch.float64)
        self.requires_grad_(True)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad()
            po, area = self(x)
            loss = ((po - target_po) / self.delay_reference).square().mean()
            loss = loss + ((area - target_area) / self.area_reference).square().mean()
            regularizer = (self.local_weights - self.prior_local).square().mean()
            regularizer = regularizer + self.interaction_weights.square().mean() + (self.area_weights - self.prior_area).square().mean()
            loss = loss + 0.002 * regularizer
            if not torch.isfinite(loss):
                raise ValueError('Non-finite calibration loss.')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
                raise ValueError('Non-finite calibration gradient.')
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        self.requires_grad_(False)
        return {'samples': len(usable), 'losses': losses,
                'parameters': {name: p.detach().tolist() for name, p in self.named_parameters()},
                'scope': 'Training fit only; new evaluations must be predicted before their labels are observed.'}


def scalar_objective(po, area, delay_reference, area_reference, mode='area_constraint',
                     area_limit_ratio=1.05, temperature=0.02, area_weight=0.2, path_weights=None):
    if mode not in {'area_constraint', 'joint'} or temperature <= 0 or area_limit_ratio <= 0:
        raise ValueError('Invalid objective configuration.')
    if not 0 < area_weight < 1:
        raise ValueError('area_weight must lie strictly between zero and one.')
    if po.numel() == 0:
        raise ValueError('Timing optimization requires at least one output.')
    normalized = po / max(delay_reference, 1e-6)
    weights = torch.ones_like(normalized) if path_weights is None else path_weights
    if weights.shape != normalized.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Path weights must be positive finite values for every PO.')
    # Every endpoint participates, including paths that become critical later.
    delay_cost = temperature * torch.logsumexp(normalized / temperature + weights.log(), dim=-1)
    area_cost = area / max(area_reference, 1e-6)
    if mode == 'area_constraint':
        return delay_cost + 20.0 * torch.relu(area_cost - area_limit_ratio).square()
    # Smooth Tchebycheff with a fixed zero reference and normalized objectives.
    terms = torch.stack([(1 - area_weight) * delay_cost, area_weight * area_cost])
    return temperature * torch.logsumexp(terms / temperature, dim=0)

"""Calibrated, port-aware Region DAG surrogate; never a replacement for STA.

All baseline and feedback labels use the same ABC Liberty timing manager.
Soft max propagates every structural predecessor, rather than summing all
regions. The baseline anchor is exact by construction, not accuracy evidence.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .arrival import prepare_arrival_layers, reference_arrivals, smooth_arrivals


class _IndependentRegionState:
    """Non-differentiable coordinate costs for one fixed baseline model.

    Only choices may change. Unaffected port costs retain their exact values;
    affected ports are recomputed from the original ordered feature sum, never
    updated by subtracting/adding candidate contributions. Features and topology
    remain immutable, as in the enclosing independent-baseline invocation.
    """
    def __init__(self, model, rid, choices=None):
        choices = {} if choices is None else choices
        model._validate_independent_choices(rid, choices)
        self.model, self.rid = model, rid
        self._choices = {index: choices.get(index, 0) for index in model.catalogue.region_decisions[rid]}
        self._ports = list(model.partition.transfers[rid])
        self._affected = {index: [] for index in self._choices}
        for position, port in enumerate(self._ports):
            for index in model.port_decisions[rid, port]:
                self._affected[index].append(position)
        self._weights = model.local_weights.detach().clone()
        self._temperature, self._depth_to_ps = model.temperature, model.depth_to_ps
        self._pending_index, self._pending_values = None, None
        with torch.no_grad():
            local = model._independent_port_options(rid, self._choices, None, self._ports, [0])[0]
            values = torch.from_numpy(local)
            self._values = torch.stack([model._independent_port_cost(rid, port, values[position])
                                        for position, port in enumerate(self._ports)])

    @property
    def choices(self):
        return dict(self._choices)

    def _check_context(self, index=None):
        if index is not None and index not in self._choices:
            raise ValueError('Independent selection includes another region.')
        if (self.model.temperature != self._temperature or self.model.depth_to_ps != self._depth_to_ps or
                not torch.equal(self.model.local_weights.detach(), self._weights)):
            raise ValueError('Independent coordinate cache requires a fixed baseline model.')

    def _candidate_values(self, index, options):
        positions = self._affected[index]
        ports = [self._ports[position] for position in positions]
        local = self.model._independent_port_options(self.rid, self._choices, index, ports, options)
        features = torch.from_numpy(local)
        values = self._values.expand(len(options), -1).clone()
        for candidate in range(len(options)):
            for offset, position in enumerate(positions):
                values[candidate, position] = self.model._independent_port_cost(
                    self.rid, self._ports[position], features[candidate, offset])
        return values

    def _arrival(self, values):
        return self._temperature * torch.logsumexp(values / self._temperature, dim=0)

    def current_arrival(self):
        self._check_context()
        with torch.no_grad():
            return self._arrival(self._values)

    def options(self, index):
        self._check_context(index)
        with torch.no_grad():
            values = self._candidate_values(index, list(range(self.model.counts[index])))
            self._pending_index, self._pending_values = index, values
            # Use the same one-dimensional reduction kernel and port order as
            # the scalar reference, including the unchanged port entries.
            return torch.stack([self._arrival(row) for row in values])

    def set_choice(self, index, choice):
        self._check_context(index)
        if type(choice) is not int or not 0 <= choice < self.model.counts[index]:
            raise ValueError('Invalid local candidate choice.')
        with torch.no_grad():
            if self._choices[index] != choice:
                if self._pending_index == index:
                    self._values = self._pending_values[choice].clone()
                else:
                    self._values = self._candidate_values(index, [choice])[0].clone()
                self._choices[index] = choice
            # A forced change on another coordinate invalidates old proposals.
            self._pending_index, self._pending_values = None, None


class TimingSurrogate(nn.Module):
    def __init__(self, partition, catalogue, scores, baseline, temperature_ratio=0.005,
                 propagation_backend='bounded_tape', defer_baseline=False):
        super().__init__()
        if propagation_backend not in {'bounded_tape', 'reference'}:
            raise ValueError('Unknown arrival propagation backend.')
        self.propagation_backend = propagation_backend
        self.partition = partition
        self.catalogue = catalogue
        self.counts = [d['candidate_count'] + 1 for d in catalogue.decisions]
        self.width = max(self.counts, default=1)
        self.n_decisions = len(self.counts)
        self.n_regions = len(partition.regions)
        self.delay_reference = max(float(baseline['delay_ps']), 1e-6)
        self.area_reference = max(float(baseline['area']), 1e-6)
        self.area_baseline = float(baseline['area'])
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
        decision_region = torch.zeros(self.n_decisions, dtype=torch.long)
        for rid, indices in catalogue.region_decisions.items():
            for i in indices:
                decision_region[i] = rid
        self.register_buffer('decision_region', decision_region)
        self.register_buffer('conflict_left', torch.tensor([a for a, _ in catalogue.conflicts], dtype=torch.long))
        self.register_buffer('conflict_right', torch.tensor([b for _, b in catalogue.conflicts], dtype=torch.long))
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
        sources, targets, depths, sparse_rows, sparse_cols = [], [], [], [], []
        levels = {}
        for rid in partition.order:
            predecessors = [partition.owner[p] for p in partition.transfers[rid] if p in partition.owner]
            levels[rid] = 1 + max((levels[p] for p in predecessors), default=-1)
        edge_by_target = {}
        for rid, transfers in enumerate(partition.transfers):
            for port, depth in transfers.items():
                edge = len(sources)
                sources.append(partition.owner.get(port, -1))
                targets.append(rid)
                depths.append(depth)
                edge_by_target.setdefault(rid, []).append(edge)
                for decision in self.port_decisions[rid, port]:
                    sparse_rows.append(edge)
                    sparse_cols.append(decision)
        self.register_buffer('edge_sources', torch.tensor(sources, dtype=torch.long))
        self.register_buffer('edge_targets', torch.tensor(targets, dtype=torch.long))
        self.register_buffer('edge_depths', torch.tensor(depths, dtype=torch.float64))
        self.register_buffer('port_incidence', torch.sparse_coo_tensor(
            torch.tensor([sparse_rows, sparse_cols], dtype=torch.long),
            torch.ones(len(sparse_rows), dtype=torch.float64),
            (len(sources), self.n_decisions)).coalesce())
        self.layers = []
        nodes_by_level = {}
        for rid in partition.order:
            nodes_by_level.setdefault(levels[rid], []).append(rid)
        for layer in sorted(nodes_by_level):
            nodes = nodes_by_level[layer]
            edges, destinations = [], []
            for position, rid in enumerate(nodes):
                edges.extend(edge_by_target[rid])
                destinations.extend([position] * len(edge_by_target[rid]))
            self.layers.append((torch.tensor(nodes, dtype=torch.long), torch.tensor(edges, dtype=torch.long),
                                torch.tensor(destinations, dtype=torch.long)))
        self.arrival_layers = prepare_arrival_layers(self.edge_sources, self.layers)
        self.register_buffer('output_regions', torch.tensor([partition.owner.get(lit >> 1, -1)
                                                            for lit in partition.aig.po_literals], dtype=torch.long))
        structural_arrivals = {}
        for rid in partition.order:
            structural_arrivals[rid] = max((structural_arrivals[partition.owner[p]] if p in partition.owner else 0) + d
                                            for p, d in partition.transfers[rid].items())
        max_depth = max((structural_arrivals[partition.owner[lit >> 1]] if (lit >> 1) in partition.owner else 0
                         for lit in partition.aig.po_literals), default=1)
        self.depth_to_ps = self.delay_reference / max(max_depth, 1)
        with torch.no_grad():
            zero = torch.zeros((1, self.n_decisions, 4), dtype=torch.float64)
            # A device subclass builds its immutable native graph before doing
            # this propagation. The existing default remains byte-for-value.
            if defer_baseline:
                baseline_dp = zero.new_zeros((1, len(self.output_regions)))
                baseline_regions = zero.new_zeros((self.n_regions, 1))
            else:
                baseline_dp, baseline_regions = self._propagate(zero)
        self.register_buffer('baseline_dp', baseline_dp[0])
        self.register_buffer('baseline_region_arrivals', baseline_regions[:, 0])

    def independent_region_arrival(self, rid, choices):
        """Exact region-root cost with all other regions fixed to no-op.

        Predecessors cannot change in this independent baseline. Computing only
        this region avoids a whole-DAG propagation for every coordinate option.
        This shortcut is never used for the joint optimizer.
        """
        allowed = set(self.catalogue.region_decisions[rid])
        if not set(choices) <= allowed:
            raise ValueError('Independent selection includes another region.')
        local_features = []
        for port in self.partition.transfers[rid]:
            local = self.features.new_zeros(4)
            for index in self.port_decisions[rid, port]:
                choice = choices.get(index, 0)
                if not 0 <= choice < self.counts[index]:
                    raise ValueError('Invalid local candidate choice.')
                local = local + self.features[index, choice]
            local_features.append(local)
        return self._independent_arrival_from_ports(rid, local_features)

    def _independent_arrival_from_ports(self, rid, local_features):
        values = [self._independent_port_cost(rid, port, local)
                  for port, local in zip(self.partition.transfers[rid], local_features)]
        return self.temperature * torch.logsumexp(torch.stack(values) / self.temperature, dim=0)

    def _independent_port_cost(self, rid, port, local):
        scale = (local * self.local_weights).sum()
        cost = self.partition.transfers[rid][port] * self.depth_to_ps * torch.exp(2 * torch.tanh(scale / 2))
        upstream = self.partition.owner.get(port)
        if upstream is not None:
            cost = cost + self.baseline_region_arrivals[upstream]
        return cost

    def _validate_independent_choices(self, rid, choices, index=None):
        if rid not in self.catalogue.region_decisions:
            raise ValueError('Unknown independent region.')
        allowed = set(self.catalogue.region_decisions[rid])
        if (index is not None and index not in allowed) or not set(choices) <= allowed:
            raise ValueError('Independent selection includes another region.')
        if any(type(c) is not int or not 0 <= c < self.counts[i] for i, c in choices.items()):
            raise ValueError('Invalid local candidate choice.')
        if self.features.device.type != 'cpu' or self.features.dtype != torch.float64:
            raise ValueError('Independent baseline batching requires float64 CPU features.')

    def independent_region_state(self, rid, choices=None):
        """Cache exact unaffected port costs for a fixed independent model."""
        return _IndependentRegionState(self, rid, choices)

    def independent_region_options(self, rid, choices, index):
        """Batch the independent baseline's coordinate options on the CPU.

        Ordered add.accumulate preserves the reference's left-to-right feature
        additions. It avoids a Python/Torch operation for every (port, decision,
        candidate) triple. No incremental cache or regrouped sum is used. This
        is a non-differentiable baseline helper, not the joint surrogate path.
        """
        self._validate_independent_choices(rid, choices, index)
        count = self.counts[index]
        ports = list(self.partition.transfers[rid])
        local = self._independent_port_options(rid, choices, index, ports, list(range(count)))
        with torch.no_grad():
            # Retained uncached reference uses the same scalar kernels.
            values = torch.from_numpy(local)
            return torch.stack([self._independent_arrival_from_ports(rid, values[c]) for c in range(count)])

    def _independent_port_options(self, rid, choices, index, ports, options):
        """Ordered sums for selected ports and options; callers validate state."""
        count = len(options)
        features = self.features.detach().numpy()
        local = np.zeros((count, len(ports), 4), dtype=np.float64)
        for position, port in enumerate(ports):
            indices = self.port_decisions[rid, port]
            if not indices:
                continue
            selected = features[indices, [choices.get(i, 0) for i in indices]]
            if index in indices:
                # Leading zero mirrors the scalar reference's initial value.
                terms = np.zeros((count, len(indices) + 1, 4), dtype=np.float64)
                terms[:, 1:, :] = selected
                terms[:, indices.index(index) + 1, :] = features[index, options]
                np.add.accumulate(terms, axis=1, out=terms)
                local[:, position, :] = terms[:, -1, :]
            else:
                terms = np.concatenate((np.zeros((1, 4), dtype=np.float64), selected))
                np.add.accumulate(terms, axis=0, out=terms)
                local[:, position, :] = terms[-1]
        return local

    def one_hot(self, choices):
        self.catalogue.validate(choices)
        result = torch.zeros((self.n_decisions, self.width), dtype=self.features.dtype, device=self.features.device)
        indices = torch.tensor(choices, dtype=torch.long, device=result.device).unsqueeze(1)
        result.scatter_(1, indices, 1.0)
        return result

    def _propagate(self, expected):
        batch = expected.shape[0]
        if self.n_regions == 0:
            return expected.new_zeros((batch, len(self.output_regions))), expected.new_zeros((0, batch))
        region_features = expected.new_zeros((batch, self.n_regions, 4)).index_add(1, self.decision_region, expected)
        # Sparse edge/decision aggregation and level-batched LSE avoid an R*D
        # dense matrix and thousands of tiny autograd operations per sample.
        flattened = expected.permute(1, 0, 2).reshape(self.n_decisions, batch * 4)
        local = torch.sparse.mm(self.port_incidence, flattened).reshape(-1, batch, 4).permute(1, 0, 2)
        upstream_features = region_features[:, self.edge_sources.clamp_min(0), :]
        upstream_features = upstream_features * (self.edge_sources >= 0)[None, :, None]
        interaction = region_features[:, self.edge_targets, :] * upstream_features
        log_scale = (local * self.local_weights).sum(-1) + (interaction * self.interaction_weights).sum(-1)
        costs = self.edge_depths[None, :] * self.depth_to_ps * torch.exp(2.0 * torch.tanh(log_scale / 2.0))
        propagate = reference_arrivals if self.propagation_backend == 'reference' else smooth_arrivals
        arrivals = propagate(costs, self.arrival_layers, self.n_regions, self.temperature)
        po = arrivals[:, self.output_regions.clamp_min(0)] * (self.output_regions >= 0)[None, :]
        return po, arrivals.T

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
        area = self.area_baseline * torch.exp(2.0 * torch.tanh((summary * self.area_weights).sum(-1) / 2.0))
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
                'arrival_propagation_backend': self.propagation_backend,
                'supported_derivative_order': 1 if self.propagation_backend == 'bounded_tape' else None,
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

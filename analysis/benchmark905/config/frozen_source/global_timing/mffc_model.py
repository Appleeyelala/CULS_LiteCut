"""Original surrogate equations, with candidate aggregation and DAG on CUDA.

The native queue propagates one MFFC per warp. No frozen CPU DAG adjoint or
multiple independent search trajectories substitute for those dependencies.
"""
import torch

from .mffc_gpu import RegionGpuPlan, region_arrivals
from .surrogate import TimingSurrogate


class MffcGpuSurrogate(TimingSurrogate):
    def __init__(self, partition, catalogue, scores, baseline, library):
        super().__init__(partition, catalogue, scores, baseline, defer_baseline=True)
        sources, targets = self.edge_sources.tolist(), self.edge_targets.tolist()
        self.to('cuda')
        self.native_plan = RegionGpuPlan(sources, targets, self.n_regions, library)
        with torch.no_grad():
            zero = self.features.new_zeros((1, self.n_decisions, 4))
            outputs, arrivals = self._propagate(zero)
            self.baseline_dp.copy_(outputs[0])
            self.baseline_region_arrivals.copy_(arrivals[:, 0])

    def _propagate(self, expected):
        batch = expected.shape[0]
        regions = expected.new_zeros((batch, self.n_regions, 4)).index_add(1, self.decision_region, expected)
        flat = expected.permute(1, 0, 2).reshape(self.n_decisions, batch * 4)
        local = torch.sparse.mm(self.port_incidence, flat).reshape(-1, batch, 4).permute(1, 0, 2)
        upstream = regions[:, self.edge_sources.clamp_min(0)] * (self.edge_sources >= 0)[None, :, None]
        pair = regions[:, self.edge_targets] * upstream
        logs = (local * self.local_weights).sum(-1) + (pair * self.interaction_weights).sum(-1)
        costs = self.edge_depths[None] * self.depth_to_ps * torch.exp(2 * torch.tanh(logs / 2))
        arrivals = region_arrivals(costs.contiguous(), self.native_plan, self.temperature)
        outputs = arrivals[:, self.output_regions.clamp_min(0)] * (self.output_regions >= 0)[None, :]
        return outputs, arrivals.T

    def fit_observations(self, observations, steps=12, learning_rate=0.025):
        usable = [row for row in observations if row.get('status') == 'PASS']
        if not usable:
            return {'samples': 0, 'losses': []}
        x = torch.stack([self.one_hot(row['choices']) for row in usable])
        target_po = self.features.new_tensor([r['metrics']['po_arrivals_ps'] for r in usable])
        target_area = self.features.new_tensor([r['metrics']['area'] for r in usable])
        self.requires_grad_(True)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad()
            po, area = self(x)
            loss = ((po - target_po) / self.delay_reference).square().mean()
            loss = loss + ((area - target_area) / self.area_reference).square().mean()
            regularizer = (self.local_weights - self.prior_local).square().mean()
            regularizer = regularizer + self.interaction_weights.square().mean()
            regularizer = regularizer + (self.area_weights - self.prior_area).square().mean()
            loss = loss + 0.002 * regularizer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.detach())
        values = torch.stack(losses).cpu()
        if not torch.isfinite(values).all() or any(not torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError('Non-finite native GPU calibration state.')
        self.requires_grad_(False)
        return {'samples': len(usable), 'losses': values.tolist(),
                'training_observation_indices': [r['index'] for r in usable],
                'parameters': {name: p.detach().cpu().tolist() for name, p in self.named_parameters()},
                'arrival_propagation_backend': 'mffc_ready_queue_cuda', 'supported_derivative_order': 1,
                'scope': 'Existing twelve coefficients only; earlier PASS labels, no Student training.'}

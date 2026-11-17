"""Causal PO context for the existing v3 global head; no new neural parameters.

Original structural sensitivities are reference features, not ABC criticality.
Only the residual head's last eight inputs change. Timing/search stay untouched.
"""
import copy
import math

import torch

from .conditional_arrival import _stage, conditional_arrivals, conditional_reference
from .learned_ports import LearnedPortSurrogate


@torch.no_grad()
def reference_influence(model):
    """Forward-mode root-delay derivatives and reachability, O(edges*decisions).

    Perturb both logical phases of each original decision root by the same
    additive delay. Propagate its derivative through the original-only LSE
    timing DAG. Reachability independently retains noncritical upstream roots.
    No output-by-edge Jacobian or candidate timing labels are constructed.
    """
    n, d = len(model.initial), model.n_decisions
    injection = model.initial.new_zeros((n, d))
    root_of_decision = {int(f)//model.width: int(r) for f,r in
                        zip(model.flat_indices.tolist(), model.root_indices.tolist())}
    for i in range(d):
        root = root_of_decision[i]
        injection[2+2*root:4+2*root, i] = 1.
    sensitivity = torch.zeros_like(injection)
    reachability = torch.zeros_like(injection)
    arrival = model.initial[None].clone()
    costs = model.edge_prior[None]
    for layer in model._layers(model.initial.device):
        nodes, options, edges, sources, groups, destinations = layer
        values, derivatives = _stage(arrival, costs, layer, model.temperature)
        original = model.option_choice[options] < 0
        edge_original = original[groups]
        destination_edges = destinations[groups]
        indices = destination_edges[:,None].expand(-1,d)
        step = model.initial.new_zeros((len(nodes),d)).scatter_add(
            0, indices, sensitivity[sources]*(derivatives[0]*edge_original)[:,None])
        reach = model.initial.new_zeros((len(nodes),d)).scatter_reduce(
            0, indices, reachability[sources]*edge_original[:,None], reduce='amax')
        sensitivity[nodes] = step + injection[nodes]
        reachability[nodes] = (reach+injection[nodes]).clamp_max(1.)
        arrival[:,nodes] = model.initial.new_zeros((1,len(nodes))).scatter_add(
            1, destinations[None], values*original[None])
    influence, reach = sensitivity[model.outputs], reachability[model.outputs]
    if not torch.isfinite(influence).all() or (influence < -1e-10).any() or (influence > 1+1e-10).any():
        raise ValueError('Invalid original root-delay sensitivity')
    if (influence > reach+1e-10).any():
        raise ValueError('Sensitivity outside original reachable cone')
    return influence.clamp(0,1), reach


class EndpointContextSurrogate(LearnedPortSurrogate):
    @classmethod
    def from_prepared(cls, base):
        """Reuse immutable prepared graph buffers and shared heads, add two buffers.

        Shallow-copy the module registries so the original graph object remains
        usable as a paired control. No checkpoint or historical cache is edited.
        """
        model = copy.copy(base)
        model.__class__ = cls
        model._buffers = base._buffers.copy()
        model._modules = base._modules.copy()
        model.metadata = dict(base.metadata)
        influence, reach = reference_influence(base)
        model.register_buffer('po_root_sensitivity', influence)
        model.register_buffer('po_root_reachability', reach)
        model.metadata.update(global_context='original_root_sensitivity_and_reachability',
                              sensitivity_shape=list(influence.shape), new_neural_parameters=0)
        return model

    def endpoint_context(self, expected):
        critical = torch.einsum('pd,bdk->bpk', self.po_root_sensitivity, expected)
        upstream = torch.einsum('pd,bdk->bpk', self.po_root_reachability, expected)
        return torch.cat([critical.tanh(),upstream.tanh()],-1)

    def propagate(self, x):
        if self.variant != 'global':
            raise ValueError('This single-change experiment supports global only')
        # Identical v3 structural timing, with original/candidate root mixing.
        expected = (x.unsqueeze(-1)*self.features[None]).sum(2)
        flat = torch.cat([x.flatten(1),x.new_zeros((len(x),1))],1)
        roots = x.new_zeros((len(x),self.root_count)).index_add(1,self.root_indices,flat[:,self.flat_indices])
        choice = self.option_choice
        weights = torch.where(choice[None]>=0,flat[:,choice.clamp_min(0)],1.-roots[:,(-1-choice).clamp_min(0)])
        costs = self.edge_prior[None].expand(len(x),-1)
        operation = conditional_arrivals if self.backend=='bounded_tape' else conditional_reference
        arrival = operation(costs,weights,self.initial,self._layers(x.device),self.temperature)[:,self.outputs]
        mean = expected.sum(1)/math.sqrt(max(self.partition.aig.and_count,1))
        weighted = (expected*self.sizes[None,:,None]).sum(1)/max(self.partition.aig.and_count,1)
        global_features = torch.cat([mean,weighted],-1)[:,None].expand(-1,len(self.outputs),-1)
        feature = torch.cat([self.po_static[None].expand(len(x),-1,-1),global_features,
                             self.endpoint_context(expected)],-1)
        dtype = next(self.shared.parameters()).dtype
        residual = self.shared.global_head(feature.to(dtype)).squeeze(-1).to(x.dtype)
        return arrival+self.delay_reference*residual

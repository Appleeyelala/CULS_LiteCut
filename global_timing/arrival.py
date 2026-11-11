"""Level-ordered smooth arrival propagation with a bounded first-order tape.

Every region is assigned once in topological order. A custom reverse pass
stores edge exponentials and region denominators instead of an autograd chain
of full arrival-vector copies. The reference uses the original Torch graph.
Only first derivatives are supported by the optimized operation.
"""
from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class ArrivalLayer:
    nodes: torch.Tensor
    edges: torch.Tensor
    destinations: torch.Tensor
    sources: torch.Tensor
    source_mask: torch.Tensor
    unique_sources: torch.Tensor
    source_destinations: torch.Tensor


def prepare_arrival_layers(edge_sources, layers):
    result = []
    for nodes, edges, destinations in layers:
        sources = edge_sources[edges]
        clamped = sources.clamp_min(0)
        unique, inverse = torch.unique(clamped, sorted=True, return_inverse=True)
        result.append(ArrivalLayer(nodes, edges, destinations, clamped, sources >= 0, unique, inverse))
    return tuple(result)


def reference_arrivals(costs, layers, regions, temperature):
    """Original differentiable implementation, retained as a numerical oracle."""
    batch = costs.shape[0]
    arrivals = costs.new_zeros((batch, regions))
    for layer in layers:
        upstream = arrivals[:, layer.sources] * layer.source_mask[None, :]
        values = upstream + costs[:, layer.edges]
        index = layer.destinations[None, :].expand(batch, -1)
        maximum = values.new_full((batch, len(layer.nodes)), -torch.inf).scatter_reduce(
            1, index, values, reduce='amax').detach()
        sums = values.new_zeros(maximum.shape).scatter_add(
            1, index, torch.exp((values - maximum.gather(1, index)) / temperature))
        stage = maximum + temperature * sums.log()
        arrivals = arrivals.index_copy(1, layer.nodes, stage)
    return arrivals


class _ArrivalPropagation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, costs, layers, regions, temperature):
        batch = costs.shape[0]
        arrivals = costs.new_zeros((batch, regions))
        exponentials = torch.empty_like(costs)
        denominators = torch.empty_like(arrivals)
        for layer in layers:
            upstream = arrivals[:, layer.sources] * layer.source_mask[None, :]
            values = upstream + costs[:, layer.edges]
            index = layer.destinations[None, :].expand(batch, -1)
            maximum = values.new_full((batch, len(layer.nodes)), -torch.inf).scatter_reduce(
                1, index, values, reduce='amax')
            exponential = torch.exp((values - maximum.gather(1, index)) / temperature)
            sums = values.new_zeros(maximum.shape).scatter_add(1, index, exponential)
            stage = maximum + temperature * sums.log()
            arrivals.index_copy_(1, layer.nodes, stage)
            exponentials.index_copy_(1, layer.edges, exponential)
            denominators.index_copy_(1, layer.nodes, sums)
        ctx.layers, ctx.temperature = layers, temperature
        ctx.save_for_backward(exponentials, denominators)
        return arrivals

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_arrivals):
        exponentials, denominators = ctx.saved_tensors
        batch = grad_arrivals.shape[0]
        adjoint = grad_arrivals.clone()
        grad_costs = torch.empty_like(exponentials)
        for layer in reversed(ctx.layers):
            # Keep the original Torch operation order, including T and 1/T,
            # instead of replacing it with an algebraically simplified weight.
            grad_sums = adjoint[:, layer.nodes] * ctx.temperature / denominators[:, layer.nodes]
            grad_values = grad_sums[:, layer.destinations] * exponentials[:, layer.edges] / ctx.temperature
            grad_costs.index_copy_(1, layer.edges, grad_values)
            # Accumulate repeated sources starting from zero, then add the
            # existing adjoint once. Adding every edge into the existing value
            # would regroup floating-point additions relative to the oracle.
            delta = grad_values.new_zeros((batch, len(layer.unique_sources)))
            index = layer.source_destinations[None, :].expand(batch, -1)
            delta.scatter_add_(1, index, grad_values * layer.source_mask[None, :])
            adjoint.index_copy_(1, layer.unique_sources, adjoint[:, layer.unique_sources] + delta)
        return grad_costs, None, None, None


def smooth_arrivals(costs, layers, regions, temperature):
    """First-order reverse mode for an already validated topological plan."""
    return _ArrivalPropagation.apply(costs, layers, regions, temperature)

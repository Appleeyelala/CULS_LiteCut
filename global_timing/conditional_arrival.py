"""Level-parallel cut-port propagation with conditional root replacements.

Each option reduces its input ports by LSE; each root mixes its original and
candidate options. The optimized tape is first-order only. No measured timing
or feasibility decisions occur in this operation.
"""
import torch
from torch.autograd.function import once_differentiable


def _stage(arrivals, costs, layer, temperature):
    nodes, options, edges, sources, groups, destinations = layer
    values = arrivals[:, sources] + costs[:, edges]
    index = groups[None].expand(values.shape[0], -1)
    maximum = values.new_full((values.shape[0], len(options)), -torch.inf)
    maximum = maximum.scatter_reduce(1, index, values, reduce='amax').detach()
    exponentials = ((values - maximum.gather(1, index)) / temperature).exp()
    sums = values.new_zeros(maximum.shape).scatter_add(1, index, exponentials)
    return maximum + temperature * sums.log(), exponentials / sums.gather(1, index)


def conditional_reference(costs, weights, initial, layers, temperature):
    arrivals = initial.expand(costs.shape[0], -1)
    for layer in layers:
        nodes, options, _, _, _, destinations = layer
        option_values, _ = _stage(arrivals, costs, layer, temperature)
        stage = costs.new_zeros((costs.shape[0], len(nodes))).scatter_add(
            1, destinations[None].expand(costs.shape[0], -1), option_values * weights[:, options])
        arrivals = arrivals.index_copy(1, nodes, stage)
    return arrivals


class _ConditionalArrival(torch.autograd.Function):
    @staticmethod
    def forward(ctx, costs, weights, initial, layers, temperature):
        arrivals = initial.expand(costs.shape[0], -1).clone()
        option_tape = torch.empty_like(weights)
        edge_tape = torch.empty_like(costs)
        for layer in layers:
            nodes, options, edges, _, _, destinations = layer
            values, derivatives = _stage(arrivals, costs, layer, temperature)
            stage = costs.new_zeros((costs.shape[0], len(nodes)))
            stage.scatter_add_(1, destinations[None].expand(costs.shape[0], -1), values * weights[:, options])
            arrivals.index_copy_(1, nodes, stage)
            option_tape.index_copy_(1, options, values)
            edge_tape.index_copy_(1, edges, derivatives)
        ctx.layers = layers
        ctx.save_for_backward(weights, option_tape, edge_tape)
        return arrivals

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        weights, values, derivatives = ctx.saved_tensors
        adjoint = gradient.clone()
        grad_costs, grad_weights = torch.empty_like(derivatives), torch.empty_like(weights)
        for nodes, options, edges, sources, groups, destinations in reversed(ctx.layers):
            grad_options = adjoint[:, nodes][:, destinations]
            grad_weights[:, options] = grad_options * values[:, options]
            grad_edges = (grad_options * weights[:, options])[:, groups] * derivatives[:, edges]
            grad_costs[:, edges] = grad_edges
            adjoint.scatter_add_(1, sources[None].expand(gradient.shape[0], -1), grad_edges)
        return grad_costs, grad_weights, None, None, None


def conditional_arrivals(costs, weights, initial, layers, temperature):
    return _ConditionalArrival.apply(costs, weights, initial, layers, temperature)

"""Trace the union of tied max-arrival paths in a hard LoadBank evaluation."""
import sys

import torch

from global_timing.liberty_cover import TimingGraph


@torch.no_grad()
def hard_timing_state(bank, vector):
    """Read the actual final pass, including its chosen cover and effective loads.

    TimingGraph's normal public result omits internal arrival times. Observe a
    completed forward frame during a diagnostic replay, after search has ended.
    The two preliminary cover-recovery passes have different arc selections.
    """
    captured = []
    code = TimingGraph.forward.__code__

    def observe(frame, event, unused):
        if event == 'return' and frame.f_code is code:
            state = frame.f_locals
            if state['self'] is bank.graph and not state['recovery']:
                captured.append(tuple(state[key] for key in ('arr', 'slew', 'loads', 'w')))

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        delay, _ = bank.forward(bank.hard(vector))
    finally:
        sys.setprofile(previous)
    if len(captured) != 1:
        raise ValueError('Expected exactly one final timing pass')
    arrivals, slew, loads, weights = captured[0]
    if not torch.equal(delay, arrivals[bank.graph.po].amax()):
        raise ValueError('Captured final pass disagrees with the hard score')
    if not torch.all((weights == 0) | (weights == 1)):
        raise ValueError('Path membership requires a discrete cover')
    return arrivals, slew, loads, weights


@torch.no_grad()
def critical_path(bank, vector):
    graph = bank.graph
    arrivals, slew, loads, weights = hard_timing_state(bank, vector)
    outputs = arrivals[graph.po]
    delay = outputs.amax()
    output_index, edge = (outputs == delay).nonzero(as_tuple=True)
    marked = torch.zeros_like(arrivals, dtype=torch.bool)
    flat = marked.view(-1)
    flat.index_fill_(0, 2 * graph.po[output_index] + edge, True)
    for stage_index in reversed(range(len(graph.stages))):
        nets, (src, dst, _), (parent, child, state), (arc, ip, op, tab), _ = graph.stages[stage_index]
        active = marked[nets]
        if not active.any():
            continue

        # A hard wire/MUX preserves its input polarity, also at scaled
        # LiteCut boundaries. Follow only the active branch of the actual cover.
        if parent is not None:
            wire, polarity = (active[child] & (weights[state, None] == 1)).nonzero(as_tuple=True)
            flat.index_fill_(0, 2 * parent[wire] + polarity, True)

        if arc is None:
            continue
        live = torch.nonzero(active[dst[arc], op], as_tuple=True)[0]
        if not live.numel():
            continue
        pin_arc = arc[live]
        source, target = src[pin_arc], nets[dst[pin_arc]]
        in_edge, out_edge = ip[live], op[live]
        if stage_index in graph.cached_stages:
            cost = graph.cached_stages[stage_index][3][live]
        else:
            cost = graph.lib.lookup(tab[live], slew[source, in_edge], loads[target, out_edge])
        value = arrivals[source, in_edge] + cost
        exact = (value == arrivals[target, out_edge])
        # The forward amax has an explicit zero floor. Every positive
        # arrival at a marked gate must have an exact driving transition.
        targets = 2 * target + out_edge
        required = torch.unique(targets[arrivals[target, out_edge] > 0])
        reached = torch.unique(targets[exact & (arrivals[target, out_edge] > 0)])
        if not torch.equal(required, reached):
            raise ValueError(f'Could not reproduce a critical gate transition at stage {stage_index}')
        flat.index_fill_(0, 2 * source[exact] + in_edge[exact], True)
    endpoints = list(zip(output_index.tolist(), edge.tolist()))
    return float(delay), endpoints, marked

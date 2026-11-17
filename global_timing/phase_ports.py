"""Literal-aware cut-port transfers for the research proxy.

The two channels mean a logical AIG literal and its complement, NOT electrical
rise/fall. Output polarity remains symbolic until the consuming edge. This is
an abstraction of AIG structural timing, never a mapped-pin correspondence.
"""
from collections import Counter
from .boolean_candidates import verify_library_truth


def merge(parts):
    result = {}
    for paths, cost in parts:
        for key, value in paths.items():
            result[key] = max(result.get(key, float('-inf')), value + cost)
    return result


def original_transfers(aig, retained, inversion_cost=.2):
    retained = set(retained) & set(aig.fanin_literals)
    uses = Counter(l >> 1 for pair in aig.fanin_literals.values() for l in pair)
    values, result = {}, {}
    for node, pair in sorted(aig.fanin_literals.items()):
        parts = []
        for lit in pair:
            source, phase = lit >> 1, lit & 1
            if source in retained or source not in aig.fanin_literals:
                parts.append(({(source, phase): 0.}, 1.))
            else:
                parts.append((values[source], 1. + inversion_cost * phase))
        paths = merge(parts)
        if node in retained:
            result[node] = (paths, {p: d + inversion_cost for p, d in paths.items()})
        values[node] = paths
        for lit in pair:
            source = lit >> 1
            uses[source] -= 1
            if uses[source] == 0: values.pop(source, None)
    return result


def candidate_transfers(decision, subgraph, library, inversion_cost=.2):
    proof = verify_library_truth(decision, subgraph, library)
    truth = int(decision['truth']) & 0xffff
    phase = library['p_phases'][truth]
    perm = library['perms4'][library['p_perms'][truth]]
    leaves = list(decision['cut_leaves'])
    if len(leaves) == 3: leaves.append(0)
    data = library['classes'][decision['truth_class']]
    values = [({(leaves[perm[k]], (phase >> k) & 1): 0.},
               {(leaves[perm[k]], ((phase >> k) & 1) ^ 1): 0.}) for k in range(4)]
    for a, b, ca, cb in data['fanins']:
        positive = merge([(values[a][ca], 1.), (values[b][cb], 1.)])
        values.append((positive, {p: d + inversion_cost for p, d in positive.items()}))
    root = data['roots'][subgraph]
    complement = int(proof['output_phase_complement'])
    return values[root][complement], values[root][complement ^ 1]


def hard_phase_arrivals(aig, decisions, choices, library, inversion_cost=.2):
    retained = {l >> 1 for l in aig.po_literals}
    for d in decisions:
        retained.add(d['root']); retained.update(d['cut_leaves'])
    original = original_transfers(aig, retained, inversion_cost)
    overrides = {}
    for d, c in zip(decisions, choices):
        if c:
            if d['root'] in overrides: raise ValueError('Conflicting root choices')
            overrides[d['root']] = candidate_transfers(d, d['candidate_subgraph_indices'][c-1], library, inversion_cost)
    arrival = {(i, phase): phase * inversion_cost for i in range(aig.pi_count+1) for phase in (0,1)}
    for node, paths in sorted(original.items()):
        for phase, terms in enumerate(overrides.get(node, paths)):
            arrival[node, phase] = max(arrival[p] + d for p,d in terms.items())
    return [arrival[l >> 1, l & 1] for l in aig.po_literals]

"""Full-global-gradient sparse joint updates; no multi-trajectory optimizer."""
import math

import torch

from .surrogate import scalar_objective


def prepare_selection_mask(model, feasibility):
    rows = [row + [False] * (model.width - len(row)) for row in feasibility.allowed()]
    allowed = torch.tensor(rows, dtype=torch.bool, device=model.features.device)
    model.register_buffer('selection_allowed', allowed & model.valid)


def gradient_blocks(model, feasibility, anchor, seen, maximum_edits=16, maximum_proposals=2):
    """Differentiate all PO timing once, then score legal joint hard proposals.

    Candidate gain is the negative directional derivative from the incumbent
    one-hot option to a replacement, including every downstream Region/PO.
    Hard projection is conservative and non-differentiable; hard scores use
    another full DAG forward, not frozen criticality or independent regions.
    """
    if maximum_edits < 1 or maximum_proposals < 1:
        raise ValueError('Positive joint update limits required.')
    feasibility.validate(anchor)
    if not anchor:
        return [], {'reason': 'No executable decisions.'}
    model.requires_grad_(False)
    values = model.one_hot(anchor).requires_grad_()
    po, area = model(values)
    objective = scalar_objective(po, area, model.delay_reference, model.area_reference)
    gradient = torch.autograd.grad(objective, values)[0]
    if not torch.isfinite(gradient).all() or not torch.isfinite(objective):
        raise ValueError('Non-finite global MFFC selection gradient.')
    indices = torch.tensor(anchor, dtype=torch.long, device=gradient.device)[:, None]
    gain = gradient.gather(1, indices) - gradient
    gain = gain.masked_fill(~model.selection_allowed, -torch.inf)
    gain.scatter_(1, indices, -torch.inf)
    best_gain, best_choice = gain.max(-1)
    count = min(len(anchor), 8 * maximum_edits)
    # Stable tie order follows the immutable catalogue, including on CUDA.
    order = torch.argsort(best_gain, descending=True, stable=True)[:count]
    records = torch.stack((order.to(torch.float64), best_choice[order].to(torch.float64), best_gain[order]), -1).detach().cpu().tolist()
    edits = [(gain, int(i), int(choice)) for i, choice, gain in records if math.isfinite(gain) and gain > 0]
    choices, budgets = [], []
    for budget in sorted({min(maximum_edits, len(anchor), b) for b in (1, 2, 4, 8, 16, maximum_edits)}):
        selection = feasibility.project(anchor, edits, budget)
        if tuple(selection) not in seen and selection not in choices:
            choices.append(selection)
            budgets.append(budget)
    anchor_value = float(objective.detach())
    proposals = []
    if choices:
        with torch.no_grad():
            outputs, areas = model(torch.stack([model.one_hot(c) for c in choices]))
            scores = scalar_objective(outputs, areas, model.delay_reference, model.area_reference)
        for selection, budget, score, arrival, mapped_area in zip(
                choices, budgets, scores.detach().cpu().tolist(), outputs.detach().cpu().tolist(), areas.detach().cpu().tolist()):
            if not math.isfinite(score) or not math.isfinite(mapped_area) or not all(math.isfinite(a) for a in arrival):
                raise ValueError('Non-finite full-proxy hard proposal score.')
            proposals.append({'choices': selection, 'source': 'mffc_global_gradient',
                              'edit_budget': budget, 'actual_edits': sum(a != b for a, b in zip(anchor, selection)),
                              'predicted_objective': score, 'anchor_objective': anchor_value,
                              'po_arrivals_ps': arrival, 'delay_ps': max(arrival), 'area': mapped_area})
    proposals.sort(key=lambda r: (r['predicted_objective'], r['actual_edits'], r['choices']))
    trace = {'schema': 'mffc_full_global_direction_v1', 'anchor': list(anchor),
             'objective': anchor_value, 'gradient_norm': float(gradient.detach().norm()),
             'candidate_gradient_entries': int(model.valid.sum()), 'transferred_edits': len(records),
             'ranked_edits': [{'decision_index': i, 'choice_index': c, 'directional_gain': g} for g, i, c in edits],
             'legal_distinct_blocks': len(proposals), 'device': str(gradient.device),
             'scope': 'Gradient of the complete fixed-topology all-PO/area proxy; no differentiation through Student, projection, rewrite or ABC.'}
    return proposals[:maximum_proposals], trace

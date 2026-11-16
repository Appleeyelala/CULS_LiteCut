"""Fixed multi-step Adam on candidate logits; no real timing feedback."""
import math

import torch

from .mffc_search import prepare_selection_mask
from .surrogate import scalar_objective


def sparse_hard(model, edits):
    """Construct the hard tensor on device, transferring at most max_edits pairs."""
    hard = model.features.new_zeros(model.features.shape[:2])
    hard[:, 0] = 1
    if edits:
        indices = torch.tensor(edits, dtype=torch.long, device=hard.device)
        hard[indices[:, 0], 0] = 0
        hard[indices[:, 0], indices[:, 1]] = 1
    return hard


def project_logits(model, feasibility, logits, maximum_edits):
    # Rank *logit margins*, never directional derivatives. Only positive margins
    # over no-op can become hard edits. Bound host transfer to 8*max_edits rows.
    masked = logits.detach().masked_fill(~model.selection_allowed, -torch.inf)
    best, choice = masked[:, 1:].max(-1)
    margins = best - masked[:, 0]
    order = torch.argsort(margins, descending=True, stable=True)[:8 * maximum_edits]
    order = order[torch.isfinite(margins[order]) & (margins[order] > 0)]
    records = torch.stack((order.to(torch.float64), (choice[order] + 1).to(torch.float64),
                           margins[order]), -1).cpu().tolist()
    vector = feasibility.project([0] * model.n_decisions,
        [(margin, int(i), int(c)) for i, c, margin in records], maximum_edits)
    edits = [(i, c) for i, c in enumerate(vector) if c]
    return vector, edits, len(records)


def ste_adam_search(model, feasibility, seed=20261117, steps=40,
                    learning_rate=0.12, maximum_edits=16):
    """Forward uses a feasible one-hot combination, backward uses softmax.

    Only logits are optimizer parameters. Keep the best proxy-scored visited
    hard combination, including original/no-op; no ABC or calibration hooks.
    """
    model.requires_grad_(False)
    prepare_selection_mask(model, feasibility)
    device = model.features.device
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = torch.nn.Parameter(torch.randn(model.features.shape[:2], dtype=torch.float64,
                                            device=device, generator=generator) * 0.01)
    with torch.no_grad():
        logits[:, 0].add_(2.0)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    initial = logits.detach().clone()
    zero = [0] * model.n_decisions
    with torch.no_grad():
        po, area = model(sparse_hard(model, []))
        anchor_error = float((po - model.baseline_po).abs().max())
        anchor_area_error = abs(float(area) - model.area_baseline)
        if anchor_error > 1e-8 or anchor_area_error > 1e-8:
            raise ValueError('Zero edits must anchor original timing and area.')
        best = {'choices': zero, 'edits': [], 'step': -1,
                'predicted_objective': float(scalar_objective(po, area, model.delay_reference, model.area_reference)),
                'delay_ps': float(po.max()), 'area': float(area), 'source': 'original_noop'}
    trace, visited, previous = [], {()}, ()
    for step in range(steps):
        temperature = 1.0 - 0.7 * step / max(steps - 1, 1)
        probabilities = torch.softmax(logits.masked_fill(~model.selection_allowed, -torch.inf) / temperature, -1)
        vector, edits, transferred = project_logits(model, feasibility, logits, maximum_edits)
        hard = sparse_hard(model, edits)
        straight_through = hard - probabilities.detach() + probabilities
        po, area = model(straight_through)
        loss = scalar_objective(po, area, model.delay_reference, model.area_reference)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite proxy loss.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise ValueError('Missing or non-finite logits gradient.')
        norm = float(torch.nn.utils.clip_grad_norm_([logits], 5.0))
        objective, delay, mapped_area = float(loss.detach()), float(po.detach().max()), float(area.detach())
        key = tuple(edits)
        visited.add(key)
        if objective < best['predicted_objective']:
            best = {'choices': vector, 'edits': edits, 'step': step, 'predicted_objective': objective,
                    'delay_ps': delay, 'area': mapped_area, 'source': 'ste_adam_proxy_best'}
        # Quantify actual updates without retaining/copying full tensors per step.
        before = logits.detach().clone()
        optimizer.step()
        update_norm = float((logits.detach() - before).norm())
        if not math.isfinite(update_norm):
            raise ValueError('Non-finite Adam update.')
        trace.append({'step': step, 'temperature': temperature, 'loss': objective,
                      'predicted_delay_ps': delay, 'predicted_area': mapped_area,
                      'logits_gradient_norm': norm, 'adam_update_norm': update_norm,
                      'hard_edits': edits, 'selection_changed': key != previous,
                      'host_candidate_rows': transferred})
        previous = key
    return best, {'schema': 'mffc_softmax_ste_adam_v1', 'seed': seed, 'steps': steps,
                  'restarts': 1, 'learning_rate': learning_rate, 'gradient_clip': 5.0,
                  'temperature': [1.0, 0.3], 'maximum_edits': maximum_edits,
                  'device': str(device), 'dtype': str(logits.dtype),
                  'optimizer_parameters': 'candidate_logits_only',
                  'proxy_parameters_frozen': all(not p.requires_grad for p in model.parameters()),
                  'anchor_po_max_error_ps': anchor_error, 'anchor_area_error': anchor_area_error,
                  'logits_total_change_norm': float((logits.detach() - initial).norm()),
                  'distinct_hard_combinations_including_noop': len(visited), 'trajectory': trace,
                  'scope': 'Proxy scores only. No measured candidate feedback or online calibration.'}

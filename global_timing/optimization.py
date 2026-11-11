"""Joint selection gradients, legal STE projection and proposal traces."""
from __future__ import annotations

import torch

from .surrogate import scalar_objective


def gradient_proposals(model, seen, seed, incumbent=None, steps=80, restarts=3,
                       mode='area_constraint', area_limit_ratio=1.05, area_weight=0.2):
    if steps <= 0 or restarts <= 0:
        raise ValueError('Positive gradient steps and restarts are required.')
    if model.n_decisions == 0:
        return [], [{'reason': 'No executable decisions; original AIG is the only feasible selection.'}]
    model.requires_grad_(False)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    proposals, trace = {}, []
    for restart in range(restarts):
        logits = torch.nn.Parameter(torch.randn(model.features.shape[:2], dtype=torch.float64, generator=generator) * 0.2)
        with torch.no_grad():
            if restart == 0:
                starting = incumbent if incumbent is not None else [0] * model.n_decisions
                logits.add_(model.one_hot(starting) * 1.0)
        optimizer = torch.optim.Adam([logits], lr=0.12)
        for step in range(steps):
            temperature = max(0.3, 1.0 - 0.7 * step / max(steps - 1, 1))
            probabilities = torch.softmax(logits.masked_fill(~model.valid, -torch.inf) / temperature, dim=-1)
            choice = model.catalogue.project([logits[i, :count].detach().tolist() for i, count in enumerate(model.counts)])
            hard = model.one_hot(choice)
            straight_through = hard - probabilities.detach() + probabilities
            po, area = model(straight_through)
            objective = scalar_objective(po, area, model.delay_reference, model.area_reference,
                                         mode, area_limit_ratio, area_weight=area_weight)
            active = 1.0 - probabilities[:, 0]
            if model.catalogue.conflicts:
                penalty = (active[model.conflict_left] * active[model.conflict_right]).mean()
            else:
                penalty = objective.new_zeros(())
            loss = objective + 0.1 * penalty
            if not torch.isfinite(loss):
                raise ValueError('Non-finite selection objective.')
            optimizer.zero_grad()
            loss.backward()
            if logits.grad is None or not torch.isfinite(logits.grad).all():
                raise ValueError('Non-finite or missing candidate-selection gradient.')
            norm = float(logits.grad.norm())
            key = tuple(choice)
            if key not in seen:
                proposals[key] = {'choices': choice, 'predicted_objective': float(objective.detach()),
                                  'predicted_delay_ps': float(po.max().detach()),
                                  'predicted_area': float(area.detach()), 'restart': restart, 'step': step}
            if step % 10 == 0 or step == steps - 1:
                trace.append({'restart': restart, 'step': step, 'temperature': temperature,
                              'objective': float(objective.detach()), 'conflict_penalty': float(penalty.detach()),
                              'gradient_norm': norm, 'gradient_by_decision': logits.grad.norm(dim=1).tolist(),
                              'gradient_values': [logits.grad[i, :count].tolist() for i, count in enumerate(model.counts)] if model.n_decisions <= 128 else None,
                              'gradient_storage': 'full_rows' if model.n_decisions <= 128 else 'norm_and_descent_choice',
                              'gradient_preferred_descent_choice': logits.grad.masked_fill(~model.valid, torch.inf).argmin(-1).tolist(),
                              'choices': choice, 'predicted_po_arrivals_ps': po.detach().tolist(),
                              'predicted_area': float(area.detach()),
                              'projection_changed_argmax': choice != logits.detach().masked_fill(~model.valid, -torch.inf).argmax(-1).tolist()})
            torch.nn.utils.clip_grad_norm_([logits], 5.0)
            optimizer.step()
    return sorted(proposals.values(), key=lambda p: (p['predicted_objective'], p['choices'])), trace

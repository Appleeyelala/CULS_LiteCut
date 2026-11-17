"""One frozen ranking-only objective for original v3; no model modifications."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def true_tie_groups(values, tolerance=.001):
    """Minimum-anchored equivalence classes matching the frozen rank protocol."""
    ordered, permutation = values.detach().sort()
    groups = torch.zeros_like(permutation)
    if len(ordered) == 0:
        return groups
    anchor = ordered[0]
    group = groups.new_zeros(())
    for i in range(1, len(ordered)):
        new_group = ordered[i]-anchor > tolerance
        group = group+new_group.long()
        anchor = torch.where(new_group, ordered[i], anchor)
        groups[i] = group
    return torch.empty_like(groups).scatter_(0, permutation, groups)


def gap_ranking_loss(predicted, target, baseline, critical):
    """Bounded-gap-weighted logistic pairs on actual global critical delays.

    The signature matches the original training loss. baseline is intentionally
    unused: no absolute PO regression and no synthetic zero-reference pair.
    Candidate predictions remain continuous, even when predicted ranks tie.
    """
    p = predicted.max(-1).values
    y = target.detach().max(-1).values
    ii, jj = torch.triu_indices(len(p), len(p), offset=1, device=p.device)
    groups = true_tie_groups(y)
    valid = groups[ii] != groups[jj]
    scale = .01*critical
    differences = (y[ii]-y[jj])[valid]
    weights = (differences.abs()/scale).clamp_max(1.)
    logits = ((p[ii]-p[jj])/scale)[valid]
    # Sum/count avoids NaN for an empty or all-tied domain, with a zero gradient.
    terms = weights*F.softplus(-differences.sign()*logits)
    return terms.sum()/max(terms.numel(), 1)

"""Isolate existing information channels without changing default behavior."""
import torch


def configure_model(model, mode):
    if mode not in ('no_score', 'no_pairs'):
        raise ValueError('Unknown model ablation.')
    with torch.no_grad():
        if mode == 'no_score':
            # Raw 19D features / Student inference are still performed and paid
            # for; only the predicted-delay delta channel is removed here.
            model.features[..., 0].zero_()
        else:
            model.interaction_weights.zero_()
            model.explicit_interactions_enabled = False
    if mode == 'no_pairs':
        # Calibration toggles requires_grad each round. A persistent zero hook
        # keeps this initially-zero coefficient zero through those optimizers.
        model._ablation_interaction_hook = model.interaction_weights.register_hook(torch.zeros_like)
    model.ablation_mode = mode
    return {'mode': mode, 'student_preparation_retained': True,
            'scope': ('Only LiteCut scalar delta removed; structural features retained.' if mode == 'no_score' else
                      'Explicit adjacent pair features removed from proxy and feedback; DAG propagation still couples endpoints.')}


def no_repair(observation):
    """Failure stays failed; the search may spend its next budget slot elsewhere."""
    return None

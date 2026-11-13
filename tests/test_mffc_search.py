"""Global direction and hard legal joint updates on analytical fixtures."""
import unittest

import torch

from global_timing.feasibility import CandidateFeasibility
from global_timing.mffc_search import gradient_blocks, prepare_selection_mask
from global_timing.surrogate import scalar_objective
from test_surrogate import model_fixture


def permissive_rule(model):
    rule = CandidateFeasibility.__new__(CandidateFeasibility)
    rule.catalogue = model.catalogue
    rule.static = [set() for _ in model.counts]
    rule.blocked = [[None] * n for n in model.counts]
    rule.references = [[set() for _ in range(n)] for n in model.counts]
    rule.forbidden = [[set() for _ in range(n)] for n in model.counts]
    return rule


class GlobalDirectionTests(unittest.TestCase):
    def test_direction_uses_full_model_and_hard_blocks_are_legal(self):
        model = model_fixture()
        rule = permissive_rule(model)
        prepare_selection_mask(model, rule)
        proposals, trace = gradient_blocks(model, rule, [0, 0], {(0, 0)}, maximum_edits=2)
        self.assertTrue(proposals)
        self.assertGreater(trace['gradient_norm'], 0)
        values = model.one_hot([0, 0]).requires_grad_()
        po, area = model(values)
        gradient = torch.autograd.grad(scalar_objective(po, area, model.delay_reference, model.area_reference), values)[0]
        for row in trace['ranked_edits']:
            i, c = row['decision_index'], row['choice_index']
            self.assertAlmostEqual(row['directional_gain'], float(gradient[i, 0] - gradient[i, c]), places=12)
        for proposal in proposals:
            rule.validate(proposal['choices'])
            po, area = model(model.one_hot(proposal['choices']))
            self.assertAlmostEqual(proposal['predicted_objective'], float(scalar_objective(
                po, area, model.delay_reference, model.area_reference)), places=12)
        self.assertTrue(any(sum(bool(c) for c in p['choices']) == 2 for p in proposals))

    def test_forbidden_candidate_mask_and_seen_points_are_respected(self):
        model = model_fixture()
        rule = permissive_rule(model)
        rule.blocked[0][2] = 'FIXTURE_FORBIDDEN'
        prepare_selection_mask(model, rule)
        proposals, _ = gradient_blocks(model, rule, [0, 0], {(0, 0)}, maximum_edits=2)
        self.assertTrue(all(p['choices'][0] != 2 for p in proposals))
        seen = {(0, 0)} | {tuple(p['choices']) for p in proposals}
        again, _ = gradient_blocks(model, rule, [0, 0], seen, maximum_edits=2)
        self.assertEqual(again, [])


if __name__ == '__main__':
    unittest.main()

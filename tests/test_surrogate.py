"""Analytical unit fixtures, not experimental timing labels."""
import unittest
import random

import torch

from global_timing.candidates import Catalogue
from global_timing.optimization import gradient_proposals
from global_timing.structure import make_partition
from global_timing.surrogate import TimingSurrogate, scalar_objective
from test_structure import fixture


def model_fixture():
    aig = fixture(3, [(2, 4), (8, 6), (8, 7), (10, 12), (14, 2)], [10, 16])
    partition = make_partition(aig)
    decisions = [{'root': root, 'cut_slot': 1, 'candidate_count': 2,
                  'candidate_subgraph_indices': [0, 1]} for root in [4, 6]]
    by_region = {i: [] for i in range(len(partition.regions))}
    for i, decision in enumerate(decisions):
        by_region[partition.owner[decision['root']]].append(i)
    catalogue = Catalogue(decisions, [], by_region)
    scores = []
    for i in range(2):
        for c in range(3):
            features = [0.] * 19
            features[8] = 3 - c * 0.4
            features[11] = 5 - c
            scores.append({'decision_index': i, 'choice_index': c, 'predicted_local_abc_delay': 20 - c * 3,
                           'continuous_features_19': features})
    return TimingSurrogate(partition, catalogue, scores, {'delay_ps': 100., 'area': 10., 'po_arrivals_ps': [60., 100.]})


class SurrogateTests(unittest.TestCase):
    def test_batched_independent_options_preserve_ordered_costs_and_ties(self):
        # A single region with many nested decision roots and differing port
        # dependencies stresses the repeated-sum case; these are unit features.
        nodes = [(2, 4)] + [(2 * (3 + i), 6 if i % 2 else 4) for i in range(1, 33)]
        partition = make_partition(fixture(3, nodes, [2 * (3 + len(nodes))]))
        decisions = [{'root': 4 + i, 'candidate_count': 3} for i in range(len(nodes))]
        catalogue = Catalogue(decisions, [], {0: list(range(len(nodes)))})
        scores = []
        rng = random.Random(73)
        for i in range(len(nodes)):
            for c in range(4):
                features = [0.] * 19
                features[8], features[11] = 1 + rng.random(), 2 + rng.random()
                scores.append({'decision_index': i, 'choice_index': c,
                               'predicted_local_abc_delay': 1 + rng.random(),
                               'continuous_features_19': features})
        model = TimingSurrogate(partition, catalogue, scores,
                                {'delay_ps': 100., 'area': 20., 'po_arrivals_ps': [100.]})
        # Include tied options; the deterministic smaller-index rule must hold.
        model.features[:, 3] = model.features[:, 2]
        choices = {i: rng.randrange(4) for i in range(len(nodes))}
        with torch.no_grad():
            for index in range(len(nodes)):
                actual = model.independent_region_options(0, choices, index)
                expected = torch.stack([model.independent_region_arrival(0, choices | {index: c})
                                        for c in range(4)])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(int(actual.argmin()), int(expected.argmin()))
        with self.assertRaises(ValueError):
            model.independent_region_options(0, {999: 1}, 0)
        with self.assertRaises(ValueError):
            model.independent_region_options(0, {0: 4}, 0)

    def test_baseline_anchor_and_multiple_endpoints(self):
        model = model_fixture()
        po, area = model(model.one_hot([0, 0]))
        torch.testing.assert_close(po, torch.tensor([60., 100.], dtype=torch.float64))
        self.assertAlmostEqual(float(area), 10.)
        altered, _ = model(model.one_hot([0, 2]))
        self.assertAlmostEqual(float(altered[0]), 60.)  # downstream branch cannot change earlier PO
        self.assertLess(float(altered[1]), 100.)

    def test_gradient_matches_finite_differences(self):
        model = model_fixture()
        x = torch.full((2, 3), 1 / 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda values: model(values), (x,), atol=1e-4, rtol=1e-3))

    def test_independent_shortcut_matches_complete_propagation(self):
        model = model_fixture()
        with torch.no_grad():
            model.interaction_weights.fill_(0.7)
            for rid, indices in model.catalogue.region_decisions.items():
                for choice in range(3):
                    local = {index: choice for index in indices}
                    full = [local.get(i, 0) for i in range(model.n_decisions)]
                    values = model.one_hot(full)
                    expected = (values.unsqueeze(-1) * model.features).sum(1).unsqueeze(0)
                    _, arrivals = model._propagate(expected)
                    torch.testing.assert_close(model.independent_region_arrival(rid, local), arrivals[rid, 0])

    def test_two_objective_modes_and_area_constraint(self):
        po = torch.tensor([80., 100.], dtype=torch.float64)
        area = torch.tensor(10., dtype=torch.float64)
        for mode in ['area_constraint', 'joint']:
            self.assertTrue(torch.isfinite(scalar_objective(po, area, 100., 10., mode)))
        allowed = scalar_objective(po, area, 100., 10., 'area_constraint')
        oversized = scalar_objective(po, area * 2, 100., 10., 'area_constraint')
        self.assertGreater(float(oversized), float(allowed))
        with self.assertRaises(ValueError):
            scalar_objective(po, area, 100., 10., temperature=0.)

    def test_joint_gradients_and_deterministic_projection(self):
        model = model_fixture()
        a, trace = gradient_proposals(model, {tuple([0, 0])}, seed=47, steps=15, restarts=2)
        b, _ = gradient_proposals(model, {tuple([0, 0])}, seed=47, steps=15, restarts=2)
        self.assertEqual(a, b)
        self.assertTrue(a)
        self.assertTrue(any(row['gradient_norm'] > 0 for row in trace))
        for row in a:
            model.catalogue.validate(row['choices'])


if __name__ == '__main__':
    unittest.main()

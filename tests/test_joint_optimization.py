"""Analytical local-model/feasibility tests; these are not timing benchmarks."""
import unittest

import torch

from global_timing.candidates import Catalogue
from global_timing.joint_optimization import (
    FrozenCriticalityObjective, JointProposalConfig, joint_proposals, project_neighborhood,
)
from global_timing.surrogate import scalar_objective
from test_surrogate import model_fixture


class JointOptimizationTests(unittest.TestCase):
    def test_local_model_value_and_gradient_match_full_surrogate_at_anchor(self):
        model = model_fixture()
        with torch.no_grad():
            model.interaction_weights.copy_(torch.tensor([0.11, -0.07, 0.03, 0.02]))
        for mode in ('area_constraint', 'joint'):
            for anchor in ([0, 0], [1, 2], [2, 1]):
                with self.subTest(mode=mode, anchor=anchor):
                    objective = FrozenCriticalityObjective(model, anchor, mode)
                    v = model.one_hot(anchor).requires_grad_()
                    po, area = model(v)
                    full = scalar_objective(po, area, model.delay_reference, model.area_reference,
                                            mode, area_weight=0.5)
                    approx = objective(v)
                    full_gradient = torch.autograd.grad(full, v)[0]
                    local_gradient = torch.autograd.grad(approx, v)[0]
                    torch.testing.assert_close(full, approx, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(full_gradient, local_gradient, rtol=2e-11, atol=1e-12)

    def test_retains_nonadditive_adjacent_region_interaction(self):
        model = model_fixture()
        with torch.no_grad():
            model.interaction_weights.fill_(0.3)
        objective = FrozenCriticalityObjective(model, [0, 0])
        self.assertGreater(objective.pair_target.numel(), 0)
        values = [objective(model.one_hot(c)) for c in ([0, 0], [1, 0], [0, 1], [1, 1])]
        cross = values[3] - values[2] - values[1] + values[0]
        self.assertGreater(abs(float(cross)), 1e-7)

    def test_batched_objective_matches_independent_values_and_gradients(self):
        model = model_fixture()
        with torch.no_grad():
            model.interaction_weights.fill_(0.1)
        objective = FrozenCriticalityObjective(model, [1, 0], 'joint')
        values = torch.stack([model.one_hot(c) for c in ([0, 0], [1, 2], [2, 1])]).requires_grad_()
        batch = objective(values)
        individual = torch.stack([objective(v) for v in values])
        torch.testing.assert_close(batch, individual, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(torch.autograd.grad(batch.sum(), values)[0],
                                   torch.autograd.grad(individual.sum(), values)[0], atol=1e-13, rtol=1e-13)

    def test_neighborhood_projection_preserves_incumbent_and_edit_budget(self):
        decisions = [{'candidate_count': 2} for _ in range(4)]
        catalogue = Catalogue(decisions, [(0, 1), (1, 2)], {0: [0, 1], 1: [2, 3]})
        anchor = [1, 0, 0, 0]
        scores = [[0, 5, 6], [0, 10, 1], [0, 4, 3], [0, 4, 3]]
        self.assertEqual(project_neighborhood(catalogue, anchor, scores, 0), anchor)
        # Decision 1 cannot silently evict unchanged incumbent decision 0.
        actual = project_neighborhood(catalogue, anchor, scores, 2)
        self.assertEqual(actual, [1, 0, 1, 1])
        catalogue.validate(actual)
        with self.assertRaises(ValueError):
            project_neighborhood(catalogue, anchor, scores, -1)
        with self.assertRaises(ValueError):
            project_neighborhood(catalogue, anchor, [[float('nan')] * 3] * 4, 2)

    def test_deactivation_can_enable_a_later_compatible_edit(self):
        catalogue = Catalogue([{'candidate_count': 1}] * 2, [(0, 1)], {0: [0, 1]})
        self.assertEqual(project_neighborhood(catalogue, [1, 0], [[10, 0], [0, 4]], 2), [0, 1])

    def test_proposals_have_finite_updates_are_legal_and_reproducible(self):
        model = model_fixture()
        config = JointProposalConfig(trajectories=8, max_changes=2)
        args = dict(model=model, seen={(0, 0)}, seed=83, incumbent=[0, 0], steps=40,
                    mode='joint', config=config)
        proposals, trace = joint_proposals(**args)
        repeated, _ = joint_proposals(**args)
        self.assertEqual(proposals, repeated)
        self.assertTrue(proposals)
        for p in proposals:
            model.catalogue.validate(p['choices'])
            self.assertLessEqual(p['actual_changes'], p['max_changes'])
            self.assertNotEqual(p['choices'], [0, 0])
        self.assertTrue(all(row[3] > 0 for row in trace[0]['sampled_values'][0]))
        self.assertNotEqual(trace[0]['sampled_values'][0], trace[0]['sampled_values'][-1])

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires Slurm GPU allocation.')
    def test_cpu_cuda_objective_gradient_and_final_selection_agree(self):
        model = model_fixture()
        with torch.no_grad():
            model.interaction_weights.fill_(0.11)
        cpu = FrozenCriticalityObjective(model, [1, 0], 'joint')
        gpu = FrozenCriticalityObjective(model, [1, 0], 'joint').cuda()
        values = torch.softmax(torch.randn((7, 2, 3), dtype=torch.float64,
                                          generator=torch.Generator().manual_seed(18)), -1).requires_grad_()
        cuda_values = values.detach().cuda().requires_grad_()
        a, b = cpu(values), gpu(cuda_values)
        torch.testing.assert_close(a, b.cpu(), rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(a.sum(), values)[0],
                                   torch.autograd.grad(b.sum(), cuda_values)[0].cpu(), rtol=1e-10, atol=1e-12)
        args = dict(model=model, seen={(0, 0)}, seed=31, incumbent=[0, 0], steps=40, mode='joint')
        a, _ = joint_proposals(**args, config=JointProposalConfig(device='cpu', trajectories=8, max_changes=2))
        b, _ = joint_proposals(**args, config=JointProposalConfig(device='cuda', trajectories=8, max_changes=2))
        self.assertEqual([p['choices'] for p in a], [p['choices'] for p in b])
        for x, y in zip(a, b):
            self.assertAlmostEqual(x['predicted_objective'], y['predicted_objective'], places=11)

    def test_configuration_rejects_bad_devices_and_invalid_limits(self):
        for config in [JointProposalConfig(device='auto'), JointProposalConfig(trajectories=0),
                       JointProposalConfig(max_changes=0), JointProposalConfig(trust_penalty=float('nan'))]:
            with self.assertRaises(ValueError):
                config.validate()

    def test_feedback_is_anchored_uses_only_pass_labels_and_reduces_training_residual(self):
        model = model_fixture()
        objective = FrozenCriticalityObjective(model, [0, 0], 'joint')
        observations = []
        for index, choice in enumerate(([0, 0], [1, 0], [0, 2])):
            amount = sum(choice)
            observations.append({'index': index, 'status': 'PASS', 'choices': list(choice),
                                 'metrics': {'po_arrivals_ps': [60.0 - amount, 100.0 - 6 * amount],
                                             'area': 10.0 - amount * 0.1}})
        observations.append({'index': 3, 'status': 'FAIL', 'choices': [2, 2],
                             'metrics': {'po_arrivals_ps': [float('nan')], 'area': float('nan')}})
        fit = objective.fit_feedback(model, observations, mode='joint')
        self.assertEqual(fit['training_observation_indices'], [0, 1, 2])
        self.assertLess(fit['training_residual_mse_after'], fit['training_residual_mse_before'])
        anchor_actual = scalar_objective(torch.tensor([60., 100.], dtype=torch.float64),
                                        torch.tensor(10., dtype=torch.float64), 100., 10., 'joint', area_weight=0.5)
        torch.testing.assert_close(objective(model.one_hot([0, 0])), anchor_actual, rtol=1e-12, atol=1e-12)
        self.assertTrue(any(abs(x) > 1e-8 for x in fit['coefficients']))
        with self.assertRaises(ValueError):
            FrozenCriticalityObjective(model, [2, 2]).fit_feedback(model, observations)

    def test_feedback_cpu_cuda_match_with_measured_history(self):
        model = model_fixture()
        history = [{'index': i, 'status': 'PASS', 'choices': c,
                    'metrics': {'po_arrivals_ps': [60., 100. - sum(c) * 6], 'area': 10. - sum(c) * 0.1}}
                   for i, c in enumerate(([0, 0], [1, 0], [0, 2]))]
        args = dict(model=model, seen={(0, 0), (1, 0), (0, 2)}, seed=31,
                    incumbent=[0, 0], steps=40, mode='joint', observations=history)
        cpu, trace = joint_proposals(**args, config=JointProposalConfig(trajectories=8, max_changes=2, feedback=True))
        self.assertEqual(trace[0]['feedback_fit']['training_observation_indices'], [0, 1, 2])
        self.assertTrue(cpu)
        if torch.cuda.is_available():
            gpu, _ = joint_proposals(**args, config=JointProposalConfig(device='cuda', trajectories=8,
                                                                      max_changes=2, feedback=True))
            self.assertEqual(cpu, gpu)


if __name__ == '__main__':
    unittest.main()

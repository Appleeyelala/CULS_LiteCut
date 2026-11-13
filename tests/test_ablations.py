"""Analytical channel/repair fixtures; no experimental timing labels."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from global_timing.ablations import configure_model, no_repair
from global_timing.joint_optimization import FrozenCriticalityObjective
from global_timing.pipeline import Experiment, SearchConfig
from test_surrogate import model_fixture


class AblationTests(unittest.TestCase):
    def test_no_score_removes_only_the_predicted_local_delta_channel(self):
        model = model_fixture()
        original = model.features.clone()
        self.assertTrue(torch.count_nonzero(original[..., 0]))
        config = configure_model(model, 'no_score')
        self.assertEqual(torch.count_nonzero(model.features[..., 0]).item(), 0)
        torch.testing.assert_close(model.features[..., 1:], original[..., 1:], rtol=0, atol=0)
        self.assertTrue(config['student_preparation_retained'])
        self.assertEqual(model.ablation_mode, 'no_score')

    def test_no_pairs_stays_disabled_through_calibration_and_feedback(self):
        model = model_fixture()
        configure_model(model, 'no_pairs')
        history = [{'index': i, 'status': 'PASS', 'choices': c, 'metrics': {
            'po_arrivals_ps': [60. - sum(c), 100. - 7 * sum(c)], 'area': 10. - .1 * sum(c)}}
            for i, c in enumerate(([0, 0], [1, 1], [2, 0]))]
        model.fit_observations(history, steps=6)
        self.assertEqual(torch.count_nonzero(model.interaction_weights).item(), 0)
        objective = FrozenCriticalityObjective(model, [0, 0])
        self.assertEqual(objective.structural_pairs, [])
        self.assertEqual(objective.pair_target.numel(), 0)
        objective.fit_feedback(model, history)
        self.assertEqual(objective.pair_target.numel(), 0)
        torch.testing.assert_close(objective.feedback_features(model.one_hot([1, 1]))[4:],
                                   torch.zeros(4, dtype=torch.float64), rtol=0, atol=0)

    def test_no_repair_retains_failure_and_budget_and_only_changes_followup(self):
        outcomes = []
        for repair_enabled in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                model = model_fixture()
                experiment = Experiment.__new__(Experiment)
                experiment.config = SearchConfig(evaluation_budget=3, warmup_evaluations=4,
                                                  calibration_steps=2, gradient_steps=2)
                experiment.output = Path(directory)
                experiment.partition, experiment.catalogue = model.partition, model.catalogue
                experiment.scores, experiment.noop = [], [0, 0]
                experiment.fresh_random = Mock(side_effect=[[1, 0], [2, 0]])
                experiment.repair_choice = Mock(side_effect=lambda row: [0, 1] if row['status'] == 'FAIL' else None)
                baseline = {'index': 0, 'choices': [0, 0], 'status': 'PASS', 'runtime_s': .1,
                            'metrics': {'delay_ps': 100., 'area': 10., 'po_arrivals_ps': [60., 100.]}}
                def evaluate(choices, method, index, prediction):
                    result = {'index': index, 'choices': choices, 'method': method,
                              'prediction_before_evaluation': prediction, 'status': 'FAIL' if index == 1 else 'PASS'}
                    if index != 1:
                        result['metrics'] = {'delay_ps': 95., 'area': 10., 'po_arrivals_ps': [60., 95.]}
                    return result
                experiment.evaluate_choice = evaluate
                with patch('global_timing.pipeline.TimingSurrogate', return_value=model):
                    result = experiment.gradient_search(baseline, 'area_constraint',
                        model_transform=lambda m: configure_model(m, 'no_score'),
                        repairer=None if repair_enabled else no_repair)
                self.assertEqual(result['evaluations_including_baseline'], 3)
                self.assertEqual(result['observations'][1]['status'], 'FAIL')
                self.assertEqual(torch.count_nonzero(model.features[..., 0]).item(), 0)
                outcomes.append(result['observations'][2])
                if not repair_enabled:
                    experiment.repair_choice.assert_not_called()
        self.assertEqual(outcomes[0]['choices'], [0, 1])
        self.assertEqual(outcomes[0]['prediction_before_evaluation']['source'], 'explicit_feasibility_repair')
        self.assertEqual(outcomes[1]['choices'], [2, 0])
        self.assertEqual(outcomes[1]['prediction_before_evaluation']['source'], 'calibration_warmup')

    def test_unknown_channel_control_is_rejected(self):
        with self.assertRaises(ValueError):
            configure_model(model_fixture(), 'unknown')


if __name__ == '__main__':
    unittest.main()

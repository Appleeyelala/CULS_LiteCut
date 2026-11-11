import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from global_timing.candidates import Catalogue
from global_timing.common import ToolError
from global_timing.pipeline import Experiment, SearchConfig


class PipelineTests(unittest.TestCase):
    def test_required_original_and_noop_failures_still_stop_search(self):
        for original_failure in [False, True]:
            with self.subTest(original_failure=original_failure), tempfile.TemporaryDirectory() as directory:
                experiment = Experiment.__new__(Experiment)
                experiment.output, experiment.aig, experiment.noop = Path(directory), Path(directory) / 'input.aig', []
                experiment.evaluator = Mock()
                if original_failure:
                    experiment.evaluator.evaluate.side_effect = ToolError('original measurement failed')
                experiment.evaluate_choice = Mock(return_value={'status': 'FAIL'})
                experiment.native_baseline = Mock()
                experiment.gradient_search = Mock()
                with self.assertRaises((ToolError, RuntimeError)):
                    experiment.run()
                experiment.native_baseline.assert_not_called()
                experiment.gradient_search.assert_not_called()

    def test_native_control_internal_errors_are_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.__new__(Experiment)
            experiment.output, experiment.aig = Path(directory), Path(directory) / 'input.aig'
            experiment.evaluator, experiment.config = Mock(), SearchConfig()
            with patch('global_timing.pipeline.run_tool', side_effect=TypeError('unit internal failure')):
                with self.assertRaises(TypeError):
                    experiment.native_baseline()
            attempt = json.loads((Path(directory) / 'native_abc/attempt.json').read_text())
            self.assertEqual(attempt['status'], 'FAIL')
            self.assertIn('TypeError', attempt['error'])
            experiment.evaluator.evaluate.assert_not_called()

    def test_invalid_search_config_rejected(self):
        for config in [SearchConfig(evaluation_budget=0), SearchConfig(gradient_steps=0),
                       SearchConfig(modes=('unknown',)), SearchConfig(area_limit_ratio=0.9)]:
            with self.assertRaises(ValueError):
                config.validate()

    def test_dynamic_rejection_repairs_only_explicitly_rejected_decisions(self):
        experiment = Experiment.__new__(Experiment)
        experiment.catalogue = Catalogue([
            {'root': 10, 'cut_slot': 2, 'candidate_count': 2},
            {'root': 20, 'cut_slot': 1, 'candidate_count': 2}], [], {0: [0], 1: [1]})
        rejected = {'choices': [1, 2], 'execution_failure': {'rejected_requests': [
            {'root': 20, 'cut_slot': 1, 'subgraph_index': 5, 'status': 'REVERTED'}]}}
        self.assertEqual(experiment.repair_choice(rejected), [1, 0])
        self.assertEqual(rejected['choices'], [1, 2])  # failed request remains immutable
        self.assertIsNone(experiment.repair_choice({'choices': [1, 2]}))


if __name__ == '__main__':
    unittest.main()

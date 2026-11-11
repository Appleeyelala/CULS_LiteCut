import unittest

from global_timing.candidates import Catalogue
from global_timing.pipeline import Experiment, SearchConfig


class PipelineTests(unittest.TestCase):
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

"""Bounded feedback/fallback control fixtures; no experimental timing labels."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from global_timing.candidates import Catalogue
from global_timing.mffc_experiment import MffcExperiment, MffcSearchConfig
from test_mffc_search import permissive_rule
from test_surrogate import model_fixture


def observation(index, choices, delay=100., failed=False, prediction=None):
    row = {'index': index, 'choices': choices, 'status': 'FAIL' if failed else 'PASS',
           'prediction_before_evaluation': prediction, 'aig': 'unit_fixture_only.aig',
           'applied_rewrites': sum(bool(c) for c in choices)}
    if failed:
        row['execution_failure'] = {'rejected_requests': [{'root': 4, 'cut_slot': 1}]}
    else:
        row['metrics'] = {'area': 10., 'delay_ps': delay, 'po_arrivals_ps': [delay * .6, delay],
                          'status': 'PASS', 'cec': 'PASS'}
    return row


class FeedbackControlTests(unittest.TestCase):
    def experiment(self, output):
        obj = MffcExperiment.__new__(MffcExperiment)
        obj.model = model_fixture()
        obj.partition, obj.catalogue = obj.model.partition, obj.model.catalogue
        for decision in obj.catalogue.decisions:
            decision['decision_owner_region'] = obj.partition.owner[decision['root']]
        obj.feasibility = permissive_rule(obj.model)
        obj.output, obj.config = Path(output), MffcSearchConfig()
        obj.noop = [0, 0]
        obj.baseline = observation(0, obj.noop)
        obj.original = {'area': 11., 'delay_ps': 100., 'po_arrivals_ps': [60., 100.]}
        obj.aig = Path('unit_fixture_original.aig')
        obj.stages, obj.native_profiles = [], []
        obj.structure_meta = {'regions': len(obj.partition.regions), 'region_edges': len(obj.partition.edges)}
        obj.backend = {'device': 'cpu', 'reason': 'Analytical control-flow fixture.'}
        return obj

    def test_failed_request_and_duplicate_repair_consume_no_free_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = self.experiment(directory)
            proposals = [
                ([{'choices': [1, 1], 'source': 'mffc_global_gradient'},
                  {'choices': [0, 1], 'source': 'mffc_global_gradient'}], {'scope': 'unit fixture'}),
                ([{'choices': [1, 0], 'source': 'mffc_global_gradient'}], {'scope': 'unit fixture'})]
            def execute(obj, choices, method, index, predictions):
                # First rejection repairs to the other already-measured point;
                # the next round must continue rather than stop on that repeat.
                return [observation(index + k, choice, delay=80. if index == 1 else 75.,
                                    failed=index == 1 and k == 0, prediction=prediction)
                        for k, (choice, prediction) in enumerate(zip(choices, predictions))]
            with patch('global_timing.mffc_experiment.gradient_blocks', side_effect=proposals) as propose, \
                    patch('global_timing.mffc_experiment.execute_many', side_effect=execute) as replay:
                result = experiment.run()
            self.assertEqual(propose.call_count, 2)
            self.assertEqual(replay.call_count, 2)
            self.assertEqual(result['requests_including_noop_failed_repair'], 4)
            self.assertEqual(result['failed_requests'], 1)
            self.assertEqual(result['repair_requests'], 0)
            self.assertEqual(result['selected']['choices'], [1, 0])
            self.assertEqual(result['selected']['metrics']['delay_ps'], 75.)

    def test_no_decisions_keeps_paid_original_separate_from_zero_choice_label(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = self.experiment(directory)
            experiment.model = None
            experiment.catalogue, experiment.noop = Catalogue([], [], {}), []
            experiment.baseline = observation(0, [])
            experiment.original = {'area': 9., 'delay_ps': 99., 'status': 'PASS', 'cec': 'PASS'}
            result = experiment.run()
            self.assertEqual(result['selected']['source'], 'original_fallback')
            self.assertIsNone(result['selected']['choices'])
            self.assertEqual(result['requests_including_noop_failed_repair'], 1)
            self.assertEqual(len(result['observations']), 1)
            self.assertEqual(result['observations'][0]['metrics']['delay_ps'], 100.)


if __name__ == '__main__':
    unittest.main()

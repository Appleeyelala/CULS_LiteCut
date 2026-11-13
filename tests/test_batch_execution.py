"""Real fixed-config execution fixtures, not gradient/QoR benchmark evidence."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

from global_timing.batch_execution import BatchedExperiment, replay_many
from global_timing.candidates import Catalogue, replay
from global_timing.evaluation import Evaluator
from global_timing.pipeline import SearchConfig
from global_timing.structure import make_partition, parse_aig


@unittest.skipUnless(os.environ.get('CULS_TEST_GPULS') and os.environ.get('CULS_TEST_OLD_CASE'),
                     'Requires validated prepared gpuls, saved c1355 catalogue and Slurm GPU.')
class BatchExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_root = Path(os.environ['CULS_TEST_RUN'])
        cls.work = cls.run_root / 'output' / ('batch-' + uuid.uuid4().hex[:8])
        cls.work.mkdir()
        cls.input = cls.run_root / 'input/design.aig'
        cls.gpuls = Path(os.environ['CULS_TEST_GPULS'])
        cls.abc = Path(os.environ['CULS_TEST_ABC'])
        cls.old_case = Path(os.environ['CULS_TEST_OLD_CASE'])
        cls.partition = make_partition(parse_aig(cls.input))
        stored = json.loads((cls.old_case / 'cut_validation/verified_catalogue.json').read_text())
        cls.catalogue = Catalogue(stored['decisions'], [tuple(p) for p in stored['conflicts']],
                                   {int(k): v for k, v in stored['region_decisions'].items()})

    def choice(self, subgraph):
        result = [0] * len(self.catalogue.decisions)
        for index, decision in enumerate(self.catalogue.decisions):
            if decision['root'] in (197, 386) and decision['cut_slot'] == 2:
                result[index] = decision['candidate_subgraph_indices'].index(subgraph) + 1
        self.assertEqual(sum(c != 0 for c in result), 2)
        self.catalogue.validate(result)
        return result

    def test_batch_mixed_acceptance_maps_only_complete_successes(self):
        # Use the real earlier dynamic-rejection configuration, not an
        # artificially invalid catalogue. The two other configurations pass.
        bad = json.loads((self.old_case / 'gradient_area_constraint/eval_0002/observation.json').read_text())['choices']
        choices = [bad, self.choice(2), [0] * len(bad)]
        output = self.work / 'mixed'
        output.mkdir()
        self.partition.write(output / 'partition')
        experiment = BatchedExperiment.__new__(BatchedExperiment)
        experiment.output, experiment.aig, experiment.gpuls = output, self.input, self.gpuls
        experiment.catalogue = self.catalogue
        experiment.config = SimpleNamespace(timeout_s=180)
        experiment.evaluator = Evaluator(self.abc, self.run_root / 'input/asap7', self.input)
        experiment.validation_workers = 2
        completions = []
        experiment.on_completion = lambda row, begin, end: completions.append((row['index'], row['status'], begin, end))
        result = experiment.evaluate_many(choices, 'fixed', 0, [{'source': 'fixed_execution_fixture'}] * 3)
        self.assertEqual([r['status'] for r in result], ['FAIL', 'PASS', 'PASS'])
        self.assertEqual([r['index'] for r in result], [0, 1, 2])
        self.assertTrue(result[0]['execution_failure']['rejected_requests'])
        self.assertFalse((output / 'fixed/eval_0000/evaluation/cec.command.json').exists())
        self.assertEqual(len(completions), 3)
        self.assertTrue(all(end >= begin for _, _, begin, end in completions))
        for index in (1, 2):
            self.assertEqual(result[index]['metrics']['cec'], 'PASS')
            self.assertEqual(result[index]['metrics']['timing_model'], 'ABC_SCL_LIBERTY')
            individual, _ = replay(self.gpuls, self.input, output / 'partition/owners.csv', self.catalogue,
                                   choices[index], output / f'individual_{index}', 180)
            self.assertEqual(Path(result[index]['aig']).read_bytes(), individual.read_bytes())
        counts = json.loads((output / 'fixed/_batch_0000/validation.json').read_text())
        self.assertEqual((counts['cec_calls'], counts['mapping_calls']), (2, 2))
        self.assertEqual(len(list((output / 'fixed/predictions').glob('*.json'))), 3)

    def test_invalid_static_batch_rejected_before_native_output(self):
        target = self.work / 'invalid_static'
        with self.assertRaises(ValueError):
            replay_many(self.gpuls, self.input, self.work / 'owners.csv', self.catalogue,
                        [[999] * len(self.catalogue.decisions)], target, [self.work / 'unused'])
        self.assertFalse(target.exists())

    def test_pipeline_callback_preserves_budget_predictions_and_request_identity(self):
        output = self.work / 'pipeline'
        experiment = BatchedExperiment(self.input, self.gpuls, self.abc, self.run_root / 'input/asap7',
            self.run_root / 'input/student_checkpoint', output,
            SearchConfig(evaluation_budget=3, warmup_evaluations=0, calibration_steps=2, gradient_steps=2))
        baseline = experiment.evaluate_choice(experiment.noop, 'replay_noop', 0)
        self.assertEqual(baseline['status'], 'PASS')
        def fixed_proposals(model, seen, seed, **kwargs):
            return [{'choices': self.choice(subgraph), 'source': 'integration_fixed_configuration'}
                    for subgraph in (0, 2)], [{'kind': 'fixed_execution_fixture_not_gradients'}]
        search = experiment.gradient_search(baseline, 'area_constraint', proposal_generator=fixed_proposals,
                                             choice_batch_evaluator=experiment.evaluate_many)
        self.assertEqual(search['evaluations_including_baseline'], 3)
        self.assertTrue(all(r['status'] == 'PASS' for r in search['observations']))
        self.assertEqual(search['gradient_proposal_evaluations'], 0)
        for row in search['observations'][1:]:
            self.assertEqual(row['prediction_before_evaluation']['training_observation_indices'], [0])
            self.assertEqual(row['prediction_before_evaluation']['source'], 'integration_fixed_configuration')
        self.assertTrue((output / 'gradient_area_constraint/_batch_0001/batch_replay.json').is_file())


if __name__ == '__main__':
    unittest.main()

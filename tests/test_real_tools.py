"""Real c1355 rewrite, CEC and Liberty regression; no mocked EDA tools."""
import json
import os
from pathlib import Path
import unittest
import uuid

from global_timing.common import legacy, read_jsonl, run_tool
from global_timing.evaluation import Evaluator
from global_timing.structure import make_partition, parse_aig


@unittest.skipUnless(os.environ.get('CULS_TEST_GPULS'), 'Requires freshly built gpuls/ABC and staged c1355 inputs in Slurm.')
class RealToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_root = Path(os.environ['CULS_TEST_RUN'])
        cls.work = cls.run_root / 'output' / ('integration-' + uuid.uuid4().hex[:8])
        cls.work.mkdir()
        cls.input = cls.run_root / 'input/design.aig'
        cls.gpuls = Path(os.environ['CULS_TEST_GPULS'])
        cls.abc = Path(os.environ['CULS_TEST_ABC'])
        cls.partition = make_partition(parse_aig(cls.input))
        cls.partition.write(cls.work / 'partition')
        cls.evaluator = Evaluator(cls.abc, cls.run_root / 'input/asap7', cls.input)

    def test_c1355_partition_matches_historical_membership(self):
        from global_timing.common import ROOT
        stored = json.loads((ROOT / 'evidence/gate3a/c1355_orig/maximal_standard_mffcs.json').read_text())
        actual = {r['root']: r['internal_nodes'] for r in self.partition.regions}
        expected = {r['root']: r['internal_nodes'] for r in stored['regions']}
        self.assertEqual(actual, expected)
        self.assertEqual(len(self.partition.edges), 128)

    def test_empty_replay_and_four_cross_region_choices(self):
        config = self.work / 'choices.txt'
        config.write_text('NONE -1 -1 -1\n' + ''.join(
            f'{name} 197 2 {a}\n{name} 386 2 {b}\n'
            for name, a, b in [('GG', 0, 0), ('GL', 0, 2), ('LG', 2, 0), ('LL', 2, 2)]))
        replay = self.work / 'replay'
        run_tool([self.gpuls, '-c', f'read {self.input}; gate3e_execution_replay -o {replay} '
                  f'--region-map {self.work / "partition/owners.csv"} --config {config}'], self.work, 'replay', 120)
        rows = read_jsonl(replay / 'traces/NONE.jsonl')
        self.assertTrue(rows)
        self.assertEqual(sum(bool(row['preselected']) for row in rows), 0)
        self.assertEqual(sum(row['final_status'] == 'ACCEPTED' for row in rows), 0)
        before = self.evaluator.evaluate(self.input, self.work / 'original')
        after = self.evaluator.evaluate(replay / 'aigs/NONE.aig', self.work / 'none')
        # Serialization can reorder the graph. Record timing without asserting
        # that equivalent serializations force identical mapper choices.
        self.assertEqual(after['cec'], 'PASS')
        self.assertEqual(parse_aig(replay / 'aigs/NONE.aig').and_count, 512)
        self.assertEqual(len(before['po_arrivals_ps']), 32)
        for name, a, b in [('GG', 0, 0), ('GL', 0, 2), ('LG', 2, 0), ('LL', 2, 2)]:
            trace = read_jsonl(replay / 'traces' / (name + '.jsonl'))
            accepted = {(row['root'], row['cut_slot'], row['selected_subgraph']) for row in trace if row['final_status'] == 'ACCEPTED'}
            self.assertEqual(accepted, {(197, 2, a), (386, 2, b)})
            metrics = self.evaluator.evaluate(replay / 'aigs' / (name + '.aig'), self.work / ('timing_' + name))
            self.assertEqual(metrics['cec'], 'PASS')
            self.assertEqual(len(metrics['po_arrivals_ps']), 32)
            self.assertAlmostEqual(max(metrics['po_arrivals_ps']), metrics['delay_ps'], places=4)

    def test_candidate_enumeration_regression(self):
        raw = self.work / 'candidates'
        run_tool([self.gpuls, '-c', f'read {self.input}; gate3c_rewrite_decisions_dump -o {raw}'], self.work, 'enumerate', 120)
        records = read_jsonl(raw / 'gpu_rewrite_cuts.jsonl')
        self.assertEqual(len(records), 1815)
        self.assertEqual(sum(row['candidate_count'] > 0 for row in records), 91)
        self.assertEqual(sum(row['candidate_count'] for row in records), 326)

    def test_litecut_batch_cache_and_all_candidate_functions(self):
        from global_timing.common import ROOT
        from global_timing.candidates import catalogue_from_raw
        from global_timing.litecut import score_catalogue
        raw = self.work / 'litecut_raw'
        run_tool([self.gpuls, '-c', f'read {self.input}; gate3c_rewrite_decisions_dump -o {raw}'], self.work, 'litecut_enumerate', 120)
        catalogue = catalogue_from_raw(read_jsonl(raw / 'gpu_rewrite_cuts.jsonl'), self.partition)
        checkpoint = self.run_root / 'input/student_checkpoint'
        rows, metadata = score_catalogue(self.input, self.partition, catalogue, checkpoint, self.work / 'cache')
        cached, hit = score_catalogue(self.input, self.partition, catalogue, checkpoint, self.work / 'cache')
        self.assertEqual(rows, cached)
        self.assertFalse(metadata['cache_hit'])
        self.assertTrue(hit['cache_hit'])
        self.assertEqual(len(rows), 326 + 91)
        original = legacy('gate3e_run').LiteCutPredictor(ROOT / 'projects/litecut', self.input, checkpoint,
                                                        ROOT / 'projects/culs/src/algorithms/rewrite_library.inc')
        for row in rows:
            if row['subgraph_index'] is None:
                continue
            self.assertTrue(row['truth_verified'])
            reference = original.score(catalogue.decisions[row['decision_index']], row['subgraph_index'])
            for a, b in zip(row['continuous_features_19'], reference['continuous_features_19']):
                self.assertAlmostEqual(a, b, delta=1e-5 * max(1, abs(b)))
            self.assertAlmostEqual(row['predicted_local_abc_delay'], reference['predicted_local_abc_delay'],
                                   delta=3e-5 * max(1, abs(reference['predicted_local_abc_delay'])))


if __name__ == '__main__':
    unittest.main()

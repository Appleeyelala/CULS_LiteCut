"""Real c1355 rewrite, CEC and Liberty regression; no mocked EDA tools."""
import json
import os
from pathlib import Path
import unittest
import uuid

from global_timing.common import ToolError, legacy, read_jsonl, run_tool
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

    def test_real_cec_rejects_inverted_output(self):
        header, first, rest = self.input.read_bytes().split(b'\n', 2)
        altered = self.work / 'inequivalent.aig'
        altered.write_bytes(header + b'\n' + str(int(first) ^ 1).encode() + b'\n' + rest)
        work = self.work / 'negative_cec'
        with self.assertRaises(ToolError):
            self.evaluator.evaluate(altered, work)
        record = json.loads((work / 'result.json').read_text())
        self.assertEqual(record['status'], 'FAIL')
        self.assertFalse((work / 'mapping.log').exists())

    def test_real_abc_mapping_failure_is_not_measurement(self):
        invalid_library = self.work / 'invalid.lib'
        invalid_library.write_text('not a Liberty library\n')
        work = self.work / 'negative_mapping'
        evaluator = Evaluator(self.abc, invalid_library, self.input)
        with self.assertRaises(ToolError):
            evaluator.evaluate(self.input, work)
        record = json.loads((work / 'result.json').read_text())
        self.assertEqual(record['status'], 'FAIL')
        self.assertEqual(record['cec'], 'PASS')
        self.assertNotIn('delay_ps', record)

    def test_named_io_is_checked_in_preserved_aiger_order(self):
        binary = b'aig 5 3 0 1 2\n10\n' + bytes([4, 2, 2, 2])
        named, unnamed = self.work / 'named.aig', self.work / 'unnamed.aig'
        named.write_bytes(binary + b'i0 alpha\ni1 beta\ni2 gamma\no0 result\nc\nunit fixture\n')
        unnamed.write_bytes(binary)
        evaluator = Evaluator(self.abc, self.run_root / 'input/asap7', named)
        result = evaluator.evaluate(unnamed, self.work / 'named_interface')
        self.assertEqual(result['cec'], 'PASS')
        self.assertIn('POSITIONAL_ORDER', result['cec_interface'])

    def test_replay_preserves_constant_literals_and_inputs(self):
        # Actual binary AIG: constant outputs, constant AND fanins and live logic.
        fanins = [(0, 2), (1, 4), (2, 4), (8, 10), (12, 6)]
        content = bytearray(b'aig 8 3 0 5 5\n0\n1\n8\n10\n16\n')
        for node, pair in enumerate(fanins, 4):
            low, high = sorted(pair)
            for value in [node * 2 - high, high - low]:
                while value >= 128:
                    content.append((value & 127) | 128)
                    value >>= 7
                content.append(value)
        aig = self.work / 'constants.aig'
        aig.write_bytes(content)
        partition = make_partition(parse_aig(aig))
        partition.write(self.work / 'constant_partition')
        config = self.work / 'constant_choices.txt'
        config.write_text('NONE -1 -1 -1\nAGAIN -1 -1 -1\n')
        raw = self.work / 'constant_replay'
        run_tool([self.gpuls, '-c', f'read {aig}; gate3e_execution_replay -o {raw} '
                  f'--region-map {self.work / "constant_partition/owners.csv"} --config {config}'],
                 self.work, 'constant_replay', 120)
        evaluator = Evaluator(self.abc, self.run_root / 'input/asap7', aig)
        for name in ['NONE', 'AGAIN']:
            result = raw / 'aigs' / (name + '.aig')
            self.assertEqual(parse_aig(result).po_literals[:2], (0, 1))
            self.assertEqual(evaluator.evaluate(result, self.work / ('constant_' + name))['cec'], 'PASS')

    def test_context_cut_cec_accepts_relations_and_rejects_wrong_function(self):
        from global_timing.candidates import Catalogue
        from global_timing.cut_validation import validate_catalogue
        from test_structure import fixture
        # u=a&b, v=c&d. a&u&v equals u&v only when original leaf relations hold.
        aig = fixture(4, [(2, 4), (6, 8), (10, 12), (2, 14)], [16])
        partition = make_partition(aig)
        decisions = [dict(root=8, cut_slot=i, cut_leaves=[1, 2, 5, 6], truth=truth,
                          candidate_count=1, candidate_subgraph_indices=[0])
                     for i, truth in enumerate([0xf000, 0xffff])]
        catalogue = Catalogue(decisions, [(0, 1)], {0: [0, 1]})
        verified, audit = validate_catalogue(self.abc, partition, catalogue, self.work / 'context_cut_audit')
        self.assertEqual(len(verified.decisions), 1)
        self.assertEqual(verified.decisions[0]['cut_slot'], 0)
        self.assertEqual(audit['entries'][0]['status'], 'PASS')
        self.assertEqual(audit['entries'][1]['status'], 'FAIL')

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
            self.assertTrue(row['original_cut_truth_verified'])
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

"""Opt-in command-local enum reuse; real installation and negative regressions."""
import json
import os
from pathlib import Path
import unittest
import uuid

from global_timing.common import ToolError, read_jsonl, run_tool
from global_timing.evaluation import Evaluator
from global_timing.structure import make_partition, parse_aig


@unittest.skipUnless(os.environ.get('CULS_TEST_GPULS'), 'Requires new gpuls and Slurm GPU allocation.')
class PreparedReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run = Path(os.environ['CULS_TEST_RUN'])
        cls.work = cls.run / 'output' / ('prepared-replay-' + uuid.uuid4().hex[:8])
        cls.work.mkdir()
        cls.aig = cls.run / 'input/design.aig'
        cls.gpuls = Path(os.environ['CULS_TEST_GPULS'])
        partition = make_partition(parse_aig(cls.aig))
        partition.write(cls.work / 'partition')
        cls.evaluator = Evaluator(Path(os.environ['CULS_TEST_ABC']), cls.run / 'input/asap7', cls.aig)

    def replay(self, config, label, prepared, gpuls=None):
        raw = self.work / label
        env = dict(os.environ)
        env.pop('CULS_REPLAY_PREPARED', None)
        if prepared:
            env['CULS_REPLAY_PREPARED'] = '1'
        command = (f'read {self.aig}; gate3e_execution_replay -o {raw} '
                   f'--region-map {self.work / "partition/owners.csv"} --config {config}')
        run_tool([gpuls or self.gpuls, '-c', command], self.work, label, 180, env)
        return raw

    def test_none_and_cross_region_configs_match_default_and_frozen_binary(self):
        config = self.work / 'cross_choices.txt'
        config.write_text('NONE -1 -1 -1\n' + ''.join(
            f'{name} 197 2 {a}\n{name} 386 2 {b}\n'
            for name, a, b in [('GG', 0, 0), ('GL', 0, 2), ('LG', 2, 0), ('LL', 2, 2)]))
        plain = self.replay(config, 'plain', False)
        prepared = self.replay(config, 'prepared', True)
        legacy = None
        if os.environ.get('CULS_TEST_LEGACY_GPULS'):
            legacy = self.replay(config, 'frozen', False, Path(os.environ['CULS_TEST_LEGACY_GPULS']))
        marker = json.loads((prepared / 'prepared_replay.json').read_text())
        self.assertEqual(marker['enumeration_count'], 1)
        self.assertEqual(marker['configuration_count'], 5)
        self.assertFalse((plain / 'prepared_replay.json').exists())
        for name in ['NONE', 'GG', 'GL', 'LG', 'LL']:
            expected_aig = (plain / 'aigs' / f'{name}.aig').read_bytes()
            expected_trace = read_jsonl(plain / 'traces' / f'{name}.jsonl')
            self.assertEqual(expected_aig, (prepared / 'aigs' / f'{name}.aig').read_bytes())
            self.assertEqual(expected_trace, read_jsonl(prepared / 'traces' / f'{name}.jsonl'))
            if legacy is not None:
                self.assertEqual(expected_aig, (legacy / 'aigs' / f'{name}.aig').read_bytes())
                self.assertEqual(expected_trace, read_jsonl(legacy / 'traces' / f'{name}.jsonl'))
            metrics = self.evaluator.evaluate(prepared / 'aigs' / f'{name}.aig', self.work / ('timing_' + name))
            self.assertEqual(metrics['cec'], 'PASS')
            self.assertEqual(metrics['timing_model'], 'ABC_SCL_LIBERTY')

    def test_invalid_candidate_still_rejected(self):
        config = self.work / 'invalid_choices.txt'
        config.write_text('BAD 197 2 999\n')
        for mode in (False, True):
            with self.assertRaises(ToolError):
                self.replay(config, 'invalid_' + str(mode), mode)

    @unittest.skipUnless(os.environ.get('CULS_TEST_OLD_CASE'), 'Requires preserved dynamic-rejection fixture.')
    def test_dynamic_rejection_is_unchanged_and_not_treated_as_success(self):
        source = Path(os.environ['CULS_TEST_OLD_CASE']) / 'gradient_area_constraint/eval_0002/rewrite/choices.txt'
        config = self.work / 'dynamic_choices.txt'
        config.write_bytes(source.read_bytes())
        plain = self.replay(config, 'dynamic_plain', False)
        prepared = self.replay(config, 'dynamic_prepared', True)
        before = read_jsonl(plain / 'traces/choice.jsonl')
        after = read_jsonl(prepared / 'traces/choice.jsonl')
        self.assertEqual(before, after)
        rejected = [r for r in after if r['preselected'] and r['final_status'] != 'ACCEPTED']
        self.assertTrue(rejected)
        # Partial output is deliberately not sent to evaluator as a successful
        # requested combination; callers must retain exact-request acceptance.


if __name__ == '__main__':
    unittest.main()

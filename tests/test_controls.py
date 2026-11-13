"""Natural-cost baseline contracts; analytical fixtures, not QoR evidence."""
import ast
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from global_timing.boolean_candidates import parse_library
from global_timing.candidates import Catalogue
from global_timing.common import ROOT, ToolError
from global_timing.controls import CandidateControl, ControlConfig, native_control


class ControlTests(unittest.TestCase):
    def test_random_execution_imports_no_unused_ml_runtime(self):
        code = ('import sys; import global_timing.controls,global_timing.execution; '
                'assert not set(("torch","numpy","networkx")) & set(sys.modules)')
        subprocess.run([sys.executable, '-B', '-c', code], check=True,
                       cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))

    def test_library_parser_matches_preserved_canonical_function_exactly(self):
        source = (ROOT / 'projects/litecut/gate2a_candidate_inference.py').read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'parse_library')
        namespace = {}
        exec('from pathlib import Path\nfrom typing import Any\nimport re\n' + ast.get_source_segment(source, node), namespace)
        library = ROOT / 'projects/culs/src/algorithms/rewrite_library.inc'
        self.assertEqual(parse_library(library), namespace['parse_library'](library))

    def fixture(self, output, budget=8):
        item = CandidateControl.__new__(CandidateControl)
        item.config, item.output = ControlConfig(evaluation_budget=budget), Path(output)
        item.catalogue = Catalogue([{'candidate_count': 2} for _ in range(4)], [(0, 1)], {0: [0, 1], 1: [2, 3]})
        item.noop, item.scores, item.on_completion = [0] * 4, None, None
        item.repair_choice = Mock(return_value=None)
        item.evaluate_choice = Mock(return_value={'index': 0, 'choices': item.noop, 'status': 'PASS',
                                                  'metrics': {'delay_ps': 100., 'area': 100.}})
        return item

    def test_sampler_preserves_frozen_variable_density_choice_sequence(self):
        from global_timing.pipeline import Experiment
        with tempfile.TemporaryDirectory() as work:
            item = self.fixture(work)
            left, right = random.Random(17), random.Random(17)
            for _ in range(40):
                self.assertEqual(item.random_choice(left), Experiment.random_choice(item, right))

    def test_random_batch_preserves_budget_identity_sources_and_area_fallback(self):
        with tempfile.TemporaryDirectory() as work:
            item = self.fixture(work)
            calls = []
            def evaluate(vectors, method, first, predictions):
                calls.append(len(vectors))
                return [{'index': first + i, 'choices': vector, 'status': 'PASS',
                         'prediction_before_evaluation': predictions[i],
                         'metrics': {'delay_ps': 50., 'area': 120.}} for i, vector in enumerate(vectors)]
            item.evaluate_many = evaluate
            result = item.run()
            self.assertEqual(calls, [1, 1, 4, 1])
            self.assertEqual(result['evaluations_including_baseline'], 8)
            self.assertEqual([r['index'] for r in result['observations']], list(range(8)))
            self.assertEqual(len({tuple(r['choices']) for r in result['observations']}), 8)
            self.assertEqual(result['best']['choices'], item.noop)  # lower delay exceeds hard area limit
            self.assertEqual(result['gradient_proposal_evaluations'], 0)
            self.assertTrue(all(r['prediction_before_evaluation']['source'] == 'random' for r in result['observations'][1:]))

    def test_native_uses_no_candidate_preparation_and_retains_failure(self):
        with tempfile.TemporaryDirectory() as work:
            evaluator = Mock()
            evaluator.evaluate.side_effect = ToolError('controlled CEC failure')
            with patch('global_timing.controls.Evaluator', return_value=evaluator), \
                 patch('global_timing.controls.enumerate_candidates', side_effect=AssertionError('unused')), \
                 patch('global_timing.controls.run_tool') as tool:
                row = native_control('original', Path(work) / 'input.aig', Path('gpuls'), Path('abc'),
                                     Path('lib'), Path(work) / 'original')
            self.assertEqual(row['status'], 'FAIL')
            self.assertNotIn('metrics', row)
            tool.assert_not_called()
            self.assertEqual(json.loads((Path(work) / 'original/observation.json').read_text())['status'], 'FAIL')


@unittest.skipUnless(os.environ.get('CULS_TEST_GPULS'), 'Requires Slurm GPU and audited real tools.')
class RealControlTests(unittest.TestCase):
    def test_real_random_greedy_and_native_controls_have_complete_cec_and_liberty(self):
        import uuid
        run = Path(os.environ['CULS_TEST_RUN'])
        work = run / 'output' / ('controls-' + uuid.uuid4().hex[:8])
        work.mkdir()
        aig, library = run / 'input/design.aig', run / 'input/asap7'
        gpuls, abc = Path(os.environ['CULS_TEST_GPULS']), Path(os.environ['CULS_TEST_ABC'])
        for method in ('original', 'native_abc', 'native_culs'):
            row = native_control(method, aig, gpuls, abc, library, work / method)
            self.assertEqual(row['status'], 'PASS', row.get('error'))
            self.assertEqual(row['metrics']['cec'], 'PASS')
            self.assertEqual(row['metrics']['timing_model'], 'ABC_SCL_LIBERTY')
        for use_litecut in (False, True):
            output = work / ('greedy' if use_litecut else 'random')
            item = CandidateControl(aig, gpuls, abc, library, run / 'input/student_checkpoint', output,
                                    ControlConfig(evaluation_budget=5), use_litecut)
            result = item.run('greedy_litecut' if use_litecut else 'random_search')
            best = result['best']
            self.assertEqual(best['status'], 'PASS')
            self.assertEqual(best['metrics']['cec'], 'PASS')
            self.assertLessEqual(best['metrics']['area'], result['observations'][0]['metrics']['area'] * 1.05 + 1e-8)
            self.assertLessEqual(len(result['observations']), 5)
            if not use_litecut:
                self.assertFalse((output / 'litecut.json').exists())
                proof = json.loads((output / 'candidate_truth.json').read_text())
                self.assertEqual(len(proof['proofs']), sum(d['candidate_count'] for d in item.catalogue.decisions))
                self.assertEqual(result['evaluations_including_baseline'], 5)


if __name__ == '__main__':
    unittest.main()

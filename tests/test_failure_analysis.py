"""Failure-report fixtures; no simulated benchmark measurements are produced."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from global_timing.failure_analysis import describe_failure


def fixture(root, status='FAIL'):
    suite = root / 'run/output/suite'
    case = suite / 'case_0000'
    case.mkdir(parents=True)
    log = root / 'run/logs/case_0000.log'
    log.parent.mkdir()
    log.write_text('TEST FIXTURE: process stopped\n')
    (case / 'config.json').write_text(json.dumps({'input_sha256': '1' * 64}))
    record = {'index': 0, 'status': status, 'sha256': '1' * 64, 'log': str(log)}
    return suite, case, record


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FailureAnalysisTests(unittest.TestCase):
    def test_whole_case_timeout_keeps_request_failures_and_partial_time_accounting_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite, case, record = fixture(root)
            record.update(timed_out=True, returncode=-9)
            command = {'elapsed_s': 3., 'returncode': -9, 'timed_out': True, 'timeout_s': 3}
            write(case / 'random_search/eval_0001/evaluation/cec.command.json', command)
            write(case / 'original/cec.command.json', {'elapsed_s': 2., 'returncode': 0})
            write(case / 'random_search/eval_0001/evaluation/result.json',
                  {'status': 'FAIL', 'error': 'fixture timeout; no equivalence result'})
            interrupted = case / 'unfinished/mapping.log'
            interrupted.parent.mkdir()
            interrupted.write_text('fixture: no command completion record')
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = describe_failure(record, suite)
            self.assertEqual(result['categories'], ['CASE_TIMEOUT'])
            self.assertEqual(result['completed_tool_counts'], {'cec': 2})
            self.assertEqual(result['completed_tool_elapsed_s'], {'cec': 5.})
            self.assertEqual(len(result['failed_tool_commands']), 1)
            self.assertEqual(len(result['failed_evaluations']), 1)
            self.assertNotIn('delay_ps', result)
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_partial_native_failure_is_explicit_and_never_gets_substitute_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite, case, record = fixture(Path(tmp), 'PARTIAL')
            summary = {'original': {'input_sha256': '1' * 64}, 'native_abc': None,
                       'native_abc_attempt': {'status': 'FAIL', 'stage': 'cec', 'error': 'fixture timeout'},
                       'greedy_litecut': {'best': {'status': 'PASS'}},
                       'independent_regions': {'best': {'status': 'PASS'}}}
            path = case / 'summary.json'
            write(path, summary)
            record.update(summary=str(path), failed_methods=['native_abc'])
            result = describe_failure(record, suite)
            self.assertEqual(result['failed_methods'], ['native_abc'])
            self.assertEqual(result['native_abc_attempt']['stage'], 'cec')
            self.assertEqual(result['categories'], ['METHOD_COMPARISON_INCOMPLETE'])
            record['failed_methods'] = []
            with self.assertRaisesRegex(ValueError, 'failed methods'):
                describe_failure(record, suite)
            record['failed_methods'] = ['native_abc']
            summary['native_abc'] = {'status': 'PASS', 'delay_ps': 1.}
            write(path, summary)
            with self.assertRaisesRegex(ValueError, 'no timing metrics'):
                describe_failure(record, suite)

    def test_report_failure_does_not_mark_verified_methods_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite, case, record = fixture(Path(tmp), 'PARTIAL')
            path = case / 'summary.json'
            write(path, {'original': {'input_sha256': '1' * 64}, 'native_abc': {'status': 'PASS'},
                         'greedy_litecut': {'best': {'status': 'PASS'}},
                         'independent_regions': {'best': {'status': 'PASS'}}})
            record.update(summary=str(path), failed_methods=[], report_status='FAIL', report_error='fixture plot error')
            result = describe_failure(record, suite)
            self.assertEqual(result['categories'], ['REPORT_FAILURE'])
            self.assertEqual(result['failed_methods'], [])
            self.assertEqual(result['report_error'], 'fixture plot error')

    def test_identity_path_escape_and_nonfinite_command_times_are_audit_errors(self):
        for error in ['identity', 'log_path', 'summary_path', 'nan_time', 'missing_log']:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                suite, case, record = fixture(root)
                if error == 'identity':
                    record['sha256'] = '2' * 64
                elif error == 'log_path':
                    record['log'] = str(case / 'config.json')
                elif error == 'summary_path':
                    record['summary'] = str(root / 'summary.json')
                elif error == 'nan_time':
                    write(case / 'cec.command.json', {'elapsed_s': float('nan'), 'returncode': 0})
                else:
                    record['log'] = str(root / 'run/logs/missing.log')
                with self.assertRaises((ValueError, FileNotFoundError)):
                    describe_failure(record, suite)

    def test_tail_retains_original_hash_and_missing_evidence_is_not_a_running_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite, _, record = fixture(Path(tmp))
            log = Path(record['log'])
            log.write_text('x' * 10000 + '\nValueError: fixture failure\n')
            result = describe_failure(record, suite)
            self.assertTrue(result['case_log_tail_truncated'])
            self.assertEqual(result['artifacts'][0]['sha256'], hashlib.sha256(log.read_bytes()).hexdigest())
            self.assertLessEqual(len(result['case_log_tail']), 6000)
            self.assertTrue(result['case_log_tail'].endswith('ValueError: fixture failure'))
            self.assertEqual(result['categories'], ['CASE_PROCESS_FAILURE'])
            self.assertEqual(result['completed_tool_counts'], {})

    def test_skip_stays_skipped_and_live_or_successful_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / 'run/output/suite'
            record = {'index': 0, 'status': 'SKIP', 'reason': 'fixture unsupported input'}
            result = describe_failure(record, suite)
            self.assertEqual(result['categories'], ['DECLARED_SKIP'])
            self.assertEqual(result['artifacts'], [])
            for status in ['RUNNING', 'PASS', 'NOT_RECORDED']:
                with self.subTest(status=status), self.assertRaises(ValueError):
                    describe_failure(dict(record, status=status), suite)


if __name__ == '__main__':
    unittest.main()

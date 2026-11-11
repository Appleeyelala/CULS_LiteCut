"""Cohort policy/provenance fixtures; these do not create experiment measurements."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from global_timing.cohort import (analysis_sources, build_cohort, complete_records,
                                  inventory_rows, read_cohort, select_attempts)


def inventory():
    return {'rows': [{'index': index, 'relative_path': f'fixture_{index}.aig',
                      'sha256': str(index + 1) * 64, 'classification': 'COMBINATIONAL'}
                     for index in range(2)]}


def metadata(indices=None):
    return {'source_commit': 'a' * 40,
            'benchmark_arguments': {'indices': indices, 'shard_count': 1, 'shard_index': 0,
                                    'evaluations': 8, 'warmup': 2, 'gradient_steps': 20, 'calibration_steps': 30},
            'binaries': {'gpuls': {'sha256': '3' * 64}, 'abc': {'sha256': '4' * 64}},
            'assets': [{'name': 'asap7', 'sha256': '5' * 64},
                       {'name': 'student_checkpoint', 'sha256': '6' * 64}]}


def row(index, status='PASS', **extra):
    return dict(inventory()['rows'][index], status=status, **extra)


def source(records, indices=None):
    return {'metadata': metadata(indices), 'records': records}


def disk_fixture(root):
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps(inventory()))
    run = root / 'run'
    suite = run / 'output/suite'
    case = suite / 'case_0000'
    case.mkdir(parents=True)
    meta = metadata()
    meta['benchmark_arguments']['manifest'] = str(manifest)
    meta['benchmark_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (run / 'run.json').write_text(json.dumps(meta))
    parameters = {'evaluation_budget': 8, 'warmup_evaluations': 2, 'gradient_steps': 20, 'calibration_steps': 30}
    config = dict(parameters, input_sha256='1' * 64, gpuls_sha256='3' * 64, checkpoint_sha256='6' * 64,
                  toolchain={'abc': {'sha256': '4' * 64}, 'library': {'sha256': '5' * 64}})
    (case / 'config.json').write_text(json.dumps(config))
    summary = {'original': {'input_sha256': '1' * 64}, 'config': parameters}
    (case / 'summary.json').write_text(json.dumps(summary))
    record = row(0, summary=str(case / 'summary.json'))
    (suite / 'records.jsonl').write_text(json.dumps(record) + '\n')
    return manifest, run, suite, case


class CohortTests(unittest.TestCase):
    def test_latest_terminal_attempt_wins_even_when_worse_or_failed(self):
        expected = inventory_rows(inventory())
        first = source([row(0, arbitrary_quality=1), row(1)])
        worse = source([row(0, arbitrary_quality=100)], '0')
        report = select_attempts(expected, [first, worse])
        self.assertEqual(report['selected_attempts'][0]['record']['arbitrary_quality'], 100)
        failed = source([row(0, 'FAIL')], '0')
        report = select_attempts(expected, [first, worse, failed])
        self.assertEqual(report['cases'][0]['status'], 'FAIL')
        self.assertEqual(report['attempt_status_counts'], {'PASS': 3, 'FAIL': 1})
        self.assertTrue(report['coverage_complete'])

    def test_pending_retry_prevents_completion_without_hiding_old_record(self):
        report = select_attempts(inventory_rows(inventory()),
                                 [source([row(0), row(1)]), source([], '0')])
        self.assertTrue(report['all_inputs_have_terminal_record'])
        self.assertFalse(report['coverage_complete'])
        self.assertEqual(report['pending_declared_attempts'], [{'source': 1, 'index': 0}])
        self.assertEqual(report['cases'][0]['status'], 'PASS')

    def test_wrong_identity_duplicate_and_undeclared_records_are_rejected(self):
        expected = inventory_rows(inventory())
        bad = [source([row(0), row(0)]), source([row(0, sha256='f' * 64)]),
               source([row(1)], '0'), source([row(0, 'RUNNING')])]
        for item in bad:
            with self.subTest(item=item), self.assertRaises(ValueError):
                select_attempts(expected, [item])

    def test_complete_prefix_is_distinct_from_a_malformed_complete_record(self):
        complete, records, ignored = complete_records(b'{"index":0}\n{"index":')
        self.assertEqual(complete, b'{"index":0}\n')
        self.assertEqual(records, [{'index': 0}])
        self.assertEqual(ignored, len(b'{"index":'))
        with self.assertRaises(json.JSONDecodeError):
            complete_records(b'not-json\n')

    def test_frozen_selection_survives_live_append_and_metadata_status_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, run, suite, _ = disk_fixture(root)
            output = root / 'cohort'
            report = build_cohort(manifest, [suite], output)
            self.assertEqual(report['case_status_counts'], {'PASS': 1, 'NOT_RECORDED': 1})
            with (suite / 'records.jsonl').open('a') as stream:
                stream.write(json.dumps(row(1, 'FAIL')) + '\n')
            meta = json.loads((run / 'run.json').read_text())
            meta['state'] = 'COMPLETED_WITH_FAILURES'
            (run / 'run.json').write_text(json.dumps(meta))
            frozen, _ = read_cohort(output / 'cohort.json')
            self.assertEqual(frozen['recorded_inputs'], 1)
            batches, _ = analysis_sources(cohort=output / 'cohort.json')
            self.assertEqual(batches[0]['suite'], str(suite))
            self.assertEqual([r['index'] for r in batches[0]['records']], [0])
            current = build_cohort(manifest, [suite], root / 'later')
            self.assertTrue(current['coverage_complete'])
            self.assertEqual(current['case_status_counts'], {'PASS': 1, 'FAIL': 1})

    def test_snapshot_selection_and_result_tampering_are_detected(self):
        for change in ['snapshot', 'selection', 'result', 'versions']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest, _, suite, case = disk_fixture(root)
                output = root / 'cohort'
                build_cohort(manifest, [suite], output)
                path = output / 'cohort.json'
                report = json.loads(path.read_text())
                if change == 'snapshot':
                    (output / report['sources'][0]['records_snapshot']['snapshot']).write_text('changed')
                elif change == 'selection':
                    report['coverage_complete'] = True
                    path.write_text(json.dumps(report))
                elif change == 'versions':
                    report['selected_source_versions'] = {'b' * 40: 1}
                    path.write_text(json.dumps(report))
                else:
                    result = json.loads((case / 'summary.json').read_text())
                    result['unexpected_edit'] = True
                    (case / 'summary.json').write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    read_cohort(path)

    def test_completed_case_requires_matching_input_tool_and_budget(self):
        for key, value in [('input_sha256', '9' * 64), ('gpuls_sha256', '9' * 64),
                           ('checkpoint_sha256', '9' * 64), ('evaluation_budget', 99)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest, _, suite, case = disk_fixture(root)
                config = json.loads((case / 'config.json').read_text())
                config[key] = value
                (case / 'config.json').write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    build_cohort(manifest, [suite], root / 'cohort')


if __name__ == '__main__':
    unittest.main()

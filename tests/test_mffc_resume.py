"""Terminal-job recovery never resubmits live work or invents a lost clock."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('mffc905_driver_test', Path(__file__).resolve().parents[1] / 'scripts/run_mffc905.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ResumeTests(unittest.TestCase):
    def test_terminal_json_publication_is_exclusive_and_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            driver.save(path, {'status': 'PASS'})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                driver.save(path, {'status': 'FAIL'})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob('*.partial')), [])

    def test_live_job_rejected_and_terminal_incomplete_attempt_not_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            (outer / 'point_attempt.json').write_text(json.dumps({'slurm_job_id': '123', 'source_commit': 'frozen',
                'input_sha256': 'input', 'method': 'Ours'}))
            row = {'index': 0, 'relative_path': 'source/input.aig', 'sha256': 'input'}
            with patch.dict(os.environ, {'SLURM_JOB_ID': '124'}):
                with patch.object(driver.subprocess, 'check_output', return_value='123|RUNNING|Unknown|\n'):
                    with self.assertRaises(RuntimeError):
                        driver.recover_point(outer, row, 'Ours', 'frozen')
                with patch.object(driver.subprocess, 'check_output', return_value='123|TIMEOUT|2026-11-14T01:00:00|\n'):
                    record = driver.recover_point(outer, row, 'Ours', 'frozen')
                self.assertEqual(record['status'], 'FAIL')
                self.assertIsNone(record['end_to_end_runtime_s'])
                self.assertIsNone(record['result_path'])

    def test_recorded_cold_timeout_recovers_its_paid_elapsed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            (outer / 'point_attempt.json').write_text(json.dumps({'slurm_job_id': '123', 'source_commit': 'frozen',
                'input_sha256': 'input', 'method': 'B'}))
            (outer / 'pre_launch.json').write_text(json.dumps({'elapsed_s': .25}))
            (outer / 'cold_method.command.json').write_text(json.dumps({'elapsed_s': 1800., 'returncode': -15, 'timed_out': True}))
            row = {'index': 0, 'relative_path': 'source/input.aig', 'sha256': 'input'}
            with patch.dict(os.environ, {'SLURM_JOB_ID': '124'}), patch.object(driver.subprocess, 'check_output',
                    return_value='123|COMPLETED|2026-11-14T01:00:00|\n'):
                record = driver.recover_point(outer, row, 'B', 'frozen')
            self.assertEqual(record['status'], 'TIMEOUT')
            self.assertEqual(record['end_to_end_runtime_s'], 1800.25)
            self.assertNotIn('metrics', record)


if __name__ == '__main__':
    unittest.main()

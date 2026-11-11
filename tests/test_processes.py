import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from global_timing.common import ToolError, run_tool


class ProcessTests(unittest.TestCase):
    def test_timeout_kills_separately_sessioned_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            program = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True); print(p.pid,flush=True);time.sleep(60)"
            with self.assertRaises(ToolError):
                run_tool([sys.executable, '-c', program], work, 'timeout', timeout=1)
            report = json.loads((work / 'timeout.command.json').read_text())
            self.assertTrue(report['timed_out'])
            child = int((work / 'timeout.log').read_text().strip())
            self.assertIn(child, report['terminated_process_ids'])
            for _ in range(50):
                stat = Path(f'/proc/{child}/stat')
                if not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                    break
                time.sleep(.02)
            else:
                self.fail('Timed-out child is still executing.')

    def test_failed_command_keeps_log_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with self.assertRaises(ToolError):
                run_tool([sys.executable, '-c', 'print("failure evidence"); raise SystemExit(7)'], work, 'failed')
            self.assertEqual(json.loads((work / 'failed.command.json').read_text())['returncode'], 7)
            with self.assertRaises(FileExistsError):
                run_tool([sys.executable, '-c', 'print("replacement")'], work, 'failed')
            self.assertIn('failure evidence', (work / 'failed.log').read_text())

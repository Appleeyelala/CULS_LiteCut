#!/usr/bin/env python3
"""Run unittest discovery and preserve individual outcomes as JSON."""
import argparse
import json
from pathlib import Path
import sys
import time
import unittest


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def startTest(self, test):
        self.started_at = time.monotonic()
        super().startTest(test)

    def record(self, test, status, reason=None):
        self.records.append({'test': test.id(), 'status': status, 'reason': reason,
                             'elapsed_s': time.monotonic() - getattr(self, 'started_at', time.monotonic())})

    def addSuccess(self, test):
        super().addSuccess(test)
        self.record(test, 'PASS')

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.record(test, 'FAIL', self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self.record(test, 'ERROR', self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.record(test, 'SKIP', reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test', action='append', help='Exact unittest name; may repeat. Omit for full discovery.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'tests'))
    suite = (unittest.defaultTestLoader.loadTestsFromNames(args.test) if args.test else
             unittest.defaultTestLoader.discover(str(root / 'tests')))
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump({'tests_run': result.testsRun, 'successful': result.wasSuccessful(),
                   'records': result.records}, stream, indent=2)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Path traces are validated as transition-connected Liberty arrival chains."""
import unittest

from global_timing.common import ToolError
from global_timing.path_analysis import parse_paths


class PathTests(unittest.TestCase):
    def setUp(self):
        self.timing = {'delay_ps': 10, 'po_arrivals_ps': [10, 9.6, 2], 'critical_po_indices': [0]}
        self.text = ('GT_PATH_META 0.95 3\n'
            'GT_PATH_BEGIN 0 0 10\nGT_PATH_NODE 0 0 5 0 10 CELL NAND\n'
            'GT_PATH_ARC 0 0 5 1 0 1 0 0 10 0 10 0 2\n'
            'GT_PATH_NODE 0 1 1 1 0 PI -\nGT_PATH_END 0 2 0\n'
            'GT_PATH_BEGIN 1 1 9.6\nGT_PATH_NODE 1 0 6 1 9.6 CELL BUF\n'
            'GT_PATH_ARC 1 0 6 2 0 1 1 0 9.6 0 9.6 0 1\n'
            'GT_PATH_NODE 1 1 2 1 0 PI -\nGT_PATH_END 1 2 1\n')

    def test_critical_and_nearcritical_transition_paths(self):
        report = parse_paths(self.text, self.timing, 2)
        self.assertEqual([row['critical'] for row in report['paths']], [True, False])
        self.assertEqual([row['pi_index'] for row in report['paths']], [0, 1])

    def test_missing_path_wrong_phase_arithmetic_and_nonfinite_are_rejected(self):
        for text in [self.text.split('GT_PATH_BEGIN 1')[0], self.text.replace('5 1 0 1 0', '5 1 0 0 0'),
                     self.text.replace('0 10 0 10', '0 8 0 10'), self.text.replace('0 9.6 0 9.6', '0 nan 0 9.6'),
                     self.text + 'GT_PATH_ERROR 0 ARRIVAL_RECONSTRUCTION\n']:
            with self.assertRaises(ToolError):
                parse_paths(text, self.timing, 2)

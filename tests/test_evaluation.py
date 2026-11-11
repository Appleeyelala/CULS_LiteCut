"""Parser fixtures are unit inputs, not experimental measurements."""
import unittest

from global_timing.common import ToolError, tool_path
from global_timing.evaluation import cec_passed, parse_timing
from pathlib import Path


class EvaluationTests(unittest.TestCase):
    def test_cec_requires_positive_certificate(self):
        self.assertTrue(cec_passed('Networks are equivalent. Time = 0.00 sec'))
        for text in ['Networks are NOT EQUIVALENT.', 'equivalent', 'UNDECIDED',
                     'Networks are equivalent.\nVerification failed', 'Error: cec failed']:
            self.assertFalse(cec_passed(text), text)

    def test_liberty_endpoint_max_and_precision(self):
        result = parse_timing('GT_META 12.234 38.125 2 7\nGT_PO 0 31.2 30.2\nGT_PO 1 37.2 38.125\n')
        self.assertEqual(result['critical_po_indices'], [1])
        self.assertEqual(result['area'], 12.234)
        self.assertEqual(result['po_arrivals_ps'], [31.2, 38.125])

    def test_rejects_missing_wrong_or_nonfinite_measurements(self):
        for text in ['Error: mapping failed', 'GT_META 12 38 2 7\nGT_PO 0 38 30\n',
                     'GT_META 12 38 1 7\nGT_PO 0 31 30\n',
                     'GT_META 12 nan 1 7\nGT_PO 0 38 30\n',
                     'GT_META 12 38 1 7\nGT_PO 0 38 30\nGT_PO 0 38 30\n']:
            with self.assertRaises(ToolError):
                parse_timing(text)

    def test_command_metacharacters_rejected(self):
        for path in ['/data/a;quit', '/data/a b', '/data/$(secret)', '/data/a\nb']:
            with self.assertRaises(ValueError):
                tool_path(Path(path))


if __name__ == '__main__':
    unittest.main()

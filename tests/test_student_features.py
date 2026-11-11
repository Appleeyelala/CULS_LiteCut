import unittest
from unittest.mock import patch

from global_timing.common import ROOT, legacy


class StudentFeatureTests(unittest.TestCase):
    def test_student_features_equal_full_prefix_without_computing_unused_columns(self):
        bridge = legacy('gate3e_run').load_mapping_free_bridge(ROOT / 'projects/litecut')
        graph = {1: [(4, 0)], 2: [(4, 1)], 3: [(5, 0)], 4: [(5, 0)], 5: []}
        extractor = bridge['CompleteCutFeatureExtractor'](graph, inverter_weight=1.5)
        full, full_names = extractor.extract_all(graph)
        self.assertEqual(len(full), 25)
        with patch.object(extractor, 'get_laplacian_eigenvalues', side_effect=AssertionError('Unused spectrum computed')), \
                patch.object(extractor, 'get_wl_signature', side_effect=AssertionError('Unused hash computed')):
            actual, names = extractor.extract_all(graph, student_only=True)
        self.assertEqual(actual, full[:19])
        self.assertEqual(names, full_names[:19])


if __name__ == '__main__':
    unittest.main()

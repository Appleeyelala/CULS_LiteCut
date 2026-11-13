"""Missing/zero/tolerance/tradeoff and fixed-denominator result semantics."""
import unittest

from global_timing.mffc_results import describe, geometric, ratio, summarize


PROTOCOL = {'delay_comparison': {'absolute_tolerance_ps': 2e-5, 'relative_tolerance': 2e-6},
            'runtime_comparison': {'absolute_tolerance_s': .01, 'relative_tolerance': .01}}


def row(index, method, delay, runtime, status='PASS', feasible=True):
    return {'index': index, 'method': method, 'status': status, 'delay_ps': delay,
            'area': 10. if status == 'PASS' else None, 'area_feasible': feasible,
            'end_to_end_runtime_s': runtime, 'device': 'cuda' if method == 'Ours' else 'cpu',
            'decisions': 2 if method == 'Ours' else None}


class SummaryTests(unittest.TestCase):
    def test_zero_denominators_never_become_identity_ratios(self):
        self.assertIsNone(ratio(0, 0))
        self.assertIsNone(ratio(3, 0))
        self.assertIsNone(ratio(None, 2))
        self.assertEqual(geometric([None, 0, 1, 4]), {'n_positive_defined': 2, 'value': 2.0})

    def test_failure_kept_in_coverage_and_runtime_but_not_quality_pairs(self):
        rows = [row(0, 'A', 100, 1), row(0, 'B', 90, 2), row(0, 'Ours', 80, 4),
                row(1, 'A', 0, 1), row(1, 'B', None, 900, 'TIMEOUT'), row(1, 'Ours', 0, 3)]
        result = summarize(rows, PROTOCOL)
        self.assertEqual(result['method_points'], 6)
        self.assertEqual(result['comparisons']['A']['successful_pairs'], 2)
        self.assertEqual(result['comparisons']['A']['delay_ratio_geometric']['n_positive_defined'], 1)
        self.assertEqual(result['comparisons']['B']['successful_pairs'], 1)
        self.assertEqual(result['methods']['B']['states'], {'PASS': 1, 'TIMEOUT': 1})
        self.assertEqual(result['methods']['B']['runtime_all_terminal']['sum'], 902)
        self.assertEqual(result['comparisons']['A']['common_three_valid_feasible_pairs'], 1)
        self.assertEqual(result['comparisons']['A']['common_feasible_relation_grid'], {'better_delay__worse_time': 1})

    def test_runtime_statistic_is_median_of_pairs_and_feasibility_is_not_imputed(self):
        rows = []
        for index, (a, ours) in enumerate([(2, 1), (100, 2), (1, 100)]):
            rows.extend([row(index, 'A', 10, a), row(index, 'B', 10, 1), row(index, 'Ours', 10, ours)])
        rows[0]['area_feasible'] = None
        result = summarize(rows, PROTOCOL)
        self.assertEqual(result['comparisons']['A']['paired_runtime_ratio']['median'], 2)
        self.assertNotEqual(2, describe([2, 100, 1])['median'] / describe([1, 2, 100])['median'])
        self.assertEqual(result['comparisons']['A']['common_three_valid_feasible_pairs'], 2)

    def test_no_extra_methods_or_duplicate_points_allowed(self):
        with self.assertRaises(ValueError):
            summarize([row(0, 'random', 10, 1)], PROTOCOL)
        with self.assertRaises(ValueError):
            summarize([row(0, 'A', 10, 1), row(0, 'A', 10, 1)], PROTOCOL)


if __name__ == '__main__':
    unittest.main()

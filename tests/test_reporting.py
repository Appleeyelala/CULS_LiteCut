"""Prevent biased cross-design aggregation and false repeatability claims."""
import copy
import unittest

from scripts.summarize_benchmarks import canonical_outcomes, paired_statistics


class ReportingTests(unittest.TestCase):
    def test_pairing_preserves_failures_and_rejects_infeasible_or_unequal_budgets(self):
        rows = [{'method': method, 'delay_ps': value, 'actual_joint_objective': value,
                 'area_constraint_feasible': True, 'evaluations_including_baseline': 8}
                for method, value in [('gradient_area_constraint', 9), ('random_search_area_constraint', 10),
                                       ('gradient_joint', .6), ('random_search_joint', .5)]]
        records = [{'index': 0, 'status': 'PASS', 'comparison': rows}, {'index': 1, 'status': 'FAIL'}]
        unequal = copy.deepcopy(records[0]); unequal['index'] = 2
        unequal['comparison'][0]['evaluations_including_baseline'] = 4
        unequal['comparison'][2]['evaluations_including_baseline'] = 4
        infeasible = copy.deepcopy(records[0]); infeasible['index'] = 3
        infeasible['comparison'][0]['area_constraint_feasible'] = False
        result = paired_statistics(records + [unequal, infeasible])['comparisons']
        area, joint = result['gradient_area_constraint'], result['gradient_joint']
        self.assertEqual(area['outcomes'], {'win': 1})
        self.assertEqual(area['omitted_count'], 3)
        self.assertEqual(joint['outcomes'], {'loss': 2})
        self.assertEqual(joint['omitted_count'], 2)
        self.assertEqual(paired_statistics([records[0], records[0]])['status'], 'NOT_POOLED')

    def test_repeat_comparison_includes_the_final_selected_configuration(self):
        row = {'status': 'PASS', 'choices': [0], 'metrics': {'delay_ps': 10}}
        group = {'observations': [row, dict(row, choices=[1])], 'best': row}
        summary = {'config': {}, 'original': {}, 'native_abc': {}, 'replay_noop': row,
                   'greedy_litecut': copy.deepcopy(group), 'independent_regions': copy.deepcopy(group),
                   'random_search': [row], 'gradient_search': {'joint': group}}
        altered = copy.deepcopy(summary)
        altered['gradient_search']['joint']['best'] = altered['gradient_search']['joint']['observations'][1]
        self.assertNotEqual(canonical_outcomes(summary), canonical_outcomes(altered))

    def test_repeat_control_failure_is_distinct_from_a_successful_measurement(self):
        row = {'status': 'PASS', 'choices': [0], 'metrics': {'delay_ps': 10}}
        group = {'observations': [row], 'best': row}
        summary = {'config': {}, 'original': {}, 'native_abc': {}, 'replay_noop': row,
                   'greedy_litecut': group, 'independent_regions': group,
                   'random_search': [row], 'gradient_search': {'joint': group}}
        altered = copy.deepcopy(summary)
        altered.update(native_abc=None, native_abc_attempt={'status': 'FAIL', 'stage': 'cec_and_mapping', 'error': 'unit timeout'})
        self.assertNotEqual(canonical_outcomes(summary), canonical_outcomes(altered))
        result = canonical_outcomes(altered)['native_abc']
        self.assertEqual(result, {'status': 'FAIL', 'stage': 'cec_and_mapping', 'metrics': None})

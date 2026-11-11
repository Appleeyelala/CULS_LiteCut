import copy
import unittest

from global_timing.search_analysis import analyze_search, aggregate_search


def summary_fixture(zero=False):
    def metrics(po):
        return {'delay_ps': max(po), 'po_arrivals_ps': po, 'area': 10., 'cec': 'PASS'}
    baseline = {'index': 0, 'status': 'PASS', 'choices': [0], 'baseline_reused': True,
                'metrics': metrics([0., 0.] if zero else [70., 100.])}
    rows = [baseline]
    for index, source, predicted, actual in [
            (1, 'calibration_warmup', [70., 80.], None),
            (2, 'explicit_feasibility_repair', [80., 100.], [75., 90.]),
            (3, 'joint_selection_gradient', [75., 85.], [80., 95.])]:
        if zero:
            predicted = [0., 0.]
            actual = [0., 0.] if actual is not None else None
        row = {'index': index, 'status': 'PASS' if actual is not None else 'FAIL', 'choices': [index],
               'prediction_before_evaluation': {'source': source, 'po_arrivals_ps': predicted,
                    'delay_ps': max(predicted), 'area': 10., 'training_observations': 1,
                    'training_observation_indices': [0]}}
        if actual is not None:
            row['metrics'] = metrics(actual)
        else:
            row['execution_failure'] = {'rejected_requests': [{'status': 'REVERTED'}]}
        rows.append(row)
    return {'original': metrics([0., 0.] if zero else [80., 110.]), 'replay_noop': baseline,
            'config': {'area_limit_ratio': 1.05}, 'gradient_search': {'area_constraint': {
                'observations': rows, 'best': rows[2], 'evaluation_budget': 4,
                'evaluations_including_baseline': 4, 'gradient_proposal_evaluations': 1}}}


class SearchAnalysisTests(unittest.TestCase):
    def test_failure_and_repair_are_distinct_from_direct_gradient_success(self):
        result = analyze_search(summary_fixture())
        method = result['methods']['area_constraint']
        self.assertEqual(method['failed_requests'], 1)
        self.assertEqual(method['selected_source'], 'explicit_feasibility_repair')
        self.assertEqual(method['direct_gradient_successful'], 1)
        self.assertEqual(method['prediction_samples'], 2)  # exclude failed request and baseline anchor
        self.assertEqual(method['global_delta_mae_ps'], 10.)
        self.assertEqual(method['endpoint_mae_ps'], 7.5)
        self.assertAlmostEqual(method['global_delta_normalized_mae'], .1)
        self.assertEqual(method['selected_delay_improvement_vs_noop_pct'], 10.)

    def test_current_future_and_failed_calibration_labels_are_rejected(self):
        for indices in [[2], [3], [1], [-1], [0, 0]]:
            summary = summary_fixture()
            pred = summary['gradient_search']['area_constraint']['observations'][2]['prediction_before_evaluation']
            pred.update(training_observation_indices=indices, training_observations=len(indices))
            with self.assertRaisesRegex(ValueError, 'calibration label'):
                analyze_search(summary)

    def test_zero_reference_and_unchanged_outputs_are_not_accuracy_evidence(self):
        result = analyze_search(summary_fixture(zero=True))
        method = result['methods']['area_constraint']
        self.assertIsNone(method['global_delta_normalized_mae'])
        self.assertIsNone(method['direction_accuracy_on_actual_changes'])
        self.assertEqual(method['actual_unchanged_samples'], 2)
        pooled = aggregate_search([{'analysis': result}])['modes']['area_constraint']
        self.assertEqual(pooled['designs_with_normalizable_predictions'], 0)
        self.assertIsNone(pooled['mean_per_design_normalized_mae'])

    def test_aggregation_weights_designs_not_request_counts(self):
        a = analyze_search(summary_fixture())
        b = copy.deepcopy(a)
        b['methods']['area_constraint'].update(global_delta_normalized_mae=.3, prediction_samples=100)
        pooled = aggregate_search([{'analysis': a}, {'analysis': b}])['modes']['area_constraint']
        self.assertAlmostEqual(pooled['mean_per_design_normalized_mae'], .2)
        self.assertEqual(pooled['prediction_samples'], 102)


if __name__ == '__main__':
    unittest.main()

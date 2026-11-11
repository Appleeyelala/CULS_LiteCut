"""Audit saved adaptive search outcomes without fitting or inventing labels."""
from collections import Counter
import math
import statistics


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Non-finite saved analysis input.')
    return value


def _mean(values):
    return statistics.mean(values) if values else None


def _source(row):
    if row.get('baseline_reused'):
        return 'replay_noop'
    return (row.get('prediction_before_evaluation') or {}).get('source', 'unrecorded')


def analyze_search(summary):
    """Separate measured improvements, proposal provenance and prior predictions.

    The prediction statistics are prospective within an adaptive run. Models
    may change between requests, so they are not fixed-model ranking scores or
    held-out-design generalization estimates. Every no-op anchor is excluded
    from the accuracy samples; failed requests have no substituted labels.
    """
    baseline = summary['replay_noop']['metrics']
    reference = _finite(baseline['delay_ps'])
    original = _finite(summary['original']['delay_ps'])
    area_reference = _finite(baseline['area'])
    tolerance = max(1e-6, abs(reference) * 1e-9)
    methods, samples = {}, []

    def sign(value):
        return 1 if value > tolerance else -1 if value < -tolerance else 0

    for mode, group in summary['gradient_search'].items():
        rows = group['observations']
        by_index = {r['index']: r for r in rows}
        if len(rows) != len(by_index) or len(rows) != group['evaluations_including_baseline']:
            raise ValueError('Observation identities/counts disagree with saved search summary.')
        if len(rows) > group['evaluation_budget']:
            raise ValueError('Search consumed more requests than its declared budget.')
        sources, source_passes, rejection_statuses = Counter(), Counter(), Counter()
        method_samples = []
        for row in rows:
            source = _source(row)
            sources[source] += 1
            if row['status'] == 'PASS':
                if row['metrics']['cec'] != 'PASS':
                    raise ValueError('Successful observation lacks a positive CEC result.')
                source_passes[source] += 1
            for rejected in row.get('execution_failure', {}).get('rejected_requests', []):
                rejection_statuses[rejected['status']] += 1
            prediction = row.get('prediction_before_evaluation') or {}
            if 'po_arrivals_ps' not in prediction or row.get('baseline_reused'):
                continue
            fitted = prediction.get('training_observation_indices')
            if fitted is None:
                raise ValueError('Prediction lacks explicit calibration-label provenance.')
            if (any(type(i) is not int or i < 0 or i >= row['index'] or i not in by_index or
                    by_index[i]['status'] != 'PASS' for i in fitted) or len(set(fitted)) != len(fitted) or
                    prediction.get('training_observations') != len(fitted)):
                raise ValueError('Prediction references a current/future, failed, missing or repeated calibration label.')
            predicted = [_finite(x) for x in prediction['po_arrivals_ps']]
            if not predicted or not math.isclose(max(predicted), _finite(prediction['delay_ps']), rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError('Saved predicted global delay does not match its endpoints.')
            if row['status'] != 'PASS':
                continue
            actual = [_finite(x) for x in row['metrics']['po_arrivals_ps']]
            if len(actual) != len(predicted):
                raise ValueError('Predicted and actual endpoint counts differ.')
            delay = _finite(row['metrics']['delay_ps'])
            pd, ad = max(predicted) - reference, delay - reference
            error = abs(pd - ad)
            po_error = statistics.mean(abs(a - b) for a, b in zip(predicted, actual))
            sample = {'mode': mode, 'index': row['index'], 'source': source,
                      'training_observation_indices': fitted, 'baseline_anchor_used': True,
                      'predicted_delta_ps': pd, 'actual_delta_ps': ad, 'absolute_error_ps': error,
                      'normalized_predicted_delta': pd / reference if reference > 0 else None,
                      'normalized_actual_delta': ad / reference if reference > 0 else None,
                      'endpoint_mae_ps': po_error,
                      'normalized_absolute_error': error / reference if reference > 0 else None,
                      'normalized_endpoint_mae': po_error / reference if reference > 0 else None,
                      'predicted_direction': sign(pd), 'actual_direction': sign(ad),
                      'actual_change_exceeds_tolerance': sign(ad) != 0,
                      'direction_correct': sign(pd) == sign(ad),
                      'area_prediction_error': abs(_finite(prediction['area']) - _finite(row['metrics']['area']))}
            samples.append(sample)
            method_samples.append(sample)
        best = group.get('best')
        if not best or best['status'] != 'PASS' or best['index'] not in by_index:
            raise ValueError('Search has no saved successful selected result.')
        recorded = by_index[best['index']]
        if best['choices'] != recorded['choices'] or best['metrics'] != recorded['metrics']:
            raise ValueError('Selected result differs from its saved observation.')
        if mode == 'area_constraint' and best['metrics']['area'] > area_reference * summary['config']['area_limit_ratio'] + 1e-8:
            raise ValueError('Selected area-constrained result exceeds its measured area limit.')
        direct = sources.get('joint_selection_gradient', 0)
        if direct != group.get('gradient_proposal_evaluations'):
            raise ValueError('Saved direct-gradient evaluation count is inconsistent.')
        changed = [r for r in method_samples if r['actual_change_exceeds_tolerance']]
        norms = [r['normalized_absolute_error'] for r in method_samples if r['normalized_absolute_error'] is not None]
        selected_delay = _finite(best['metrics']['delay_ps'])
        methods[mode] = {
            'budget': group['evaluation_budget'], 'consumed_requests_including_baseline': len(rows),
            'failed_requests': sum(r['status'] != 'PASS' for r in rows),
            'requests_by_source': dict(sources), 'successful_requests_by_source': dict(source_passes),
            'rejected_decisions_by_status': dict(rejection_statuses),
            'direct_gradient_requested': direct, 'direct_gradient_successful': source_passes.get('joint_selection_gradient', 0),
            'selected_index': best['index'], 'selected_source': _source(best),
            'selected_parent_source': (best.get('prediction_before_evaluation') or {}).get('parent_source'),
            'selected_delay_ps': selected_delay, 'selected_area': best['metrics']['area'],
            'selected_delay_improvement_vs_noop_pct': 100 * (reference - selected_delay) / reference if reference > 0 else None,
            'selected_delay_improvement_vs_original_pct': 100 * (original - selected_delay) / original if original > 0 else None,
            'prediction_samples': len(method_samples), 'actual_changed_samples': len(changed),
            'actual_unchanged_samples': len(method_samples) - len(changed),
            'direction_accuracy_on_actual_changes': _mean([int(r['direction_correct']) for r in changed]),
            'false_change_predictions_on_unchanged': sum(not r['actual_change_exceeds_tolerance'] and
                                                        r['predicted_direction'] != 0 for r in method_samples),
            'global_delta_mae_ps': _mean([r['absolute_error_ps'] for r in method_samples]),
            'global_delta_normalized_mae': _mean(norms),
            'endpoint_mae_ps': _mean([r['endpoint_mae_ps'] for r in method_samples]),
            'normalization_reference': 'measured replay-noop global delay; zero references omitted',
        }
    return {'methods': methods, 'prediction_samples': samples, 'direction_tolerance_ps': tolerance,
            'serialization_delay_change_pct': 100 * (reference - original) / original if original > 0 else None,
            'scope': 'Pre-evaluation predictions from changing within-design models, conditioned on evaluated candidates. '
                     'No baseline-anchor accuracy samples, failed-request imputations, fixed-model ranking or held-out-design claims.'}


def aggregate_search(records):
    """Weight each measured design once; preserve the request-level tables."""
    result = {}
    for mode in sorted({mode for record in records for mode in record['analysis']['methods']}):
        methods = [record['analysis']['methods'][mode] for record in records if mode in record['analysis']['methods']]
        normalized = [r['global_delta_normalized_mae'] for r in methods if r['global_delta_normalized_mae'] is not None]
        directional = [r['direction_accuracy_on_actual_changes'] for r in methods if r['direction_accuracy_on_actual_changes'] is not None]
        result[mode] = {'design_count': len(methods), 'prediction_samples': sum(r['prediction_samples'] for r in methods),
                        'designs_with_normalizable_predictions': len(normalized),
                        'mean_per_design_normalized_mae': _mean(normalized),
                        'median_per_design_normalized_mae': statistics.median(normalized) if normalized else None,
                        'designs_with_actual_changed_samples': len(directional),
                        'mean_per_design_direction_accuracy_on_actual_changes': _mean(directional),
                        'final_selected_sources': dict(Counter(r['selected_source'] for r in methods)),
                        'direct_gradient_requested': sum(r['direct_gradient_requested'] for r in methods),
                        'direct_gradient_successful': sum(r['direct_gradient_successful'] for r in methods),
                        'failed_requests': sum(r['failed_requests'] for r in methods)}
    return {'modes': result, 'scope': 'Descriptive equal-design weighting within the explicitly supplied run cohort. '
            'Failures remain in coverage; missing values are not imputed. Adaptive samples and related benchmark variants '
            'are not independent statistical replicates; no significance or held-out generalization claim.'}

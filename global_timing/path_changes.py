"""Compare validated Liberty arrival traces using preserved PI/PO identities.

Mapped node IDs are deliberately absent from comparisons. Each PO supplies one
representative maximizing chain, not an enumeration of tied logical paths.
"""
from collections import Counter
import math
import statistics

from .search_analysis import analyze_search
from .evaluation import native_baseline_metrics


def references(summary):
    """All successful observations expected in a complete saved path audit."""
    groups = {'replay_noop': [summary['replay_noop']], 'random_search': summary['random_search']}
    groups.update({name: summary[name]['observations'] for name in ['greedy_litecut', 'independent_regions']})
    groups.update({'gradient_' + mode: item['observations'] for mode, item in summary['gradient_search'].items()})
    best = {name: summary[name]['best'] for name in ['greedy_litecut', 'independent_regions']}
    best.update({'gradient_' + mode: item['best'] for mode, item in summary['gradient_search'].items()})
    controls = ['original'] + (['native_abc'] if native_baseline_metrics(summary) is not None else [])
    items = [{'method': name, 'index': None, 'metrics': summary[name], 'observation': None,
              'baseline_anchor': name == 'original', 'is_saved_final_selection': False,
              'failed_requests_since_previous_success': 0} for name in controls]
    for name, rows in groups.items():
        indices = [row['index'] for row in rows]
        if indices != sorted(set(indices)):
            raise ValueError('Observation indices must be unique and ordered within a method.')
        selected = best.get(name)
        if selected is not None:
            matches = [row for row in rows if row['index'] == selected['index']]
            if (len(matches) != 1 or matches[0]['status'] != 'PASS' or
                    matches[0]['metrics'] != selected['metrics'] or matches[0]['choices'] != selected['choices']):
                raise ValueError('Saved final selection does not identify a successful observation.')
        failed = 0
        for row in rows:
            if row['status'] != 'PASS':
                failed += 1
                continue
            items.append({'method': name, 'index': row['index'], 'metrics': row['metrics'], 'observation': row,
                          'baseline_anchor': name == 'replay_noop' or bool(row.get('baseline_reused')),
                          'is_saved_final_selection': selected is not None and selected['index'] == row['index'],
                          'failed_requests_since_previous_success': failed})
            failed = 0
    return items


def endpoint_state(trace, metrics):
    """Compact interface signature after parse_paths has validated the chain."""
    critical = [path for path in trace['paths'] if path['critical']]
    if {path['po_index'] for path in critical} != set(metrics['critical_po_indices']):
        raise ValueError('Critical path flags do not match the measured endpoint set.')
    if not critical or trace['near_critical_ratio'] != .95:
        raise ValueError('Missing critical or incompatible near-critical path coverage.')
    if not all(path['arrival_reconstruction'] == 'PASS' for path in trace['paths']):
        raise ValueError('An arrival path was not validated.')
    return {'delay_ps': metrics['delay_ps'], 'area': metrics['area'],
            'critical_po': sorted({path['po_index'] for path in critical}),
            'critical_pairs': sorted([path['pi_index'], path['po_index']] for path in critical),
            'critical_transitions': sorted([path['pi_index'], path['po_index'], path['output_phase']] for path in critical),
            'near_critical_po': sorted(path['po_index'] for path in trace['paths'])}


def compare_states(before, after):
    for state in [before, after]:
        if any(not math.isfinite(state[key]) or state[key] < 0 for key in ['area', 'delay_ps']):
            raise ValueError('Invalid measured timing/area in path comparison.')
    pairs_changed = before['critical_pairs'] != after['critical_pairs']
    po_changed = before['critical_po'] != after['critical_po']
    return {'delay_delta_ps': after['delay_ps'] - before['delay_ps'],
            'delay_improvement_pct': (100 * (before['delay_ps'] - after['delay_ps']) / before['delay_ps']
                                      if before['delay_ps'] > 0 else None),
            'critical_po_changed': po_changed, 'observed_critical_pairs_changed': pairs_changed,
            'observed_pair_change_with_same_po_set': pairs_changed and not po_changed,
            'observed_critical_transitions_changed': before['critical_transitions'] != after['critical_transitions'],
            'near_critical_po_changed': before['near_critical_po'] != after['near_critical_po']}


def analyze_case(summary, path_report, states):
    """Join every successful request to its audited trace; failures stay gaps."""
    if path_report['status'] != 'PASS':
        raise ValueError('A failed path audit cannot become a complete analyzed case.')
    expected = {(item['method'], item['index']): item for item in references(summary)}
    links = {}
    for link in path_report['selection_path_changes']:
        key = link['method'], link['index']
        if key in links:
            raise ValueError('Duplicate path reference for an observation.')
        links[key] = link
    if set(links) != set(expected):
        raise ValueError('Path references omit or invent successful observations.')
    if set(states) != {link['path_evidence'] for link in links.values()}:
        raise ValueError('Validated measurements do not match the referenced traces.')
    for key, link in links.items():
        item, state = expected[key], states[link['path_evidence']]
        if (link['input_sha256'] != item['metrics']['input_sha256'] or
                sorted(link['critical_pairs']) != state['critical_pairs'] or
                any(state[field] != item['metrics'][field] for field in ['delay_ps', 'area'])):
            raise ValueError('Observation and validated path measurement disagree.')
    original = states[links['original', None]['path_evidence']]
    baseline = states[links['replay_noop', summary['replay_noop']['index']]['path_evidence']]
    observations, transitions, previous = [], [], {}
    for key, item in expected.items():
        link = links[key]
        state = states[link['path_evidence']]
        row = item['observation']
        identity = {'method': item['method'], 'index': item['index'], 'path_evidence': link['path_evidence'],
                    'input_sha256': link['input_sha256'], 'baseline_anchor': item['baseline_anchor'],
                    'is_saved_final_selection': item['is_saved_final_selection'],
                    'active_decisions': sum(choice != 0 for choice in row['choices']) if row else None,
                    'proposal_source': ((row.get('prediction_before_evaluation') or {}).get('source') if row else None),
                    'failed_requests_since_previous_success': item['failed_requests_since_previous_success']}
        comparisons = compare_states(baseline, state)
        observations.append(dict(identity, **state, **{'vs_noop_' + k: v for k, v in comparisons.items()}))
        prior = previous.get(item['method'])
        if prior is not None:
            prior_index, prior_state = prior
            transition = compare_states(prior_state, state)
            if link['critical_source_endpoint_pairs_changed'] != transition['observed_critical_pairs_changed']:
                raise ValueError('Saved path-change flag disagrees with its successful predecessor.')
            transitions.append(dict(identity, previous_successful_index=prior_index, **transition))
        elif link['critical_source_endpoint_pairs_changed']:
            raise ValueError('First method observation cannot have a saved predecessor change.')
        previous[item['method']] = item['index'], state
    counts = Counter(row['method'] for row in observations)
    return {'observations': observations, 'transitions': transitions,
            'successful_reference_counts': dict(counts),
            'serialization_control': compare_states(original, baseline),
            'scope': 'Changes of observed PI/PO sets and representative transition chains; '
                     'mapped node IDs are never compared across mappings. Failed requests have no path label.'}


def aggregate_cases(cases):
    methods = sorted({row['method'] for case in cases for row in case['analysis']['observations']
                      if row['method'] not in {'original', 'native_abc', 'replay_noop'}})
    result = {}
    for method in methods:
        rows = [row for case in cases for row in case['analysis']['observations']
                if row['method'] == method and not row['baseline_anchor']]
        selected = [row for case in cases for row in case['analysis']['observations']
                    if row['method'] == method and row['is_saved_final_selection']]
        transitions = [row for case in cases for row in case['analysis']['transitions'] if row['method'] == method]
        result[method] = {'successful_nonanchor_observations': len(rows),
            'observed_pair_changes_vs_noop': sum(row['vs_noop_observed_critical_pairs_changed'] for row in rows),
            'critical_po_changes_vs_noop': sum(row['vs_noop_critical_po_changed'] for row in rows),
            'observed_pair_changes_with_same_po_set_vs_noop': sum(row['vs_noop_observed_pair_change_with_same_po_set'] for row in rows),
            'observed_transition_changes_vs_noop': sum(row['vs_noop_observed_critical_transitions_changed'] for row in rows),
            'near_critical_po_changes_vs_noop': sum(row['vs_noop_near_critical_po_changed'] for row in rows),
            'successful_transitions': len(transitions),
            'observed_pair_changes_between_successes': sum(row['observed_critical_pairs_changed'] for row in transitions),
            'transitions_spanning_failed_requests': sum(row['failed_requests_since_previous_success'] > 0 for row in transitions),
            'saved_final_selections': len(selected),
            'saved_final_observed_pair_changes_vs_noop': sum(row['vs_noop_observed_critical_pairs_changed'] for row in selected)}
    return {'design_count': len(cases), 'methods': result,
            'prediction_error_by_endpoint_change': aggregate_prediction_changes(cases),
            'serialization_controls_with_observed_pair_change': sum(case['analysis']['serialization_control']['observed_critical_pairs_changed'] for case in cases),
            'scope': 'Descriptive counts of audited successful observations, not independent replicates or '
                     'proof of unique logical critical-path identity. Final selection markers cover saved '
                     'gradient and deterministic results; random search has two separate objective rankings.'}


def prediction_changes(summary, path_analysis):
    """Join prospective predictions to this exact observation's audited paths."""
    paths = {(row['method'], row['index']): row for row in path_analysis['observations']}
    if len(paths) != len(path_analysis['observations']):
        raise ValueError('Duplicate observations in path/prediction join.')
    result = []
    for sample in analyze_search(summary)['prediction_samples']:
        key = 'gradient_' + sample['mode'], sample['index']
        if key not in paths:
            raise ValueError('Prediction sample lacks its exact path observation.')
        path = paths[key]
        observation = next(row for row in summary['gradient_search'][sample['mode']]['observations']
                           if row['index'] == sample['index'])
        if (path['baseline_anchor'] or path['input_sha256'] != observation['metrics']['input_sha256'] or
                path['vs_noop_delay_delta_ps'] != sample['actual_delta_ps']):
            raise ValueError('Prediction/path join identifies another observation or baseline.')
        category = ('CRITICAL_PO_SET_CHANGED' if path['vs_noop_critical_po_changed'] else
                    'PI_CHANGED_WITH_SAME_PO_SET' if path['vs_noop_observed_critical_pairs_changed'] else
                    'SAME_OBSERVED_ENDPOINT_PAIR_SET')
        result.append(dict(sample, endpoint_change=category,
            observed_transition_changed=path['vs_noop_observed_critical_transitions_changed']))
    return result


def aggregate_prediction_changes(cases):
    """Equal design weight within each category; no causal/path identity claim."""
    keys = sorted({(row['mode'], row['endpoint_change']) for case in cases
                   for row in case['analysis'].get('prediction_samples', [])})
    result = []
    for mode, category in keys:
        per_design = [[row for row in case['analysis'].get('prediction_samples', [])
                       if row['mode'] == mode and row['endpoint_change'] == category] for case in cases]
        nonempty = [rows for rows in per_design if rows]
        errors, directions = [], []
        for rows in nonempty:
            values = [row['normalized_absolute_error'] for row in rows if row['normalized_absolute_error'] is not None]
            if values:
                errors.append(statistics.mean(values))
            values = [int(row['direction_correct']) for row in rows if row['actual_change_exceeds_tolerance']]
            if values:
                directions.append(statistics.mean(values))
        result.append({'mode': mode, 'endpoint_change': category, 'design_count': len(nonempty),
            'sample_count': sum(map(len, nonempty)), 'normalizable_designs': len(errors),
            'mean_per_design_normalized_mae': statistics.mean(errors) if errors else None,
            'designs_with_actual_delay_change': len(directions),
            'mean_per_design_direction_accuracy': statistics.mean(directions) if directions else None})
    return result

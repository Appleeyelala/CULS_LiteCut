"""Three-method statistics with explicit missing/zero denominators."""
from collections import Counter, defaultdict
import math
import statistics


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def describe(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return {'n': 0, 'median': None, 'p90': None, 'sum': None}
    position = .9 * (len(values) - 1)
    left = int(position)
    p90 = values[left] + (position - left) * (values[min(left + 1, len(values) - 1)] - values[left])
    return {'n': len(values), 'median': statistics.median(values), 'p90': p90, 'sum': math.fsum(values)}


def geometric(values):
    usable = [v for v in values if v is not None and math.isfinite(v) and v > 0]
    return {'n_positive_defined': len(usable),
            'value': math.exp(math.fsum(map(math.log, usable)) / len(usable)) if usable else None}


def relation(left, right, absolute, relative):
    if left is None or right is None:
        return None
    tolerance = max(absolute, relative * max(abs(left), abs(right)))
    return 'tie' if abs(left - right) <= tolerance else 'better' if left < right else 'worse'


def paired(records, protocol):
    grouped = defaultdict(dict)
    for row in records:
        if row['method'] in grouped[row['index']]:
            raise ValueError('Duplicate terminal method point.')
        grouped[row['index']][row['method']] = row
    comparisons = {}
    for baseline in ['A', 'B']:
        points, grid = [], Counter()
        common = 0
        for index, methods in sorted(grouped.items()):
            ours, other = methods.get('Ours'), methods.get(baseline)
            if not ours or not other or ours['status'] != 'PASS' or other['status'] != 'PASS':
                continue
            dc, tc = protocol['delay_comparison'], protocol['runtime_comparison']
            dr = relation(ours['delay_ps'], other['delay_ps'], dc['absolute_tolerance_ps'], dc['relative_tolerance'])
            tr = relation(ours['end_to_end_runtime_s'], other['end_to_end_runtime_s'], tc['absolute_tolerance_s'], tc['relative_tolerance'])
            delay_ratio = ratio(ours['delay_ps'], other['delay_ps'])
            point = {'index': index, 'delay_relation': dr, 'runtime_relation': tr,
                     'delay_ratio': delay_ratio, 'improvement_percent': None if delay_ratio is None else 100 * (1 - delay_ratio),
                     'runtime_ratio_baseline_over_ours': ratio(other['end_to_end_runtime_s'], ours['end_to_end_runtime_s']),
                     'area_ratio_ours_over_baseline': ratio(ours['area'], other['area'])}
            points.append(point)
            if (set(methods) == {'A', 'B', 'Ours'} and
                    all(r['status'] == 'PASS' and r.get('area_feasible') is True for r in methods.values())):
                common += 1
                grid[dr + '_delay__' + tr + '_time'] += 1
        improvements = describe([r['improvement_percent'] for r in points])
        comparisons[baseline] = {'successful_pairs': len(points),
            'delay_win_tie_loss': dict(Counter(p['delay_relation'] for p in points)),
            'improvement_percent': improvements,
            'delay_ratio_geometric': geometric([p['delay_ratio'] for p in points]),
            'paired_runtime_ratio': describe([p['runtime_ratio_baseline_over_ours'] for p in points]),
            'paired_area_ratio': describe([p['area_ratio_ours_over_baseline'] for p in points]),
            'common_three_valid_feasible_pairs': common, 'common_feasible_relation_grid': dict(grid),
            'points': points}
    return comparisons


def summarize(records, protocol):
    expected = {'A', 'B', 'Ours'}
    if any(r['method'] not in expected or r['status'] not in {'PASS', 'FAIL', 'TIMEOUT'} for r in records):
        raise ValueError('Unexpected method or nonterminal state.')
    comparisons = paired(records, protocol)
    methods = {}
    for method in ['A', 'B', 'Ours']:
        rows = [r for r in records if r['method'] == method]
        successes = [r for r in rows if r['status'] == 'PASS']
        stages = defaultdict(list)
        for r in rows:
            for name, value in r.get('stage_seconds', {}).items():
                stages[name].append(value)
        methods[method] = {'attempted': len(rows), 'states': dict(Counter(r['status'] for r in rows)),
            'runtime_unavailable': sum(r.get('end_to_end_runtime_s') is None for r in rows),
            'runtime_all_terminal': describe([r['end_to_end_runtime_s'] for r in rows]),
            'runtime_successful': describe([r['end_to_end_runtime_s'] for r in successes]),
            'delay_ps': describe([r.get('delay_ps') for r in successes]),
            'area': describe([r.get('area') for r in successes]),
            'area_feasible': sum(r.get('area_feasible') is True for r in successes),
            'area_infeasible': sum(r.get('area_feasible') is False for r in successes),
            'area_reference_unavailable': sum(r.get('area_feasible') is None for r in successes),
            'final_sources': dict(Counter(r.get('final_source') for r in successes)),
            'gradient_devices': dict(Counter(r.get('device') for r in successes)),
            'inclusive_stages_s': {name: describe(values) for name, values in stages.items()},
            'cec_commands': sum(r.get('cec_commands', 0) for r in rows),
            'mapping_commands': sum(r.get('mapping_commands', 0) for r in rows),
            'replay_commands': sum(r.get('replay_commands', 0) for r in rows),
            'failed_requests': sum(r.get('failed_requests', 0) or 0 for r in rows),
            'repair_requests': sum(r.get('repair_requests', 0) or 0 for r in rows)}
        methods[method]['incomplete_internal_requests'] = sum(r.get('incomplete_internal_requests', 0) for r in rows)
        methods[method]['unreadable_partial_command_files'] = sum(len(r.get('unreadable_partial_command_files', [])) for r in rows)
    return {'designs': len({r['index'] for r in records}), 'method_points': len(records), 'methods': methods,
            'comparisons': comparisons,
            'known_no_candidate_designs': len({r['index'] for r in records if r['method'] == 'Ours' and r.get('decisions') == 0}),
            'unknown_candidate_count_designs': len({r['index'] for r in records if r['method'] == 'Ours' and r.get('decisions') is None}),
            'scope': 'A/B relative timing is an overall method comparison, not causal GPU acceleration. All-terminal runtime includes failures/timeouts. Inclusive tool/stage durations may overlap; geometric statistics have separate positive denominators.'}

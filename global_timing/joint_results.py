"""Descriptive final-panel statistics; no EDA, model fitting or new labels.

The protocol gate uses the originally specified seed-pair distribution.
Design-level summaries additionally keep repeated seeds from masquerading as
independent circuits. Missing/infeasible outcomes never acquire delay values.
"""
from collections import Counter
import math
import statistics


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def ratio(left, right):
    if not finite_nonnegative(left) or not finite_nonnegative(right):
        raise ValueError('Ratios require finite nonnegative measured values.')
    if right == 0:
        return 1.0 if left == 0 else None  # explicitly unbounded, not dropped silently
    return left / right


def geomean(values):
    if not values:
        return None
    if any(v is None for v in values):
        return None
    if any(not finite_nonnegative(v) for v in values):
        raise ValueError('Non-finite values cannot enter a reported geometric mean.')
    return 0.0 if any(v == 0 for v in values) else math.exp(statistics.mean(math.log(v) for v in values))


def percentile(values, quantile):
    if not values or any(v is None for v in values):
        return None
    ordered = sorted(values)
    p = quantile * (len(ordered) - 1)
    left = int(p)
    return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (p - left)


def matrix(rows, panel, protocol):
    expected = {(r['index'], seed, variant) for r in panel['rows']
                for seed in protocol['seeds'] for variant in protocol['variants']}
    lookup = {}
    names = {r['index']: r for r in panel['rows']}
    for row in rows:
        key = row['index'], row['seed'], row['variant']
        if key not in expected or key in lookup:
            raise ValueError('Unexpected or duplicate final-panel method point.')
        if row['status'] not in ('PASS', 'FAIL'):
            raise ValueError('Only terminal method outcomes may enter final statistics.')
        if row['input'] != names[row['index']]['relative_path']:
            raise ValueError('Final method input name differs from the frozen panel.')
        if row['status'] == 'PASS':
            if row.get('input_sha256') != names[row['index']]['sha256']:
                raise ValueError('Final method input SHA differs from the frozen panel.')
            if row.get('timing_model') != 'ABC_SCL_LIBERTY':
                raise ValueError('Final statistics require the common Liberty timing contract.')
            if row.get('installation_gain_policy', 'positive_gain') != protocol.get('installation_gain_policy', 'positive_gain'):
                raise ValueError('Successful point uses a different installation policy.')
            if not finite_nonnegative(row['delay_ps']) or not finite_nonnegative(row['area']):
                raise ValueError('Successful method has invalid actual timing/area.')
            if not finite_nonnegative(row['end_to_end_with_input_staging_s']) or row['end_to_end_with_input_staging_s'] == 0:
                raise ValueError('Successful method has no valid complete runtime.')
        lookup[key] = row
    if set(lookup) != expected:
        raise ValueError('Final-panel coverage is incomplete; do not summarize a prefix as final.')
    return lookup


def outcome(left, right, tolerance=2e-5):
    return 'win' if left < right - tolerance else 'loss' if left > right + tolerance else 'tie'


def paired(lookup, panel, protocol, proposed, control):
    pairs = []
    for design in panel['rows']:
        for seed in protocol['seeds']:
            a, b = lookup[design['index'], seed, proposed], lookup[design['index'], seed, control]
            item = {'index': design['index'], 'input': design['relative_path'], 'seed': seed,
                    'proposed': proposed, 'control': control}
            if a['status'] != 'PASS' or b['status'] != 'PASS':
                item['status'] = 'FAILED_METHOD'
            elif not a.get('area_feasible_1p05') or not b.get('area_feasible_1p05'):
                item['status'] = 'AREA_INFEASIBLE_OR_REFERENCE_MISSING'
            else:
                job = a.get('slurm_job_id')
                shared = bool(job) and job == b.get('slurm_job_id')
                recipe = all(a.get(k) is not None and a.get(k) == b.get(k)
                             for k in ('algorithm_sha256', 'checkpoint_sha256', 'domain_sha256'))
                configs = [{k: v for k, v in (r.get('joint_proposal_config') or {}).items() if k != 'device'} for r in (a, b)]
                calls = [r.get('stages', {}).get('gradient_proposals', {}).get('calls') for r in (a, b)]
                same_work = (recipe and configs[0] == configs[1] and a.get('torch_num_threads') == b.get('torch_num_threads')
                             and calls[0] is not None and calls[0] == calls[1])
                same_eda = all(a.get(k) is not None and a.get(k) == b.get(k)
                               for k in ('native_replay_commands', 'cec_commands', 'mapping_commands'))
                item.update(status='MEASURED', delay_ratio=ratio(a['delay_ps'], b['delay_ps']),
                    delay_outcome=outcome(a['delay_ps'], b['delay_ps']),
                    proposed_delay_ps=a['delay_ps'], control_delay_ps=b['delay_ps'],
                    shared_allocation=shared,
                    same_algorithm_and_inner_work=same_work, same_eda_call_counts=same_eda,
                    matched_consumed_requests=a['requests'] == b['requests'],
                    speed_ratio=b['end_to_end_with_input_staging_s'] / a['end_to_end_with_input_staging_s'] if shared else None,
                    primary_cuda_used='cuda' in (a.get('proposal_devices') or []),
                    same_request_sequence=bool(a.get('request_sequence_sha256')) and a.get('request_sequence_sha256') == b.get('request_sequence_sha256'),
                    same_final_choice=bool(a.get('best_choices_sha256')) and a.get('best_choices_sha256') == b.get('best_choices_sha256'))
            pairs.append(item)
    valid = [p for p in pairs if p['status'] == 'MEASURED']
    values = [p['delay_ratio'] for p in valid]
    speeds = [p['speed_ratio'] for p in valid if p['speed_ratio'] is not None]
    designs = []
    for design in panel['rows']:
        selected = [p for p in valid if p['index'] == design['index']]
        designs.append({'index': design['index'], 'input': design['relative_path'],
            'measured_seed_pairs': len(selected), 'complete_seeds': len(selected) == len(protocol['seeds']),
            'delay_ratio_geomean': geomean([p['delay_ratio'] for p in selected]),
            'median_shared_allocation_speed': statistics.median([p['speed_ratio'] for p in selected if p['speed_ratio'] is not None])
                if any(p['speed_ratio'] is not None for p in selected) else None})
    return {'proposed': proposed, 'control': control, 'expected_seed_pairs': len(pairs),
        'measured_area_feasible_seed_pairs': len(valid), 'status_counts': dict(Counter(p['status'] for p in pairs)),
        'outcomes': dict(Counter(p['delay_outcome'] for p in valid)),
        'geomean_delay_ratio_seed_pairs': geomean(values), 'p90_delay_ratio_seed_pairs': percentile(values, .9),
        'unbounded_delay_ratios': sum(v is None for v in values),
        'median_shared_allocation_speed': statistics.median(speeds) if speeds else None,
        'complete_designs': sum(d['complete_seeds'] for d in designs), 'designs': designs, 'pairs': pairs}


def cutoff_comparison(lookup, panel, protocol, proposed, control, cutoff):
    pairs, ratios = [], []
    for design in panel['rows']:
        for seed in protocol['seeds']:
            a, b = lookup[design['index'], seed, proposed], lookup[design['index'], seed, control]
            left = a.get(f'best_at_{cutoff}s_ps') if a['status'] == 'PASS' else None
            right = b.get(f'best_at_{cutoff}s_ps') if b['status'] == 'PASS' else None
            if left is None or right is None:
                label = 'neither_available' if left is None and right is None else 'control_only' if left is None else 'proposed_only'
            else:
                label = outcome(left, right)
                ratios.append(ratio(left, right))
            pairs.append({'index': design['index'], 'seed': seed, 'outcome': label,
                          'proposed_delay_ps': left, 'control_delay_ps': right})
    return {'cutoff_s': cutoff, 'expected_seed_pairs': len(pairs),
            'outcomes': dict(Counter(p['outcome'] for p in pairs)),
            'ratio_geomean_when_both_available': geomean(ratios), 'pairs': pairs,
            'scope': 'Only actually completed area-feasible measurements; absence by a cutoff is explicit, not a no-op or infinity label.'}


def summarize(rows, panel, protocol):
    lookup = matrix(rows, panel, protocol)
    primary = protocol['primary_variant']
    controls = [v for v in protocol['variants'] if v != primary]
    comparisons = {control: paired(lookup, panel, protocol, primary, control) for control in controls}
    # CPU primary versus true non-gradient control separates the GPU policy.
    cpu_gradient = paired(lookup, panel, protocol, 'adaptive_batch_cpu', 'random_batch')
    gpu = comparisons['adaptive_batch_cpu']
    gpu_valid = [p for p in gpu['pairs'] if p['status'] == 'MEASURED' and p['primary_cuda_used']]
    gpu_identical = [p for p in gpu_valid if p['same_request_sequence'] and p['shared_allocation']
                     and p['same_algorithm_and_inner_work'] and p['same_eda_call_counts']]
    cpu_only = [p for p in gpu['pairs'] if p['status'] == 'MEASURED' and not p['primary_cuda_used']]
    gpu_summary = {'actual_cuda_seed_pairs': len(gpu_valid),
        'actual_cuda_designs': len({p['index'] for p in gpu_valid}),
        'identical_sequence_shared_allocation_seed_pairs': len(gpu_identical),
        'identical_sequence_shared_allocation_designs': len({p['index'] for p in gpu_identical}),
        'median_end_to_end_speed_identical_cuda_pairs': statistics.median([p['speed_ratio'] for p in gpu_identical]) if gpu_identical else None,
        'cpu_or_no_proposal_seed_pairs': len(cpu_only), 'pairs': gpu['pairs'],
        'scope': 'Only observed CUDA work, matching code/model/domain/config/call counts, identical request sequences and shared allocation enter pure GPU speed attribution. CPU-only/no-proposal and other mismatched rows remain explicit.'}
    methods = {}
    for variant in protocol['variants']:
        selected = [r for r in rows if r['variant'] == variant]
        successful = [r for r in selected if r['status'] == 'PASS']
        feasible = [r for r in successful if r.get('area_feasible_1p05')]
        directs = sum(r.get('direct_requests', 0) for r in successful)
        accepted = sum(r.get('direct_successes', 0) for r in successful)
        improving_direct = sum(p['source'] == 'joint_selection_gradient' and p['area'] <= 1.05 * r['noop_area'] + 1e-8
            and p['delay_ps'] < r['noop_delay_ps'] - 2e-5 for r in successful if 'noop_area' in r for p in r['points'])
        hits = [r['time_to_quality_s'] for r in successful if r.get('time_to_quality_s') is not None]
        runtime = [r['end_to_end_with_input_staging_s'] for r in successful]
        methods[variant] = {'expected_method_points': len(selected), 'successful_points': len(successful),
            'area_feasible_final_points': len(feasible), 'failed_method_points': len(selected) - len(successful),
            'requests': sum(r.get('requests', 0) for r in successful),
            'failed_requests_in_successful_methods': sum(r.get('failed_requests', 0) for r in successful),
            'direct_requests': directs, 'successful_direct_requests': accepted,
            'area_feasible_direct_points_better_than_noop': improving_direct,
            'final_sources': dict(Counter(r.get('final_source', 'unknown') for r in successful)),
            'median_end_to_end_s_successful_methods': statistics.median(runtime) if runtime else None,
            'quality_target_reached': len(hits), 'quality_target_not_reached_or_failed': len(selected) - len(hits),
            'median_time_to_quality_among_hits_s': statistics.median(hits) if hits else None,
            'native_replay_commands': sum(r.get('native_replay_commands', 0) for r in successful),
            'cec_commands': sum(r.get('cec_commands', 0) for r in successful),
            'mapping_commands': sum(r.get('mapping_commands', 0) for r in successful)}
    legacy = comparisons['legacy_cpu']
    complete = legacy['measured_area_feasible_seed_pairs'] == legacy['expected_seed_pairs']
    geometry, p90 = legacy['geomean_delay_ratio_seed_pairs'], legacy['p90_delay_ratio_seed_pairs']
    limits = protocol['qor_noninferiority']
    if not complete:
        qor_gate = 'INCOMPLETE'
    else:
        qor_gate = ('MET_ON_THIS_PANEL' if geometry is not None and p90 is not None
            and geometry <= limits['geometric_mean_delay_ratio_to_legacy_max'] + 1e-12
            and p90 <= limits['p90_delay_ratio_to_legacy_max'] + 1e-12 else 'NOT_MET_ON_THIS_PANEL')
    speed = legacy['median_shared_allocation_speed']
    runtime_gate = 'INCOMPLETE' if not complete or speed is None else 'MET_ON_THIS_PANEL' if speed >= 1.5 else 'NOT_MET_ON_THIS_PANEL'
    equal_time = {control: [cutoff_comparison(lookup, panel, protocol, primary, control, cut)
                 for cut in protocol['cutoffs_s']] for control in ('legacy_cpu', 'random_batch', 'native_abc', 'native_culs')}
    groups = {}
    for group in sorted({tuple(r['and_bin']) for r in panel['rows']}):
        subpanel = dict(panel, rows=[r for r in panel['rows'] if tuple(r['and_bin']) == group])
        groups[str(group)] = paired(lookup, subpanel, protocol, primary, 'legacy_cpu')
    return {'coverage': {'designs': len(panel['rows']), 'seeds': protocol['seeds'],
                        'variants': protocol['variants'], 'method_points': len(rows)},
        'methods': methods, 'comparisons': comparisons, 'cpu_gradient_vs_random': cpu_gradient,
        'gpu': gpu_summary, 'equal_time': equal_time, 'scale_groups_vs_legacy': groups,
        'protocol_assessment': {'qor_noninferiority_vs_legacy': qor_gate,
            'median_speed_at_least_1p5_vs_legacy': runtime_gate,
            'equal_time_alternative': 'DESCRIPTIVE_ONLY; no aggregate binary threshold was prospectively specified',
            'scope': 'Thresholds are the frozen seed-pair protocol. Additional design-level summaries are descriptive, not independent significance tests.'},
        'limitations': ['Already analyzed frozen panel; not held-out generalization.',
            'Three seeds are repeated runs of each design, not three independent circuits.',
            'No-inner control retains surrogate calibration and the large-branch adjoint.',
            'Training loss, local Student predictions and actual Liberty timing are different quantities.',
            'Source/failure counts describe correlated adaptive observations; no causal attribution of all gains to gradients.',
            'Time-to-quality median among hits is conditional; retain the full reached/not-reached denominator.',
            'Tool-call totals currently cover successful method processes; failed processes remain explicit omissions, never zero-cost successes.']}

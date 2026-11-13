"""Descriptive delivery supplements; no optimizer, timing or experiment calls."""
from collections import Counter

from .mffc_results import describe


def named_tradeoffs(grid):
    """Named views of the frozen nine-cell grid; first two may overlap."""
    return {
        'delay_better_and_not_slower': grid.get('better_delay__better_time', 0) + grid.get('better_delay__tie_time', 0),
        'faster_and_delay_not_worse': grid.get('better_delay__better_time', 0) + grid.get('tie_delay__better_time', 0),
        'faster_but_quality_worse': grid.get('worse_delay__better_time', 0),
        'slower_but_quality_better': grid.get('better_delay__worse_time', 0),
    }


def installation_details(requests):
    """Keep full installation, partial work and final measurement distinct.

    Each request contains optional saved observation/replay dictionaries.
    A replay PASS followed by a failed CEC/map is still a full installation,
    but it is never a successful measured quality observation.
    """
    states, rejections = Counter(), Counter()
    nonempty_pass, accepted_full, accepted_partial, unknown_gain = 0, 0, 0, 0
    failed_times, repair_times = [], []
    missing_observations = 0
    for request in requests:
        replay, observation = request.get('replay'), request.get('observation')
        status = replay.get('status') if replay else 'UNKNOWN'
        status = status if status in {'PASS', 'FAIL'} else 'UNKNOWN'
        states[status] += 1
        if replay:
            requested = replay.get('requested', [])
            applied = replay.get('applied_rewrites')
            if status == 'PASS':
                if applied != len(requested) or replay.get('rejected_requests'):
                    raise ValueError('A PASS replay is not a complete installation.')
                nonempty_pass += bool(requested)
                accepted_full += applied
            elif applied is not None:
                accepted_partial += applied
            for rejected in replay.get('rejected_requests', []):
                rejections[rejected['status']] += 1
            if applied is None:
                unknown_gain += 1
        else:
            unknown_gain += 1
        if observation is None:
            missing_observations += 1
            continue
        if observation['status'] == 'FAIL':
            if observation.get('metrics'):
                raise ValueError('Failed observation must not supply a quality label.')
            failed_times.append(observation.get('runtime_s'))
        prediction = observation.get('prediction_before_evaluation') or {}
        if prediction.get('source') == 'explicit_feasibility_repair':
            repair_times.append(observation.get('runtime_s'))
    return {'replay_requests_with_directory': len(requests), 'installation_states': dict(states),
            'nonempty_complete_installations': nonempty_pass,
            'accepted_rewrites_in_complete_installations': accepted_full,
            'accepted_rewrites_in_failed_partial_installations': accepted_partial,
            'replay_applied_count_unavailable': unknown_gain,
            'rejection_statuses': dict(rejections), 'missing_terminal_observations': missing_observations,
            'failed_request_inclusive_s': describe(failed_times), 'repair_request_inclusive_s': describe(repair_times),
            'scope': 'PASS includes no-op; nonempty count excludes it. Partial installed work has no quality label. Request times may overlap shared native replay and concurrent validation; sums are not exclusive wall fractions.'}


def shape_summary(records):
    """Recorded GPU costs by structural shape, with explicit scope counts."""
    rows = [r for r in records if r['method'] == 'Ours']
    fields = ['regions', 'region_ports', 'region_dag_layers', 'max_layer_width', 'decisions',
              'candidates', 'native_forward_calls', 'native_backward_calls', 'native_kernel_ms', 'end_to_end_runtime_s']
    def group(data):
        return {'method_points': len(data), 'status_counts': dict(Counter(r['status'] for r in data)),
                'recorded': {field: describe([r.get(field) for r in data]) for field in fields}}
    devices = {device: group([r for r in rows if r.get('device', 'unknown') == device])
               for device in sorted({r.get('device', 'unknown') for r in rows})}
    cuda = [r for r in rows if r.get('device') == 'cuda' and r.get('native_forward_calls', 0) > 0]
    bands = {}
    for field, edges in [('region_dag_layers', [0, 1, 32, 512, 4096]),
                         ('max_layer_width', [0, 32, 512, 8192])]:
        bands[field] = {}
        for i, low in enumerate(edges):
            high = edges[i + 1] if i + 1 < len(edges) else None
            selected = [r for r in cuda if r.get(field) is not None and r[field] >= low
                        and (high is None or r[field] < high)]
            bands[field][str(low) + '_' + (str(high) if high is not None else 'plus')] = group(selected)
    return {'devices': devices, 'cuda_shape_bands': bands, 'observed_cuda_method_points': len(cuda),
            'scope': 'Descriptive structural groups, not controlled GPU/CPU experiments. Native CUDA event time excludes tensor aggregation/context/EDA, includes all saved forward/backward calls with possibly different batch sizes; cannot be a whole-method acceleration ratio.'}


def tool_cost_summary(records):
    """Account for saved real tool durations, including failed method points.

    Missing command clocks are absent observations, never free computation.
    Each duration is a per-method-point sum of that tool's recorded commands;
    concurrent validations may overlap, so this is not an exclusive wall share.
    """
    categories = {'cec': 'cec_commands', 'mapping': 'mapping_commands',
                  'rewrite': 'replay_commands', 'native_rewrite': 'abc_native_rewrite_commands'}
    methods = {}
    for method in ('A', 'B', 'Ours'):
        rows = [r for r in records if r['method'] == method]
        tools = {}
        for category, count_field in categories.items():
            counts = [r.get(count_field) for r in rows]
            clocks = [r.get('tool_seconds_inclusive', {}).get(category) for r in rows]
            tools[category] = {
                'recorded_command_counts': describe(counts),
                'per_point_inclusive_s': describe(clocks),
                'points_with_recorded_commands_but_no_clock': sum(
                    count is not None and count > 0 and clock is None
                    for count, clock in zip(counts, clocks)),
            }
        methods[method] = {
            'terminal_method_points': len(rows),
            'status_counts': dict(Counter(r['status'] for r in rows)),
            'cold_wall_s': describe([r.get('end_to_end_runtime_s') for r in rows]),
            'unreadable_command_files': sum(len(r.get('unreadable_partial_command_files', [])) for r in rows),
            'tools': tools,
        }
    return {'methods': methods,
            'scope': 'Recorded real tool commands and per-point inclusive sums, including failed/timed-out methods. Missing clocks are excluded with n, not replaced by zero. Counts can be incomplete when command files are unreadable. Tool/stage sums may overlap and are not exclusive percentages of cold wall.'}

#!/usr/bin/env python3
"""Audit saved real evidence and summarize exactly the frozen A/B/Ours cohort."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='New /data analysis run.')
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    run, experiment = args.run.resolve(), args.experiment.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not all(p.is_relative_to('/data/wushangge') for p in [run, experiment]):
        parser.error('Analyze in a committed /data Slurm workspace.')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from global_timing.common import save_json
    from global_timing.evaluation import cec_passed, parse_timing
    from global_timing.mffc_results import paired, summarize
    protocol = json.loads(args.protocol.read_text())
    preflight = json.loads((experiment / 'input/mffc905_preflight.json').read_text())
    if not args.development and preflight['protocol'] != protocol:
        raise ValueError('Formal analysis protocol differs from the frozen executed protocol.')
    inputs = {r['index']: r for r in preflight['rows']}
    completion_file = experiment / 'output' / ('development_completion.json' if args.development else 'formal_completion.json')
    completion = json.loads(completion_file.read_text())
    if (completion['status'] != 'TERMINAL_ALL_REQUESTED' or completion['method_points'] != 3 * len(inputs)
            or (not args.development and (len(inputs) != 905 or not completion['formal'] or protocol['state'] != 'FROZEN'))):
        raise ValueError('Missing complete terminal coverage for the claimed scope.')
    output = run / 'output' / ('development_' + experiment.name if args.development else 'benchmark905_delivery')
    output.mkdir(exist_ok=False)
    hashes, records = {}, []
    counts = {'actual_evaluations': 0, 'successful_replay_observations': 0, 'failed_requests': 0}
    def read(path):
        path = Path(path)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    def evaluation(work, metrics, input_row, candidate):
        raw = read(work / 'result.json')
        if raw != metrics or raw['status'] != 'PASS' or raw['cec'] != 'PASS':
            raise ValueError('Saved real metrics mismatch: ' + str(work))
        cec = work / 'cec.log'
        if not cec_passed(cec.read_text()):
            raise ValueError('No explicit raw CEC proof: ' + str(cec))
        parsed = parse_timing((work / 'mapping.log').read_text())
        for key in parsed:
            if parsed[key] != raw[key]:
                raise ValueError('Raw Liberty disagreement: ' + str(work) + ' / ' + key)
        for path, expected in [(Path(candidate), raw['input_sha256']), (work / 'mapped.v', raw['mapped_sha256'])]:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashes[str(path)] != expected:
                raise ValueError('Actual candidate/netlist bytes changed: ' + str(path))
        if raw['provenance']['original']['sha256'] != input_row['sha256']:
            raise ValueError('Actual CEC used a different original input.')
        for name in ['abc', 'library']:
            if raw['provenance'][name]['sha256'] != preflight['tools'][name]['sha256']:
                raise ValueError('Actual timing tool/library differs from frozen identity.')
        for path in [cec, work / 'mapping.log', work / 'cec.command.json', work / 'mapping.command.json']:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts['actual_evaluations'] += 1
    def audit_ours(work, result, input_row):
        evaluation(work / 'original', result['original'], input_row, work.parent / 'input.aig')
        catalogue = read(work / 'cut_validation/verified_catalogue.json')['decisions']
        for observation in result['observations']:
            if observation['status'] != 'PASS':
                if observation.get('metrics') is not None:
                    raise ValueError('Failed request acquired a timing label.')
                counts['failed_requests'] += 1
                continue
            index = observation['index']
            request = work / ('replay_noop' if index == 0 else 'optimization') / f'eval_{index:04d}'
            replay = read(request / 'rewrite/replay.json')
            wanted = {(d['root'], d['cut_slot'], d['candidate_subgraph_indices'][c - 1])
                      for d, c in zip(catalogue, observation['choices']) if c}
            if len(observation['choices']) != len(catalogue) or wanted != {tuple(r) for r in replay['requested']}:
                raise ValueError('Requested library choices differ from the measured vector.')
            trace_path = request / 'rewrite/raw/traces/choice.jsonl'
            selected = []
            with trace_path.open() as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        if row['preselected']:
                            selected.append(row)
            got = {(r['root'], r['cut_slot'], r['selected_subgraph']) for r in selected}
            if (got != wanted or len(selected) != len(wanted) or replay['applied_rewrites'] != len(wanted)
                    or any(r['final_status'] != 'ACCEPTED' for r in selected)):
                raise ValueError('Partial/substituted installation in a successful observation.')
            hashes[str(trace_path)] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            prediction = observation.get('prediction_before_evaluation') or {}
            if any(i >= index for i in prediction.get('training_observation_indices', [])):
                raise ValueError('Prediction used a current/future observation index.')
            evaluation(request / 'evaluation', observation['metrics'], input_row, observation['aig'])
            counts['successful_replay_observations'] += 1
        selected = result['selected']
        if selected['source'] == 'original_fallback':
            if selected['metrics'] != result['original'] or selected['choices'] is not None:
                raise ValueError('Original fallback substituted another measurement.')
        else:
            best = next(r for r in result['observations'] if r['index'] == selected['observation_index'])
            if best['status'] != 'PASS' or best['choices'] != selected['choices'] or best['metrics'] != selected['metrics']:
                raise ValueError('Selected final is not its successful real observation.')
        if selected['metrics']['area'] > result['replay_noop']['metrics']['area'] * 1.05 + 1e-8:
            raise ValueError('Selected actual area violates the declared bound.')
    seen = set()
    for path in sorted((experiment / 'output/records').glob('*.json')):
        point = read(path)
        key = (point['index'], point['method'])
        if key in seen or point['index'] not in inputs or point['method'] not in ['A', 'B', 'Ours']:
            raise ValueError('Unexpected or duplicate terminal record.')
        seen.add(key)
        input_row = inputs[point['index']]
        if point['input_sha256'] != input_row['sha256'] or point['status'] not in ['PASS', 'FAIL', 'TIMEOUT']:
            raise ValueError('Wrong input or nonterminal record.')
        work = Path(point['detail_path']) / 'work'
        record = {**{k: point[k] for k in ['index', 'input', 'input_sha256', 'method', 'status', 'error', 'source_commit', 'slurm_job_id', 'end_to_end_runtime_s']},
                  'source': input_row['relative_path'].split('/')[0], 'and_count': input_row['and_count'],
                  'pi_count': input_row['pi_count'], 'po_count': input_row['po_count'], 'detail_path': str(work),
                  'delay_ps': None, 'area': None, 'cec': None, 'final_source': None, 'area_feasible': None}
        commands, seconds = Counter(), Counter()
        for command_path in work.rglob('*.command.json'):
            command = read(command_path)
            category = command_path.name.removesuffix('.command.json')
            commands[category] += 1
            seconds[category] += command['elapsed_s']
        record.update(cec_commands=commands['cec'], mapping_commands=commands['mapping'], replay_commands=commands['rewrite'],
                      abc_native_rewrite_commands=commands['native_rewrite'],
                      tool_seconds_inclusive=dict(seconds))
        if point['status'] == 'PASS':
            result = read(point['result_path'])
            if result['tools'] != preflight['tools'] or result['source_commit'] != preflight['source_commit']:
                raise ValueError('Recorded method tools/source differ from preflight.')
            if point['method'] == 'Ours':
                audit_ours(work, result, input_row)
                for name in ['decisions', 'candidates', 'selected_original_mffcs', 'failed_requests', 'repair_requests',
                             'direct_requests', 'requests_including_noop_failed_repair', 'gradient_rounds']:
                    record[name] = result[name]
                record.update(result['structure'])
                stage_seconds = Counter()
                for stage in result['stages']:
                    stage_seconds[stage['stage']] += stage['elapsed_s']
                record['stage_seconds'] = dict(stage_seconds)
                profiles = result['native_profiles']
                record['native_forward_calls'] = profiles[-1]['calls_cumulative']['forward'] if profiles else 0
                record['native_backward_calls'] = profiles[-1]['calls_cumulative']['backward'] if profiles else 0
                record['native_kernel_ms'] = sum(sum(p['kernel_ms_since_collect'].values()) for p in profiles)
                record['noop_area'], record['noop_delay_ps'] = result['replay_noop']['metrics']['area'], result['replay_noop']['metrics']['delay_ps']
                record['accepted_rewrites'] = result['selected']['applied_rewrites']
                choices = result['selected']['choices']
                record['selected_choices_sha256'] = hashlib.sha256(json.dumps(choices).encode()).hexdigest() if choices is not None else None
            else:
                evaluation(work / 'evaluation', result['selected']['metrics'], input_row, result['selected']['aig'])
            record.update(delay_ps=result['selected']['metrics']['delay_ps'], area=result['selected']['metrics']['area'],
                          cec=result['selected']['metrics']['cec'], final_source=result['selected']['source'],
                          device=result['backend']['device'], device_reason=result['backend']['reason'])
        records.append(record)
    expected = {(i, m) for i in inputs for m in ['A', 'B', 'Ours']}
    if seen != expected:
        raise ValueError('The requested full cohort does not have exactly all terminal method points.')
    grouped = defaultdict(dict)
    for r in records:
        grouped[r['index']][r['method']] = r
    for methods in grouped.values():
        ours = methods['Ours']
        reference = ours.get('noop_area')
        for r in methods.values():
            if r['status'] == 'PASS' and reference is not None:
                r['area_feasible'] = r['area'] <= reference * 1.05 + 1e-8
            if r['method'] != 'Ours':
                for name in ['regions', 'region_edges', 'region_ports', 'region_dag_layers', 'max_layer_width', 'decisions', 'candidates']:
                    r[name] = ours.get(name)
    summary = summarize(records, protocol)
    summary.update(formal=not args.development, experiment=str(experiment), source_commit=preflight['source_commit'],
                   analysis_source_commit=json.loads((run / 'run.json').read_text())['source_commit'],
                   suite_completion=completion, audit=counts, protocol=protocol)
    summary['by_source'] = {source: summarize([r for r in records if r['source'] == source], protocol)
                            for source in sorted({r['source'] for r in records})}
    bins = protocol['and_size_bins']
    summary['by_and_size'] = {}
    for i, low in enumerate(bins):
        high = bins[i + 1] if i + 1 < len(bins) else None
        group = [r for r in records if r['and_count'] >= low and (high is None or r['and_count'] < high)]
        summary['by_and_size'][f'{low}_{high if high is not None else "plus"}'] = summarize(group, protocol)
    comparisons = paired(records, protocol)
    for baseline in ['A', 'B']:
        for pair in comparisons[baseline]['points']:
            r = grouped[pair['index']]['Ours']
            for name in ['improvement_percent', 'delay_ratio', 'runtime_ratio_baseline_over_ours', 'area_ratio_ours_over_baseline']:
                r[name + '_' + baseline] = pair[name]
    prefix = 'DEVELOPMENT' if args.development else 'BENCHMARK905'
    fields = sorted({k for r in records for k in r})
    with (output / (prefix + '_RESULTS.csv')).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in r.items()}
                         for r in records)
    save_json(output / (prefix + '_SUMMARY.json'), summary)
    save_json(output / 'audited_records.json', records)
    save_json(output / 'evidence_hashes.json', hashes)
    from global_timing.mffc_report import write_report
    write_report(output, prefix, records, summary)
    save_json(output / 'analysis_completion.json', {'status': 'AUDITED_AND_REPORTED', 'formal': not args.development,
              'inputs': len(inputs), 'method_points': len(records), 'audit': counts,
              'artifacts_sha256': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in output.rglob('*') if p.is_file()}})
    print(json.dumps({'status': 'AUDITED_AND_REPORTED', 'output': str(output), 'coverage': len(records), 'audit': counts}))


if __name__ == '__main__':
    main()

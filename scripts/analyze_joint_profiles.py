#!/usr/bin/env python3
"""Audit completed joint profile controls and report real QoR/runtime points."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--profile-run', type=Path, action='append', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('Requires a prepared /data run and Slurm allocation.')
    from global_timing.common import digest, read_jsonl, save_json
    from global_timing.evaluation import cec_passed, parse_timing
    from global_timing.candidates import Catalogue
    output = run / 'output/profile_analysis'
    output.mkdir(exist_ok=False)
    metadata_path = run / 'run.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(state='RUNNING', experiment_executed=True, command=sys.argv,
                    slurm_job_id=os.environ['SLURM_JOB_ID'])
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    rows, inputs, signatures, parity = [], {}, {}, []
    try:
        for profile in args.profile_run:
            parent = profile / 'output/profile_summary.json'
            saved = json.loads(parent.read_text())
            if saved['status'] != 'PASS' or json.loads((profile / 'run.json').read_text())['state'] != 'COMPLETED':
                raise ValueError('Only completed, fully reported profile runs may be analyzed: ' + str(profile))
            inputs[str(parent)] = digest(parent)
            for record in saved['records']:
                detail = Path(record['detail'])
                if digest(detail) != record['detail_sha256']:
                    raise ValueError('Profile detail hash mismatch.')
                inputs[str(detail)] = digest(detail)
                item = json.loads(detail.read_text())
                if item['status'] != 'PASS' or record['returncode'] != 0:
                    raise ValueError('Failed child cannot become a successful data point.')
                case = detail.parent / f'case_{record["index"]:04d}'
                search_file = case / 'gradient_area_constraint/summary.json'
                search = json.loads(search_file.read_text())
                inputs[str(search_file)] = digest(search_file)
                best = search['best']
                if best['choices'] != item['selected']['choices'] or best['metrics'] != item['selected']['metrics']:
                    raise ValueError('Selected point differs between profile and search summary.')
                candidate = Path(best['aig'])
                request = candidate.parents[3]  # Keep request alias, not resolved shared batch path.
                result_path = request / 'evaluation/result.json'
                measured = json.loads(result_path.read_text())
                if measured['status'] != 'PASS' or measured['cec'] != 'PASS' or measured['timing_model'] != 'ABC_SCL_LIBERTY':
                    raise ValueError('Final result is not a genuine Liberty/CEC PASS.')
                if digest(candidate) != measured['input_sha256']:
                    raise ValueError('Final AIG differs from the evaluated input.')
                cec_log, mapping_log = request / 'evaluation/cec.log', request / 'evaluation/mapping.log'
                if not cec_passed(cec_log.read_text()):
                    raise ValueError('CEC raw log does not explicitly prove equivalence.')
                parsed = parse_timing(mapping_log.read_text())
                for name in ('delay_ps', 'area', 'po_arrivals_ps'):
                    if parsed[name] != measured[name] or measured[name] != best['metrics'][name]:
                        raise ValueError('Timing raw log, result and selected metrics disagree: ' + name)
                catalogue_path = case / 'cut_validation/verified_catalogue.json'
                stored = json.loads(catalogue_path.read_text())
                domain = Catalogue(stored['decisions'], [tuple(p) for p in stored['conflicts']],
                                   {int(k): v for k, v in stored['region_decisions'].items()})
                domain.validate(best['choices'])
                catalogue = domain.decisions
                requested = {(d['root'], d['cut_slot'], d['candidate_subgraph_indices'][c - 1])
                             for d, c in zip(catalogue, best['choices']) if c}
                trace_path = request / 'rewrite/raw/traces/choice.jsonl'
                trace = read_jsonl(trace_path)
                selected = [r for r in trace if r['preselected']]
                if len(selected) != len(requested) or {(r['root'], r['cut_slot'], r['selected_subgraph']) for r in selected} != requested:
                    raise ValueError('Final selected raw trace differs from exact requested candidates.')
                if any(r['final_status'] != 'ACCEPTED' for r in selected) or sum(r['final_status'] == 'ACCEPTED' for r in trace) != len(requested):
                    raise ValueError('Final requested combination was only partially installed.')
                for path in [result_path, candidate, cec_log, mapping_log, catalogue_path, trace_path]:
                    inputs[str(path)] = digest(path)
                observations = search['observations']
                direct = [o for o in observations if (o.get('prediction_before_evaluation') or {}).get('source') == 'joint_selection_gradient']
                signature = [(o['choices'], o['status'],
                              {k: o['metrics'][k] for k in ('delay_ps', 'area', 'po_arrivals_ps')} if o['status'] == 'PASS' else None)
                             for o in observations]
                signatures[str(profile), record['index'], record['variant']] = signature
                commands = Counter()
                for path in case.rglob('*.command.json'):
                    label = path.name.removesuffix('.command.json')
                    commands[label] += 1
                clock_valid, event_count = None, None
                if item.get('timeline'):
                    path = Path(item['timeline'])
                    if digest(path) != item['timeline_sha256']:
                        raise ValueError('Timeline hash mismatch.')
                    inputs[str(path)] = digest(path)
                    events = json.loads(path.read_text())['events']
                    start, wall = record['subprocess_started_monotonic'], record['full_process_wall_s']
                    if not all(start <= e['started_monotonic'] <= e['completed_monotonic'] <= start + wall for e in events):
                        raise ValueError('Event lies outside actual child process.')
                    requests = [e for e in events if e.get('kind') == 'request_completed']
                    if len(requests) != item['requests']:
                        raise ValueError('Missing actual request-completion events.')
                    clock_valid, event_count = True, len(requests)
                row = {'run': str(profile), 'index': record['index'], 'input': item['input_name'],
                       'variant': record['variant'], 'delay_ps': measured['delay_ps'], 'area': measured['area'],
                       'no_op_delay_ps': item['no_op']['metrics']['delay_ps'],
                       'full_process_wall_s': record['full_process_wall_s'],
                       'input_staging_s': record.get('input_staging_s'),
                       'end_to_end_with_input_staging_s': record.get('end_to_end_with_input_staging_s'),
                       'area_feasible_1p05': measured['area'] <= 1.05 * item['no_op']['metrics']['area'] + 1e-8,
                       'requests': len(observations), 'failed_requests': item['failed_requests'],
                       'direct_requested': len(direct), 'direct_successful': sum(o['status'] == 'PASS' for o in direct),
                       'final_source': (best.get('prediction_before_evaluation') or {}).get('source', 'replay_noop'),
                       'accepted_rewrites': len(requested),
                       'selected_original_mffcs': len({d['decision_owner_region'] for d, c in zip(catalogue, best['choices']) if c}),
                       'native_replay_commands': commands['rewrite'], 'cec_commands': commands['cec'],
                       'mapping_commands': commands['mapping'],
                       'timeline_valid': clock_valid, 'request_completion_events': event_count,
                       'litecut_cache_hit': item.get('litecut_cache_hit'), 'cuda_cache_regime': item.get('cuda_cache_regime'),
                       'actual_cec': 'PASS', 'timing_model': 'ABC_SCL_LIBERTY'}
                rows.append(row)
            for index in {r['index'] for r in saved['records']}:
                key = (str(profile), index, 'feedback_cpu')
                if key in signatures:
                    for variant in ('feedback_batch_cpu', 'feedback_batch_cuda'):
                        other = (str(profile), index, variant)
                        if other in signatures:
                            parity.append({'run': str(profile), 'index': index, 'control': 'feedback_cpu',
                                           'variant': variant, 'all_request_choices_status_metrics_equal': signatures[key] == signatures[other]})
        if not rows:
            raise ValueError('No complete profile data points were provided.')
        fields = list(rows[0])
        with (output / 'results.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        save_json(output / 'summary.json', {'status': 'PASS', 'rows': rows, 'engineering_parity': parity,
            'scope': 'Final raw AIG/exact acceptance/CEC/Liberty audit plus descriptive cold-process controls. Single trials are not stable speed distributions; no random/greedy superiority or independent gradient benefit is inferred.',
            'limitations': 'Staging is charged once per standalone method; scheduling/source archival excluded. Older runs without timelines cannot supply exact intermediate time-to-quality.'})
        save_json(output / 'input_hashes.json', inputs)
        header = '| 输入 | 方法 | Delay ps | Area | 冷进程 s | 请求/失败 | replay/CEC/map 调用 | 最终来源 |\n|---|---|---:|---:|---:|---|---|---|\n'
        table = ''.join(f'| {r["input"]} | {r["variant"]} | {r["delay_ps"]:.6f} | {r["area"]:.6f} | {r["full_process_wall_s"]:.3f} | {r["requests"]}/{r["failed_requests"]} | {r["native_replay_commands"]}/{r["cec_commands"]}/{r["mapping_commands"]} | {r["final_source"]} |\n' for r in rows)
        (output / 'REPORT.md').write_text('# 真实开发对照审计\n\n'+header+table+'\n所有最终点重新读取原始 trace、CEC 与统一 Liberty 日志核对；面积可行性和逐请求等价见 summary.json。单次计时、开发设计与失败点均保留，不能代替最终面板和消融。\n')
        metadata.update(state='COMPLETED', outputs=str(output / 'summary.json'))
    except Exception as exc:
        metadata.update(state='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()

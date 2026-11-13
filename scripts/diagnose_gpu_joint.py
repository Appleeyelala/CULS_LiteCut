#!/usr/bin/env python3
"""Inspect retained timings and graph duplicates; perform no EDA computation."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import digest, save_json


FAMILIES = {'c1355', 'ss_pcm', 'dma', 'div', 'hyp', 'voter',
            'back_end_with_decode', 'division'}


def family(name):
    stem = Path(name).stem.lower()
    for suffix in ('_orig', '_comb'):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def inspect_case(case):
    summary = json.loads((case / 'summary.json').read_text())
    cfg = json.loads((case / 'config.json').read_text())
    costs = defaultdict(lambda: {'calls': 0, 'elapsed_s': 0.0, 'failed': 0})
    sources = {}
    for path in sorted(case.rglob('*.command.json')):
        record = json.loads(path.read_text())
        relative = path.relative_to(case)
        phase = relative.parts[0]
        name = path.name.removesuffix('.command.json')
        slot = costs[phase + '/' + name]
        slot['calls'] += 1
        slot['elapsed_s'] += record['elapsed_s']
        slot['failed'] += int(bool(record.get('timed_out') or record.get('returncode')))
    graphs = defaultdict(list)
    for path in sorted(case.rglob('evaluation/result.json')):
        record = json.loads(path.read_text())
        if record['status'] == 'PASS':
            graphs[record['input_sha256']].append(str(path.relative_to(case)))
    # Original is evaluated directly in original/result.json.
    original = case / 'original/result.json'
    if original.exists():
        record = json.loads(original.read_text())
        if record['status'] == 'PASS':
            graphs[record['input_sha256']].append(str(original.relative_to(case)))
    gradient = {}
    for mode, result in summary['gradient_search'].items():
        observations = result['observations']
        direct = [o for o in observations
                  if (o.get('prediction_before_evaluation') or {}).get('source')
                  == 'joint_selection_gradient']
        best = result['best']
        gradient[mode] = {
            'requests': len(observations),
            'direct_requests': len(direct),
            'direct_successes': sum(o['status'] == 'PASS' for o in direct),
            'failed_requests': sum(o['status'] != 'PASS' for o in observations),
            'exclusive_method_runtime_s': result['exclusive_method_runtime_s'],
            'best_index': best['index'],
            'best_source': (best.get('prediction_before_evaluation') or {}).get('source', 'replay_noop'),
            'best_delay_ps': best['metrics']['delay_ps'],
            'best_area': best['metrics']['area'],
        }
    trace_files = sorted(case.rglob('rewrite/raw/traces/*.jsonl'))
    sources[str(case / 'summary.json')] = digest(case / 'summary.json')
    sources[str(case / 'config.json')] = digest(case / 'config.json')
    with Path(cfg['input']).open('rb') as stream:
        header = stream.readline().decode().strip().split()
    return {
        'case': str(case), 'source_hashes': sources, 'input': cfg['input'],
        'input_sha256': cfg['input_sha256'],
        'pi_count': int(header[2]), 'po_count': int(header[4]), 'and_count': int(header[5]),
        'regions': summary['region_count'],
        'decisions': summary['cut_validation_counts']['verified_decisions'],
        'candidates': summary['candidate_count'],
        'preparation_runtime_s': summary['preparation_runtime_s'],
        'pipeline_all_methods_runtime_s': summary['total_runtime_including_preparation_s'],
        'gradient': gradient, 'recorded_closed_commands': dict(costs),
        'successful_measurements': sum(map(len, graphs.values())),
        'unique_measured_aig_byte_hashes': len(graphs),
        'duplicate_measurements_by_byte_hash': sum(len(v) - 1 for v in graphs.values()),
        'duplicate_groups': {h: v for h, v in graphs.items() if len(v) > 1},
        'replay_trace_count': len(trace_files),
        'replay_trace_bytes': sum(p.stat().st_size for p in trace_files),
        'scope': 'Saved mixed-hardware multi-method run, not a new runtime or GPU speed measurement. '
                 'Closed command sums exclude preparation/optimizer and may include timeouts. '
                 'Byte duplicates indicate potential reuse, not permission to omit provenance or validation.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--comparison-csv', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('Use a Slurm allocation and a prepared /data run.')
    import csv
    started = time.monotonic()
    metadata_path = run / 'run.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(state='RUNNING', slurm_job_id=os.environ['SLURM_JOB_ID'],
                    command=sys.argv, experiment_executed=False,
                    environment={'python': sys.version,
                                 'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')}})
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    output = run / 'output/gpu_joint_diagnostics'
    output.mkdir(exist_ok=False)
    try:
        chosen = {}
        with args.comparison_csv.open() as stream:
            for row in csv.DictReader(stream):
                if family(row['input_name']) in FAMILIES:
                    chosen[int(row['input_index'])] = row
        records = []
        for index, row in sorted(chosen.items()):
            case = Path(row['source_summary']).parent
            record = inspect_case(case)
            record.update(index=index, input_name=row['input_name'])
            save_json(output / f'case_{index:04d}.json', record)
            records.append({k: v for k, v in record.items()
                            if k not in {'duplicate_groups', 'source_hashes'}})
            print(json.dumps({'index': index, 'name': row['input_name'],
                              'duplicates': record['duplicate_measurements_by_byte_hash'],
                              'trace_bytes': record['replay_trace_bytes']}), flush=True)
        save_json(output / 'summary.json', {
            'status': 'PASS', 'cases': records, 'case_count': len(records),
            'input_csv': str(args.comparison_csv),
            'input_csv_sha256': digest(args.comparison_csv),
            'source_commit': metadata['source_commit'],
            'elapsed_s': time.monotonic() - started,
            'scope': 'Purposeful read-only diagnostics of fixed development families; '
                     'no new search, training, rewrite, CEC or mapping.',
        })
        metadata.update(state='COMPLETED', outputs=str(output / 'summary.json'))
    except Exception as exc:
        metadata.update(state='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()

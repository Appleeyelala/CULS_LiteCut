#!/usr/bin/env python3
"""Reconstruct saved selections' critical/near-critical Liberty arrival paths.

Uses a separate observer build and fresh CEC/mapping in a new /data directory.
Every timing/area/PO value and mapped Verilog body must match the saved result.
Original search results remain untouched. No new candidates are searched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import digest, save_json
from global_timing.evaluation import Evaluator
from global_timing.path_analysis import parse_paths
from global_timing.structure import parse_aig


def netlist_body(path):
    return '\n'.join(line for line in path.read_text().splitlines() if not line.lstrip().startswith('//'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    parser.add_argument('--benchmarks', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not str(run).startswith('/data/wushangge/'):
        raise SystemExit('Path analysis requires a prepared Slurm /data run.')
    binary = args.tool_run / 'build/framework_tools/abc/abc'
    build = json.loads((args.tool_run / 'build/framework_tools/build_report.json').read_text())
    if digest(binary) != build['binaries']['abc']['sha256']:
        raise SystemExit('Observer binary checksum changed.')
    output = run / 'output/arrival_paths'
    output.mkdir()
    os.environ['CULS_GLOBAL_TIMING_PATH_REPORT'] = '1'
    records = []
    for benchmark in args.benchmarks:
        summary = json.loads((benchmark / 'summary.json').read_text())
        original = Path(summary['original']['provenance']['original']['path'])
        library = Path(summary['original']['provenance']['library']['path'])
        for key, path in [('original', original), ('library', library)]:
            if digest(path) != summary['original']['provenance'][key]['sha256']:
                raise ValueError('Saved input/library bytes changed.')
        pi_count = parse_aig(original).pi_count
        tag = hashlib.sha256(str(benchmark.resolve()).encode()).hexdigest()[:12]
        work = output / tag
        work.mkdir()
        evaluator = Evaluator(binary, library, original)
        groups = {'replay_noop': [summary['replay_noop']], 'random_search': summary['random_search']}
        groups.update({name: summary[name]['observations'] for name in ['greedy_litecut', 'independent_regions']})
        groups.update({'gradient_' + name: item['observations'] for name, item in summary['gradient_search'].items()})
        requests = [('original', None, original, summary['original'], benchmark / 'original/mapped.v'),
                    ('native_abc', None, benchmark / 'native_abc/rewritten.aig', summary['native_abc'],
                     benchmark / 'native_abc/evaluation/mapped.v')]
        for name, rows in groups.items():
            for row in rows:
                if row['status'] == 'PASS':
                    mapped = benchmark / row['method'] / f'eval_{row["index"]:04d}' / 'evaluation/mapped.v'
                    requests.append((name, row['index'], Path(row['aig']), row['metrics'], mapped))
        cached, measurements, changes, previous = {}, [], [], {}
        for name, index, aig, reference, mapped in requests:
            input_hash = digest(aig)
            mapped_hash = hashlib.sha256(netlist_body(mapped).encode()).hexdigest()
            key = input_hash + ':' + mapped_hash
            if input_hash != reference['input_sha256'] or reference['cec'] != 'PASS':
                raise ValueError('Saved CEC/input provenance mismatch.')
            if key not in cached:
                destination = work / f'graph_{len(cached):04d}'
                record = {'input_sha256': input_hash, 'mapped_reference_body_sha256': mapped_hash,
                          'aig': str(aig), 'source_benchmark': str(benchmark),
                          'reference_abc': reference['provenance']['abc'], 'observer_abc_sha256': digest(binary)}
                try:
                    actual = evaluator.evaluate(aig, destination)
                    for field in ['area', 'delay_ps', 'mapped_gates', 'po_arrivals_ps']:
                        if actual[field] != reference[field]:
                            raise ValueError('Observer changed saved measurement: ' + field)
                    if netlist_body(destination / 'mapped.v') != netlist_body(mapped):
                        raise ValueError('Observer mapped netlist body differs from saved mapping.')
                    trace = parse_paths((destination / 'mapping.log').read_text(), actual, pi_count)
                    save_json(destination / 'paths.json', trace)
                    record.update(status='PASS', paths=str(destination / 'paths.json'),
                                  measured_metrics_match=True, mapped_netlist_body_match=True, fresh_cec='PASS')
                    record['critical_pairs'] = [[row['pi_index'], row['po_index']] for row in trace['paths'] if row['critical']]
                    record['near_critical_path_count'] = len(trace['paths'])
                    record['timing'] = {field: actual[field] for field in ['area', 'delay_ps', 'po_arrivals_ps']}
                except Exception as exc:
                    record.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
                cached[key] = record
                measurements.append(record)
            record = cached[key]
            if record['status'] == 'PASS':
                if any(record['timing'][field] != reference[field] for field in ['area', 'delay_ps', 'po_arrivals_ps']):
                    raise ValueError('Identical AIG bytes have conflicting saved measurements.')
                pairs = record['critical_pairs']
                changes.append({'method': name, 'index': index, 'input_sha256': input_hash, 'critical_pairs': pairs,
                    'critical_source_endpoint_pairs_changed': name in previous and previous[name] != pairs,
                    'path_evidence': record['paths']})
                previous[name] = pairs
        report = {'source_benchmark': str(benchmark), 'measurements': measurements, 'selection_path_changes': changes,
                  'status': 'PASS' if all(row['status'] == 'PASS' for row in measurements) else 'FAIL',
                  'scope': 'Maximizing Liberty arrival chains, one per near-critical PO. PI/PO indices are preserved interface identities. Mapped node IDs are not aligned across mappings; changed endpoint pairs do not assert unique logical path identity.'}
        save_json(work / 'summary.json', report)
        records.append({'benchmark': str(benchmark), 'status': report['status'], 'summary': str(work / 'summary.json'),
                        'unique_aigs': len(measurements), 'failed': sum(row['status'] != 'PASS' for row in measurements)})
        print(json.dumps(records[-1]), flush=True)
    save_json(output / 'summary.json', {'records': records, 'observer_abc_sha256': digest(binary),
                                      'source_commit': json.loads((run / 'run.json').read_text())['source_commit']})
    return int(any(row['status'] != 'PASS' for row in records))


if __name__ == '__main__':
    raise SystemExit(main())

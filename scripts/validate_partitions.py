#!/usr/bin/env python3
"""Certify every staged input's MFFC partition without rewrite or mapping."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import digest, save_json
from global_timing.cohort import inventory_rows
from global_timing.partition_validation import validate_partition
from global_timing.structure import make_partition, parse_aig


def certify(row, output):
    started = time.monotonic()
    record = {key: row[key] for key in ['index', 'relative_path', 'sha256', 'staged_path']}
    try:
        path = Path(row['staged_path']).resolve()
        if not path.is_relative_to('/data/wushangge') or digest(path) != row['sha256']:
            raise ValueError('Changed or unstaged input.')
        if row['classification'] != 'COMBINATIONAL':
            raise ValueError('Structural certificate requires a combinational AIG.')
        partition = make_partition(parse_aig(path))
        record.update(validate_partition(partition))
        directory = output / f'case_{row["index"]:04d}'
        partition.write(directory)
        if digest(path) != row['sha256']:
            raise ValueError('Input changed during partition certification.')
        record['artifacts'] = {str(directory / name): digest(directory / name)
                               for name in ['partition.json', 'owners.csv']}
    except Exception as exc:
        record.update(status='FAIL', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    record['elapsed_s'] = time.monotonic() - started
    save_json(output / f'case_{row["index"]:04d}' / 'certificate.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('Use a prepared /data run and Slurm.')
    if not 1 <= args.workers <= int(os.environ.get('SLURM_CPUS_PER_TASK', '1')):
        parser.error('Workers must fit the actual CPU allocation.')
    raw = args.manifest.read_bytes()
    inventory = inventory_rows(json.loads(raw))
    output = run / 'output/partition_validation'
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'input_manifest.json').open('xb') as stream:
        stream.write(raw)
    started = time.monotonic()
    records = []
    with (output / 'records.jsonl').open('x') as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(certify, row, output) for row in inventory.values()]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            stream.write(json.dumps(record, sort_keys=True) + '\n')
            stream.flush()
            print(json.dumps({key: record.get(key) for key in
                             ['index', 'status', 'and_count', 'region_count', 'elapsed_s', 'error']}), flush=True)
    complete = {row['index'] for row in records} == set(inventory)
    report = {'status': 'PASS' if complete and all(row['status'] == 'PASS' for row in records) else 'FAIL',
              'expected_inputs': len(inventory), 'recorded_inputs': len(records), 'coverage_complete': complete,
              'case_status_counts': dict(Counter(row['status'] for row in records)),
              'source_commit': json.loads((run / 'run.json').read_text())['source_commit'],
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'workers': args.workers,
              'input_manifest_sha256': digest(output / 'input_manifest.json'),
              'elapsed_s': time.monotonic() - started, 'records': sorted(records, key=lambda row: row['index']),
              'scope': 'All declared inventory inputs. Independent fanout-closure and reverse-distance certificates; '
                       'no GPU, rewrite, CEC, mapping, Student inference or search. Does not replace full benchmark comparisons.'}
    save_json(output / 'summary.json', report)
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Snapshot every AIG/AAG in the configured dataset and record actual headers."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not str(run).startswith('/data/wushangge/'):
        raise SystemExit('Dataset staging requires Slurm and /data.')
    output = run / 'input/benchmarks'
    output.mkdir(exist_ok=False)
    rows = []
    for source in sorted(args.dataset_root.rglob('*')):
        if not source.is_file() or source.suffix.lower() not in {'.aig', '.aag'}:
            continue
        row = {'index': len(rows), 'source': str(source), 'relative_path': str(source.relative_to(args.dataset_root)),
               'size_bytes': source.stat().st_size, 'sha256': digest(source)}
        destination = output / (f'{len(rows):04d}' + source.suffix.lower())
        with source.open('rb') as incoming, destination.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if digest(destination) != row['sha256']:
            raise RuntimeError('Dataset changed while staging: ' + str(source))
        row['staged_path'] = str(destination)
        try:
            with destination.open('rb') as stream:
                header = stream.readline(1024).decode('ascii').split()
            if len(header) < 6 or header[0] not in {'aig', 'aag'}:
                raise ValueError('Not an AIGER header.')
            values = list(map(int, header[1:]))
            if min(values) < 0:
                raise ValueError('Negative header count.')
            row.update(format=header[0], maximum_variable=values[0], pi_count=values[1],
                       latch_count=values[2], po_count=values[3], and_count=values[4],
                       extra_header_fields=values[5:], header=' '.join(header),
                       classification='SEQUENTIAL' if values[2] else 'COMBINATIONAL')
        except (ValueError, UnicodeError) as exc:
            row.update(classification='INVALID_HEADER', error=str(exc))
        rows.append(row)
    summary = {'count': len(rows), 'size_bytes': sum(r['size_bytes'] for r in rows),
               'classifications': dict(Counter(r['classification'] for r in rows)),
               'formats': dict(Counter(r.get('format', 'invalid') for r in rows)),
               'job_id': os.environ['SLURM_JOB_ID'], 'rows': rows}
    with (run / 'output/benchmark_manifest.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run real comparisons on every selected staged design, with per-case status.

Two long-lived Slurm shards can cover the full dataset without placing hundreds
of GPU requests in PENDING and exceeding the user's concurrent request budget.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--indices', help='comma-separated exact inventory indices; omitted means full assigned shard')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--evaluations', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--gradient-steps', type=int, default=20)
    parser.add_argument('--calibration-steps', type=int, default=30)
    parser.add_argument('--case-timeout', type=int, default=3600)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not str(run).startswith('/data/wushangge/'):
        raise SystemExit('Benchmark suite requires Slurm and a prepared /data run.')
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count or args.case_timeout <= 0:
        parser.error('Invalid shard or time limit.')
    manifest = json.loads(args.manifest.read_text())
    indices = set(map(int, args.indices.split(','))) if args.indices else None
    if indices is not None and not indices <= {r['index'] for r in manifest['rows']}:
        parser.error('Requested index is absent from the staged dataset inventory.')
    selected = [r for r in manifest['rows'] if (indices is None or r['index'] in indices)
                and r['index'] % args.shard_count == args.shard_index]
    subprocess.run([sys.executable, '-B', str(run / 'src/scripts/test_framework_run.py'),
                    '--run', str(run), '--tool-run', str(args.tool_run)], cwd=run, check=True)
    metadata_path = run / 'run.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(state='RUNNING', benchmark_manifest=str(args.manifest),
                    benchmark_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    benchmark_arguments=vars(args) | {'run': str(run), 'tool_run': str(args.tool_run), 'manifest': str(args.manifest)})
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    suite = run / 'output/suite'
    suite.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(run / 'src'))
    sys.path.insert(0, str(run / 'src'))
    from global_timing.reporting import write_report
    records = []
    try:
        with (suite / 'records.jsonl').open('x') as stream:
            for row in selected:
                started = time.monotonic()
                record = dict(row, status='RUNNING')
                source = Path(row['staged_path'])
                output = suite / f'case_{row["index"]:04d}'
                try:
                    if not str(source.resolve()).startswith('/data/wushangge/'):
                        raise ValueError('Unstaged input outside /data.')
                    if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                        raise ValueError('Staged benchmark checksum changed.')
                    if row['classification'] != 'COMBINATIONAL':
                        record.update(status='SKIP', reason='Current CULS/LiteCut contract requires a combinational AIG.')
                    else:
                        command = [sys.executable, '-B', '-m', 'global_timing', '--input', str(source),
                            '--gpuls', metadata['binaries']['gpuls']['path'], '--abc', metadata['binaries']['abc']['path'],
                            '--library', str(run / 'input/asap7'), '--checkpoint', str(run / 'input/student_checkpoint'),
                            '--output', str(output), '--evaluations', str(args.evaluations), '--warmup', str(args.warmup),
                            '--gradient-steps', str(args.gradient_steps), '--calibration-steps', str(args.calibration_steps),
                            '--mode', 'both']
                        record['command'] = command
                        log_path = run / 'logs' / f'case_{row["index"]:04d}.log'
                        record['log'] = str(log_path)
                        with log_path.open('x') as log:
                            process = subprocess.Popen(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                       start_new_session=True)
                            try:
                                process.wait(timeout=args.case_timeout)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                                record['timed_out'] = True
                        record['returncode'] = process.returncode
                        if process.returncode or not (output / 'summary.json').is_file():
                            raise RuntimeError('Case failed or timed out; inspect preserved log and intermediate results.')
                        comparison = write_report(output, output / 'report')
                        record.update(status='PASS', summary=str(output / 'summary.json'), comparison=comparison)
                except Exception as exc:
                    record.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}')
                record['elapsed_s'] = time.monotonic() - started
                records.append(record)
                stream.write(json.dumps(record, sort_keys=True) + '\n')
                stream.flush()
                print(json.dumps({k: record.get(k) for k in ['index', 'relative_path', 'status', 'elapsed_s', 'reason']}), flush=True)
        summary = {'selected_count': len(selected), 'recorded_count': len(records),
                   'counts': dict(Counter(r['status'] for r in records)), 'records': records,
                   'scope': 'explicit index subset' if indices is not None else 'full assigned dataset shard',
                   'shard_index': args.shard_index, 'shard_count': args.shard_count}
        (suite / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        metadata.update(state='COMPLETED_WITH_FAILURES' if any(r['status'] == 'FAIL' for r in records) else 'COMPLETED',
                        outputs=str(suite / 'summary.json'))
    except Exception as exc:
        metadata.update(state='FAILED', error=str(exc))
        (suite / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()

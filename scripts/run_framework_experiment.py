#!/usr/bin/env python3
"""Validate a committed snapshot, then run a real c1355 optimization benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    parser.add_argument('--evaluations', type=int, default=24)
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--gradient-steps', type=int, default=60)
    parser.add_argument('--calibration-steps', type=int, default=80)
    parser.add_argument('--mode', choices=['area_constraint', 'joint', 'both'], default='both')
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not str(run).startswith('/data/wushangge/'):
        raise SystemExit('Experiment requires Slurm and a /data workspace.')
    validation = [sys.executable, '-B', str(run / 'src/scripts/test_framework_run.py'),
                  '--run', str(run), '--tool-run', str(args.tool_run)]
    subprocess.run(validation, check=True, cwd=run)
    metadata_path = run / 'run.json'
    metadata = json.loads(metadata_path.read_text())
    env = dict(os.environ, PYTHONPATH=str(run / 'src'))
    command = [sys.executable, '-B', '-m', 'global_timing', '--input', str(run / 'input/design.aig'),
               '--gpuls', metadata['binaries']['gpuls']['path'], '--abc', metadata['binaries']['abc']['path'],
               '--library', str(run / 'input/asap7'), '--checkpoint', str(run / 'input/student_checkpoint'),
               '--output', str(run / 'output/benchmark'), '--evaluations', str(args.evaluations),
               '--warmup', str(args.warmup), '--gradient-steps', str(args.gradient_steps),
               '--calibration-steps', str(args.calibration_steps), '--mode', args.mode]
    metadata.update(state='RUNNING', experiment_command=command)
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    try:
        with (run / 'logs/benchmark.log').open('x') as log:
            result = subprocess.run(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Benchmark failed; see logs/benchmark.log')
        metadata.update(state='COMPLETED', outputs=str(run / 'output/benchmark/summary.json'))
    except Exception as exc:
        metadata.update(state='FAILED', error=str(exc))
        (run / 'output/experiment_failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run the current committed test snapshot against hashed /data tool builds."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    args = parser.parse_args()
    run, build_run = args.run.resolve(), args.tool_run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not str(run).startswith('/data/wushangge/'):
        raise SystemExit('Tests require a prepared /data run and Slurm allocation.')
    report = json.loads((build_run / 'build/framework_tools/build_report.json').read_text())
    for item in report['binaries'].values():
        path = Path(item['path']).resolve()
        if not str(path).startswith('/data/wushangge/') or digest(path) != item['sha256']:
            raise SystemExit('Invalid or changed executable build artifact.')
    # Verify the complete native source trees; Python can evolve independently.
    for prefix in ['projects/culs', 'third_party/abc', 'framework/abc']:
        for source in (run / 'src' / prefix).rglob('*'):
            if not source.is_file() or source.suffix not in {'.c', '.cpp', '.cc', '.cu', '.h', '.hpp', '.cuh', '.inc'}:
                continue
            relative = source.relative_to(run / 'src')
            prior = build_run / 'src' / relative
            if not prior.is_file() or digest(source) != digest(prior):
                raise SystemExit('Rebuild changed native tool source before testing: ' + str(relative))
    metadata = json.loads((run / 'run.json').read_text())
    packages = {}
    for package in ['torch', 'numpy', 'scipy', 'pandas', 'networkx', 'matplotlib', 'scikit-learn']:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True)
    metadata.update(state='RUNNING', experiment_executed=True, slurm_job_id=os.environ['SLURM_JOB_ID'],
                    tool_build_run=str(build_run), binaries=report['binaries'],
                    environment={'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                                 'packages': packages, 'node_gpu_inventory': gpu.stdout.strip(),
                                 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 'thread_limits': {key: os.environ.get(key) for key in
                                                   ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
                                 'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')}})
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    env = dict(os.environ, CULS_TEST_RUN=str(run), CULS_TEST_GPULS=report['binaries']['gpuls']['path'],
               CULS_TEST_ABC=report['binaries']['abc']['path'], PYTHONPATH=str(run / 'src'))
    command = [sys.executable, '-B', str(run / 'src/scripts/run_tests.py'), '--output', str(run / 'output/tests.json')]
    result = subprocess.run(command, cwd=run, env=env)
    metadata.update(state='COMPLETED' if result.returncode == 0 else 'FAILED', outputs=str(run / 'output/tests.json'))
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())

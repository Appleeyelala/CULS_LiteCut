#!/usr/bin/env python3
"""Compile the standalone CUDA LSE library and verify device graph derivatives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('Requires a committed-source /data workspace and a Slurm GPU allocation.')
    build = run / 'build/mffc_gpu'
    build.mkdir(exist_ok=False)
    source = run / 'src/framework/mffc_gpu/region_lse.cu'
    library = build / 'libmffc_region.so'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    commands = [
        ['/usr/local/cuda-11.8/bin/nvcc', '-O3', '-std=c++17', '-shared', '-Xcompiler', '-fPIC',
         '-ccbin', '/usr/local/bin/g++', '-gencode', 'arch=compute_80,code=sm_80',
         '-gencode', 'arch=compute_80,code=compute_80', str(source), '-o', str(library),
         '-Xlinker', '-rpath', '-Xlinker', '/usr/local/cuda-11.8/lib64'],
        [sys.executable, '-B', str(run / 'src/scripts/run_tests.py'), '--output',
         str(run / 'output/mffc_gpu_tests.json'), '--test', 'test_mffc_gpu',
         '--test', 'test_mffc_model', '--test', 'test_mffc_search', '--test', 'test_feasibility',
         '--test', 'test_surrogate', '--test', 'test_arrival', '--test', 'test_candidates'],
    ]
    report = {'status': 'RUNNING', 'source_sha256': sha(source), 'commands': [],
              'source_commit': json.loads((run / 'run.json').read_text())['source_commit'],
              'job_id': os.environ['SLURM_JOB_ID'], 'library': str(library)}
    try:
        for index, command in enumerate(commands):
            started = time.perf_counter()
            log = run / 'logs' / f'mffc_gpu_build_{index}.log'
            environment = dict(os.environ, PYTHONPATH=str(run / 'src'), MFFC_GPU_LIBRARY=str(library))
            with log.open('x') as stream:
                result = subprocess.run(command, cwd=run, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            report['commands'].append({'argv': command, 'elapsed_s': time.perf_counter() - started,
                                       'returncode': result.returncode, 'log': str(log)})
            if result.returncode:
                raise RuntimeError('Build/validation failed; inspect ' + str(log))
        tests = json.loads((run / 'output/mffc_gpu_tests.json').read_text())
        if not tests['successful'] or any(r['status'] != 'PASS' for r in tests['records']):
            raise RuntimeError('Native device validation must pass without skipped cases.')
        report.update(status='PASS', library_sha256=sha(library), tests=tests['tests_run'],
                      compiler=subprocess.check_output([commands[0][0], '--version'], text=True),
                      gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True))
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        with (build / 'build_report.json').open('x') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps({'status': report['status'], 'library': str(library), 'tests': report['tests']}))


if __name__ == '__main__':
    main()

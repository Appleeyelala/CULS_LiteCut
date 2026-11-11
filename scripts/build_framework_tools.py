#!/usr/bin/env python3
"""Build current CULS and an independently instrumented ABC inside a run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--build-only', action='store_true',
                        help='Compile without GPU runtime tests; records BUILT_NOT_RUNTIME_VALIDATED explicitly.')
    args = parser.parse_args()
    run = args.run.resolve()
    if not str(run).startswith('/data/wushangge/') or not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Build requires a Slurm allocation and a /data run.')
    if not 1 <= args.jobs <= int(os.environ.get('SLURM_CPUS_PER_TASK', '1')):
        raise SystemExit('--jobs exceeds the CPU allocation.')
    src = run / 'src'
    metadata = json.loads((run / 'run.json').read_text())
    metadata.update(state='RUNNING', slurm_job_id=os.environ['SLURM_JOB_ID'], experiment_executed=True,
                    environment={'python': sys.version, 'path': os.environ.get('PATH'),
                                 'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')}})
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    root = run / 'build/framework_tools'
    root.mkdir(exist_ok=False)
    abc_src = root / 'abc'
    shutil.copytree(src / 'third_party/abc', abc_src)
    timing = abc_src / 'src/map/scl/sclSize.c'
    content = timing.read_text()
    declaration = 'void Abc_SclTimePerformInt('
    invocation = '    Abc_SclTimeNtkPrint( p, fShowAll, fPrintPath );'
    if content.count(declaration) != 1 or content.count(invocation) != 1:
        raise RuntimeError('ABC timing instrumentation anchor changed.')
    content = content.replace(declaration, (src / 'framework/abc/endpoint_report.c.inc').read_text() + '\n' + declaration)
    content = content.replace(invocation, invocation + '\n    CulsGlobalTimingReport( p );')
    timing.write_text(content)
    commands = [
        [sys.executable, '-B', str(src / 'scripts/run_tests.py'), '--output', str(run / 'output/unit_tests.json')],
        ['cmake', '-S', str(src / 'projects/culs'), '-B', str(root / 'culs'),
         '-DCMAKE_C_COMPILER=/usr/local/bin/gcc', '-DCMAKE_CXX_COMPILER=/usr/local/bin/g++',
         '-DCMAKE_CUDA_HOST_COMPILER=/usr/local/bin/g++', '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc',
         '-DCMAKE_BUILD_TYPE=Release'],
        ['cmake', '--build', str(root / 'culs'), '-j', str(args.jobs)],
        ['make', '-C', str(abc_src), '-j', str(args.jobs), 'ABC_USE_NO_READLINE=1',
         'CC=/usr/local/bin/gcc', 'CXX=/usr/local/bin/g++', 'LD=/usr/local/bin/g++'],
    ]
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'commands': [], 'state': 'RUNNING'}
    try:
        for index, command in enumerate(commands):
            start = time.monotonic()
            path = run / 'logs' / f'build_{index}.log'
            with path.open('x') as log:
                result = subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT)
            report['commands'].append({'argv': command, 'log': str(path), 'returncode': result.returncode,
                                       'elapsed_s': time.monotonic() - start})
            if result.returncode:
                raise RuntimeError(f'Build command failed; see {path}')
        report['binaries'] = {}
        for name, path in {'gpuls': root / 'culs/gpuls', 'abc': abc_src / 'abc'}.items():
            report['binaries'][name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if args.build_only:
            report['state'] = 'BUILT_NOT_RUNTIME_VALIDATED'
            return
        env = dict(os.environ, CULS_TEST_RUN=str(run), CULS_TEST_GPULS=str(root / 'culs/gpuls'),
                   CULS_TEST_ABC=str(abc_src / 'abc'), PYTHONPATH=str(src))
        with (run / 'logs/framework_tests.log').open('x') as log:
            test = subprocess.run([sys.executable, '-B', str(src / 'scripts/run_tests.py'),
                                  '--output', str(run / 'output/tests.json')], cwd=run,
                                  env=env, stdout=log, stderr=subprocess.STDOUT)
        report['test_exit_code'] = test.returncode
        if test.returncode:
            raise RuntimeError('Framework regression failed; see logs/framework_tests.log')
        report['state'] = 'COMPLETED'
    except Exception as exc:
        report.update(state='FAILED', error=str(exc))
        raise
    finally:
        (root / 'build_report.json').write_text(json.dumps(report, indent=2) + '\n')
        metadata.update(state=report['state'], outputs=str(root / 'build_report.json'))
        (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()

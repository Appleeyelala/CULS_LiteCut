#!/usr/bin/env python3
"""Build opt-in CULS replay changes and regress them with frozen ABC tools."""
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
    parser.add_argument('--tool-run', type=Path, required=True)
    parser.add_argument('--old-case', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('Build/runtime validation requires /data and Slurm GPU allocation.')
    if not 1 <= args.jobs <= int(os.environ.get('SLURM_CPUS_PER_TASK', '1')):
        parser.error('Build workers exceed allocation.')
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    old = json.loads((args.tool_run / 'build/framework_tools/build_report.json').read_text())
    for tool in old['binaries'].values():
        if sha(tool['path']) != tool['sha256']:
            raise ValueError('Frozen baseline binary changed.')
    root = run / 'build/framework_tools'
    root.mkdir(exist_ok=False)
    gpuls = root / 'culs/gpuls'
    report = {'state': 'RUNNING', 'commands': [], 'binaries': {}, 'abc_rebuilt': False,
              'job_id': os.environ['SLURM_JOB_ID']}
    metadata_path = run / 'run.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(state='RUNNING', experiment_executed=True, command=sys.argv,
                    slurm_job_id=os.environ['SLURM_JOB_ID'],
                    environment={'python': sys.version,
                                 'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')}})
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    commands = [
        ['cmake', '-S', str(run / 'src/projects/culs'), '-B', str(root / 'culs'),
         '-DCMAKE_C_COMPILER=/usr/local/bin/gcc', '-DCMAKE_CXX_COMPILER=/usr/local/bin/g++',
         '-DCMAKE_CUDA_HOST_COMPILER=/usr/local/bin/g++', '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc',
         '-DCMAKE_BUILD_TYPE=Release'],
        ['cmake', '--build', str(root / 'culs'), '-j', str(args.jobs)],
        [sys.executable, '-B', str(run / 'src/scripts/run_tests.py'), '--output', str(run / 'output/replay_tests.json'),
         '--test', 'test_prepared_replay', '--test', 'test_real_tools'],
        [sys.executable, '-B', str(run / 'src/scripts/run_tests.py'), '--output', str(run / 'output/prepared_regressions.json'),
         '--test', 'test_real_tools'],
    ]
    try:
        for index, command in enumerate(commands):
            env = dict(os.environ, PYTHONPATH=str(run / 'src'), CULS_TEST_RUN=str(run),
                       CULS_TEST_GPULS=str(gpuls), CULS_TEST_ABC=old['binaries']['abc']['path'],
                       CULS_TEST_LEGACY_GPULS=old['binaries']['gpuls']['path'], CULS_TEST_OLD_CASE=str(args.old_case))
            env.pop('CULS_REPLAY_PREPARED', None)
            if index == 3:
                env['CULS_REPLAY_PREPARED'] = '1'
            start = time.perf_counter()
            log = run / 'logs' / f'replay_build_{index}.log'
            with log.open('x') as stream:
                result = subprocess.run(command, cwd=run, env=env, stdout=stream, stderr=subprocess.STDOUT)
            report['commands'].append({'argv': command, 'returncode': result.returncode,
                                       'log': str(log), 'elapsed_s': time.perf_counter() - start})
            if result.returncode:
                raise RuntimeError('Build/runtime regression failed; inspect ' + str(log))
        report['binaries'] = {'gpuls': {'path': str(gpuls), 'sha256': sha(gpuls)}, 'abc': old['binaries']['abc']}
        report['state'] = 'COMPLETED'
    except Exception as exc:
        report.update(state='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (root / 'build_report.json').write_text(json.dumps(report, indent=2) + '\n')
        metadata.update(state=report['state'], outputs=str(root / 'build_report.json'))
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()

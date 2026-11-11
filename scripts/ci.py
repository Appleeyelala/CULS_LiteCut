#!/usr/bin/env python3
"""Local Slurm CI: exact-source regression, real smoke, repository matrix.

No remote CI service, hook, external remote, package install or publication is
configured. Prepare a committed run first, then invoke this within its job.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import run_tool, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    parser.add_argument('--unit-only', action='store_true', help='CPU core tests only; GPU omissions remain explicit.')
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not str(run).startswith('/data/wushangge/'):
        raise SystemExit('CI requires a prepared Slurm /data run.')
    metadata_path = run / 'run.json'
    metadata = json.loads(metadata_path.read_text())
    started = time.monotonic()
    report = {'source_commit': metadata['source_commit'], 'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'mode': 'unit_only' if args.unit_only else 'real_e2e_and_program_matrix', 'steps': [], 'state': 'RUNNING'}
    commands = []
    if args.unit_only:
        commands.append(('unit', [sys.executable, '-B', str(run / 'src/scripts/run_tests.py'),
                                  '--output', str(run / 'output/tests.json')]))
    else:
        command = [sys.executable, '-B', str(run / 'src/scripts/run_framework_experiment.py'),
                   '--run', str(run), '--tool-run', str(args.tool_run), '--evaluations', '8', '--warmup', '2',
                   '--gradient-steps', '20', '--calibration-steps', '30']
        commands.append(('e2e', command))
        for name in ['test_native_writer', 'test_imported_programs', 'test_repository']:
            commands.append((name, [sys.executable, '-B', str(run / 'src/scripts' / (name + '.py')),
                                   '--run', str(run), '--tool-run', str(args.tool_run)]))
    try:
        for name, command in commands:
            run_tool(command, run / 'logs/ci', name, timeout=3600)
            report['steps'].append({'name': name, 'status': 'PASS', 'command': command})
        if not args.unit_only:
            # Import numerical plotting dependencies only after subprocesses;
            # their loader may set MKL_THREADING_LAYER in the parent process.
            from global_timing.reporting import write_report
            report['benchmark_comparison'] = write_report(run / 'output/benchmark', run / 'output/benchmark/report')
            report['repository_matrix'] = json.loads((run / 'output/repository_tests/matrix_summary.json').read_text())
        report['core_tests'] = json.loads((run / 'output/tests.json').read_text())
        report['state'] = 'COMPLETED_WITH_EXPLICIT_OMISSIONS'
        report['scope'] = 'Core executed tests and actual smoke pass; imported runtime/import omissions remain in the matrix.'
    except Exception as exc:
        report.update(state='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_s'] = time.monotonic() - started
        save_json(run / 'output/ci_summary.json', report)
        metadata = json.loads(metadata_path.read_text())
        metadata.update(state=report['state'], ci_summary=str(run / 'output/ci_summary.json'),
                        slurm_job_id=os.environ['SLURM_JOB_ID'], experiment_executed=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()

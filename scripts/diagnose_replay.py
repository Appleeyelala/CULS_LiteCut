#!/usr/bin/env python3
"""Reproduce a saved no-op replay failure under GDB in an isolated Slurm run."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import digest, read_jsonl, run_tool, save_json, tool_path
from global_timing.evaluation import Evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    parser.add_argument('--saved-case', type=Path, required=True)
    args = parser.parse_args()
    run, tools, case = [p.resolve() for p in [args.run, args.tool_run, args.saved_case]]
    if not os.environ.get('SLURM_JOB_ID') or not all(p.is_relative_to('/data/wushangge') for p in [run, tools, case]):
        parser.error('Use prepared Slurm /data paths.')
    output = run / 'output/replay_diagnosis'
    output.mkdir(exist_ok=False)
    config = json.loads((case / 'config.json').read_text())
    metadata = json.loads((run / 'run.json').read_text())
    binary_report = json.loads((tools / 'build/framework_tools/build_report.json').read_text())
    gpuls, abc = [Path(binary_report['binaries'][key]['path']) for key in ['gpuls', 'abc']]
    for key, path in [('gpuls', gpuls), ('abc', abc)]:
        if digest(path) != binary_report['binaries'][key]['sha256']:
            raise ValueError('Binary checksum changed: ' + key)
    aig = Path(metadata['dataset']['staged_path'])
    if digest(aig) != config['input_sha256']:
        raise ValueError('Prepared input differs from the failed saved case.')
    owner = run / 'input/owners.csv'
    with (case / 'partition/owners.csv').open('rb') as incoming, owner.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    choice = run / 'input/noop.txt'
    choice.write_text('choice -1 -1 -1\n')
    raw = output / 'raw'
    command = (f'read {tool_path(aig)}; gate3e_execution_replay -o {tool_path(raw)} '
               f'--region-map {tool_path(owner)} --config {tool_path(choice)}')
    argv = ['gdb', '--quiet', '--batch', '-ex', 'set pagination off',
            '-ex', 'set print frame-arguments none', '-ex', 'run',
            '-ex', 'thread apply all bt 30', '--args', str(gpuls), '-c', command]
    metadata.update(state='RUNNING', command=argv, experiment_executed=True,
                    slurm_job_id=os.environ['SLURM_JOB_ID'], binaries=binary_report['binaries'],
                    environment={'python': sys.version, 'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')}})
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    record = {'status': 'RUNNING', 'saved_case': str(case), 'tool_build_run': str(tools),
              'inputs': [{'path': str(p), 'sha256': digest(p)} for p in [aig, owner, choice]]}
    try:
        log = run_tool(argv, output, 'gdb', 300)
        record['signals'] = re.findall(r'received signal (SIG\w+)', log)
        if record['signals']:
            raise RuntimeError('Native replay received ' + ', '.join(record['signals']) + '; inspect saved GDB backtrace.')
        trace = read_jsonl(raw / 'traces/choice.jsonl')
        if any(row['preselected'] or row['final_status'] == 'ACCEPTED' for row in trace):
            raise ValueError('No-op diagnostic selected a rewrite.')
        record['metrics'] = Evaluator(abc, run / 'input/asap7', aig).evaluate(raw / 'aigs/choice.aig', output / 'evaluation')
        record.update(status='PASS_NOT_REPRODUCED', trace_rows=len(trace))
    except Exception as exc:
        record.update(status='FAIL_REPRODUCED_OR_DIAGNOSTIC_ERROR', error=f'{type(exc).__name__}: {exc}')
    save_json(output / 'summary.json', record)
    metadata.update(state='COMPLETED_DIAGNOSTIC', outputs=str(output / 'summary.json'))
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'metrics'}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

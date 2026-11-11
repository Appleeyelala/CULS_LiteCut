#!/usr/bin/env python3
"""Exercise retained R9 analysis programs in new directories using real evidence.

This tests existing programs and saved inputs, not a new scientific experiment
or an independent validation split. Historical sources and results stay intact.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import digest, run_tool, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not str(run).startswith('/data/wushangge/'):
        raise SystemExit('Historical analysis tests require Slurm and a prepared /data run.')
    history = Path('/data/wushangge/runs')
    mechanism = history / 'CULS-R9-mechanism-20261012-9CY36o'
    pin = history / 'CULS-R9-pin-audit-20261014-iLFKEK'
    local = history / 'CULS-R9-local-arrival-20261014-ZM1dee'
    prior = history / 'CULS-mapper-internal-20261003-34U6OZ'
    original_abc = history / 'CULS-source-pair-20260930-qU1wxi/bin/abc_pair_timing'
    output = run / 'output/historical_analysis_tests'
    output.mkdir()
    shared = [prior / ('input/' + name) for name in ['c1355_orig.aig', 'S002.aig', 'S002.jsonl', 'asap7.lib']]
    shared += [prior / 'bin/abc_pair_timing', original_abc, mechanism / 'mechanism_evidence.json',
               pin / 'pin_audit_evidence.json', mechanism / 'abc/abc']
    shared += [prior / f'output/{variant}_original.blif' for variant in ['baseline', 'candidate']]
    before = {str(path): digest(path) for path in shared if path.is_file()}
    save_json(output / 'read_only_dependencies.json', before)
    records = []
    specs = [
        ('run_r9_mechanism', mechanism, ['cec.log', 'baseline.log', 'candidate.log',
            'baseline.blif', 'candidate.blif', 'baseline_instrumented.log', 'candidate_instrumented.log',
            'baseline_instrumented.blif', 'candidate_instrumented.blif']),
        ('run_r9_pin_audit', pin, ['baseline.log', 'candidate.log', 'baseline.blif', 'candidate.blif']),
        ('run_r9_local_frontier', local, []),
        ('finalize_r9_local_arrival', local, ['local_prediction.log', 'baseline_observed.blif', 'frontier_structure.json'])]
    for name, retained, inputs in specs:
        work = output / name
        work.mkdir()
        relative = f'projects/culs/experiments/path_delay_model/{name}.py'
        source = run / 'src' / relative
        record = {'source_relative': relative, 'source_sha256': digest(source), 'status': 'RUNNING',
                  'scope': 'Existing script assertions on retained R9 inputs, with new CEC/ABC calls where the script requires them; not fresh independent prediction evidence.',
                  'copied_inputs': []}
        try:
            for filename in inputs:
                incoming, staged = retained / filename, work / filename
                shutil.copyfile(incoming, staged)
                record['copied_inputs'].append({'source': str(incoming), 'staged': str(staged), 'sha256': digest(staged)})
            if name == 'run_r9_mechanism':
                (work / 'abc').mkdir()
                (work / 'abc/abc').symlink_to(mechanism / 'abc/abc')
            env = dict(os.environ, PYTHONPATH=str(run / 'src/projects/culs/tools'))
            run_tool([sys.executable, '-B', source, work], work, 'execution', timeout=300, env=env)
            record['status'] = 'PASS'
        except FileNotFoundError as exc:
            record.update(status='SKIP', reason=f'Missing retained dependency: {exc}')
        except Exception as exc:
            record.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}')
        save_json(work / 'test_result.json', record)
        records.append(record)
        print(json.dumps({k: record[k] for k in ['source_relative', 'status']}), flush=True)
    after = {path: digest(Path(path)) for path in before}
    if before != after:
        raise AssertionError('Historical dependency bytes changed during testing.')
    save_json(output / 'summary.json', {'records': records, 'historical_dependencies_unchanged': True,
                                      'slurm_job_id': os.environ['SLURM_JOB_ID']})
    return int(any(record['status'] == 'FAIL' for record in records))


if __name__ == '__main__':
    raise SystemExit(main())

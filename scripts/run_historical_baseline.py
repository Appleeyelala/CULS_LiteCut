#!/usr/bin/env python3
"""Re-run the existing c1355 Gate3 pipeline in a fresh prepared Slurm run.

Uses original imported programs with explicit paths; never edits old evidence.
Every subprocess and its outcome is recorded, including ABC's semantic errors.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not str(run).startswith('/data/wushangge/') or not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('This baseline must run inside Slurm under /data/wushangge.')
    source = run / 'src'
    tools = source / 'projects/culs/tools'
    output = run / 'output/historical_baseline'
    output.mkdir(exist_ok=False)
    logs = run / 'logs'
    meta_path = run / 'run.json'
    meta = json.loads(meta_path.read_text())
    meta.update(state='RUNNING', experiment_executed=True, slurm_job_id=os.environ['SLURM_JOB_ID'])
    env = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
           'hostname': platform.node(), 'packages': {}, 'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')}}
    for package in ['torch', 'numpy', 'scipy', 'networkx', 'pandas', 'scikit-learn', 'torch-geometric']:
        try:
            env['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            env['packages'][package] = None
    meta['environment'] = env
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    commands = []

    def execute(name, argv, timeout=600):
        start = time.monotonic()
        row = {'name': name, 'argv': [str(x) for x in argv], 'cwd': str(run)}
        try:
            proc = subprocess.run(row['argv'], cwd=run, capture_output=True, text=True, timeout=timeout)
            row.update(returncode=proc.returncode, elapsed_s=time.monotonic() - start)
            log = logs / (name + '.log')
            log.write_text(proc.stdout + '\n--- stderr ---\n' + proc.stderr)
            row['log'] = str(log)
            if proc.returncode:
                raise RuntimeError(f'{name} returned {proc.returncode}; see {log}')
            return proc.stdout
        except Exception as exc:
            row['error'] = str(exc)
            raise
        finally:
            commands.append(row)
            (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')

    py = sys.executable
    gpuls = run / 'input/gpuls_gate3f'
    abc = run / 'input/abc_gate3f'
    aig = run / 'input/design.aig'
    aroot = output / 'gate3a'
    a = aroot / 'c1355_orig'
    b = output / 'gate3b'
    c = output / 'gate3c'
    d = output / 'gate3d'
    e = output / 'gate3e'
    f = output / 'gate3f'
    owner = a / 'node_to_strict_mffc_region.csv'
    regions = a / 'maximal_standard_mffcs.json'
    try:
        execute('program_audit', [py, source / 'scripts/audit_programs.py', '--root', source,
                                 '--output', output / 'program_audit'])
        execute('gpu_environment', ['nvidia-smi'])
        for label in ['run1', 'run2']:
            execute('gate3a_' + label, [gpuls, '-c', f'read {aig}; gate3a_mffc_partition_dump -o {a / label}'])
        execute('gate3a_analysis', [py, tools / 'gate3a_analyze.py', '--dataset-root', run / 'input',
                '--original-input', aig, '--output-root', aroot, '--design', 'c1355_orig',
                '--culs-head', meta['source_commit'], '--culs-log', 'prepared committed snapshot', '--culs-status', 'clean'])
        for label in ['run1', 'run2']:
            execute('gate3b_' + label, [py, tools / 'gate3b_analyze.py', 'run', '--input', aig,
                '--maximal-mffcs', regions, '--owner-csv', owner, '--run-dir', b / 'c1355_orig' / label,
                '--design', 'c1355_orig', '--run-label', label])
        execute('gate3b_finalize', [py, tools / 'gate3b_analyze.py', 'finalize', '--output-root', b, '--design', 'c1355_orig'])
        execute('gate3c_dump', [gpuls, '-c', f'read {aig}; gate3c_rewrite_decisions_dump -o {c / "raw"}'])
        execute('gate3c_analysis', [py, tools / 'gate3c_analyze.py', '--raw-dir', c / 'raw',
                '--maximal-mffcs', regions, '--owner-map', owner, '--output', c, '--input-manifest', a / 'input_manifest.json'])
        for label in ['raw', 'raw_second']:
            execute('gate3d_' + label, [gpuls, '-c', f'read {aig}; gate3d_rewrite_execution_audit -o {d / label} --region-map {owner}'])
        execute('gate3d_analysis', [py, tools / 'gate3d_analyze.py', '--raw', d / 'raw', '--raw-second', d / 'raw_second',
                '--gate3c', c, '--gate3a', a, '--source', source / 'projects/culs/src/algorithms/rewrite.cu', '--output', d, '--input', aig])
        execute('gate3e', [py, tools / 'gate3e_run.py', '--cutlib', source / 'projects/litecut', '--original-aig', aig,
                '--checkpoint', run / 'input/student_checkpoint', '--library', source / 'projects/culs/src/algorithms/rewrite_library.inc',
                '--gate3c', c, '--gate3b', b / 'c1355_orig', '--gate3d', d, '--region-map', owner,
                '--gpuls', gpuls, '--abc', abc, '--out-dir', e])
        execute('gate3f', [py, tools / 'gate3f_run.py', '--original-aig', aig, '--library', run / 'input/asap7',
                '--gpuls', gpuls, '--region-map', owner, '--gate3e', e, '--abc', abc, '--out-dir', f])
        # Independently verify all emitted AIGs. Negative text takes precedence.
        cec = []
        emitted = sorted(set(e.rglob('*.aig')) | set(f.rglob('*.aig')))
        if not emitted:
            raise RuntimeError('No replay AIGs were produced.')
        for index, candidate in enumerate(emitted):
            text = execute(f'cec_strict_{index}', [abc, '-c', f'cec {aig} {candidate}'])
            passed = bool(re.search(r'Networks are equivalent\b', text, re.I)) and not re.search(r'NOT EQUIVALENT|not equivalent|verification failed', text, re.I)
            cec.append({'aig': str(candidate), 'sha256': sha256(candidate), 'pass': passed})
            if not passed:
                raise RuntimeError(f'Strict CEC failed for {candidate}')
        (output / 'independent_cec.json').write_text(json.dumps(cec, indent=2) + '\n')
        meta['state'] = 'COMPLETED'
    except Exception as exc:
        meta.update(state='FAILED', error=str(exc))
        (output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        meta['outputs'] = str(output)
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()

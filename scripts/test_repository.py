#!/usr/bin/env python3
"""Inventory and test every source, with explicit limited scopes and omissions."""
import argparse
from collections import Counter
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from audit_programs import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not str(run).startswith('/data/wushangge/'):
        raise SystemExit('Repository tests require Slurm and /data.')
    root = run / 'src'
    sys.path.insert(0, str(root))
    from global_timing.common import run_tool
    output = run / 'output/repository_tests'
    audit(root, output)
    rows = json.loads((output / 'program_inventory.json').read_text())
    build = args.tool_run / 'build/framework_tools'
    compile_records = []
    for path in (build / 'abc').rglob('*.o'):
        relative = path.relative_to(build / 'abc')
        for suffix in ['.c', '.cpp', '.cc']:
            source = Path('third_party/abc') / relative.with_suffix(suffix)
            if (root / source).is_file():
                compile_records.append(str(source))
    for path in (build / 'culs').rglob('*.o'):
        text = str(path.relative_to(build / 'culs'))
        if '/src/' in text:
            compile_records.append('projects/culs/src/' + text.split('/src/', 1)[1][:-2])
    compiled = {relative for relative in compile_records
                if (args.tool_run / 'src' / relative).is_file() and
                hashlib.sha256((root / relative).read_bytes()).digest() ==
                hashlib.sha256((args.tool_run / 'src' / relative).read_bytes()).digest()}
    for index, row in enumerate(rows):
        path = root / row['path']
        row['functional'] = 'NOT_MEASURED_PER_FILE'
        row['functional_reason'] = 'Integration coverage is recorded by named tests; no per-line/function coverage claim.'
        if row['kind'] == '.py' and row['syntax'] == 'PASS':
            modes = ['import']
            source = path.read_text()
            if row['entrypoint'] and 'argparse' in source and ('parse_args(' in source or 'parse_runtime_args(' in source):
                modes.append('help')
            for mode in modes:
                scratch = output / 'probes' / f'{index:04d}_{mode}'
                command = [sys.executable, '-B', str(root / 'scripts/probe_python.py'), '--root', str(root),
                           '--target', str(path), '--scratch', str(scratch), '--mode', mode]
                try:
                    run_tool(command, output / 'logs', f'{index:04d}_{mode}', timeout=35)
                    result = json.loads((scratch / 'result.json').read_text())
                    row[mode] = result['status']
                    row[mode + '_reason'] = result.get('reason')
                    row[mode + '_evidence'] = str(scratch / 'result.json')
                except Exception as exc:
                    row[mode] = 'FAIL'
                    row[mode + '_reason'] = str(exc)
            row['runtime'] = row.get('help', 'SKIP')
            row['runtime_reason'] = ('Isolated --help smoke only; no workload execution.' if 'help' in row else
                'No safe standalone CLI smoke; actual import outcome recorded separately. Workload requires its datasets/configuration.')
        elif row['path'] in compiled:
            row.update(syntax='PASS', syntax_reason='Translation unit compiled in recorded framework build.',
                       build='PASS', build_evidence=str(build / 'build_report.json'),
                       runtime='SKIP', runtime_reason='Linked native translation unit; real binary integration tests exist, per-file runtime coverage unmeasured.')
        elif row['kind'] in {'.sh', '.slurm'}:
            literals = sorted(set(re.findall(r'(?:/home|/data|/usr/local)/[^\s"\';]+', path.read_text())))
            row.update(runtime='SKIP', runtime_reason='Historical wrapper retains fixed run/output or compiler paths; executing it could overwrite preserved evidence. bash -n checked; inspect required paths.',
                       required_paths=literals)
        elif row['path'].startswith('history/abc_variants/'):
            row.update(runtime='SKIP', runtime_reason='Variant overlay requires its own ABC baseline build and exact historical experiment; not the framework binary.')
        elif row['path'].startswith('legacy/'):
            row.update(runtime='SKIP', runtime_reason='Historical Gate1 snapshot has a separate unbuilt executable and fixed historical replay inputs; not the current CULS build.')
        else:
            row.update(runtime='SKIP', runtime_reason='Header/build support/optional translation unit or notebook; no isolated executable. Compiler coverage only asserted where object evidence exists.')
        print(json.dumps({'path': row['path'], 'syntax': row['syntax'], 'import': row['import'], 'runtime': row['runtime']}), flush=True)
    with (output / 'test_matrix.json').open('x') as stream:
        json.dump(rows, stream, indent=2)
    summary = {'count': len(rows), 'entrypoints': sum(r['entrypoint'] for r in rows),
               'syntax': dict(Counter(r['syntax'] for r in rows)), 'import': dict(Counter(r['import'] for r in rows)),
               'runtime_smoke': dict(Counter(r['runtime'] for r in rows)),
               'compiled_translation_units': len(compiled), 'all_files_recorded': True,
               'scope': 'Every snapshot source recorded; runtime smoke is not full functionality. Failures and skipped historical workloads remain explicit.',
               'database': json.loads((output / 'summary.json').read_text())['database_hits']}
    with (output / 'matrix_summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
    with (output / 'environment_packages.json').open('x') as stream:
        json.dump(sorted([{'name': d.metadata['Name'], 'version': d.version} for d in importlib.metadata.distributions()],
                         key=lambda item: item['name'].lower()), stream, indent=2)
    metadata_path = run / 'run.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(state='COMPLETED_WITH_RECORDED_OMISSIONS', outputs=str(output / 'matrix_summary.json'),
                    slurm_job_id=os.environ['SLURM_JOB_ID'], experiment_executed=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()

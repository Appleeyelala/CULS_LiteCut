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


def functional_evidence(root, runs):
    """Attach named executed tests only when the tested source bytes match."""
    evidence = {}

    def add(run, relative, report, name, status, scope):
        current, tested = root / relative, run / 'src' / relative
        if not current.is_file() or not tested.is_file() or current.read_bytes() != tested.read_bytes():
            return
        evidence.setdefault(relative, []).append({'test': name, 'status': status, 'scope': scope,
            'report': str(report), 'source_sha256': hashlib.sha256(current.read_bytes()).hexdigest()})

    modules = {'test_structure': ['structure'], 'test_candidates': ['candidates'], 'test_arrival': ['arrival'],
               'test_cohort': ['cohort'],
               'test_evaluation': ['evaluation'], 'test_pipeline': ['pipeline'],
               'test_processes': ['processes', 'common'], 'test_surrogate': ['surrogate', 'optimization'],
               'test_truth': ['truth', 'litecut'], 'test_paths': ['path_analysis'],
               'test_search_analysis': ['search_analysis'], 'test_path_changes': ['path_changes']}
    specific = {
        'test_liberty_paths_reconstruct_arrivals_without_changing_mapping':
            ['framework/abc/path_report.c.inc', 'framework/abc/endpoint_report.c.inc', 'global_timing/path_analysis.py'],
        'test_context_cut_cec_accepts_relations_and_rejects_wrong_function': ['global_timing/cut_validation.py'],
        'test_litecut_batch_cache_and_all_candidate_functions': ['global_timing/litecut.py',
            'projects/litecut/gate2a_candidate_inference.py', 'projects/litecut/src/extract_cutlib_student/modeling.py',
            'projects/litecut/src/data_preprocess/feat_extraction_v3.py'],
        'test_reused_baseline_resolves_record_provenance_at_variable_depth': ['scripts/analyze_timing_paths.py']}
    specific['test_student_features_equal_full_prefix_without_computing_unused_columns'] = [
        'projects/litecut/src/data_preprocess/feat_extraction_v3.py']
    specific['test_native_control_failure_preserves_real_other_method_evaluations'] = [
        'global_timing/pipeline.py', 'global_timing/evaluation.py', 'global_timing/reporting.py',
        'global_timing/path_changes.py', 'scripts/summarize_benchmarks.py']
    for name in ['test_raw_measurement_validation_rejects_tampered_cec_paths_and_netlist',
                 'test_report_priority_preserves_failure_and_exact_source_path']:
        specific[name] = ['scripts/analyze_path_changes.py']
    imported = {'gate1c_r_io_signature': 'legacy/gate1_aigrewrite/gate1c_r_io_signature.py',
                'gate1c_select': 'legacy/gate1_aigrewrite/gate1c_select.py',
                'upstream_gia_test': 'third_party/abc/test/gia/gia_test.cc'}
    for run in dict.fromkeys(runs):
        report = run / 'output/tests.json'
        if report.is_file():
            for record in json.loads(report.read_text())['records']:
                module = record['test'].split('.')[0]
                add(run, f'tests/{module}.py', report, record['test'], record['status'], 'Executed named test, not line coverage.')
                for target in modules.get(module, []):
                    add(run, f'global_timing/{target}.py', report, record['test'], record['status'],
                        'Related named unit test; only its assertions are covered, not the entire module.')
                for target in specific.get(record['test'].split('.')[-1], []):
                    add(run, target, report, record['test'], record['status'], 'Named assertion evidence; no whole-file or line-coverage claim.')
        report = run / 'output/imported_tests/summary.json'
        if report.is_file():
            for record in json.loads(report.read_text()):
                add(run, imported[record['program']], report, record['program'], record['status'],
                    'Real retained-input CLI assertions or four unchanged upstream Gia unit tests.')
        report = run / 'output/native_writer/summary.json'
        if report.is_file():
            record = json.loads(report.read_text())
            for relative in ['projects/culs/src/algorithms/rewrite.cu', 'projects/culs/src/algorithms/rewrite.h',
                             'tests/native/replay_constants.cpp']:
                add(run, relative, report, 'native_writer_constants', record['status'], record['scope'])
        report = run / 'output/cycle_regression.json'
        if report.is_file():
            record = json.loads(report.read_text())
            for relative in ['projects/culs/src/algorithms/rewrite.cu', 'projects/culs/src/algorithms/rewrite.h',
                             'global_timing/candidates.py', 'global_timing/pipeline.py']:
                add(run, relative, report, 'retained_transitive_cycle_repair', record['status'], record['scope'])
        report = run / 'output/historical_analysis_tests/summary.json'
        if report.is_file():
            for record in json.loads(report.read_text())['records']:
                add(run, record['source_relative'], report, 'retained_R9_analysis', record['status'], record['scope'])
        report = run / 'output/native_validation/summary.json'
        if report.is_file():
            for record in json.loads(report.read_text())['records']:
                targets = ['scripts/test_native_variants.py']
                if record['variant'] == 'gate1_aigrewrite':
                    targets += ['legacy/gate1_aigrewrite/src/' + name for name in
                                ['algorithms/rewrite.cu', 'aig_manager.cu', 'command_manager.cpp']]
                    scope = 'Actual normalization and four retained dependent two-decision replays, exact provenance/graph comparison and original-input CEC.'
                else:
                    targets += ['scripts/build_native_variants.py']
                    variants = json.loads((run / 'src/provenance/abc_variants.json').read_text())
                    variant = next(v for v in variants if Path(v['overlay']).name == record['variant'])
                    targets += [str(Path(variant['overlay']) / name) for name in
                                variant['changed_files'] + variant['extra_source_files']]
                    scope = 'Actual mapped CEC and reachable/unreachable pair checks; optional mapper/local-arrival diagnostics. Named command coverage only; build script helper exercised without rebuilding.'
                for relative in targets:
                    add(run, relative, report, 'rebuilt_native:' + record['variant'], record['status'], scope)
        for directory, targets, label in [
            ('path_changes', ['scripts/analyze_path_changes.py', 'global_timing/path_changes.py'],
             'Retained CEC/timing/netlist/trace integrity and cross-selection endpoint comparisons; no new mapping.'),
            ('search_quality', ['scripts/analyze_search_quality.py', 'global_timing/search_analysis.py'],
             'Saved search-label provenance, error and selection-source audit; no new timing measurements.'),
            ('gradient_profile', ['scripts/profile_gradient.py'],
             'Exact saved calibration and gradient trace reconstruction with CPU profiling.')]:
            report = run / 'output' / directory / 'summary.json'
            if report.is_file():
                record = json.loads(report.read_text())
                status = ('PASS' if record['audit_failure_count'] == 0 else 'FAIL') if directory in {'search_quality', 'path_changes'} else record['status']
                if directory == 'path_changes' and (record['invalid_report_snapshots'] or not record['analyzed_inputs']):
                    status = 'FAIL'
                for relative in targets:
                    add(run, relative, report, directory, status, label)
        for directory, targets, scope in [
            ('selection_profiles', ['scripts/profile_selection.py', 'global_timing/candidates.py',
                'global_timing/surrogate.py', 'global_timing/pipeline.py'],
             'Saved-input catalogue/ordered-cost checks; prior selections compared only where explicitly available.'),
            ('arrival_paths', ['scripts/analyze_timing_paths.py', 'global_timing/path_analysis.py',
                'framework/abc/path_report.c.inc'],
             'Fresh CEC, unchanged saved measurements/netlist, and reconstructed near-critical arrival chains.')]:
            report = run / 'output' / directory / 'summary.json'
            if report.is_file():
                for record in json.loads(report.read_text())['records']:
                    for relative in targets:
                        # A catalogue-only profile does not exercise selection.
                        if directory == 'selection_profiles' and relative in {
                                'global_timing/surrogate.py', 'global_timing/pipeline.py'} and 'option_checks' not in record:
                            continue
                        if (directory == 'selection_profiles' and relative == 'global_timing/pipeline.py' and
                                record.get('full_independent_selection_executed') is False):
                            continue
                        add(run, relative, report, directory + ':' + record['benchmark'], record['status'], scope)
        report = run / 'output/historical_baseline/commands.json'
        if report.is_file():
            for record in json.loads(report.read_text()):
                for argument in record['argv']:
                    path = Path(argument)
                    if path.suffix == '.py' and path.is_relative_to(run / 'src'):
                        add(run, str(path.relative_to(run / 'src')), report, record['name'],
                            'PASS' if record.get('returncode') == 0 and 'error' not in record else 'FAIL',
                            'Actual historical Gate3 invocation; inspect stage outputs and independent CEC report.')
    return evidence


def native_compile_evidence(root, runs):
    """Record actual native objects only for unchanged source snapshots."""
    evidence = {}
    for run in runs:
        report = run / 'output/imported_native/summary.json'
        if not report.is_file():
            raise FileNotFoundError('Native build has no final report: ' + str(report))
        records = json.loads(report.read_text())['records']
        variants = {Path(v['overlay']).name: v for v in
                    json.loads((run / 'src/provenance/abc_variants.json').read_text())}
        for record in records:
            if record.get('build') != 'PASS':
                continue
            binary = Path(record['binary'])
            if hashlib.sha256(binary.read_bytes()).hexdigest() != record['binary_sha256']:
                raise ValueError('Recorded native binary hash changed: ' + str(binary))
            sources = []
            if record['variant'] == 'gate1_aigrewrite':
                for obj in binary.parent.rglob('*.o'):
                    text = obj.relative_to(binary.parent).as_posix()
                    if text.startswith('CMakeFiles/abcg.dir/src/'):
                        sources.append('legacy/gate1_aigrewrite/src/' + text.split('/src/', 1)[1][:-2])
            else:
                variant = variants[record['variant']]
                for obj in binary.parent.rglob('*.o'):
                    for suffix in ['.c', '.cpp', '.cc']:
                        relative = obj.relative_to(binary.parent).with_suffix(suffix)
                        overlay = Path(variant['overlay']) / relative
                        selected = overlay if (run / 'src' / overlay).is_file() else Path(variant['base']) / relative
                        if (run / 'src' / selected).is_file():
                            sources.append(str(selected))
            for relative in sources:
                current, tested = root / relative, run / 'src' / relative
                if current.is_file() and current.read_bytes() == tested.read_bytes():
                    evidence.setdefault(relative, []).append({'report': str(report), 'variant': record['variant'],
                        'binary_sha256': record['binary_sha256'],
                        'source_sha256': hashlib.sha256(current.read_bytes()).hexdigest()})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    parser.add_argument('--evidence-run', type=Path, action='append', default=[],
                        help='Prior actual tests to attach only for byte-identical source files.')
    parser.add_argument('--native-build-run', type=Path, action='append', default=[],
                        help='Completed isolated historical native builds with object and source evidence.')
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
    native_compiled = native_compile_evidence(root, args.native_build_run)
    compiled.update(native_compiled)
    functional = functional_evidence(root, [run, *args.evidence_run])
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
            if row['path'] in native_compiled:
                row.update(syntax_reason='Translation unit compiled in an isolated historical native build; exact source bytes checked.',
                           build_evidence=native_compiled[row['path']])
        elif row['kind'] in {'.sh', '.slurm'}:
            literals = sorted(set(re.findall(r'(?:/home|/data|/usr/local)/[^\s"\';]+', path.read_text())))
            row.update(runtime='SKIP', runtime_reason='Historical wrapper retains fixed run/output or compiler paths; executing it could overwrite preserved evidence. bash -n checked; inspect required paths.',
                       required_paths=literals)
        elif row['path'].startswith('history/abc_variants/'):
            row.update(runtime='SKIP', runtime_reason='Variant header or support file has no standalone runtime; named native command evidence is attached where supplied.')
        elif row['path'].startswith('legacy/'):
            row.update(runtime='SKIP', runtime_reason='Historical header or optional entry has no isolated runtime; separate Gate1 binary evidence is attached where supplied.')
        else:
            row.update(runtime='SKIP', runtime_reason='Header/build support/optional translation unit or notebook; no isolated executable. Compiler coverage only asserted where object evidence exists.')
        if row['path'] in functional:
            evidence = functional[row['path']]
            row['functional_evidence'] = evidence
            statuses = {item['status'] for item in evidence}
            row['functional'] = ('FAIL' if statuses & {'FAIL', 'ERROR'} else
                                 'NAMED_TESTS_PASS' if 'PASS' in statuses else 'SKIP')
            row['functional_reason'] = 'Named, source-matched evidence only; compilation, import and CLI smoke remain separate.'
        print(json.dumps({'path': row['path'], 'syntax': row['syntax'], 'import': row['import'], 'runtime': row['runtime']}), flush=True)
    with (output / 'test_matrix.json').open('x') as stream:
        json.dump(rows, stream, indent=2)
    summary = {'count': len(rows), 'entrypoints': sum(r['entrypoint'] for r in rows),
               'syntax': dict(Counter(r['syntax'] for r in rows)), 'import': dict(Counter(r['import'] for r in rows)),
               'runtime_smoke': dict(Counter(r['runtime'] for r in rows)),
               'functional': dict(Counter(r['functional'] for r in rows)),
               'compiled_translation_units': len(compiled), 'all_files_recorded': True,
               'historical_native_translation_units': len(native_compiled),
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

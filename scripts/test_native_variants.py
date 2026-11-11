#!/usr/bin/env python3
"""Validate rebuilt historical tools in a new run without rebuilding or overwriting evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import digest, run_tool, save_json, tool_path
from global_timing.evaluation import cec_passed
from global_timing.structure import aig_structure_digest, parse_aig
from build_native_variants import abc_regression


def verify_sources(source, built_source, prefixes):
    """Reuse binaries only when their recorded native source snapshot matches.

    ABC's archive-expanded .gitcommit is build metadata, not functional source;
    preserve both values explicitly instead of pretending their bytes match.
    """
    checked, metadata = {}, {}
    for prefix in prefixes:
        left, right = source / prefix, built_source / prefix
        left_files = {p.relative_to(left) for p in left.rglob('*') if p.is_file()}
        right_files = {p.relative_to(right) for p in right.rglob('*') if p.is_file()}
        if left_files != right_files:
            raise ValueError('Native source file set changed: ' + prefix)
        for relative in sorted(left_files):
            a, b = left / relative, right / relative
            key = str(Path(prefix) / relative)
            if relative.name == '.gitcommit':
                metadata[key] = {'validation_snapshot': a.read_text(), 'build_snapshot': b.read_text()}
                continue
            expected = digest(a)
            if expected != digest(b):
                raise ValueError('Native source differs from binary build snapshot: ' + key)
            checked[key] = expected
    return {'identical_source_files': checked, 'archive_commit_metadata': metadata}


def verify_binary(record):
    binary = Path(record['binary']).resolve(strict=True)
    if record.get('build') != 'PASS' or digest(binary) != record['binary_sha256']:
        raise ValueError('Native binary build provenance/hash mismatch.')
    return binary


def gate1_regression(binary, abc, baseline, retained, output):
    output.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GATE1A_', 'GATE1B_', 'GATE1C_'))}
    env['LD_LIBRARY_PATH'] = str(binary.parent) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    original = parse_aig(baseline)
    if (original.pi_count, original.po_count, original.and_count) != (41, 32, 502):
        raise ValueError('Gate1 regression requires the retained 502-AND c1355 input.')
    records = []

    def prove(candidate, work, name):
        parsed = parse_aig(candidate)
        if (parsed.pi_count, parsed.po_count) != (original.pi_count, original.po_count):
            raise ValueError('Gate1 changed the PI/PO interface.')
        log = run_tool([abc, '-c', f'cec -n {tool_path(baseline)} {tool_path(candidate)}'], work, name, 120)
        if not cec_passed(log):
            raise ValueError('Gate1 output was not proved equivalent to the original input.')
        return {'cec': 'PASS', 'and_count': parsed.and_count, 'sha256': digest(candidate),
                'structure_sha256': aig_structure_digest(candidate)}

    normalized = output / 'normalized_original.aig'
    run_tool([binary, '-c', f'gread {tool_path(baseline)}; gwrite {tool_path(normalized)}'], output, 'normalize', 120, env)
    normalization = prove(normalized, output, 'normalize_cec')
    for label, x, y in [('AC', 0, 0), ('AD', 0, 1), ('BC', 1, 0), ('BD', 1, 1)]:
        work = output / label
        work.mkdir()
        output_aig, provenance = work / 'rewritten.aig', work / 'provenance.json'
        spec = f'146:0:{x},226:0:{y}'
        forced = dict(env, GATE1C_SPEC=spec, GATE1A_ROOT='146', GATE1A_CUT='0',
                      GATE1A_SUB=str(x), GATE1A_PROVENANCE=str(provenance))
        run_tool([binary, '-c', f'gread {tool_path(baseline)}; grw; gwrite {tool_path(output_aig)}'],
                 work, 'rewrite', 180, forced)
        actual = json.loads(provenance.read_text())
        expected_choices = [(146, 0, x), (226, 0, y)]
        observed_choices = [(d['target_root'], d['cut_slot'], d['subgraph_index']) for d in actual['decisions']]
        if actual['forced_spec'] != spec or observed_choices != expected_choices or actual['rewritten_root_count'] != 2:
            raise ValueError('Gate1 did not select exactly the requested two decisions.')
        if actual['unexpected_rewritten_roots'] or actual['conflicts']:
            raise ValueError('Gate1 reported an unexpected rewrite or conflict.')
        for row in actual['decisions']:
            if not all(row[k] is True for k in ['candidate_selected', 'seen', 'evaluated', 'built', 'installed']):
                raise ValueError('Gate1 did not install a requested selection.')
            if row['installation_kind'] not in {'PHYSICAL_REPLACEMENT', 'NO_OP_FULL_CORRESPONDENCE'}:
                raise ValueError('Gate1 reported an unrecognized installation result.')
        result = dict(configuration=label, requested_choices=expected_choices, **prove(output_aig, work, 'cec_original'))
        retained_json, retained_aig = retained / (label + '.json'), retained / (label + '.aig')
        expected = json.loads(retained_json.read_text())
        if actual != expected:
            raise ValueError('Gate1 provenance differs from the retained historical configuration: ' + label)
        if result['structure_sha256'] != aig_structure_digest(retained_aig):
            raise ValueError('Gate1 serialized graph differs from the retained historical configuration: ' + label)
        result.update(status='PASS', provenance=actual, retained_provenance_sha256=digest(retained_json),
                      retained_aig_sha256=digest(retained_aig), retained_structure_match=True,
                      note='installed is the historical Gate1 field; full correspondence can be a selected no-op, not a physical replacement.')
        save_json(work / 'result.json', result)
        records.append(result)
    return {'normalization': normalization, 'configurations': records,
            'scope': 'Retained dependent two-decision Gate1 replay; not general independent-decision feasibility.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--build-run', type=Path, required=True)
    parser.add_argument('--scope', choices=['abc', 'gate1', 'all'], default='all')
    parser.add_argument('--abc', type=Path, help='CEC binary for the GPU Gate1 scope')
    args = parser.parse_args()
    run, build = args.run.resolve(), args.build_run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not all(p.is_relative_to('/data/wushangge') for p in [run, build]):
        parser.error('Native validation requires Slurm and prepared /data workspaces.')
    if args.scope != 'abc' and args.abc is None:
        parser.error('--abc is required for Gate1 CEC.')
    output = run / 'output/native_validation'
    output.mkdir(exist_ok=False)
    metadata = json.loads((run / 'run.json').read_text())
    build_metadata = json.loads((build / 'run.json').read_text())
    metadata.update(state='RUNNING', experiment_executed=True, slurm_job_id=os.environ['SLURM_JOB_ID'],
                    build_run=str(build), build_source_commit=build_metadata['source_commit'],
                    environment={'python': sys.version, 'path': os.environ.get('PATH'),
                                 'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')}})
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    records = []
    input_paths = [run / 'input/design.aig', run / 'input/asap7']
    if args.scope in {'abc', 'all'}:
        candidate = build / 'input/retained_R9_candidate.aig'
        if digest(candidate) != '59dbe54c03d28c336775785e42d3e4781713e66e96769ac688fd38977672b5e1':
            raise ValueError('Retained R9 candidate checksum changed.')
        input_paths.append(candidate)
        for variant in json.loads((run / 'src/provenance/abc_variants.json').read_text()):
            label = Path(variant['overlay']).name
            record = {'variant': label, 'status': 'RUNNING'}
            try:
                original = json.loads((build / 'output/imported_native' / (label + '.json')).read_text())
                binary = verify_binary(original)
                sources = verify_sources(run / 'src', build / 'src', [variant['base'], variant['overlay']])
                save_json(output / (label + '.sources.json'), sources)
                record.update(binary=str(binary), binary_sha256=digest(binary),
                              prior_validation_status=original['status'], prior_validation_error=original.get('error'))
                record['functional'] = abc_regression(binary, output / label, run / 'input/design.aig',
                    run / 'input/asap7', bool(variant['extra_source_files']), 'local-arrival' in label, candidate)
                record['status'] = 'PASS'
            except Exception as exc:
                record.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            save_json(output / (label + '.json'), record)
            records.append(record)
            print(json.dumps(record), flush=True)
    if args.scope in {'gate1', 'all'}:
        record = {'variant': 'gate1_aigrewrite', 'status': 'RUNNING'}
        try:
            original = json.loads((build / 'output/imported_native/gate1.json').read_text())
            binary = verify_binary(original)
            sources = verify_sources(run / 'src', build / 'src', ['third_party/abc', 'legacy/gate1_aigrewrite'])
            save_json(output / 'gate1.sources.json', sources)
            retained = Path('/data/wushangge/CULS_LiteCut/history/imports/gate1/gate1c_r_out')
            baseline_source = Path('/data/public/AIGRewrite/case/c1355.aig')
            baseline = run / 'input/gate1_c1355.aig'
            with baseline_source.open('rb') as incoming, baseline.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            if digest(baseline) != '88df0a75b9328378b869843ba675c316d05a8bd812b7de22effc12af51f430ea':
                raise ValueError('Historical Gate1 input checksum changed.')
            input_paths.append(baseline)
            record.update(binary=str(binary), binary_sha256=digest(binary), abc=str(args.abc),
                          abc_sha256=digest(args.abc), baseline_sha256=digest(baseline))
            record['functional'] = gate1_regression(binary, args.abc, baseline, retained, output / 'gate1')
            record['status'] = 'PASS'
        except Exception as exc:
            record.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        save_json(output / 'gate1.json', record)
        records.append(record)
        print(json.dumps(record), flush=True)
    save_json(output / 'summary.json', {'records': records, 'source_commit': metadata['source_commit'],
        'build_run': str(build), 'build_source_commit': build_metadata['source_commit'],
        'inputs': [{'path': str(p), 'sha256': digest(p)} for p in input_paths],
        'scope': args.scope, 'slurm_job_id': os.environ['SLURM_JOB_ID']})
    failed = any(row['status'] != 'PASS' for row in records)
    metadata.update(state='COMPLETED_WITH_FAILURES' if failed else 'COMPLETED',
                    outputs=str(output / 'summary.json'))
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())

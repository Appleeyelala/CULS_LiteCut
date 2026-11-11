#!/usr/bin/env python3
"""Rebuild imported ABC overlays and Gate1 without changing historical runs."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.common import digest, run_tool, save_json, tool_path
from global_timing.evaluation import cec_passed


def blif_body(path):
    return '\n'.join(line for line in path.read_text().splitlines() if not line.lstrip().startswith('#'))


def abc_regression(binary, work, baseline, library, has_debug, has_local, candidate):
    work.mkdir(parents=True, exist_ok=False)
    measurements = []
    for observed in ([False, True] if has_debug else [False]):
        output = work / ('observed' if observed else 'plain')
        output.mkdir()
        env = dict(os.environ)
        for key in ['AIGREWRITE_MAP_DEBUG_NODE', 'AIGREWRITE_MAP_DEBUG_IDS', 'AIGREWRITE_R9_LOCAL']:
            env.pop(key, None)
        if observed:
            env.update(AIGREWRITE_MAP_DEBUG_NODE='213', AIGREWRITE_MAP_DEBUG_IDS='17,33,151,188,197,209,213')
            if has_local:
                env['AIGREWRITE_R9_LOCAL'] = '1'
        command = f'read_lib {tool_path(library)}; read_aiger {tool_path(baseline)}; map; '
        command += 'print_pair_delay -tv po11 pi04; '
        if observed and has_local:
            command += f'local_r9_arrival {tool_path(baseline)} {tool_path(candidate)}; '
        mapped = output / 'mapped.blif'
        command += f'write_blif {tool_path(mapped)}'
        log = run_tool([binary, '-c', command], output, 'mapping', 300, env=env)
        matches = re.findall(r'PAIR_RESULT .*?reachable=1 .*?pair_delay=([0-9.eE+-]+)', log)
        if len(matches) != 1 or 'upper_bound=PASS single_source_consistent=PASS' not in log or 'PAIR_TRACE_ERROR' in log:
            raise ValueError('Rebuilt source-pair command did not pass its actual timing checks.')
        cec = run_tool([binary, '-c', f'cec -n {tool_path(baseline)} {tool_path(mapped)}'], output, 'cec', 120)
        if not cec_passed(cec):
            raise ValueError('Rebuilt mapped network did not pass CEC.')
        item = {'observed': observed, 'pair_delay_ps': float(matches[0]), 'cec': 'PASS',
                'mapped_sha256': digest(mapped), 'metric': 'source-restricted fixed-MIO pair timing, not Liberty global timing'}
        if observed:
            events = [json.loads(line[7:]) for line in log.splitlines() if line.startswith('MAPDBG ')]
            if not events or not any(e.get('type') == 'node' and e.get('aiger_var') == 213 for e in events):
                raise ValueError('Mapper diagnostic did not identify the requested original function point.')
            item['mapper_events'] = len(events)
            if blif_body(mapped) != blif_body(work / 'plain/mapped.blif') or item['pair_delay_ps'] != measurements[0]['pair_delay_ps']:
                raise ValueError('Diagnostic instrumentation changed the mapped network or pair measurement.')
            if has_local:
                local = [json.loads(line[8:]) for line in log.splitlines() if line.startswith('R9LOCAL ')]
                if 'R9CHECK baseline_local=PASS' not in log or len(local) != 2 or 'R9STOP' in log:
                    raise ValueError('Local arrival entry did not pass its baseline sanity check.')
                expected = {'baseline': 109.239990, 'candidate': 106.219986}
                if any(abs(row['arrival'] - expected[row['variant']]) > 3e-5 for row in local):
                    raise ValueError('Rebuilt local arrival differs from the retained R9 reference.')
                item['local_arrivals'] = local
        measurements.append(item)
    fixture = work / 'unreachable.blif'
    fixture.write_text('.model unreachable\n.inputs pi0 pi1\n.outputs po0\n.names pi0 po0\n1 1\n.end\n')
    log = run_tool([binary, '-c', f'read_lib {tool_path(library)}; read_blif {tool_path(fixture)}; '
                   'strash; map; print_pair_delay po0 pi1'], work, 'unreachable', 120)
    if 'reachable=0' not in log or 'pair_delay=NA' not in log:
        raise ValueError('Unreachable pair fixture was not reported without a numeric label.')
    return {'measurements': measurements, 'unreachable_pair': 'PASS'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('Native builds require Slurm and a prepared /data run.')
    if not 1 <= args.jobs <= int(os.environ.get('SLURM_CPUS_PER_TASK', '1')):
        parser.error('Build parallelism must fit the allocation.')
    source = run / 'src'
    root = run / 'build/imported_native'
    root.mkdir(exist_ok=False)
    output = run / 'output/imported_native'
    output.mkdir()
    metadata = json.loads((run / 'run.json').read_text())
    metadata.update(state='RUNNING', slurm_job_id=os.environ['SLURM_JOB_ID'], experiment_executed=True,
                    environment={'python': sys.version, 'path': os.environ.get('PATH'),
                                 'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')}})
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    candidate_source = Path('/data/wushangge/runs/CULS-mapper-internal-20261003-34U6OZ/input/S002.aig')
    candidate = run / 'input/retained_R9_candidate.aig'
    shutil.copyfile(candidate_source, candidate)
    if digest(candidate) != '59dbe54c03d28c336775785e42d3e4781713e66e96769ac688fd38977672b5e1':
        raise ValueError('Retained R9 candidate checksum changed.')
    records = []
    for variant in json.loads((source / 'provenance/abc_variants.json').read_text()):
        label = Path(variant['overlay']).name
        work = root / label
        record = {'variant': label, 'source_overlay': variant['overlay'], 'status': 'RUNNING',
                  'base': variant['base'], 'base_upstream_commit': 'c738ed6e86e6ca10bcef0d4c649cc07d925b4be8'}
        try:
            shutil.copytree(source / variant['base'], work)
            paths = variant['changed_files'] + variant['extra_source_files']
            record['overlay_hashes'] = {}
            for relative in paths:
                incoming = source / variant['overlay'] / relative
                destination = work / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(incoming, destination)
                record['overlay_hashes'][relative] = digest(incoming)
            if variant['missing_base_files']:
                raise ValueError('Missing-base-file overlays need an explicit reconstruction policy.')
            logs = output / label
            logs.mkdir()
            run_tool(['make', '-C', work, '-j', str(args.jobs), 'ABC_USE_NO_READLINE=1',
                      'CC=/usr/local/bin/gcc', 'CXX=/usr/local/bin/g++', 'LD=/usr/local/bin/g++'], logs, 'build', 1800)
            binary = work / 'abc'
            record.update(build='PASS', binary=str(binary), binary_sha256=digest(binary))
            record['functional'] = abc_regression(binary, logs / 'regression', run / 'input/design.aig',
                run / 'input/asap7', bool(variant['extra_source_files']), 'local-arrival' in label, candidate)
            record['status'] = 'PASS'
        except Exception as exc:
            record.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        save_json(output / (label + '.json'), record)
        records.append(record)
        print(json.dumps({key: record.get(key) for key in ['variant', 'status', 'error']}), flush=True)
    legacy = {'variant': 'gate1_aigrewrite', 'status': 'RUNNING', 'source': 'legacy/gate1_aigrewrite',
              'abc_dependency': 'third_party/abc at recorded clean c738ed6 baseline; matches retained Gate1 abc HEAD/status'}
    try:
        work = root / 'gate1'
        shutil.copytree(source / 'legacy/gate1_aigrewrite', work / 'src')
        shutil.copytree(source / 'third_party/abc', work / 'src/abc')
        logs = output / 'gate1'
        logs.mkdir()
        run_tool(['cmake', '-S', work / 'src', '-B', work / 'build', '-DCMAKE_BUILD_TYPE=Release',
                  '-DCMAKE_C_COMPILER=/usr/local/bin/gcc', '-DCMAKE_CXX_COMPILER=/usr/local/bin/g++',
                  '-DCMAKE_CUDA_HOST_COMPILER=/usr/local/bin/g++', '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc',
                  '-DCMAKE_CUDA_ARCHITECTURES=86;89'], logs, 'configure', 300)
        run_tool(['cmake', '--build', work / 'build', '--target', 'abcg', '-j', str(args.jobs)], logs, 'build', 1800)
        binary = work / 'build/abcg'
        legacy.update(status='BUILT_NOT_GPU_VALIDATED', build='PASS', binary=str(binary), binary_sha256=digest(binary))
    except Exception as exc:
        legacy.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
    records.append(legacy)
    save_json(output / 'gate1.json', legacy)
    failed = any(row['status'] == 'FAIL' for row in records)
    report = {'records': records, 'source_commit': metadata['source_commit'], 'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'inputs': [{'path': str(p), 'sha256': digest(p)} for p in [run / 'input/design.aig', run / 'input/asap7', candidate]],
              'scope': 'Rebuilt historical overlays and their actual commands, kept separate from new-framework timing labels.'}
    save_json(output / 'summary.json', report)
    metadata.update(state='COMPLETED_WITH_FAILURES' if failed else 'COMPLETED_WITH_PENDING_GATE1_GPU_TEST',
                    outputs=str(output / 'summary.json'))
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())

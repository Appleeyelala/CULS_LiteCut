#!/usr/bin/env python3
"""Cold A/B/Ours driver; developer subsets precede a separately frozen full905."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ENTRY_TIME = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # A crash may leave an unlinked .partial file, never a half terminal JSON.
    # Hard-link publication preserves exclusive creation without overwrites.
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def recover_point(outer, row, method, source_commit):
    """A terminal old job may leave a point directory but no parent record.

    Never restart this attempt or infer success from incomplete artifacts. If
    its cold-command clock is absent, elapsed runtime is honestly unavailable.
    """
    attempt = json.loads((outer / 'point_attempt.json').read_text())
    if (attempt['source_commit'] != source_commit or attempt['input_sha256'] != row['sha256']
            or attempt['method'] != method):
        raise ValueError('Interrupted point belongs to another source/input/method.')
    previous = str(attempt['slurm_job_id'])
    if previous == os.environ['SLURM_JOB_ID']:
        raise RuntimeError('Existing point belongs to this live allocation; refusing a duplicate.')
    account = subprocess.check_output(['sacct', '-j', previous, '--format=JobIDRaw,State,End', '-n', '-P'], text=True)
    fields = next((line.split('|') for line in account.splitlines() if line.split('|')[0] == previous), None)
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
    if fields is None or fields[1].split()[0].rstrip('+') not in terminal:
        raise RuntimeError('Previous attempt job is not confirmed terminal; do not restart it.')
    command_file = outer / 'cold_method.command.json'
    command = json.loads(command_file.read_text()) if command_file.exists() else None
    result_path = outer / 'child_result.json'
    result = json.loads(result_path.read_text()) if command and command['returncode'] == 0 and result_path.exists() else None
    good = bool(result and result.get('status') == 'PASS' and result['selected']['metrics'].get('cec') == 'PASS')
    clock_path = outer / 'pre_launch.json'
    staging_s = json.loads(clock_path.read_text())['elapsed_s'] if clock_path.exists() else None
    record = {'index': row['index'], 'input': row['relative_path'], 'input_sha256': row['sha256'],
              'method': method, 'status': 'PASS' if good else 'TIMEOUT' if command and command.get('timed_out') else 'FAIL',
              'error': None if good else 'Interrupted prior terminal job; no substituted metrics or automatic retry.',
              'end_to_end_runtime_s': command['elapsed_s'] + staging_s if command and staging_s is not None else None,
              'runtime_scope': 'Recovered staging plus recorded cold-command wall; missing clock is N/A, not zero.',
              'result_path': str(result_path) if good else None, 'detail_path': str(outer),
              'source_commit': source_commit, 'slurm_job_id': previous,
              'recovered_from_terminal_job': {'job_id': previous, 'state': fields[1], 'end': fields[2]}}
    if good:
        record.update(metrics=result['selected']['metrics'], final_source=result['selected']['source'], backend=result['backend'])
    return record


def child(args):
    # These shared validation modules do not import Torch. Only Ours pays for
    # importing/preparing the Student and gradient stack in its cold child.
    from global_timing.evaluation import Evaluator
    from global_timing.common import run_tool, tool_path
    context = json.loads(args.context.read_text())
    outer = args.context.parent
    work = outer / 'work'
    source = outer / 'input.aig'
    if sha(source) != context['input']['sha256']:
        raise ValueError('Cold child input identity changed.')
    tools = context['tools']
    if args.method in ('A', 'B'):
        work.mkdir(exist_ok=False)
        evaluator = Evaluator(Path(tools['abc']['path']), Path(tools['library']['path']), source,
                              context['search']['timeout_s'])
        candidate = source
        if args.method == 'B':
            candidate = work / 'rewritten.aig'
            command = f'read_aiger {tool_path(source)}; strash; rewrite; write_aiger {tool_path(candidate)}'
            run_tool([str(evaluator.abc), '-c', command], work, 'native_rewrite', evaluator.timeout)
        metrics = evaluator.evaluate(candidate, work / 'evaluation')
        result = {'status': 'PASS', 'selected': {'metrics': metrics, 'aig': str(candidate),
                  'choices': None, 'source': 'original' if args.method == 'A' else 'abc_single_native_rewrite',
                  'applied_rewrites': 0 if args.method == 'A' else None},
                  'backend': {'device': 'cpu', 'reason': 'ABC baseline; no candidate/ML preparation.'}}
    else:
        import torch
        from global_timing.mffc_experiment import MffcExperiment, MffcSearchConfig
        torch.set_num_threads(1)
        experiment = MffcExperiment(source, tools['gpuls']['path'], tools['abc']['path'],
            tools['library']['path'], tools['checkpoint']['path'], tools['native_library']['path'], work,
            MffcSearchConfig(**context['search']), validation_workers=context['validation_workers'])
        result = experiment.run()
        result['torch'] = {'version': torch.__version__, 'threads': torch.get_num_threads(),
                           'cuda_device': torch.cuda.get_device_name() if torch.cuda.is_initialized() else None}
    counts, seconds = Counter(), Counter()
    for path in work.rglob('*.command.json'):
        command = json.loads(path.read_text())
        category = path.name.removesuffix('.command.json')
        counts[category] += 1
        seconds[category] += command['elapsed_s']
    result.update(method=args.method, input=context['input'], tools=tools,
                  tool_commands=dict(counts), tool_inclusive_seconds=dict(seconds),
                  source_commit=context['source_commit'], child_entry_to_result_s=time.perf_counter() - ENTRY_TIME,
                  slurm_job_id=os.environ['SLURM_JOB_ID'],
                  runtime_scope='Outer cold wall adds process startup/exit and input staging; tool sums may overlap.')
    save(outer / 'child_result.json', result)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--tool-run', type=Path)
    parser.add_argument('--native-build-run', type=Path)
    parser.add_argument('--indices', help='Explicit development subset; omitted only with a frozen full905 protocol.')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--check-only', action='store_true', help='Validate full905 preflight; never launch method children.')
    parser.add_argument('--case-timeout', type=int, default=1800)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--context', type=Path)
    parser.add_argument('--method', choices=['A', 'B', 'Ours'])
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):
        parser.error('Execute the committed /data source through Slurm.')
    if args.child:
        return child(args)
    if not all((args.run, args.manifest, args.tool_run, args.native_build_run)):
        parser.error('Run, frozen inventory, tool and native build directories are required.')
    run = args.run.resolve()
    if not run.is_relative_to('/data/wushangge') or args.case_timeout < 1:
        parser.error('Requires a data run and positive cold method timeout.')
    from global_timing.common import run_tool
    from global_timing.mffc_experiment import MffcSearchConfig
    lock = (run / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metadata = json.loads((run / 'run.json').read_text())
    inventory = json.loads(args.manifest.read_text())
    rows = inventory['rows']
    from dataclasses import asdict
    protocol = json.loads(args.protocol.read_text()) if args.protocol else {'search': asdict(MffcSearchConfig()), 'validation_workers': 2}
    MffcSearchConfig(**protocol['search']).validate()
    if not args.indices:
        if ((protocol.get('state') != 'FROZEN' and not args.check_only) or len(rows) != 905 or
                protocol.get('input_inventory_sha256') != sha(args.manifest)):
            parser.error('Full905 requires a frozen matching protocol; development must name indices.')
        if args.case_timeout != protocol['cold_method_timeout_s']:
            raise ValueError('Cold method timeout differs from the frozen protocol.')
        if os.environ.get('SLURM_JOB_NODELIST') != protocol['required_node']:
            raise ValueError('Formal/resumed suite must use the frozen host.')
    else:
        wanted = [int(i) for i in args.indices.split(',')]
        by_index = {r['index']: r for r in rows}
        if len(wanted) != len(set(wanted)) or any(i not in by_index for i in wanted):
            parser.error('Unknown/duplicate development indices.')
        rows = [by_index[i] for i in wanted]
    if protocol['validation_workers'] > int(os.environ.get('SLURM_CPUS_PER_TASK', '1')):
        raise ValueError('Actual validation workers exceed CPU allocation.')
    build = json.loads((args.tool_run / 'build/framework_tools/build_report.json').read_text())
    native = json.loads((args.native_build_run / 'build/mffc_gpu/build_report.json').read_text())
    if native['status'] != 'PASS' or native['source_sha256'] != sha(ROOT / 'framework/mffc_gpu/region_lse.cu'):
        raise ValueError('Native GPU library lacks successful exact-source validation.')
    tools = {k: build['binaries'][k] for k in ('gpuls', 'abc')}
    tools.update(library={'path': str(run / 'input/asap7')}, checkpoint={'path': str(run / 'input/student_checkpoint')},
                 native_library={'path': native['library'], 'sha256': native['library_sha256']})
    for name, tool in tools.items():
        actual = sha(tool['path'])
        if tool.get('sha256', actual) != actual:
            raise ValueError('Tool/asset identity changed: ' + name)
        tool['sha256'] = actual
    if not args.indices and {k: v['sha256'] for k, v in tools.items()} != protocol['frozen_tool_sha256']:
        raise ValueError('Formal tools/model/library differ from the frozen protocol.')
    gpu = subprocess.check_output(['nvidia-smi', '--id=' + os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
        '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True).strip()
    if not args.indices and protocol['required_gpu_name'] not in gpu:
        raise ValueError('Formal/resumed suite requires the frozen GPU type.')
    session = run / 'input' / ('session_' + os.environ['SLURM_JOB_ID'] + '.json')
    if not session.exists():
        save(session, {'job_id': os.environ['SLURM_JOB_ID'], 'node': os.environ.get('SLURM_JOB_NODELIST'),
                      'gpu': gpu, 'driver_argv': sys.argv, 'slurm_cpus': os.environ.get('SLURM_CPUS_PER_TASK'),
                      'environment': {k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']}})
    frozen = {'input_inventory': str(args.manifest), 'input_inventory_sha256': sha(args.manifest),
              'rows': rows, 'protocol': protocol, 'tools': tools, 'source_commit': metadata['source_commit'],
              'scope': 'Development subset' if args.indices else 'Complete905 three-method formal cohort'}
    preflight = run / 'input/mffc905_preflight.json'
    if preflight.exists():
        if json.loads(preflight.read_text()) != frozen:
            raise ValueError('Resume would change frozen input/protocol/tool identity.')
    else:
        save(preflight, frozen)
    if args.check_only:
        save(run / 'output/preflight_check.json', {'status': 'CHECKED_NOT_EXECUTED', 'inputs': len(rows),
                                                  'methods_launched': 0, 'tools': tools, 'gpu': gpu})
        print(json.dumps({'status': 'CHECKED_NOT_EXECUTED', 'inputs': len(rows), 'methods_launched': 0}))
        return 0
    began = time.perf_counter()
    records = []
    for position, row in enumerate(rows):
        methods = ['A', 'B', 'Ours']
        shift = position % 3
        for method in methods[shift:] + methods[:shift]:
            record_path = run / 'output/records' / f'{row["index"]:04d}_{method}.json'
            if record_path.exists():
                records.append(json.loads(record_path.read_text()))
                continue
            start = time.perf_counter()
            outer = run / 'output' / f'case_{row["index"]:04d}_{method}'
            if outer.exists() and (outer / 'point_attempt.json').exists():
                record = recover_point(outer, row, method, metadata['source_commit'])
                save(record_path, record)
                records.append(record)
                continue
            outer.mkdir(parents=True, exist_ok=False)
            attempt = {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'started_epoch_s': time.time(),
                       'started_monotonic_s': start, 'source_commit': metadata['source_commit'],
                       'input_sha256': row['sha256'], 'method': method}
            save(outer / 'point_attempt.json', attempt)
            source = Path(row['staged_path'])
            if sha(source) != row['sha256']:
                raise ValueError('Frozen staged AIG changed before cold method.')
            shutil.copyfile(source, outer / 'input.aig')
            context = outer / 'context.json'
            save(context, {'input': row, 'tools': tools, 'search': protocol['search'],
                           'validation_workers': protocol['validation_workers'], 'source_commit': metadata['source_commit']})
            cache = run / 'cache' / f'{row["index"]:04d}_{method}'
            cache.mkdir()
            environment = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_CACHE_PATH=str(cache),
                               XDG_CACHE_HOME=str(cache), TORCH_EXTENSIONS_DIR=str(cache / 'torch'),
                               PYTHONDONTWRITEBYTECODE='1')
            save(outer / 'pre_launch.json', {'elapsed_s': time.perf_counter() - start})
            # Persist separately so a later allocation can recover the already
            # attempted point without editing its immutable start record.
            status, error = 'PASS', None
            try:
                run_tool([sys.executable, '-B', str(Path(__file__).resolve()), '--child', '--context', str(context),
                          '--method', method], outer, 'cold_method', args.case_timeout, environment)
                result = json.loads((outer / 'child_result.json').read_text())
                if result['status'] != 'PASS' or result['selected']['metrics']['cec'] != 'PASS':
                    raise RuntimeError('Cold method did not return a fully validated final.')
            except Exception as exc:
                status, error, result = 'FAIL', f'{type(exc).__name__}: {exc}', None
                command_file = outer / 'cold_method.command.json'
                if command_file.exists() and json.loads(command_file.read_text()).get('timed_out'):
                    status = 'TIMEOUT'
            record = {'index': row['index'], 'input': row['relative_path'], 'input_sha256': row['sha256'],
                      'method': method, 'status': status, 'error': error,
                      'end_to_end_runtime_s': time.perf_counter() - start,
                      'result_path': str(outer / 'child_result.json') if result else None,
                      'detail_path': str(outer), 'source_commit': metadata['source_commit'],
                      'slurm_job_id': os.environ['SLURM_JOB_ID']}
            if result:
                record.update(metrics=result['selected']['metrics'], final_source=result['selected']['source'],
                              backend=result['backend'])
            save(record_path, record)
            records.append(record)
            print(json.dumps({k: record[k] for k in ('index', 'method', 'status', 'end_to_end_runtime_s')}) , flush=True)
    completion = {'status': 'TERMINAL_ALL_REQUESTED', 'expected_inputs': len(rows), 'method_points': len(records),
                  'states': dict(Counter(r['status'] for r in records)), 'three_methods_only': ['A', 'B', 'Ours'],
                  'session_elapsed_s': time.perf_counter() - began, 'formal': not bool(args.indices)}
    save(run / 'output' / ('formal_completion.json' if not args.indices else 'development_completion.json'), completion)
    print(json.dumps(completion), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise

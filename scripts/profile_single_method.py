#!/usr/bin/env python3
"""Cold-process, single-circuit legacy gradient baseline with stage accounting."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from threading import Lock

ENTRY_TIME = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_case(args):
    # Import startup is part of this child's wall clock, not hidden setup.
    run = args.run.resolve()
    cuda_cache = run / 'cache' / f'cuda_{args.variant}_{args.case_index:04d}'
    cuda_cache.mkdir(exist_ok=False)
    os.environ['CUDA_CACHE_PATH'] = str(cuda_cache)
    if args.source_root is not None:
        if not args.source_root.resolve().is_relative_to('/data/wushangge'):
            raise ValueError('Source override must be an immutable /data snapshot.')
        sys.path.insert(0, str(args.source_root.resolve()))
    from global_timing import pipeline
    from global_timing.common import digest, save_json
    from global_timing.evaluation import Evaluator
    from global_timing.surrogate import TimingSurrogate
    imported = time.perf_counter()
    rows = json.loads((run / 'input/profile_cases.json').read_text())
    row = next(r for r in rows if r['index'] == args.case_index)
    output = args.output_base.resolve() if args.output_base else run / 'output'
    if not output.is_relative_to(run / 'output'):
        raise ValueError('Profile output must stay inside its run.')
    work = output / f'case_{row["index"]:04d}'
    stages = defaultdict(lambda: {'calls': 0, 'inclusive_s': 0.0})
    timeline = []
    accounting_lock = Lock()

    def instrument(owner, name, label):
        original = getattr(owner, name)

        def wrapped(*items, **kw):
            started = time.perf_counter()
            result = None
            try:
                result = original(*items, **kw)
                return result
            finally:
                completed = time.perf_counter()
                event = {'stage': label, 'started_monotonic': started,
                         'completed_monotonic': completed, 'elapsed_s': completed - started}
                if label == 'request_replay_and_validation' and isinstance(result, dict):
                    event.update(kind='request_completed', index=result['index'], method=result['method'],
                                 status=result['status'],
                                 source=(result.get('prediction_before_evaluation') or {}).get('source', 'replay_noop'))
                    if result.get('metrics'):
                        event['metrics'] = {k: result['metrics'][k] for k in
                                            ['cec', 'timing_model', 'delay_ps', 'area']}
                with accounting_lock:
                    stages[label]['calls'] += 1
                    stages[label]['inclusive_s'] += completed - started
                    timeline.append(event)
        setattr(owner, name, wrapped)

    for name in ['parse_aig', 'make_partition', 'enumerate_candidates',
                 'validate_catalogue', 'score_catalogue', 'gradient_proposals']:
        instrument(pipeline, name, name)
    instrument(TimingSurrogate, '__init__', 'surrogate_initialization')
    instrument(TimingSurrogate, 'fit_observations', 'surrogate_calibration')
    instrument(pipeline.Experiment, 'evaluate_choice', 'request_replay_and_validation')
    instrument(Evaluator, 'evaluate', 'cec_and_mapping')
    source = Path(row['staged_path'])
    if digest(source) != row['sha256']:
        raise ValueError('Frozen circuit hash mismatch.')
    report = json.loads((args.tool_run / 'build/framework_tools/build_report.json').read_text())
    tools = {k: Path(v['path']) for k, v in report['binaries'].items()}
    for name, path in tools.items():
        if digest(path) != report['binaries'][name]['sha256']:
            raise ValueError('Frozen binary hash mismatch: ' + name)
    config = pipeline.SearchConfig(
        seed=args.seed, evaluation_budget=args.evaluations, warmup_evaluations=2,
        gradient_steps=20 if args.variant == 'legacy_cpu' else args.proposal_steps,
        gradient_restarts=2, calibration_steps=30,
        feedback_batch=4, modes=('area_constraint',), timeout_s=args.tool_timeout)
    record = {'status': 'RUNNING', 'index': row['index'], 'input_name': row['relative_path'],
              'input_sha256': row['sha256'], 'strategy': args.variant,
              'source_root': str(Path(pipeline.__file__).resolve().parents[1]),
              'algorithm_sha256': {name: digest(Path(pipeline.__file__).with_name(name + '.py'))
                                   for name in ['pipeline', 'optimization', 'surrogate', 'arrival', 'litecut', 'candidates']},
              'child_import_s': imported - ENTRY_TIME, 'stages': stages,
              'cuda_cache': str(cuda_cache), 'cuda_cache_regime': 'isolated_per_cold_child',
              'torch_version': str(pipeline.torch.__version__),
              'torch_num_threads': pipeline.torch.get_num_threads(),
              'scope': 'One cold child process, one circuit, one gradient strategy; '
                       'includes required preparation and original/noop measurements. '
                       'No execution of unrelated greedy/random/joint controls.'}
    try:
        experiment_class, extra = pipeline.Experiment, {}
        if '_batch_' in args.variant:
            from global_timing.batch_execution import BatchedExperiment
            def completed(result, began, ended):
                event = {'stage': 'request_replay_and_validation', 'kind': 'request_completed',
                         'started_monotonic': began, 'completed_monotonic': ended, 'elapsed_s': ended - began,
                         'index': result['index'], 'method': result['method'], 'status': result['status'],
                         'source': (result.get('prediction_before_evaluation') or {}).get('source', 'unknown')}
                if result.get('metrics'):
                    event['metrics'] = {k: result['metrics'][k] for k in ['cec', 'timing_model', 'delay_ps', 'area']}
                with accounting_lock:
                    stages['request_replay_and_validation']['calls'] += 1
                    stages['request_replay_and_validation']['inclusive_s'] += ended - began
                    timeline.append(event)
            experiment_class = BatchedExperiment
            extra = {'validation_workers': args.validation_workers, 'on_completion': completed}
            record['validation_workers'] = args.validation_workers
            record['algorithm_sha256']['batch_execution'] = digest(
                Path(pipeline.__file__).with_name('batch_execution.py'))
        experiment = experiment_class(
            source, tools['gpuls'], tools['abc'], run / 'input/asap7',
            run / 'input/student_checkpoint', work, config, **extra)
        original = experiment.evaluator.evaluate(source, work / 'original')
        baseline = experiment.evaluate_choice(experiment.noop, 'replay_noop', 0)
        if baseline['status'] != 'PASS':
            raise RuntimeError('Required no-op failed.')
        if args.variant == 'legacy_cpu':
            search = experiment.gradient_search(baseline, 'area_constraint')
        else:
            from functools import partial
            from global_timing import joint_optimization
            device = 'cuda' if args.variant.endswith('_cuda') else 'cpu'
            proposal_config = joint_optimization.JointProposalConfig(device=device, trajectories=args.trajectories,
                                                                     max_changes=args.max_changes,
                                                                     feedback=args.variant.startswith('feedback_'))
            record['joint_proposal_config'] = dict(proposal_config.__dict__)
            record['algorithm_sha256']['joint_optimization'] = digest(
                Path(pipeline.__file__).with_name('joint_optimization.py'))
            instrument(joint_optimization, 'joint_proposals', 'gradient_proposals')
            search = experiment.gradient_search(baseline, 'area_constraint',
                proposal_generator=partial(joint_optimization.joint_proposals, config=proposal_config),
                choice_batch_evaluator=experiment.evaluate_many if '_batch_' in args.variant else None)
        record.update(status='PASS', original=original, no_op=baseline,
                      selected=search['best'], preparation_runtime_s=experiment.preparation_runtime_s,
                      method_runtime_s=search['exclusive_method_runtime_s'],
                      requests=search['evaluations_including_baseline'],
                      direct_gradient_requests=search['gradient_proposal_evaluations'],
                      failed_requests=sum(o['status'] != 'PASS' for o in search['observations']),
                      regions=len(experiment.partition.regions),
                      decisions=len(experiment.catalogue.decisions),
                      candidates=sum(d['candidate_count'] for d in experiment.catalogue.decisions),
                      litecut_cache_hit=experiment.cache['cache_hit'])
        if args.variant.endswith('_cuda'):
            record['python_cuda_device'] = pipeline.torch.cuda.get_device_name(0)
    except Exception as exc:
        record.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['cold_child_wall_s'] = time.perf_counter() - ENTRY_TIME
        record['stages'] = dict(stages)
        record['clock'] = dict(time.get_clock_info('perf_counter').__dict__)
        record['script_entry_monotonic'] = ENTRY_TIME
        timeline_path = output / f'case_{row["index"]:04d}_timeline.json'
        save_json(timeline_path, {'clock': record['clock'], 'script_entry_monotonic': ENTRY_TIME,
                                 'events': timeline, 'scope': 'Completion times, not inferred from file mtimes. Parent subprocess start supplies the cold-process origin; nested stages overlap.'})
        record['timeline'] = str(timeline_path)
        record['timeline_sha256'] = digest(timeline_path)
        record['timing_scope'] = ('Stage values are inclusive and overlap; do not sum them. '
                                  'The parent records full subprocess wall time including interpreter startup. '
                                  'The deployment baseline excludes separately evaluated controls, equally for future variants.')
        save_json(output / f'case_{row["index"]:04d}_profile.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tool-run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--indices', default='0,47,80')
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--evaluations', type=int, default=8)
    parser.add_argument('--tool-timeout', type=int, default=900)
    parser.add_argument('--case-timeout', type=int, default=1800)
    parser.add_argument('--variants', default='legacy_cpu',
                        help='Sequential controls: legacy_cpu,criticality_cpu,criticality_cuda,feedback_cpu,feedback_cuda')
    parser.add_argument('--variant', choices=['legacy_cpu', 'criticality_cpu', 'criticality_cuda', 'feedback_cpu', 'feedback_cuda',
                                            'feedback_batch_cpu', 'feedback_batch_cuda'], default='legacy_cpu')
    parser.add_argument('--legacy-source', type=Path, help='Frozen old /data source for legacy control.')
    parser.add_argument('--legacy-tool-run', type=Path, help='Frozen original binaries for legacy_cpu only.')
    parser.add_argument('--source-root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output-base', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--proposal-steps', type=int, default=40)
    parser.add_argument('--trajectories', type=int, default=32)
    parser.add_argument('--max-changes', type=int, default=32)
    parser.add_argument('--validation-workers', type=int, default=2)
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.tool_run = args.tool_run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not args.run.is_relative_to('/data/wushangge'):
        parser.error('Requires a prepared /data run and a Slurm allocation.')
    if args.case_index is not None:
        run_case(args)
        return
    variants = args.variants.split(',')
    if len(set(variants)) != len(variants) or not set(variants) <= {'legacy_cpu', 'criticality_cpu', 'criticality_cuda', 'feedback_cpu', 'feedback_cuda',
                                                                'feedback_batch_cpu', 'feedback_batch_cuda'}:
        parser.error('Unknown or repeated variant.')
    if args.legacy_source is not None and (not args.legacy_source.resolve().is_relative_to('/data/wushangge')
                                          or not (args.legacy_source / 'global_timing/pipeline.py').is_file()):
        parser.error('Legacy source must be a frozen /data source snapshot.')
    from global_timing.common import digest, save_json
    if args.manifest is None:
        parser.error('--manifest is required for parent mode.')
    manifest = json.loads(args.manifest.read_text())
    indices = {int(v) for v in args.indices.split(',')}
    selected = [r for r in manifest['rows'] if r['index'] in indices]
    if len(selected) != len(indices):
        parser.error('Unknown input index.')
    inputs = args.run / 'input/profile_designs'
    inputs.mkdir(exist_ok=False)
    for row in selected:
        staging_started = time.perf_counter()
        source = Path(row['staged_path'])
        if digest(source) != row['sha256']:
            raise ValueError('Input inventory changed.')
        destination = inputs / (f'{row["index"]:04d}' + source.suffix)
        with source.open('rb') as incoming, destination.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if digest(destination) != row['sha256']:
            raise ValueError('Staged input hash mismatch.')
        row['staged_path'] = str(destination)
        row['staging_runtime_s'] = time.perf_counter() - staging_started
    save_json(args.run / 'input/profile_cases.json', selected)
    metadata_path = args.run / 'run.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(state='RUNNING', slurm_job_id=os.environ['SLURM_JOB_ID'],
                    command=sys.argv, experiment_executed=True,
                    input_manifest=str(args.manifest), input_manifest_sha256=digest(args.manifest),
                    environment={'python': sys.version,
                                 'threads': {k: os.environ.get(k) for k in
                                             ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
                                 'cuda_visibility': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')}})
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    records = []
    try:
        # Variant order rotates per circuit to expose simple order/cache bias;
        # these first profiles are diagnostic, not repeated final speed trials.
        tasks = [(row, variant) for position, row in enumerate(selected)
                 for variant in variants[position % len(variants):] + variants[:position % len(variants)]]
        for row, variant in tasks:
            base = args.run / 'output' if variants == ['legacy_cpu'] else args.run / 'output' / variant
            base.mkdir(exist_ok=True)
            source = args.legacy_source.resolve() if variant == 'legacy_cpu' and args.legacy_source else args.run / 'src'
            tool_source = args.legacy_tool_run.resolve() if variant == 'legacy_cpu' and args.legacy_tool_run else args.tool_run
            command = [sys.executable, '-B', str(Path(__file__).resolve()),
                       '--run', str(args.run), '--tool-run', str(tool_source),
                       '--case-index', str(row['index']), '--seed', str(args.seed),
                       '--evaluations', str(args.evaluations), '--tool-timeout', str(args.tool_timeout),
                       '--variant', variant, '--source-root', str(source), '--output-base', str(base),
                       '--proposal-steps', str(args.proposal_steps), '--trajectories', str(args.trajectories),
                       '--max-changes', str(args.max_changes), '--validation-workers', str(args.validation_workers)]
            started = time.perf_counter()
            log = args.run / 'logs' / f'profile_{variant}_{row["index"]:04d}.log'
            with log.open('x') as stream:
                process = subprocess.Popen(command, cwd=args.run, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    process.wait(timeout=args.case_timeout)
                except subprocess.TimeoutExpired:
                    from global_timing.processes import terminate_tree
                    terminate_tree(process.pid)
                    process.wait()
            record = {'index': row['index'], 'input_name': row['relative_path'], 'variant': variant,
                      'command': command, 'returncode': process.returncode,
                      'subprocess_started_monotonic': started,
                      'full_process_wall_s': time.perf_counter() - started, 'log': str(log)}
            record['input_staging_s'] = row['staging_runtime_s']
            record['end_to_end_with_input_staging_s'] = record['full_process_wall_s'] + row['staging_runtime_s']
            path = base / f'case_{row["index"]:04d}_profile.json'
            if path.exists():
                detail = json.loads(path.read_text())
                record.update(status=detail['status'], detail=str(path), detail_sha256=digest(path))
            else:
                record.update(status='FAIL', reason='Child did not finish its profile; retained log and partial output.')
            records.append(record)
            print(json.dumps(record), flush=True)
        summary = {'records': records, 'status': 'PASS' if all(
            r['status'] == 'PASS' and r['returncode'] == 0 for r in records) else 'PARTIAL',
            'source_commit': metadata['source_commit'],
            'cache_policy': 'Isolated CUDA cache per child under /data; independent per-variant LiteCut directories; filesystem page cache not flushed.',
            'scope': 'Sequential per-circuit cold-process controls on a single allocation; '
                     'no cross-benchmark parallel throughput speedup claim.'}
        save_json(args.run / 'output/profile_summary.json', summary)
        metadata.update(state='COMPLETED' if summary['status'] == 'PASS'
                        else 'COMPLETED_WITH_EXPLICIT_FAILURES',
                        outputs=str(args.run / 'output/profile_summary.json'))
    except Exception as exc:
        metadata.update(state='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()

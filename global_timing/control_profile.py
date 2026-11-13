"""Cold-child accounting for standalone controls; ML is optional and lazy."""
from collections import defaultdict
import json
from pathlib import Path
import time
from threading import Lock

from . import controls
from .common import digest, save_json


def run_control_case(args, entry_time):
    run = args.run.resolve()
    rows = json.loads((run / 'input/profile_cases.json').read_text())
    row = next(r for r in rows if r['index'] == args.case_index)
    output = args.output_base.resolve() if args.output_base else run / 'output'
    if not output.is_relative_to(run / 'output'):
        raise ValueError('Control output must stay inside the prepared run.')
    case = output / f'case_{row["index"]:04d}'
    source = Path(row['staged_path'])
    if digest(source) != row['sha256']:
        raise ValueError('Frozen standalone input hash mismatch.')
    report = json.loads((args.tool_run / 'build/framework_tools/build_report.json').read_text())
    tools = {k: Path(v['path']) for k, v in report['binaries'].items()}
    if any(digest(path) != report['binaries'][k]['sha256'] for k, path in tools.items()):
        raise ValueError('Frozen standalone binary hash mismatch.')
    stages, events, lock = defaultdict(lambda: {'calls': 0, 'inclusive_s': 0.0}), [], Lock()

    def completed(result, begin, end):
        event = {'stage': 'request_replay_and_validation', 'kind': 'request_completed',
                 'started_monotonic': begin, 'completed_monotonic': end, 'elapsed_s': end - begin,
                 'index': result['index'], 'method': result['method'], 'status': result['status'],
                 'source': (result.get('prediction_before_evaluation') or {}).get('source', 'replay_noop')}
        if result.get('metrics'):
            event['metrics'] = {k: result['metrics'][k] for k in ['cec', 'timing_model', 'delay_ps', 'area']}
        with lock:
            stages['request_replay_and_validation']['calls'] += 1
            stages['request_replay_and_validation']['inclusive_s'] += end - begin
            events.append(event)

    record = {'status': 'RUNNING', 'index': row['index'], 'seed': args.seed, 'input_name': row['relative_path'],
              'input_sha256': row['sha256'], 'strategy': args.variant,
              'source_root': str(Path(__file__).resolve().parents[1]),
              'algorithm_sha256': {n: digest(Path(__file__).with_name(n + '.py')) for n in
                  ['controls', 'control_profile', 'execution', 'boolean_candidates', 'candidates', 'cut_validation', 'evaluation']},
              'cuda_cache_regime': 'isolated_directory_unused_by_nonML_control',
              'installation_gain_policy': 'allow_nonpositive' if 'physical' in args.variant else 'positive_gain',
              'scope': 'Standalone natural-cost baseline: random omits unused ML; native omits candidate preparation. Required proof/replay/CEC/mapping remains complete.'}
    try:
        if args.variant in ('original', 'native_abc', 'native_culs'):
            result = controls.native_control(args.variant, source, tools['gpuls'], tools['abc'],
                run / 'input/asap7', case, args.tool_timeout, completed)
            summary = {'method': args.variant, 'best': result, 'observations': [result],
                       'evaluations_including_baseline': 1, 'gradient_proposal_evaluations': 0}
            summary_path = case / 'summary.json'
            save_json(summary_path, summary)
            record.update(record_kind='native_control', selected=result, requests=1,
                failed_requests=int(result['status'] != 'PASS'), preparation_runtime_s=0.0,
                method_runtime_s=result['runtime_s'], litecut_cache_hit=None)
        else:
            use_litecut = args.variant.startswith('greedy_')
            config = controls.ControlConfig(seed=args.seed, evaluation_budget=args.evaluations,
                timeout_s=args.tool_timeout, allow_nonpositive_gain='physical' in args.variant)
            experiment = controls.CandidateControl(source, tools['gpuls'], tools['abc'], run / 'input/asap7',
                run / 'input/student_checkpoint', case, config, use_litecut,
                args.validation_workers, completed)
            method = 'greedy_litecut' if use_litecut else 'random_search'
            summary = experiment.run(method, batched=args.variant != 'random_cpu')
            summary_path = case / method / 'summary.json'
            record.update(record_kind='candidate_search', selected=summary['best'],
                no_op=summary['observations'][0], requests=len(summary['observations']),
                failed_requests=sum(r['status'] != 'PASS' for r in summary['observations']),
                direct_gradient_requests=0, preparation_runtime_s=experiment.preparation_runtime_s,
                method_runtime_s=summary['exclusive_method_runtime_s'],
                regions=len(experiment.partition.regions), decisions=len(experiment.catalogue.decisions),
                candidates=sum(d['candidate_count'] for d in experiment.catalogue.decisions),
                litecut_cache_hit=experiment.cache['cache_hit'], control_config=config.__dict__)
        record['search_summary'] = str(summary_path)
        if summary['best']['status'] != 'PASS':
            raise RuntimeError('Standalone native control failed; retained explicit failed evidence.')
        record['status'] = 'PASS'
    except Exception as exc:
        record.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        import sys
        record.update(cold_child_wall_s=time.perf_counter() - entry_time,
            stages=dict(stages), clock=dict(time.get_clock_info('perf_counter').__dict__),
            script_entry_monotonic=entry_time,
            torch_loaded='torch' in sys.modules, numpy_loaded='numpy' in sys.modules)
        if 'torch' in sys.modules:
            record.update(torch_version=str(sys.modules['torch'].__version__),
                          torch_num_threads=sys.modules['torch'].get_num_threads())
        timeline = output / f'case_{row["index"]:04d}_timeline.json'
        save_json(timeline, {'clock': record['clock'], 'script_entry_monotonic': entry_time,
                            'events': events, 'scope': 'Actual request completion times; nested/concurrent intervals overlap.'})
        record.update(timeline=str(timeline), timeline_sha256=digest(timeline))
        save_json(output / f'case_{row["index"]:04d}_profile.json', record)

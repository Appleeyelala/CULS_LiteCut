#!/usr/bin/env python3
"""Check selection optimizations against saved real catalogues and scalar costs.

Read-only reuse of prior inputs. This does not invoke rewriting or mapping;
new profiles and selections are written only in the prepared Slurm run.
"""
import argparse
import cProfile
import json
import os
from pathlib import Path
import random
import pstats
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def profile(benchmark, output, catalogue_only, student_checkpoint=None, options_only=False, option_checks=8):
    import torch
    from global_timing.candidates import Catalogue, catalogue_from_raw
    from global_timing.common import digest, legacy, read_jsonl, save_json
    from global_timing.pipeline import Experiment
    from global_timing.structure import make_partition, parse_aig
    from global_timing.surrogate import TimingSurrogate

    record = {'benchmark': str(benchmark), 'status': 'RUNNING', 'source_hashes': {}}
    output.mkdir(parents=True, exist_ok=False)

    def read(relative):
        path = benchmark / relative
        record['source_hashes'][str(path)] = digest(path)
        return json.loads(path.read_text())

    try:
        config = read('config.json')
        aig = Path(config['input'])
        if digest(aig) != config['input_sha256']:
            raise ValueError('Prior input changed.')
        record['source_hashes'][str(aig)] = config['input_sha256']
        partition = make_partition(parse_aig(aig))
        raw_path = benchmark / 'candidates/raw/gpu_rewrite_cuts.jsonl'
        record['source_hashes'][str(raw_path)] = digest(raw_path)
        rows = read_jsonl(raw_path)
        started = time.monotonic()
        current = catalogue_from_raw(rows, partition)
        record.update(catalogue_s=time.monotonic() - started, legal_cuts=len(rows),
                      executable_decisions=len(current.decisions), regions=len(partition.regions))
        prior_path = benchmark / 'candidates/catalogue.json'
        if prior_path.is_file():
            prior = read('candidates/catalogue.json')
            if current.decisions != prior['decisions'] or current.conflicts != [
                    (r['left'], r['right']) for r in prior['conflicts']] or current.region_decisions != {
                    int(k): v for k, v in prior['region_decisions'].items()}:
                raise AssertionError('Optimized catalogue differs from saved output.')
            record['catalogue_comparison'] = 'EXACT_MATCH'
            del prior
        else:
            record['catalogue_comparison'] = 'NO_SAVED_REFERENCE'
        # Bound the deliberately inefficient reference to 512 cuts. Timings
        # describe this actual subset, not an extrapolated whole-program gain.
        sample = rows[:512]
        normalizer = legacy('gate3c_analyze').normalize_decision
        started = time.monotonic()
        slow = [normalizer(row, f'd{i}', partition.owner, set(range(len(partition.regions))))
                for i, row in enumerate(sample)]
        record['normalization_reference_512_s'] = time.monotonic() - started
        region_ids = frozenset(range(len(partition.regions)))
        started = time.monotonic()
        fast = [normalizer(row, f'd{i}', partition.owner, region_ids) for i, row in enumerate(sample)]
        record['normalization_optimized_512_s'] = time.monotonic() - started
        if fast != slow:
            raise AssertionError('Normalization reference mismatch.')
        record['normalization_sample_count'] = len(sample)
        del rows, sample, slow, fast, current
        if not catalogue_only:
            saved = read('cut_validation/verified_catalogue.json')
            catalogue = Catalogue(saved['decisions'], [tuple(pair) for pair in saved['conflicts']],
                                  {int(k): v for k, v in saved['region_decisions'].items()})
            scores = read('litecut.json')['scores']
            if student_checkpoint is not None:
                from global_timing.litecut import score_catalogue
                if digest(student_checkpoint) != config['checkpoint_sha256']:
                    raise ValueError('Student checkpoint differs from saved input.')
                record['source_hashes'][str(student_checkpoint)] = config['checkpoint_sha256']
                started = time.monotonic()
                rescored, cache = score_catalogue(aig, partition, catalogue, student_checkpoint, output / 'student_cache')
                record.update(student_rescore_s=time.monotonic() - started, student_rescore_count=len(rescored),
                              student_rescore_cache=cache, student_scores_exact_match=rescored == scores)
                if rescored != scores:
                    raise AssertionError('Student-only features/predictions differ from saved full-extractor scores.')
                del rescored
            baseline = read('replay_noop/eval_0000/observation.json')['metrics']
            started = time.monotonic()
            model = TimingSurrogate(partition, catalogue, scores, baseline)
            record['model_init_s'] = time.monotonic() - started
            rid, indices = max(catalogue.region_decisions.items(), key=lambda item: len(item[1]))
            record['largest_region'] = {'region': rid, 'decisions': len(indices),
                'boundary_ports': len(partition.transfers[rid]),
                'port_decision_incidences': sum(len(model.port_decisions[rid, port]) for port in partition.transfers[rid])}
            local = {i: random.Random(i + 17).randrange(model.counts[i]) for i in indices}
            option_records = []
            with torch.no_grad():
                for index in indices[:option_checks]:
                    profiler = cProfile.Profile() if options_only else None
                    started = time.monotonic()
                    if profiler is not None:
                        profiler.enable()
                    actual = model.independent_region_options(rid, local, index)
                    if profiler is not None:
                        profiler.disable()
                    fast_s = time.monotonic() - started
                    if profiler is not None:
                        profiler.dump_stats(str(output / f'option_{index}.prof'))
                        with (output / f'option_{index}.txt').open('x') as stream:
                            pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(40)
                    started = time.monotonic()
                    expected = torch.stack([model.independent_region_arrival(rid, local | {index: c})
                                            for c in range(model.counts[index])])
                    slow_s = time.monotonic() - started
                    row = {'region': rid, 'region_decisions': len(indices), 'decision': index,
                           'options': model.counts[index], 'reference_s': slow_s, 'optimized_s': fast_s,
                           'optimized_time_includes_cprofile': profiler is not None,
                           'exact_equal': bool(torch.equal(actual, expected)),
                           'max_abs_error': float((actual - expected).abs().max())}
                    option_records.append(row)
                    if not row['exact_equal']:
                        save_json(output / 'option_mismatch.json', {'checks': option_records,
                                  'actual': actual.tolist(), 'reference': expected.tolist()})
                        raise AssertionError('Batched independent costs differ from ordered scalar reference.')
            record['option_checks'] = option_records
            del model
            record['full_independent_selection_executed'] = not options_only
            if options_only:
                record['selection_comparison'] = 'NOT_RUN_OPTIONS_ONLY'
            else:
                experiment = Experiment.__new__(Experiment)
                experiment.partition, experiment.catalogue, experiment.scores = partition, catalogue, scores
                experiment.noop = [0] * len(catalogue.decisions)
                started = time.monotonic()
                choices = experiment.independent_choices(baseline)
                record['full_independent_selection_s'] = time.monotonic() - started
                catalogue.validate(choices)
                save_json(output / 'independent_choices.json', {'choices': choices})
                record['selection_sha256'] = digest(output / 'independent_choices.json')
                prior_path = benchmark / 'independent_regions/eval_0000/observation.json'
                if prior_path.is_file():
                    reference = read('independent_regions/eval_0000/observation.json')['choices']
                    if choices != reference:
                        raise AssertionError('Independent selection differs from saved pre-repair request.')
                    record['selection_comparison'] = 'EXACT_MATCH'
                else:
                    record['selection_comparison'] = 'NO_SAVED_REFERENCE'
        record['status'] = 'PASS'
    except Exception as exc:
        record.update(status='FAIL', error=str(exc), traceback=traceback.format_exc())
    save_json(output / 'profile.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--benchmarks', type=Path, nargs='*', default=[])
    parser.add_argument('--catalogue-only', type=Path, nargs='*', default=[])
    parser.add_argument('--student-rescore', action='store_true',
                        help='Recompute all saved scores with the prepared checkpoint and require exact equality.')
    parser.add_argument('--options-only', action='store_true',
                        help='Profile/check bounded coordinate options; do not execute the complete independent selection.')
    parser.add_argument('--option-checks', type=int, default=8, help='Number of leading decisions in the largest region to check.')
    args = parser.parse_args()
    if args.option_checks < 1:
        parser.error('--option-checks must be positive.')
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('Profiles require a prepared /data run and Slurm allocation.')
    from global_timing.common import save_json
    import hashlib
    records = []
    metadata = json.loads((run / 'run.json').read_text())
    metadata.update(state='RUNNING', experiment_executed=True, command=sys.argv,
                    slurm_job_id=os.environ['SLURM_JOB_ID'])
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for benchmark, catalogue_only in [(p, False) for p in args.benchmarks] + [(p, True) for p in args.catalogue_only]:
        benchmark = benchmark.resolve()
        key = hashlib.sha256(str(benchmark).encode()).hexdigest()[:12]
        record = profile(benchmark, run / 'output/selection_profiles' / key, catalogue_only,
                         run / 'input/student_checkpoint' if args.student_rescore else None,
                         options_only=args.options_only, option_checks=args.option_checks)
        records.append(record)
        print(json.dumps({k: record.get(k) for k in ['benchmark', 'status', 'error', 'full_independent_selection_s']}), flush=True)
    save_json(run / 'output/selection_profiles/summary.json', {'source_commit': metadata['source_commit'],
              'scope': 'Saved real inputs; no rewrite, mapping or new timing labels.', 'records': records})
    failed = any(r['status'] != 'PASS' for r in records)
    metadata.update(state='FAILED' if failed else 'COMPLETED')
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())

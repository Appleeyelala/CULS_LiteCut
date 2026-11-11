#!/usr/bin/env python3
"""Profile an exact saved gradient round without rewriting, mapping or new labels."""
import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--mode', choices=['area_constraint', 'joint'], default='area_constraint')
    parser.add_argument('--iteration', type=int, default=2)
    args = parser.parse_args()
    run, benchmark = args.run.resolve(), args.benchmark.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not all(p.is_relative_to('/data/wushangge') for p in [run, benchmark]):
        parser.error('Profiling requires Slurm and prepared /data paths.')
    from global_timing.candidates import Catalogue
    from global_timing.common import digest, save_json
    from global_timing.optimization import gradient_proposals
    from global_timing.pipeline import Experiment, SearchConfig
    from global_timing.structure import make_partition, parse_aig
    from global_timing.surrogate import TimingSurrogate
    output = run / 'output/gradient_profile'
    output.mkdir(exist_ok=False)
    source_hashes = {}
    stages = []

    def read(path):
        source_hashes[str(path)] = digest(path)
        return json.loads(path.read_text())

    def profiled(name, function):
        profiler = cProfile.Profile()
        started = time.monotonic()
        with profiler:
            result = function()
        elapsed = time.monotonic() - started
        profiler.dump_stats(str(output / (name + '.prof')))
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(45)
        (output / (name + '.txt')).write_text(stream.getvalue())
        stages.append({'stage': name, 'elapsed_with_profiler_s': elapsed})
        print(json.dumps(stages[-1]), flush=True)
        return result

    metadata = read(run / 'run.json')
    metadata.update(state='RUNNING', experiment_executed=True, slurm_job_id=os.environ['SLURM_JOB_ID'],
                    command=sys.argv, environment={'python': sys.version,
                        'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')}})
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    record = {'status': 'RUNNING', 'benchmark': str(benchmark), 'mode': args.mode, 'iteration': args.iteration,
              'scope': 'Saved observations and exact saved gradient settings; no rewrite/mapping/new timing labels. Profiled wall time includes instrumentation.'}
    try:
        cfg = read(benchmark / 'config.json')
        config = SearchConfig(**{k: cfg[k] for k in SearchConfig.__dataclass_fields__})
        config.validate()
        aig = Path(cfg['input'])
        if digest(aig) != cfg['input_sha256']:
            raise ValueError('Saved benchmark input checksum changed.')
        source_hashes[str(aig)] = cfg['input_sha256']
        partition = profiled('partition', lambda: make_partition(parse_aig(aig)))
        saved = read(benchmark / 'cut_validation/verified_catalogue.json')
        catalogue = Catalogue(saved['decisions'], [tuple(pair) for pair in saved['conflicts']],
                              {int(k): v for k, v in saved['region_decisions'].items()})
        scores = read(benchmark / 'litecut.json')['scores']
        baseline = read(benchmark / 'replay_noop/eval_0000/observation.json')
        method = benchmark / ('gradient_' + args.mode)
        reference = read(method / f'calibration_{args.iteration:04d}.json')
        last = max(reference['training_observation_indices'])
        observations = [baseline] + [read(method / f'eval_{i:04d}/observation.json') for i in range(1, last + 1)]
        actual_training = [row['index'] for row in observations if row['status'] == 'PASS']
        if actual_training != reference['training_observation_indices']:
            raise ValueError('Saved training-index provenance mismatch.')
        model = profiled('model_initialization', lambda: TimingSurrogate(partition, catalogue, scores, baseline['metrics']))
        fit = profiled('calibration', lambda: model.fit_observations(observations, steps=config.calibration_steps))
        if fit['parameters'] != reference['parameters'] or fit['losses'] != reference['losses']:
            raise ValueError('Reconstructed first calibration differs from the saved round; later rounds require prior model state.')
        experiment = Experiment.__new__(Experiment)
        experiment.config = config
        best = min(observations, key=lambda row: experiment.rank_actual(row, baseline['metrics'], args.mode))
        proposals, trace = profiled('gradient_proposals', lambda: gradient_proposals(model,
            {tuple(row['choices']) for row in observations}, config.seed + args.iteration,
            incumbent=best['choices'], steps=config.gradient_steps, restarts=config.gradient_restarts,
            mode=args.mode, area_limit_ratio=config.area_limit_ratio, area_weight=config.area_weight))
        saved_trace = read(method / f'gradients_{args.iteration:04d}.json')
        if trace != saved_trace:
            save_json(output / 'actual_trace.json', trace)
            raise ValueError('Reconstructed gradient trace differs from the saved real search.')
        record.update(status='PASS', calibration_exact_match=True, gradient_trace_exact_match=True,
            regions=model.n_regions, decisions=model.n_decisions, padded_candidate_width=model.width,
            levels=len(model.layers), edges=len(model.edge_sources), port_incidence_nnz=model.port_incidence._nnz(),
            proposal_count=len(proposals), samples=fit['samples'], steps=config.gradient_steps,
            restarts=config.gradient_restarts, calibration_steps=config.calibration_steps)
    except Exception as exc:
        record.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
    record.update(stages=stages, input_sha256=source_hashes, source_commit=metadata['source_commit'])
    save_json(output / 'summary.json', record)
    metadata.update(state='COMPLETED' if record['status'] == 'PASS' else 'FAILED', outputs=str(output / 'summary.json'))
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'input_sha256'}), flush=True)
    return int(record['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())

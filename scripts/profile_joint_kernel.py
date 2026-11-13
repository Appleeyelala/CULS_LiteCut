#!/usr/bin/env python3
"""Same-work CPU/CUDA proposals on saved large single-circuit proxy inputs.

This is a warm/mixed-context component benchmark, never an EDA or end-to-end
QoR result. Every call includes fresh adjoint construction and transfers.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--indices', default='20,22,38')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--reference-summary', type=Path,
                        help='Optional saved same-input proposal hashes from an earlier implementation.')
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('Requires a prepared /data run and Slurm allocation.')
    if args.repeats < 1:
        parser.error('Positive repeat count required.')
    import torch
    from global_timing.candidates import Catalogue
    from global_timing.common import digest, save_json
    from global_timing.joint_optimization import JointProposalConfig, joint_proposals
    from global_timing.structure import make_partition, parse_aig
    from global_timing.surrogate import TimingSurrogate
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the controlled component benchmark.')
    indices = {int(v) for v in args.indices.split(',')}
    rows = [r for r in json.loads(args.diagnostic.read_text())['cases'] if r['index'] in indices]
    if len(rows) != len(indices):
        parser.error('Missing diagnostic case.')
    reference = {}
    if args.reference_summary:
        for case in json.loads(args.reference_summary.read_text())['results']:
            for call in case['calls']:
                reference[case['index'], call['repeat'], call['device']] = (case['inputs'], call['choices_sha256'])
    metadata_path = run / 'run.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(state='RUNNING', slurm_job_id=os.environ['SLURM_JOB_ID'], experiment_executed=True,
                    command=sys.argv, environment={'python': sys.version, 'torch': torch.__version__,
                    'gpu': torch.cuda.get_device_name(0),
                    'threads': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
                    'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')}},
                    diagnostic_sha256=digest(args.diagnostic))
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    results = []
    try:
        for row in rows:
            case = Path(row['case'])
            start = time.perf_counter()
            summary_path = case / 'summary.json'
            summary = json.loads(summary_path.read_text())
            config_path = case / 'config.json'
            input_path = Path(json.loads(config_path.read_text())['input'])
            catalogue_path = case / 'cut_validation/verified_catalogue.json'
            scores_path = case / 'litecut.json'
            stored = json.loads(catalogue_path.read_text())
            catalogue = Catalogue(stored['decisions'], [tuple(p) for p in stored['conflicts']],
                                  {int(k): v for k, v in stored['region_decisions'].items()})
            scores = json.loads(scores_path.read_text())['scores']
            partition = make_partition(parse_aig(input_path))
            model = TimingSurrogate(partition, catalogue, scores, summary['replay_noop']['metrics'])
            calibration = sorted((case / 'gradient_area_constraint').glob('calibration_*.json'))[-1]
            parameters = json.loads(calibration.read_text())['parameters']
            with torch.no_grad():
                for name, value in parameters.items():
                    getattr(model, name).copy_(torch.tensor(value, dtype=torch.float64))
            model.requires_grad_(False)
            item = {'index': row['index'], 'name': row['input_name'],
                    'regions': model.n_regions, 'decisions': model.n_decisions,
                    'width': model.width, 'source_case': str(case),
                    'setup_s': time.perf_counter() - start,
                    'inputs': {str(p): digest(p) for p in [summary_path, config_path, input_path, catalogue_path, scores_path, calibration]},
                    'calls': [], 'pairs': []}
            for repeat in range(args.repeats):
                by_device = {}
                for device in (('cpu', 'cuda') if repeat % 2 else ('cuda', 'cpu')):
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    proposals, trace = joint_proposals(model, {tuple([0] * model.n_decisions)},
                        seed=20261111 + repeat, incumbent=[0] * model.n_decisions,
                        steps=40, config=JointProposalConfig(device=device))
                    elapsed = time.perf_counter() - begin
                    choice_bytes = json.dumps([p['choices'] for p in proposals], separators=(',', ':')).encode()
                    import hashlib
                    measured = {'repeat': repeat, 'device': device, 'wall_s': elapsed,
                                'proposal_count': len(proposals), 'choices_sha256': hashlib.sha256(choice_bytes).hexdigest(),
                                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'trace': trace}
                    if args.reference_summary:
                        old_inputs, old_choices = reference[row['index'], repeat, device]
                        if old_inputs != item['inputs']:
                            raise ValueError('Reference proxy inputs differ; comparison would be invalid.')
                        measured['same_choices_as_reference'] = measured['choices_sha256'] == old_choices
                    item['calls'].append(measured)
                    by_device[device] = (proposals, elapsed)
                    print(json.dumps({'index': row['index'], 'repeat': repeat, 'device': device,
                                      'wall_s': elapsed, 'proposals': len(proposals)}), flush=True)
                cpu, gpu = by_device['cpu'], by_device['cuda']
                agreement = [p['choices'] for p in cpu[0]] == [p['choices'] for p in gpu[0]]
                item['pairs'].append({'repeat': repeat, 'same_choices_in_order': agreement,
                                      'cpu_over_cuda_wall_ratio': cpu[1] / gpu[1],
                                      'scope': 'Same float64 algorithm, trajectories and steps; component call, not end-to-end speedup.'})
            save_json(run / 'output' / f'kernel_case_{row["index"]:04d}.json', item)
            results.append(item)
        save_json(run / 'output/kernel_summary.json', {
            'status': 'COMPLETED', 'source_commit': metadata['source_commit'],
            'results': results, 'scope': __doc__,
            'all_cpu_cuda_selections_equal': all(p['same_choices_in_order'] for r in results for p in r['pairs']),
            'all_reference_selection_hashes_equal': (all(c['same_choices_as_reference'] for r in results for c in r['calls'])
                                                     if args.reference_summary else None),
            'no_eda_executed': True,
            'warning': 'Historical calibrated proxy inputs only; no mapping-delay or true-rewrite benefit is inferred. CUDA context is shared across calls; setup/import and model construction are reported separately.'})
        metadata.update(state='COMPLETED', outputs=str(run / 'output/kernel_summary.json'))
    except Exception as exc:
        metadata.update(state='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()

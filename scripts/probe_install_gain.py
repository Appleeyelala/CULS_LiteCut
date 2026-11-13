#!/usr/bin/env python3
"""Controlled real replay/timing of archived DMA proposals under two gain policies."""
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
    parser.add_argument('--source-case', type=Path, required=True)
    parser.add_argument('--gpuls', type=Path, required=True)
    parser.add_argument('--abc', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not run.is_relative_to('/data/wushangge'):
        parser.error('Requires /data and a Slurm GPU allocation.')
    from global_timing.batch_execution import replay_many
    from global_timing.candidates import Catalogue
    from global_timing.common import digest, read_jsonl, save_json
    from global_timing.evaluation import Evaluator
    output = run / 'output/gain_probe'
    output.mkdir(exist_ok=False)
    start = time.perf_counter()
    source = args.source_case.resolve()
    config = json.loads((source / 'config.json').read_text())
    aig = Path(config['input'])
    if not aig.is_relative_to('/data/wushangge') or digest(aig) != config['input_sha256']:
        raise ValueError('Archived input identity differs.')
    stored = json.loads((source / 'cut_validation/verified_catalogue.json').read_text())
    catalogue = Catalogue(stored['decisions'], [tuple(p) for p in stored['conflicts']],
                          {int(k): v for k, v in stored['region_decisions'].items()})
    archived = [json.loads((source / f'gradient_area_constraint/eval_{i:04d}/observation.json').read_text())
                for i in (3, 4, 5, 6)]
    if any(row['status'] != 'FAIL' for row in archived):
        raise ValueError('Probe expects the four explicitly archived failed requests.')
    repair = json.loads((source / 'gradient_area_constraint/summary.json').read_text())['best']
    evaluator = Evaluator(args.abc, args.library, aig, 300)
    records = []
    for policy, vectors, labels in [
        ('positive_gain', [[0] * len(catalogue.decisions), repair['choices']] + [r['choices'] for r in archived],
         ['noop', 'archived_repair'] + [f'archived_eval_{i}' for i in (3, 4, 5, 6)]),
        ('allow_nonpositive', [r['choices'] for r in archived], [f'archived_eval_{i}' for i in (3, 4, 5, 6)]),
    ]:
        root = output / policy
        executions = replay_many(args.gpuls, aig, source / 'partition/owners.csv', catalogue, vectors,
            root / 'native', [root / name / 'rewrite' for name in labels], 300,
            allow_nonpositive_gain=policy == 'allow_nonpositive')
        for label, vector, execution in zip(labels, vectors, executions):
            row = {'policy': policy, 'label': label, 'choices': vector, 'status': execution['status'],
                   'execution': execution, 'selected_original_mffcs': len({d['decision_owner_region']
                     for d, c in zip(catalogue.decisions, vector) if c})}
            if execution['status'] == 'PASS':
                trace = read_jsonl(Path(execution['aig']).parents[1] / 'traces/choice.jsonl')
                row['accepted_nonpositive_actions'] = [{k: t[k] for k in
                    ['root', 'cut_slot', 'selected_subgraph', 'saved', 'added', 'status_reason']}
                    for t in trace if t['preselected'] and t['final_status'] == 'ACCEPTED' and t['saved'] <= t['added']]
                row['metrics'] = evaluator.evaluate(Path(execution['aig']), root / label / 'evaluation')
                row['status'] = 'PASS'
            records.append(row)
    noop = next(r['metrics'] for r in records if r['label'] == 'noop')
    for row in records:
        if row['status'] == 'PASS':
            row['area_feasible_1p05'] = row['metrics']['area'] <= noop['area'] * 1.05 + 1e-8
    commands = list(output.rglob('*.command.json'))
    counts = {name: sum(p.name == name + '.command.json' for p in commands)
              for name in ['rewrite', 'cec', 'mapping']}
    save_json(output / 'summary.json', {'status': 'COMPLETED', 'records': records,
        'source_case': str(source), 'elapsed_s': time.perf_counter() - start,
        'actual_command_counts': counts,
        'identities': {str(p): digest(p) for p in [aig, args.gpuls, args.abc, args.library,
                                                  source / 'config.json', source / 'cut_validation/verified_catalogue.json']},
        'scope': 'New controlled real rewrite/CEC/Liberty measurements of existing developer proposals; no new gradient search, training or optimizer speed claim. All ten planned configurations and all tool calls are charged to this probe.'})
    for row in records:
        metrics = row.get('metrics', {})
        print(json.dumps({'policy': row['policy'], 'label': row['label'], 'status': row['status'],
                          'delay_ps': metrics.get('delay_ps'), 'area': metrics.get('area'),
                          'nonpositive_accepted': row.get('accepted_nonpositive_actions')}), flush=True)


if __name__ == '__main__':
    main()

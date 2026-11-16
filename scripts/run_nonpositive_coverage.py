#!/usr/bin/env python3
"""Cold, candidate-only fixed100 audit. No Mapping, optimizer or A/B runs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_timing.common import digest, run_tool, save_json, tool_path

ZERO17 = [124,188,191,410,412,414,461,518,565,586,593,638,662,704,729,792,887]


def counts(catalogue, feasibility=None):
    signs = Counter()
    decisions = 0
    for i, d in enumerate(catalogue.decisions):
        active = [j for j in range(d['candidate_count'])
                  if feasibility is None or feasibility.blocked[i][j + 1] is None]
        decisions += bool(active)
        signs.update(d['candidate_gain_signs'][j] for j in active)
    return {'decisions': decisions, 'candidates': sum(signs.values()),
            'positive': signs[1], 'zero': signs[0], 'negative': signs[-1]}


def probe(row, tools, partition, catalogue, feasibility, output):
    from global_timing.candidates import replay
    from global_timing.evaluation import cec_passed
    results = []
    for sign in (0, -1):
        found = next(((i, j + 1) for i, d in enumerate(catalogue.decisions)
                      for j, gain in enumerate(d['candidate_gain_signs'])
                      if gain == sign and feasibility.blocked[i][j + 1] is None), None)
        if found is None:
            continue
        choices = [0] * len(catalogue.decisions)
        choices[found[0]] = found[1]
        feasibility.validate(choices)
        for prepared in (False, True):
            work = output / f'probe_gain_{sign}_prepared_{int(prepared)}'
            saved = os.environ.get('CULS_REPLAY_PREPARED')
            if prepared:
                os.environ['CULS_REPLAY_PREPARED'] = '1'
            else:
                os.environ.pop('CULS_REPLAY_PREPARED', None)
            record = {'gain_sign': sign, 'prepared': prepared, 'decision_index': found[0], 'choice': found[1]}
            try:
                aig, result = replay(Path(tools['gpuls']['path']), Path(row['staged_path']),
                    output / 'partition/owners.csv', catalogue, choices, work / 'rewrite', 900,
                    allow_nonpositive_gain=True, allow_nonpositive_candidates=True)
                text = run_tool([tools['abc']['path'], '-c',
                    f'cec -n {tool_path(Path(row["staged_path"]))} {tool_path(aig)}'], work, 'cec', 900)
                if not cec_passed(text):
                    raise RuntimeError('CEC did not prove the new candidate.')
                record.update(status='PASS', cec='PASS', applied=result['applied_rewrites'])
            except Exception as exc:
                record.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            finally:
                if saved is None:
                    os.environ.pop('CULS_REPLAY_PREPARED', None)
                else:
                    os.environ['CULS_REPLAY_PREPARED'] = saved
            save_json(work / 'probe.json', record)
            results.append(record)
    return results


def child(args):
    from global_timing.candidates import enumerate_candidates
    from global_timing.cut_validation import validate_catalogue
    from global_timing.feasibility import CandidateFeasibility
    from global_timing.structure import make_partition, parse_aig
    context = json.loads(args.context.read_text())
    row, tools = context['row'], context['tools']
    output = args.context.parent / 'work'
    output.mkdir(exist_ok=False)
    partition = make_partition(parse_aig(Path(row['staged_path'])))
    partition.write(output / 'partition')
    catalogue = enumerate_candidates(Path(tools['gpuls']['path']), Path(row['staged_path']),
        partition, output / 'candidates', 900, allow_nonpositive_candidates=True)
    raw_counts = counts(catalogue)
    verified, audit = validate_catalogue(Path(tools['abc']['path']), partition, catalogue,
                                        output / 'cut_validation', 900)
    feasibility = CandidateFeasibility(partition.aig, verified, ROOT / 'projects/culs/src/algorithms/rewrite_library.inc')
    save_json(output / 'candidate_feasibility.json', feasibility.payload())
    result = {'index': row['index'], 'status': 'PASS', 'raw': raw_counts,
              'verified': counts(verified), 'feasible': counts(verified, feasibility),
              'boolean_excluded_decisions': audit['excluded_decisions'],
              'blocked_reasons': dict(Counter(x for a in feasibility.blocked for x in a if x))}
    # At most the two requested representatives and one positive-control input.
    if context.get('probe'):
        result['probes'] = probe(row, tools, partition, verified, feasibility, output)
    save_json(output / 'coverage.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--gpuls', type=Path)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--context', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):
        parser.error('Requires a committed /data snapshot and Slurm allocation.')
    if args.child:
        return child(args)
    run = args.run.resolve()
    old = json.loads((args.reference_run / 'input/ste100_preflight.json').read_text())
    tools = dict(old['tools'])
    tools['gpuls'] = {'path': str(args.gpuls), 'sha256': digest(args.gpuls)}
    for item in tools.values():
        assert digest(Path(item['path'])) == item['sha256']
    rows = {r['index']: r for r in old['rows']}
    wanted = sorted(random.Random(20261116).sample(range(905),100))
    assert sorted(rows) == wanted
    order = [792,124] + [i for i in ZERO17 if i not in (792,124)] + [i for i in wanted if i not in ZERO17]
    meta = json.loads((run / 'run.json').read_text())
    save_json(run / 'input/coverage_protocol.json', {'source_commit': meta['source_commit'], 'tools': tools,
        'rows': [rows[i] for i in wanted], 'seed': 20261116, 'original_zero17': ZERO17,
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'node': os.environ.get('SLURMD_NODENAME'),
        'cpu_count': os.environ.get('SLURM_CPUS_PER_TASK'), 'case_timeout_s': 1800,
        'candidate_rule': 'allow_nonpositive_gain', 'mapping_calls_authorized': 0})
    records = []
    for i in order:
        row = rows[i]
        outer = run / 'output' / f'case_{i:04d}'
        outer.mkdir(exist_ok=False)
        assert digest(Path(row['staged_path'])) == row['sha256']
        context = outer / 'context.json'
        save_json(context, {'row': row, 'tools': tools, 'probe': i in (792,124,3)})
        began = time.perf_counter()
        error = None
        try:
            run_tool([sys.executable, '-B', __file__, '--child', '--context', str(context)],
                     outer, 'cold_coverage', 1800)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        elapsed = time.perf_counter() - began
        path = outer / 'work/coverage.json'
        command = json.loads((outer / 'cold_coverage.command.json').read_text())
        result = json.loads(path.read_text()) if path.exists() else {'index': i, 'status': 'TIMEOUT' if command.get('timed_out') else 'FAIL'}
        result.update(input=row['relative_path'], input_sha256=row['sha256'], error=error,
                      coverage_runtime_s=elapsed, evidence=str(outer), source_commit=meta['source_commit'])
        save_json(run / 'output/records' / f'{i:04d}.json', result)
        records.append(result)
        print(json.dumps({k: result[k] for k in ('index','status','coverage_runtime_s')} | {'counts':result.get('feasible')}), flush=True)
        if len(records) == 17:
            gate = all(r['status']=='PASS' and all(r[k]['candidates']>0 for k in ('raw','verified','feasible')) for r in records)
            save_json(run / 'output/GATE17.json', {'GO_905': gate, 'records': records,
                       'scope': 'All17 raw, Boolean and feasible positive counts; fixed100 still required.'})
    gate = json.loads((run / 'output/GATE17.json').read_text())['GO_905']
    save_json(run / 'output/COVERAGE_STATUS.json', {'completed':len(records), 'counts':dict(Counter(r['status'] for r in records)),
        'GO_905':gate and all(r['status']=='PASS' for r in records), 'gate17_passed':gate,
        'mapping_calls':len(list((run/'output').rglob('mapping.command.json'))), 'records':records})

if __name__ == '__main__':
    main()

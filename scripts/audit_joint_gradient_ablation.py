#!/usr/bin/env python3
"""Read-only development audit after proxy choices are frozen; never searches or maps."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_timing.research_cohort import compare_delay
from global_timing.structure import parse_aig
from global_timing.evaluation import parse_timing, cec_passed
from summarize_joint_gradient_panel import summarize


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def wins(values):
    c = Counter(values)
    return dict(win=c[-1], tie=c[0], loss=c[1])


def fanout_outputs(aig, roots):
    users = defaultdict(list)
    for node, literals in aig.fanin_literals.items():
        for literal in literals:
            users[literal >> 1].append(node)
    result = []
    for root in roots:
        seen, pending = {root}, [root]
        while pending:
            for node in users[pending.pop()]:
                if node not in seen:
                    seen.add(node)
                    pending.append(node)
        result.append({i for i, literal in enumerate(aig.po_literals) if literal >> 1 in seen})
    return result


def audit(panel, ablation):
    summary, rows, methods, pairs = summarize(panel, ablation)
    checks, cases, choices, locality = [], [], [], []
    checked_tools = set()
    mapping_times = []
    for folder in sorted(panel.glob('case_*')):
        frozen = read(folder / 'frozen_predictions.json')
        ab = read(ablation / folder.name / 'ablation.json')
        source = Path(frozen['source'])
        context = read(source / 'context.json')
        baseline = read(folder / 'baseline.json')
        raw = read(folder / 'catalogue.json')
        measurement = read(folder / 'measurements.json')
        assert sha(source / 'input.aig') == frozen['input_sha256'] == ab['input_sha256']
        assert sha(folder / 'frozen_predictions.json') == ab['panel_sha256'] == measurement['frozen_predictions_sha256']
        assert set(ab['methods']) == {'old_ste', 'old_aligned_ste', 'old_joint', 'ports_ste', 'ports_aligned_ste', 'ports_joint'}
        assert ab['new_mapping_calls'] == 0
        assert ab['methods']['old_ste']['mask'] == frozen['ste_best_mask']
        for tool in context['tools'].values():
            if not tool['path'] or tool['path'] in checked_tools:
                continue
            assert sha(Path(tool['path'])) == tool['sha256'], tool['path']
            checked_tools.add(tool['path'])
            checks.append(dict(path=tool['path'], sha256=tool['sha256']))
        actual = {r['mask']: r for r in measurement['results'] if r['status'] == 'PASS' and r['metrics']['cec'] == 'PASS'}
        cid = frozen['circuit']
        cone = fanout_outputs(parse_aig(source / 'input.aig'), [d['root'] for d in raw['decisions']])
        actual_delays = [r['metrics']['delay_ps'] for r in actual.values()]
        tol = max(2e-5, 2e-6 * baseline['delay_ps'])
        for mask, observation in actual.items():
            if mask == 0:
                continue
            metrics = observation['metrics']
            changed = {i for i, (a, b) in enumerate(zip(metrics['po_arrivals_ps'], baseline['po_arrivals_ps'])) if abs(a-b) > tol}
            reachable = set().union(*(cone[i] for i in range(4) if mask >> i & 1))
            outside = changed - reachable
            locality.append(dict(circuit_id=cid, mask=mask, structural_reachable_pos=len(reachable),
                changed_pos=len(changed), changed_outside_structural_fanout=len(outside),
                maximum_outside_change_ps=max((abs(metrics['po_arrivals_ps'][i]-baseline['po_arrivals_ps'][i]) for i in outside), default=0.)))
            evaluation = folder / 'measurements' / f'mask_{mask:02d}' / 'evaluation'
            command = read(evaluation / 'mapping.command.json')
            assert command['returncode'] == 0 and not command.get('timed_out')
            cec = read(evaluation / 'cec.command.json')
            assert cec['returncode'] == 0 and not cec.get('timed_out')
            assert cec_passed((evaluation / 'cec.log').read_text())
            parsed = parse_timing((evaluation / 'mapping.log').read_text())
            assert parsed['delay_ps'] == metrics['delay_ps'] and parsed['area'] == metrics['area']
            assert sha(evaluation / 'mapped.v') == metrics['mapped_sha256']
            assert sha(Path(observation['execution']['aig'])) == metrics['input_sha256']
            mapping_times.append(dict(circuit_id=cid, mask=mask, command=command))
        for name, predictions in ab['models'].items():
            estimates = [p['delay_ps'] for p in predictions]
            assert abs(predictions[0]['delay_ps'] - baseline['delay_ps']) < 1e-8
            min_mask = min(range(16), key=lambda k: (estimates[k], k.bit_count(), k))
            feasible = [r for r in actual.values() if r['metrics']['area'] <= baseline['area'] * 1.05]
            oracle = min(feasible, key=lambda r: r['metrics']['delay_ps'])
            cases.append(dict(circuit_id=cid, model=name, complete_panel=len(actual)==16,
                predicted_span_ps=max(estimates)-min(estimates), measured_span_ps=max(actual_delays)-min(actual_delays),
                all_predictions_tied=max(estimates)-min(estimates)<=tol, proxy_best_mask=min_mask,
                observed_oracle_mask=oracle['mask'], observed_oracle_delay_ps=oracle['metrics']['delay_ps'],
                observed_real_improvements=sum(compare_delay(r['metrics']['delay_ps'],baseline['delay_ps'])<0 for r in actual.values())))
        for name, method in ab['methods'].items():
            mask = method['mask']; trace = method['trace']; trajectory = trace['trajectory']
            observed = actual.get(mask)
            old = actual.get(ab['methods']['old_ste']['mask'])
            proposals = [p for step in trajectory for p in step.get('proposals', [])]
            choices.append(dict(circuit_id=cid, method=name, mask=mask, edits=mask.bit_count(),
                verified=observed is not None, delay_vs_original=compare_delay(observed['metrics']['delay_ps'], baseline['delay_ps']) if observed else None,
                delay_vs_old_ste=compare_delay(observed['metrics']['delay_ps'],old['metrics']['delay_ps']) if observed and old else None,
                backward_calls=trace.get('backward_calls',trace.get('steps')), forward_calls=trace.get('forward_calls',trace.get('steps',0)+1),
                accepted_joint_steps=sum(step.get('accepted') is not None for step in trajectory),
                distinct_proposals=len({tuple(p['choices']) for p in proposals}), termination=trace.get('termination'),
                search_wall_s=method['search_wall_s']))
    summary['gate'] = {name: dict(
        ranking_pass=v['median_spearman'] is not None and v['median_spearman'] >= .5,
        precision_pass=v['improvement_precision'] is not None and v['improvement_precision'] >= .7,
        scope='Development diagnostic gate only; passing would not establish independent-validation or deployment quality.')
        for name, v in summary['models'].items()}
    summary['any_proxy_meets_development_gate'] = any(g['ranking_pass'] and g['precision_pass'] for g in summary['gate'].values())
    summary['choices'] = {}
    for name in sorted({r['method'] for r in choices}):
        selected = [r for r in choices if r['method']==name]
        summary['choices'][name] = dict(noops=sum(r['mask']==0 for r in selected),
            vs_original=wins([r['delay_vs_original'] for r in selected if r['verified']]),
            vs_old_ste=wins([r['delay_vs_old_ste'] for r in selected if r['delay_vs_old_ste'] is not None]),
            median_backward_calls=statistics.median(r['backward_calls'] for r in selected),
            median_forward_calls=statistics.median(r['forward_calls'] for r in selected))
    summary['locality'] = dict(measured_nonempty=len(locality),
        combinations_with_changes_outside_structural_fanout=sum(r['changed_outside_structural_fanout']>0 for r in locality),
        caveat='Describes observed mapping changes, not causal attribution. No serialized-replay no-op control was mapped; rewrite, serialization and global remapping effects are not separated.')
    summary['verified_tool_hashes'] = checks
    summary['runtime_scope'] = 'Search-only timings; reused candidate scores and original Mapping. No measured end-to-end optimization Runtime or speedup claim.'
    return summary, cases, choices, locality, mapping_times


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-root', type=Path, required=True)
    p.add_argument('--ablation-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):
        p.error('Run the input graph audit under Slurm from /data.')
    summary, cases, choices, locality, mapping = audit(args.panel_root, args.ablation_root)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (args.output/'mapping_commands.json').write_text(json.dumps(mapping,indent=2,allow_nan=False)+'\n')
    for name, data in [('mechanisms',cases),('selection_trace',choices),('structural_locality',locality)]:
        with (args.output/f'{name}.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('cases','evidence','verified_tool_hashes')},indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Stage 0: read existing records and freeze data splits; no model or ABC calls."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from global_timing.research_cohort import grouped_split, blind_panel, compare_delay, rank_key


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):
        p.error('Committed /data source and Slurm required.')
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = read(args.reference_run/'input/full905_manifest.json')['rows']
    if len(inventory) != 905:
        raise ValueError('Expected the complete original inventory.')
    splits = grouped_split(inventory)
    assignments = {r['circuit_id']: r for r in splits}
    evidence = []
    public = ROOT/'analysis/ste_adam_100/nonpositive_gain/full905/FULL905_OURS_RESULTS.csv'
    with public.open() as f:
        prior = {int(r['circuit_id']): r for r in csv.DictReader(f)}
    if set(prior) != set(assignments):
        raise ValueError('Inventory/result IDs mismatch.')
    for item in inventory:
        i = item['index']
        outer = args.reference_run/'output'/f'case_{i:04d}_Ours'
        summary = outer/'work/summary.json'
        row = dict(assignments[i], status=prior[i]['status'], evidence_path=str(summary),
                   original_delay_ps=float(prior[i]['original_delay_ps']) if prior[i]['original_delay_ps'] else None,
                   candidate_delay_ps=float(prior[i]['candidate_delay_ps']) if prior[i]['candidate_delay_ps'] else None,
                   mechanism='MISSING_COMPLETED_SEARCH')
        if summary.exists():
            data = read(summary)
            best = data.get('proxy_best')
            row.update(summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
                       decisions=data['decisions'], regions=data['regions'], candidates=data['candidates'])
            if best:
                trajectory = read(outer/'work/optimization/trajectory.json')
                selected = best['edits']
                row.update(predicted_delay_ps=best['delay_ps'],
                           proxy_loss_delta=best['predicted_objective']-trajectory['trajectory'][0]['loss'],
                           predicted_area_ratio=best['area']/data['original']['area'],
                           edits=len(selected), distinct=trajectory['distinct_hard_combinations_including_noop'],
                           predicted_comparison=compare_delay(best['delay_ps'], data['original']['delay_ps']))
                if not selected:
                    row['mechanism'] = 'NOOP'
                elif row['candidate_delay_ps'] is not None:
                    real = compare_delay(row['candidate_delay_ps'], row['original_delay_ps'])
                    row.update(real_comparison=real,
                               actual_area_ratio=data['final_candidate']['metrics']['area']/data['original']['area'])
                    row['mechanism'] = ('SMOOTH_ONLY' if row['predicted_comparison'] >= 0 else
                                        'PREDICTED_DOWN_REAL_UP' if real > 0 else
                                        'BOTH_DOWN' if real < 0 else 'PREDICTED_DOWN_REAL_TIE')
                else:
                    row['mechanism'] = 'EXECUTION_FAILURE'
        evidence.append(row)
    panels = {}
    used_groups = set()
    for mechanism in ['PREDICTED_DOWN_REAL_UP', 'SMOOTH_ONLY', 'BOTH_DOWN']:
        pool = [r for r in evidence if r['split']=='train' and r['mechanism']==mechanism
                and r.get('edits',0)>=4 and r.get('regions',0)>=2]
        chosen=[]
        # Two area failures are included where available, explicitly outcome-selected development cases.
        if mechanism=='PREDICTED_DOWN_REAL_UP':
            for r in sorted(pool, key=lambda r: (r['and_count'], rank_key(20261117, r['circuit_id']))):
                if r['status']=='AREA_LIMIT_EXCEEDED' and r['group_id'] not in used_groups:
                    chosen.append(r); used_groups.add(r['group_id'])
                    if len(chosen)==2: break
        for size in range(4):
            candidates=[r for r in pool if r['size_bin']==size and r['group_id'] not in used_groups]
            if candidates and len(chosen)<4:
                r=min(candidates,key=lambda r:rank_key(20261117,r['circuit_id']))
                chosen.append(r); used_groups.add(r['group_id'])
        for r in sorted(pool,key=lambda r:rank_key(20261117,r['circuit_id'])):
            if len(chosen)<4 and r['group_id'] not in used_groups:
                chosen.append(r); used_groups.add(r['group_id'])
        if len(chosen)!=4: raise ValueError('Insufficient development mechanism stratum: '+mechanism)
        panels[mechanism]=[r['circuit_id'] for r in chosen]
    manifest={'schema':'joint_gradient_v2_split_v1','seed':20261117,'circuits':splits,
              'counts':dict(Counter(r['split'] for r in splits)),
              'development12':panels,'validation24':blind_panel(splits,'validation',24),
              'test100':blind_panel(splits,'test',100),
              'limitations':'Historical outcomes have been studied. Development is outcome-selected; validation/test panel selection is label-blind. Family normalization is conservative, not a proof of RTL independence.'}
    save(args.output/'cohort_manifest.json',manifest)
    save(args.output/'evidence.json',evidence)
    fields=sorted(set().union(*(r.keys() for r in evidence)))
    with (args.output/'evidence.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(evidence)
    measured=[r for r in evidence if r.get('real_comparison') is not None]
    stats={'rows':len(evidence),'unique_ids':len(assignments),'status':dict(Counter(r['status'] for r in evidence)),
           'mechanisms':dict(Counter(r['mechanism'] for r in evidence)),
           'measured_nonempty':len(measured),'real_delay':dict(Counter(r['real_comparison'] for r in measured)),
           'predicted_real_table':dict(Counter(f'{r["predicted_comparison"]}/{r["real_comparison"]}' for r in measured)),
           'mapping_calls':0,'source_commit':read(ROOT.parent/'run.json')['source_commit'],
           'slurm_job_id':os.environ['SLURM_JOB_ID'],'cohort_counts':manifest['counts'],'development12':panels}
    save(args.output/'summary.json',stats)
    print(json.dumps(stats),flush=True)


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Read saved predictions: inherited structural reversals and objective mismatch."""
import argparse
from collections import Counter
import csv
import itertools
import json
import math
from pathlib import Path
import statistics as stats
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_v3_ranking import ranks, sha, write_csv


def ranking_value(pred, actual, original):
    ar=ranks(actual,.001);terms=[]
    for i,j in itertools.combinations(range(8),2):
        if ar[i]==ar[j]:continue
        gap=actual[i]-actual[j];scale=.01*original
        z=-math.copysign(1.,gap)*(pred[i]-pred[j])/scale
        terms.append(min(abs(gap)/scale,1.)*(max(z,0.)+math.log1p(math.exp(-abs(z)))))
    return stats.mean(terms) if terms else 0.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();folder=a.fit_run/'output/report'
    source=ROOT/'analysis/joint_gradient_v3/ranking_improvement/diagnosis/combinations.csv'
    structural={(int(r['circuit_id']),int(r['index'])):r for r in csv.DictReader(source.open()) if r['seed']=='20261117'}
    combo_path=folder/'all_combinations.csv'
    saved=list(csv.DictReader(combo_path.open()));losses=[];pairs=[];cases=[]
    for seed in [20261117,20261118,20261119]:
        summary=json.loads((a.fit_run/'output'/f'gap_ranking_{seed}'/'summary.json').read_text())
        for role,summary_key in [('train','train'),('development','validation')]:
            ids=list(dict.fromkeys(int(r['circuit_id']) for r in saved if r['role']==role))
            objectives={'v3_global':[],'gap_ranking':[]}
            for cid in ids:
                series={m:sorted([r for r in saved if r['role']==role and int(r['seed'])==seed and int(r['circuit_id'])==cid and r['model']==m],key=lambda r:int(r['index'])) for m in objectives}
                y=[float(r['actual_delay_ps']) for r in series['gap_ranking']]
                fixed=[float(structural[cid,i]['fixed_critical_ps']) for i in range(1,9)]
                original=float(structural[cid,1]['original_delay_ps'])
                ar,fr=ranks(y,.001),ranks(fixed,.001)
                for model in objectives:
                    pred=[float(r['predicted_delay_ps']) for r in series[model]]
                    objectives[model].append(ranking_value(pred,y,original)*8/9)
                    if model!='gap_ranking':continue
                    pr=ranks(pred,.001)
                    cases.append(dict(role=role,seed=seed,circuit_id=cid,
                        maximum_absolute_critical_correction_ps=max(abs(u-v) for u,v in zip(pred,fixed)),
                        mean_absolute_critical_correction_ps=stats.mean(abs(u-v) for u,v in zip(pred,fixed)),
                        fixed_range_ps=max(fixed)-min(fixed),predicted_range_ps=max(pred)-min(pred),actual_range_ps=max(y)-min(y)))
                    for i,j in itertools.combinations(range(8),2):
                        if abs(y[i]-y[j])/min(y)<=.05 or ar[i]==ar[j]:continue
                        if (pr[i]-pr[j])*(ar[i]-ar[j])>=0:continue
                        f=(fr[i]-fr[j])*(ar[i]-ar[j])
                        pairs.append(dict(role=role,seed=seed,circuit_id=cid,index_a=i+1,index_b=j+1,
                            fixed_classification='wrong' if f<0 else 'correct' if f>0 else 'predicted_tie_only',
                            new_classification='wrong',actual_gap_percent=100*abs(y[i]-y[j])/min(y)))
            measured=summary[summary_key]['mean_circuit_loss']
            recomputed=stats.mean(objectives['gap_ranking'])
            assert abs(measured-recomputed)<1e-9,(seed,role,measured,recomputed)
            losses.append(dict(seed=seed,role=role,old_predictions_under_new_objective=stats.mean(objectives['v3_global']),
                new_predictions_under_new_objective=recomputed,selected_epoch=summary['best_epoch'],
                epochs=len(summary['history']),first_train_loss=summary['history'][0]['train_loss'],
                last_train_loss=summary['history'][-1]['train_loss'],first_development_loss=summary['history'][0]['validation_loss'],
                last_development_loss=summary['history'][-1]['validation_loss']))
    a.output.mkdir(parents=True,exist_ok=False)
    for name,rs in [('same_objective',losses),('critical_corrections',cases),('severe_reversal_origins',pairs)]:write_csv(a.output/(name+'.csv'),rs)
    out=dict(severe_development_origins=dict(Counter(r['fixed_classification'] for r in pairs if r['role']=='development')),
        files=[dict(path=str(f),sha256=sha(f)) for f in [source,combo_path]],new_training_runs=0,new_mapping_calls=0,
        caveat='Critical correction is difference of two maxima, not an aligned per-PO residual or a true timing error.')
    (a.output/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Read-only factorial evidence join; preserve missing/partial measurements."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from global_timing.research_cohort import compare_delay


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def median(v):return statistics.median(v) if v else None
def gm(v):return math.exp(statistics.mean(map(math.log,v))) if v else None


def ranks(values):
    """Average ranks; ties use the protocol measurement tolerance."""
    order=sorted(range(len(values)),key=lambda i:values[i]);result=[0.]*len(values);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and compare_delay(values[order[end]],values[order[start]])==0:end+=1
        for i in order[start:end]:result[i]=(start+end-1)/2
        start=end
    return result


def spearman(x,y):
    if len(x)<3:return None
    a=ranks(x);b=ranks(y);ma=statistics.mean(a);mb=statistics.mean(b)
    denominator=math.sqrt(sum((v-ma)**2 for v in a)*sum((v-mb)**2 for v in b))
    return sum((v-ma)*(w-mb) for v,w in zip(a,b))/denominator if denominator else None


def summarize(panel_root,ablation_root=None):
    rows=[];cases=[];method_rows=[];pairs=[];evidence=[]
    baselines={(int(r['index']),r['method']):r for r in csv.DictReader((ROOT/'BENCHMARK905_RESULTS.csv').open()) if r['method'] in ('A','B')}
    for folder in sorted(panel_root.glob('case_*')):
        frozen=folder/'frozen_predictions.json';f=read(frozen);baseline=read(folder/'baseline.json');cid=f['circuit']
        observations={0:{'status':'PASS','metrics':baseline}};full=folder/'measurements.json'
        if full.exists():
            m=read(full)
            if m['frozen_predictions_sha256']!=sha(frozen):raise ValueError('Prediction hash mismatch.')
            observations.update({r['mask']:r for r in m['results']})
        else:
            for mask in range(1,16):
                d=folder/'measurements'/f'mask_{mask:02d}'
                p=d/'observation_resumed.json'
                if not p.exists():p=d/'observation.json'
                if p.exists():
                    try:observations[mask]=read(p);continue
                    except json.JSONDecodeError:pass
                if (d/'evaluation').exists():observations[mask]={'status':'INCOMPLETE_RECORD'}
        models={'old':[dict(mask=r['mask'],delay_ps=r['predicted_delay_ps'],area=r['predicted_area'],objective=r['predicted_loss']) for r in f['combinations']]}
        methods={'old_ste':{'mask':f['ste_best_mask'],'best':f['ste_best']}}
        if ablation_root:
            p=ablation_root/folder.name/'ablation.json'
            if p.exists():
                ablation=read(p)
                if ablation['panel_sha256']!=sha(frozen):raise ValueError('Ablation panel changed.')
                models=ablation['models'];methods=ablation['methods'];evidence.append({'path':str(p),'sha256':sha(p)})
        actual={k:o['metrics'] for k,o in observations.items() if o['status']=='PASS' and o.get('metrics',{}).get('cec')=='PASS'}
        feasible={k:v for k,v in actual.items() if v['area']<=baseline['area']*1.05}
        oracle=min(feasible,key=lambda k:feasible[k]['delay_ps']) if feasible else None
        case={'circuit_id':cid,'complete_panel':len(actual)==16,'verified_nonempty':len(actual)-int(0 in actual),
              'nonempty_area_violations':sum(k!=0 and k not in feasible for k in actual),'models':{}}
        for model,predictions in models.items():
            pred={r['mask']:r for r in predictions};valid=sorted(actual)
            rho=spearman([pred[k]['delay_ps'] for k in valid],[actual[k]['delay_ps'] for k in valid])
            predicted_down=[k for k in valid if k and compare_delay(pred[k]['delay_ps'],baseline['delay_ps'])<0]
            correct=sum(compare_delay(actual[k]['delay_ps'],baseline['delay_ps'])<0 for k in predicted_down)
            case['models'][model]={'spearman':rho,'ranking_pairs':len(valid),'predicted_improvements':len(predicted_down),
                                   'correct_improvements':correct,'improvement_precision':correct/len(predicted_down) if predicted_down else None}
            for mask in range(16):
                real=actual.get(mask);status=observations.get(mask,{}).get('status','NOT_RUN')
                rows.append({'circuit_id':cid,'model':model,'mask':mask,'status':status,'predicted_delay_ps':pred[mask]['delay_ps'],
                             'predicted_area':pred[mask]['area'],'predicted_objective':pred[mask]['objective'],
                             'mapped_delay_ps':real['delay_ps'] if real else None,'mapped_area':real['area'] if real else None,
                             'real_area_feasible':mask in feasible if real else None,'input_sha256':f['input_sha256']})
            for i in range(4):
                for j in range(i):
                    a,b=1<<i,1<<j;m=a|b
                    if all(k in actual for k in (0,a,b,m)):
                        real=actual[m]['delay_ps']-actual[a]['delay_ps']-actual[b]['delay_ps']+actual[0]['delay_ps']
                        estimate=pred[m]['delay_ps']-pred[a]['delay_ps']-pred[b]['delay_ps']+pred[0]['delay_ps']
                        pairs.append({'circuit_id':cid,'model':model,'decision_a':i,'decision_b':j,'real_interaction_ps':real,
                                      'predicted_interaction_ps':estimate,'absolute_error_ps':abs(real-estimate)})
        for method,selection in methods.items():
            mask=selection['mask'];real=actual.get(mask);model=method.split('_')[0]
            # Objective scales differ for the aligned controls; compare hard
            # predicted max separately from real oracle, never deploy oracle.
            predicted=models[model];proxy_min=min(r['delay_ps'] for r in predicted if r['area']<=baseline['area']*1.05)
            b=baselines.get((cid,'B'));a=baselines.get((cid,'A'))
            for entry in (a,b):
                if entry and entry['input_sha256']!=f['input_sha256']:raise ValueError('Baseline input mismatch.')
            method_rows.append({'circuit_id':cid,'method':method,'mask':mask,'measured_status':observations.get(mask,{}).get('status','NOT_RUN'),
                'predicted_search_gap_ps':selection['best']['delay_ps']-proxy_min,
                'delay_ps':real['delay_ps'] if real else None,'area_feasible':mask in feasible if real else None,
                'ratio_A':real['delay_ps']/float(a['delay_ps']) if real and a and a['status']=='PASS' else None,
                'ratio_B':real['delay_ps']/float(b['delay_ps']) if real and b and b['status']=='PASS' else None,
                'observed_oracle_gap_ps':real['delay_ps']-feasible[oracle]['delay_ps'] if real and oracle is not None else None,
                'oracle_complete':len(actual)==16,'search_wall_s':selection.get('search_wall_s')})
        cases.append(case);evidence.append({'path':str(frozen),'sha256':sha(frozen)})
        if full.exists():evidence.append({'path':str(full),'sha256':sha(full)})
    summary={'scope':'Development-only restricted binary domains. Incomplete panels excluded from complete-panel model aggregates. No deployment quality claim.',
        'cases':cases,'expected_nonempty':15*len(cases),'verified_nonempty':sum(c['verified_nonempty'] for c in cases),
        'complete_panels':sum(c['complete_panel'] for c in cases),'mapping_log_count':len(list(panel_root.rglob('mapping.log'))),
        'mapping_command_count':len(list(panel_root.rglob('mapping.command.json'))),'models':{},'methods':{},'evidence':evidence}
    for model in sorted({name for c in cases for name in c['models']}):
        valid=[c['models'][model] for c in cases if c['complete_panel'] and model in c['models']]
        correlations=[r['spearman'] for r in valid if r['spearman'] is not None]
        positives=sum(r['predicted_improvements'] for r in valid);correct=sum(r['correct_improvements'] for r in valid)
        summary['models'][model]={'complete_circuits':len(valid),'defined_correlations':len(correlations),'median_spearman':median(correlations),
            'predicted_improvements':positives,'correct_improvements':correct,'improvement_precision':correct/positives if positives else None}
    for method in sorted({r['method'] for r in method_rows}):
        selected=[r for r in method_rows if r['method']==method and r['delay_ps'] is not None]
        summary['methods'][method]={'measured_choices':len(selected),'area_violations':sum(not r['area_feasible'] for r in selected),
            'geomean_ratio_A':gm([r['ratio_A'] for r in selected if r['ratio_A'] is not None]),
            'geomean_ratio_B':gm([r['ratio_B'] for r in selected if r['ratio_B'] is not None]),
            'predicted_search_gap_median_ps':median([r['predicted_search_gap_ps'] for r in selected]),
            'search_runtime_median_s':median([r['search_wall_s'] for r in selected if r['search_wall_s'] is not None])}
    return summary,rows,method_rows,pairs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel-root',type=Path,required=True)
    p.add_argument('--ablation-root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summary,rows,methods,pairs=summarize(a.panel_root,a.ablation_root);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for name,values in [('combinations',rows),('methods',methods),('pair_interactions',pairs)]:
        if not values:continue
        with (a.output/f'{name}.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('evidence','cases')},indent=2))


if __name__=='__main__':main()

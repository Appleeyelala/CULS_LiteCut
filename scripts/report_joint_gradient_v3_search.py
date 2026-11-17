#!/usr/bin/env python3
"""Join frozen development search outputs with existing measurements afterwards."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint_gradient_panel import read,save,sha
from report_joint_gradient_v3 import csv_file


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--baselines',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    baseline={}
    with a.baselines.open() as f:
        for row in csv.DictReader(f):baseline[int(row['circuit_id'])]=row
    rows=[];evidence=[];predictions=[]
    for path in sorted((a.run/'output').glob('case_*.json')):
        result=read(path);cid=result['circuit'];folder=a.panel/f'case_{cid:04d}'
        if result['panel_sha256']!=sha(folder/'frozen_predictions.json'):raise ValueError('Changed panel')
        measured={r['mask']:r for r in read(folder/'measurements.json')['results']}
        original=read(folder/'baseline.json');b=baseline[cid]
        frozen=read(folder/'frozen_predictions.json')
        if b['input_sha256']!=frozen['input_sha256'] or abs(float(b['A_delay_ps'])-original['delay_ps'])>2e-5:
            raise ValueError('Baseline provenance mismatch')
        for prediction in result['predictions']:
            observation=measured[prediction['mask']];metrics=observation.get('metrics',{})
            predictions.append(dict(circuit_id=cid,variant=result['variant'],mask=prediction['mask'],
                predicted_delay_ps=prediction['delay_ps'],measured_status=observation['status'],
                measured_delay_ps=metrics.get('delay_ps'),
                area_feasible=metrics['area']<=original['area']*1.05 if observation['status']=='PASS' and 'area' in metrics else None))
        feasible_predicted=[r['delay_ps'] for r in result['predictions'] if r['area']<=original['area']*1.05]
        for solver,selection in result['methods'].items():
            mask=selection['mask'];observation=measured[mask];metrics=observation.get('metrics',{})
            # Incomplete observations stay incomplete; never promote diagnostic-only timing.
            success=observation['status']=='PASS' and metrics.get('cec')=='PASS'
            best=selection['best'];trace=selection['trace'];area=metrics.get('area') if success else None
            rows.append(dict(circuit_id=cid,variant=result['variant'],solver=solver,mask=mask,edits=len(best['edits']),
                status=observation['status'],cec=metrics.get('cec') if success else None,
                original_delay_ps=original['delay_ps'],A_delay_ps=float(b['A_delay_ps']),B_delay_ps=float(b['B_delay_ps']),
                selected_delay_ps=metrics['delay_ps'] if success else None,selected_area=area,
                area_feasible=area<=original['area']*1.05 if area is not None else None,
                predicted_delay_ps=best['delay_ps'],proxy_optimum_gap_ps=best['delay_ps']-min(feasible_predicted),
                search_wall_s=selection['search_wall_s'],backward_calls=trace.get('backward_calls'),forward_calls=trace.get('forward_calls')))
        evidence.append({'path':str(path),'sha256':sha(path),'measurement_path':str(folder/'measurements.json'),
                         'measurement_sha256':sha(folder/'measurements.json')})
    methods={}
    for variant in ['fixed','ports','global','combined']:
        for solver in ['aligned_ste','joint']:
            selected=[r for r in rows if r['variant']==variant and r['solver']==solver]
            complete=[r for r in selected if r['selected_delay_ps'] is not None]
            win=tie=loss=0
            for r in complete:
                delta=r['selected_delay_ps']-r['A_delay_ps'];tol=max(2e-5,2e-6*r['A_delay_ps'])
                win+=delta < -tol;loss+=delta > tol;tie+=abs(delta)<=tol
            methods[variant+'_'+solver]={'records':len(selected),'complete':len(complete),
                'delay_geomean_over_A':math.exp(statistics.mean(math.log(r['selected_delay_ps']/r['A_delay_ps']) for r in complete)) if complete else None,
                'delay_geomean_over_B':math.exp(statistics.mean(math.log(r['selected_delay_ps']/r['B_delay_ps']) for r in complete)) if complete else None,
                'win_tie_loss_over_A':[win,tie,loss],'noop':sum(r['mask']==0 for r in selected),
                'area_violations':sum(r['area_feasible'] is False for r in selected),
                'proxy_optimum_misses':sum(r['proxy_optimum_gap_ps']>max(2e-5,2e-6*r['original_delay_ps']) for r in selected),
                'search_wall_median_s':statistics.median(r['search_wall_s'] for r in selected) if selected else None}
    common=set(baseline)
    for row in rows:
        if row['selected_delay_ps'] is None or not row['area_feasible']:common.discard(row['circuit_id'])
    common_metrics={}
    for name in methods:
        variant,solver=name.split('_',1)
        subset=[r for r in rows if r['variant']==variant and r['solver']==solver and r['circuit_id'] in common]
        common_metrics[name]={'count':len(subset),'delay_geomean_over_B':math.exp(statistics.mean(math.log(r['selected_delay_ps']/r['B_delay_ps']) for r in subset)) if subset else None}
    csv_file(a.output/'selected_results.csv',rows);csv_file(a.output/'frozen_predictions.csv',predictions)
    save(a.output/'summary.json',{'methods':methods,'new_mapping_calls':0,'expected_selection_records':96,
         'actual_selection_records':len(rows),'scope':'Previous 12 development circuits, four binary choices each; not independent evaluation or full-domain search.'})
    save(a.output/'evidence_index.json',{'run':str(a.run),'baselines_path':str(a.baselines),'baselines_sha256':sha(a.baselines),'files':evidence})
    save(a.output/'common_feasible.json',{'circuit_ids':sorted(common),'methods':common_metrics,
         'scope':'Post-hoc common feasible cohort diagnostic; does not repair failed full-cohort area constraints.'})
    print(json.dumps(methods,indent=2))


if __name__=='__main__':main()

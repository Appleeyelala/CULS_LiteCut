#!/usr/bin/env python3
"""Publish small v3 evidence from immutable completed run records."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint_gradient_panel import read,save,sha


def csv_file(path,rows):
    if not rows:return
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-run',type=Path,required=True)
    p.add_argument('--learning-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    data=read(a.data_run/'run.json');learning=read(a.learning_run/'run.json')
    manifest_path=Path(data['source_snapshot'])/'analysis/joint_gradient_v3/learning_manifest.json';manifest=read(manifest_path)
    rows=[];cases=[];evidence=[];history=[]
    for item in manifest['selected']:
        folder=a.data_run/'output'/f'case_{item["circuit_id"]:04d}'
        baseline=read(folder/'baseline.json');frozen=read(folder/'frozen.json');measured=read(folder/'measurements.json')
        if measured['frozen_sha256']!=sha(folder/'frozen.json'):raise ValueError('Freeze digest changed')
        noop=next((r.get('metrics',{}).get('delay_ps') for r in measured['results'] if r['index']==0 and r['status']=='PASS'),None)
        cases.append(dict(circuit_id=item['circuit_id'],role=item['role'],family=item['family'],input=item['input'],input_sha256=item['input_sha256'],
            combinations=len(measured['results']),mapping_calls=measured['new_mapping_calls'],original_delay_ps=baseline['delay_ps'],
            serialized_noop_delay_ps=noop,serialized_noop_ratio=noop/baseline['delay_ps'] if noop is not None else None))
        for record in measured['results']:
            metrics=record.get('metrics',{});delay=metrics.get('delay_ps');area=metrics.get('area')
            rows.append(dict(circuit_id=item['circuit_id'],role=item['role'],index=record['index'],kind=record['kind'],status=record['status'],
                cec=metrics.get('cec'),edits=sum(bool(c) for c in record['choices']),original_delay_ps=baseline['delay_ps'],
                candidate_delay_ps=delay,ratio_to_original=delay/baseline['delay_ps'] if delay is not None else None,
                ratio_to_serialized_noop=delay/noop if delay is not None and noop else None,
                area=area,area_ratio=area/baseline['area'] if area is not None else None,area_feasible=record.get('area_feasible'),
                structural_error=record.get('structural_max_error'),measured_evaluation_runtime_s=metrics.get('runtime_s')))
        for name in ['frozen.json','measurements.json','baseline.json','historical.json','historical_excluded.json']:
            f=folder/name
            if f.exists():evidence.append({'path':str(f),'sha256':sha(f)})
        if (folder/'historical.json').exists():
            h=read(folder/'historical.json');history.append(dict(circuit_id=item['circuit_id'],structural_error=h['structural_max_error'],
                cec=h['metrics']['cec'],area_feasible=h['area_feasible'],source=h['source_summary']))
    csv_file(a.output/'measured_combinations.csv',rows);csv_file(a.output/'circuit_accounting.csv',cases);csv_file(a.output/'historical_reuse.csv',history)
    training=[]
    for summary in sorted((a.learning_run/'output').glob('*/summary.json')):
        s=read(summary);v=s['validation']
        training.append(dict(variant=s['variant'],seed=s['seed'],best_epoch=s['best_epoch'],epochs=len(s['history']),
            median_spearman=v['median_spearman_with_undefined_as_zero'],precision=v['improvement_precision'],coverage=v['improvement_coverage'],
            precision_vs_serialized_noop=v['precision_against_serialized_noop'],validation_loss=v['mean_circuit_loss'],
            gate_pass=v['gate_pass'],elapsed_s=s['elapsed_s'],source=str(summary)))
        evidence.append({'path':str(summary),'sha256':sha(summary)})
        save(a.output/f'{s["variant"]}_{s["seed"]}_validation.json',v)
    csv_file(a.output/'learning_metrics.csv',training)
    gates={}
    for variant in ['ports','global','combined']:
        selected=[r for r in training if r['variant']==variant]
        complete={r['seed'] for r in selected}=={20261117,20261118,20261119}
        rho=statistics.median(r['median_spearman'] or 0. for r in selected) if selected else None
        precision=statistics.median(r['precision'] or 0. for r in selected) if selected else None
        passed=complete and rho>=.5 and precision>=.7 and any(r['gate_pass'] for r in selected)
        gates[variant]={'complete':complete,'median_seed_spearman':rho,'median_seed_precision':precision,'pass':passed}
    nonempty=[r for r in rows if r['index'] and r['status']=='PASS']
    summary={'data_source_commit':data['source_commit'],'learning_source_commit':learning['source_commit'],
        'circuits':len(cases),'new_mapping_calls':sum(c['mapping_calls'] for c in cases),
        'new_records':len(rows),'successful_records':sum(r['status']=='PASS' for r in rows),
        'nonempty_successful':len(nonempty),'area_violations':sum(r['area_feasible'] is False for r in rows),
        'structural_mismatches':sum((r['structural_error'] or 0)>1e-8 for r in rows),
        'historical_combinations_reused':len(history),'seed_gates':gates,
        'independent24_authorized_by_gate':gates['ports']['pass'] or gates['combined']['pass'],
        'serialized_noop_ratio_median':statistics.median(c['serialized_noop_ratio'] for c in cases if c['serialized_noop_ratio'] is not None),
        'serialized_noop_max_absolute_relative_drift':max(abs(c['serialized_noop_ratio']-1) for c in cases if c['serialized_noop_ratio'] is not None),
        'scope':'Bounded offline-learning pilot, not end-to-end Ours performance; no candidate timing feedback in search.'}
    save(a.output/'summary.json',summary);save(a.output/'evidence_index.json',{'data_run':str(a.data_run),'learning_run':str(a.learning_run),'files':evidence})
    if summary['new_mapping_calls']>216:raise ValueError('Mapping budget exceeded')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

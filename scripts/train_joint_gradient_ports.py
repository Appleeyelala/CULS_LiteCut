#!/usr/bin/env python3
"""Offline v3 PO-delta/ranking learning; no ABC or replay calls in training.

Each optimizer update averages examples within one circuit, then circuits are
visited once per epoch. Validation selects checkpoints; test labels are never
opened. All unsuccessful/area-invalid records remain in the input audit.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint_gradient_panel import read,save,sha


def move_graph(model, device):
    """Move per-circuit buffers without moving shared parameters/Adam state."""
    for name,buffer in model._buffers.items():
        if buffer is not None:model._buffers[name]=buffer.to(device)
    return model


def timing_loss(predicted, target, baseline, critical):
    import torch
    # A 1% reference-delay scale prevents a zero-delta solution from winning
    # numerically just because all normalized differences are small.
    p=(predicted-baseline[None])/critical/.01
    y=(target-baseline[None])/critical/.01
    po=torch.nn.functional.smooth_l1_loss(p,y,beta=1.)
    pd=(predicted.max(-1).values-critical)/critical/.01
    yd=(target.max(-1).values-critical)/critical/.01
    delay=torch.nn.functional.smooth_l1_loss(pd,yd,beta=1.)
    # Original paid no-op is an exact zero reference, not a new replay label.
    pd=torch.cat([pd,pd.new_zeros(1)]);yd=torch.cat([yd,yd.new_zeros(1)])
    pairs=torch.triu_indices(len(pd),len(pd),offset=1,device=pd.device)
    differences=yd[pairs[0]]-yd[pairs[1]];valid=differences.abs()>2e-4
    logits=(pd[pairs[0]]-pd[pairs[1]])[valid]
    sign=differences[valid].sign()
    ranking=torch.nn.functional.softplus(-sign*logits).mean() if len(logits) else pd.sum()*0.
    return po+delay+.1*ranking


def load_examples(data, manifest, shared, variant):
    import torch
    from global_timing.candidates import Catalogue
    from global_timing.structure import make_partition,parse_aig
    from global_timing.learned_ports import LearnedPortSurrogate,make_port_reference
    examples=[];audit=[]
    lib=ROOT/'projects/culs/src/algorithms/rewrite_library.inc'
    for entry in manifest['selected']:
        folder=data/f'case_{entry["circuit_id"]:04d}'
        frozen=read(folder/'frozen.json');baseline=read(folder/'baseline.json')
        if sha(Path(frozen['source'])/'input.aig')!=entry['input_sha256']:raise ValueError('Input hash changed')
        measured=read(folder/'measurements.json')
        if measured['frozen_sha256']!=sha(folder/'frozen.json'):raise ValueError('Measurement/freeze mismatch')
        if any(r.get('structural_max_error',0)>1e-8 for r in measured['results']):
            raise ValueError('Unresolved hard-composition discrepancy; refusing training')
        partition=make_partition(parse_aig(Path(frozen['source'])/'input.aig'))
        raw=read(folder/'catalogue.json');scores=read(folder/'scores.json')['scores']
        domains=[('sampled',raw,scores,measured['results'])]
        if (folder/'historical.json').exists():
            history=read(folder/'historical.json')
            if history['structural_max_error']>1e-8:raise ValueError('Historical structural gate failed')
            domains.append(('historical',history['catalogue'],history['scores'],
                [dict(index=1,status='PASS',choices=history['choices'],metrics=history['metrics'],area_feasible=history['area_feasible'])]))
        case={'circuit_id':entry['circuit_id'],'role':entry['role'],'family':entry['family'],'domains':[]}
        for domain,raw,scores,records in domains:
            accepted=[]
            for record in records:
                metrics=record.get('metrics',{})
                eligible=bool(record['index'] and record['status']=='PASS' and metrics.get('cec')=='PASS' and metrics.get('timing_model')=='ABC_SCL_LIBERTY')
                audit.append(dict(circuit_id=entry['circuit_id'],domain=domain,index=record['index'],status=record['status'],
                    timing_label_eligible=eligible,area_feasible=record.get('area_feasible'),
                    serialized_noop_delta_ps=metrics.get('delay_ps',baseline['delay_ps'])-baseline['delay_ps'] if record['index']==0 and metrics else None))
                if eligible:accepted.append(record)
            if not accepted:continue
            cat=Catalogue(raw['decisions'],raw['conflicts'],{int(k):v for k,v in raw['region_decisions'].items()})
            cat.region_decisions={i:cat.region_decisions.get(i,[]) for i in range(len(partition.regions))}
            reference=make_port_reference(partition,cat,scores,baseline)
            model=LearnedPortSurrogate(reference,scores,lib,shared,variant=variant)
            x=torch.stack([model.one_hot(r['choices']) for r in accepted])
            target=torch.tensor([r['metrics']['po_arrivals_ps'] for r in accepted],dtype=torch.double)
            if target.shape!=(len(accepted),len(reference.baseline_po)) or not torch.isfinite(target).all():raise ValueError('Invalid PO labels')
            case['domains'].append({'name':domain,'model':model,'x':x,'target':target,'records':accepted})
        if case['domains']:examples.append(case)
    return examples,audit


def evaluate(examples,role,device):
    import torch
    import numpy as np
    from scipy.stats import spearmanr
    rows=[];losses=[];case_statistics=[]
    with torch.no_grad():
        for case in examples:
            if case['role']!=role:continue
            case_loss=[]
            for domain in case['domains']:
                model=move_graph(domain['model'],device);x=domain['x'].to(device);target=domain['target'].to(device)
                po,_=model(x)
                case_loss.append((float(timing_loss(po,target,model.baseline_po,model.delay_reference)),len(x)))
                predicted=po.max(-1).values.cpu().tolist();actual=target.max(-1).values.cpu().tolist()
                tol=max(2e-5,2e-6*model.delay_reference)
                for p,a,record in zip(predicted,actual,domain['records']):
                    rows.append(dict(circuit_id=case['circuit_id'],domain=domain['name'],index=record['index'],
                        predicted_delay_ps=p,actual_delay_ps=a,original_delay_ps=model.delay_reference,
                        predicted_improvement=p<model.delay_reference-tol,actual_improvement=a<model.delay_reference-tol,
                        area_feasible=record.get('area_feasible')))
                if domain['name']=='sampled':
                    # Quantized ties use the declared timing tolerance.
                    pq=np.rint(np.array(predicted)/tol);aq=np.rint(np.array(actual)/tol)
                    rho=float(spearmanr(pq,aq).statistic) if len(set(pq))>1 and len(set(aq))>1 else None
                    case_statistics.append(dict(circuit_id=case['circuit_id'],spearman=rho,combinations=len(predicted),
                        unique_predictions=len(set(pq)),unique_actual=len(set(aq))))
                move_graph(model,'cpu')
            losses.append(sum(v*n for v,n in case_loss)/sum(n for v,n in case_loss))
    announced=[r for r in rows if r['domain']=='sampled' and r['predicted_improvement']]
    eligible=[r for r in rows if r['domain']=='sampled']
    correlations=[r['spearman'] if r['spearman'] is not None else 0. for r in case_statistics]
    precision=sum(r['actual_improvement'] for r in announced)/len(announced) if announced else None
    median=float(np.median(correlations)) if correlations else None
    return {'mean_circuit_loss':float(np.mean(losses)) if losses else None,'cases':case_statistics,'predictions':rows,
            'median_spearman_with_undefined_as_zero':median,'improvement_precision':precision,
            'improvement_coverage':len(announced)/len(eligible) if eligible else 0.,
            'gate_pass':median is not None and median>=.5 and precision is not None and precision>=.7}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=['fixed','ports','global','combined'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):p.error('Committed /data source and Slurm required')
    import torch
    from global_timing.learned_ports import PortTimingParameters
    torch.set_num_threads(1);torch.manual_seed(args.seed);random.seed(args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=read(args.manifest);shared=PortTimingParameters().to(args.device)  # float32 heads; float64 timing tape.
    started=time.perf_counter();examples,audit=load_examples(args.data,manifest,shared,args.variant)
    save(args.output/'label_audit.json',audit)
    if {c['circuit_id'] for c in examples}!={r['circuit_id'] for r in manifest['selected']}:
        raise ValueError('A frozen circuit has no usable labels; report it, do not silently replace it')
    parameters=shared.parameters() if args.variant=='combined' else shared.port.parameters() if args.variant=='ports' else shared.global_head.parameters()
    optimizer=torch.optim.AdamW(parameters,lr=1e-3,weight_decay=1e-4)
    history=[];best=float('inf');best_state={k:v.detach().cpu().clone() for k,v in shared.state_dict().items()};stale=0;best_epoch=-1
    if args.variant!='fixed':
        for epoch in range(200):
            training=[c for c in examples if c['role']=='train'];random.Random(args.seed+epoch).shuffle(training)
            losses=[]
            for case in training:
                optimizer.zero_grad(set_to_none=True);total=sum(len(d['x']) for d in case['domains']);case_loss=0.
                # Each domain uses the same shared heads; update once per circuit.
                for domain in case['domains']:
                    model=move_graph(domain['model'],args.device);x=domain['x'].to(args.device);target=domain['target'].to(args.device)
                    po,_=model(x);loss=timing_loss(po,target,model.baseline_po,model.delay_reference)*len(x)/total
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();case_loss+=float(loss.detach())
                    move_graph(model,'cpu')
                torch.nn.utils.clip_grad_norm_(shared.parameters(),5.)
                optimizer.step();losses.append(case_loss)
            validation=evaluate(examples,'validation',args.device);score=validation['mean_circuit_loss']
            row={'epoch':epoch,'train_loss':sum(losses)/len(losses),'validation_loss':score,
                 'validation_spearman':validation['median_spearman_with_undefined_as_zero'],
                 'validation_precision':validation['improvement_precision']};history.append(row);print(json.dumps(row),flush=True)
            if score<best-1e-6:
                best=score;best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in shared.state_dict().items()}
            else:stale+=1
            if stale>=20:break
    shared.load_state_dict(best_state)
    torch.save({'state_dict':best_state,'variant':args.variant,'seed':args.seed,'best_epoch':best_epoch,
                'manifest_sha256':sha(args.manifest),'source_commit':os.environ.get('V3_SOURCE_COMMIT')},args.output/'weights.pt')
    result={'variant':args.variant,'seed':args.seed,'best_epoch':best_epoch,'history':history,
        'validation':evaluate(examples,'validation',args.device),'train':evaluate(examples,'train',args.device),
        'elapsed_s':time.perf_counter()-started,'new_mapping_calls':0,'manifest_sha256':sha(args.manifest),
        'training':{'optimizer':'AdamW','lr':1e-3,'weight_decay':1e-4,'max_epochs':200,'patience':20,
                    'loss':'circuit-equal PO-delta Huber + Critical-delta Huber + 0.1 pairwise logistic; 1% scale',
                    'heads_dtype':'float32','timing_dtype':'float64'}}
    save(args.output/'summary.json',result)


if __name__=='__main__':main()

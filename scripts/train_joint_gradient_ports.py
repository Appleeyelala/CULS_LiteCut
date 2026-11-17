#!/usr/bin/env python3
"""Offline v3 PO-delta/ranking learning; no ABC or replay calls in training.

Each optimizer update averages examples within one circuit, then circuits are
visited once per epoch. Validation selects checkpoints; test labels are never
opened. All unsuccessful/area-invalid records remain in the input audit.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint_gradient_panel import read,save,sha


def load_prepared_graphs(path):
    import torch
    from global_timing.common import legacy
    # parse_aig returns this dynamically registered, historical dataclass.
    # A fresh process must register it before trusted local cache unpickling.
    legacy('gate3b_analyze')
    return torch.load(path,map_location='cpu',weights_only=False)


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
        noop=next((r.get('metrics',{}).get('delay_ps') for r in measured['results'] if r['index']==0 and r['status']=='PASS'),None)
        case={'circuit_id':entry['circuit_id'],'role':entry['role'],'family':entry['family'],'domains':[],
              'serialized_noop_delay_ps':noop}
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


def evaluate(examples,role,device,loss_function=timing_loss):
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
                case_loss.append((float(loss_function(po,target,model.baseline_po,model.delay_reference)),len(x)))
                predicted=po.max(-1).values.cpu().tolist();actual=target.max(-1).values.cpu().tolist()
                tol=max(2e-5,2e-6*model.delay_reference)
                for p,a,record in zip(predicted,actual,domain['records']):
                    rows.append(dict(circuit_id=case['circuit_id'],domain=domain['name'],index=record['index'],
                        predicted_delay_ps=p,actual_delay_ps=a,original_delay_ps=model.delay_reference,
                        predicted_improvement=p<model.delay_reference-tol,actual_improvement=a<model.delay_reference-tol,
                        serialized_noop_delay_ps=case['serialized_noop_delay_ps'],
                        improves_serialized_noop=a<case['serialized_noop_delay_ps']-tol if case['serialized_noop_delay_ps'] is not None else None,
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
    measured_noop=[r for r in announced if r['improves_serialized_noop'] is not None]
    median=float(np.median(correlations)) if correlations else None
    return {'mean_circuit_loss':float(np.mean(losses)) if losses else None,'cases':case_statistics,'predictions':rows,
            'median_spearman_with_undefined_as_zero':median,'improvement_precision':precision,
            'improvement_coverage':len(announced)/len(eligible) if eligible else 0.,
            'precision_against_serialized_noop':sum(r['improves_serialized_noop'] for r in measured_noop)/len(measured_noop) if measured_noop else None,
            'gate_pass':median is not None and median>=.5 and precision is not None and precision>=.7}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=['fixed','ports','global','combined'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--prepared',type=Path)
    p.add_argument('--global-context',choices=['original','endpoint'],default='original',
                   help='Single-change research ablation; original remains the default')
    p.add_argument('--loss-mode',choices=['original','gap-ranking'],default='original')
    args=p.parse_args()
    loss_function=timing_loss
    if args.loss_mode=='gap-ranking':
        if args.variant!='global' or args.global_context!='original':
            p.error('Ranking-loss experiment requires original global structure and features')
        from global_timing.ranking_loss import gap_ranking_loss
        loss_function=gap_ranking_loss
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):p.error('Committed /data source and Slurm required')
    import torch
    from global_timing.learned_ports import PortTimingParameters
    torch.set_num_threads(1);torch.manual_seed(args.seed);random.seed(args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=read(args.manifest);started=time.perf_counter()
    code_hashes={n:sha(ROOT/'global_timing'/n) for n in ['learned_ports.py','phase_ports.py','conditional_arrival.py']}
    if args.prepared:
        prepared=load_prepared_graphs(args.prepared)
        if prepared['manifest_sha256']!=sha(args.manifest) or prepared['code_hashes']!=code_hashes:
            raise ValueError('Prepared graph provenance mismatch')
        examples,audit=prepared['examples'],prepared['audit'];shared=examples[0]['domains'][0]['model'].shared
        shared.load_state_dict(PortTimingParameters().state_dict())
        for case in examples:
            for domain in case['domains']:
                model=domain['model']
                if model.shared is not shared:raise ValueError('Prepared models lost shared parameter identity')
                model.variant=args.variant;model.metadata['variant']=args.variant
    else:
        shared=PortTimingParameters()
        examples,audit=load_examples(args.data,manifest,shared,args.variant)
    save(args.output/'label_audit.json',audit)
    if {c['circuit_id'] for c in examples}!={r['circuit_id'] for r in manifest['selected']}:
        raise ValueError('A frozen circuit has no usable labels; report it, do not silently replace it')
    if args.prepare_only:
        torch.save({'examples':examples,'audit':audit,'manifest_sha256':sha(args.manifest),'code_hashes':code_hashes},args.output/'prepared.pt')
        save(args.output/'preparation.json',{'cases':len(examples),'domains':sum(len(c['domains']) for c in examples),
             'code_hashes':code_hashes,'manifest_sha256':sha(args.manifest),'elapsed_s':time.perf_counter()-started,
             'model_metadata':[{'circuit_id':c['circuit_id'],'domains':[d['model'].metadata for d in c['domains']]} for c in examples]})
        return
    if args.global_context=='endpoint':
        if args.variant!='global':raise ValueError('Endpoint-context ablation requires global variant')
        from global_timing.endpoint_context import EndpointContextSurrogate
        for case in examples:
            for domain in case['domains']:
                model=move_graph(domain['model'],args.device)
                domain['model']=move_graph(EndpointContextSurrogate.from_prepared(model),'cpu')
    shared.to(args.device)  # Move shared parameters once, before creating AdamW.
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
                    po,_=model(x);loss=loss_function(po,target,model.baseline_po,model.delay_reference)*len(x)/total
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();case_loss+=float(loss.detach())
                    move_graph(model,'cpu')
                torch.nn.utils.clip_grad_norm_(shared.parameters(),5.)
                optimizer.step();losses.append(case_loss)
            validation=evaluate(examples,'validation',args.device,loss_function);score=validation['mean_circuit_loss']
            row={'epoch':epoch,'train_loss':sum(losses)/len(losses),'validation_loss':score,
                 'validation_spearman':validation['median_spearman_with_undefined_as_zero'],
                 'validation_precision':validation['improvement_precision']};history.append(row);print(json.dumps(row),flush=True)
            if score<best-1e-6:
                best=score;best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in shared.state_dict().items()}
            else:stale+=1
            if stale>=20:break
    shared.load_state_dict(best_state)
    torch.save({'state_dict':best_state,'variant':args.variant,'global_context':args.global_context,
                'loss_mode':args.loss_mode,
                'loss_code_sha256':sha(ROOT/'global_timing/ranking_loss.py') if args.loss_mode=='gap-ranking' else None,
                'context_code_sha256':sha(ROOT/'global_timing/endpoint_context.py') if args.global_context=='endpoint' else None,
                'seed':args.seed,'best_epoch':best_epoch,
                'manifest_sha256':sha(args.manifest),'source_commit':os.environ.get('V3_SOURCE_COMMIT')},args.output/'weights.pt')
    result={'variant':args.variant,'global_context':args.global_context,'loss_mode':args.loss_mode,
        'seed':args.seed,'best_epoch':best_epoch,'history':history,
        'validation':evaluate(examples,'validation',args.device,loss_function),'train':evaluate(examples,'train',args.device,loss_function),
        'elapsed_s':time.perf_counter()-started,'new_mapping_calls':0,'manifest_sha256':sha(args.manifest),
        'training':{'optimizer':'AdamW','lr':1e-3,'weight_decay':1e-4,'max_epochs':200,'patience':20,
                    'loss':('bounded-gap pairwise logistic, 1% scale; no absolute fitting' if args.loss_mode=='gap-ranking'
                            else 'circuit-equal PO-delta Huber + Critical-delta Huber + 0.1 pairwise logistic; 1% scale'),
                    'heads_dtype':'float32','timing_dtype':'float64'}}
    save(args.output/'summary.json',result)


if __name__=='__main__':main()

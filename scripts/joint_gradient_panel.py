#!/usr/bin/env python3
"""Freeze factorial proxy panels before measuring any new candidate labels."""
import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def read(p):return json.loads(Path(p).read_text())


def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_subset(source):
    from global_timing.candidates import Catalogue,conflict_reason
    from global_timing.saved_arrays import iter_saved_array
    best=read(source/'work/optimization/proxy_best.json')
    edits=dict(best['edits'])
    decisions=[];source_indices=[]
    for i,d in enumerate(iter_saved_array(source/'work/cut_validation/verified_catalogue.json','decisions')):
        if i in edits:
            d=copy.deepcopy(d)
            d['candidate_subgraph_indices']=[d['candidate_subgraph_indices'][edits[i]-1]]
            d['candidate_count']=1
            decisions.append(d);source_indices.append(i)
        if len(decisions)==len(edits):break
    if len(decisions)!=len(edits) or len(decisions)<4:raise ValueError('Incomplete or too small archived best domain.')
    lookup={source:i for i,source in enumerate(source_indices)}
    score_path=read(source/'work/litecut.json')['scores_path']
    scores=[]
    for row in iter_saved_array(score_path,'scores'):
        i=row['decision_index'];c=row['choice_index']
        if i in edits and c in (0,edits[i]):
            row=dict(row,decision_index=lookup[i],choice_index=int(c!=0));scores.append(row)
        if len(scores)==2*len(edits):break
    if len(scores)!=2*len(edits):raise ValueError('Missing saved Student scores.')
    conflicts=[(i,j) for i in range(len(decisions)) for j in range(i)
               if conflict_reason(decisions[i],decisions[j])]
    groups={}
    for i,d in enumerate(decisions):groups.setdefault(d['decision_owner_region'],[]).append(i)
    return Catalogue(decisions,conflicts,groups),scores,source_indices,score_path


def restrict(catalogue,scores,indices,regions):
    from global_timing.candidates import Catalogue
    lookup={old:new for new,old in enumerate(indices)}
    groups={r:[] for r in range(regions)}
    for new,old in enumerate(indices):groups[catalogue.decisions[old]['decision_owner_region']].append(new)
    cat=Catalogue([catalogue.decisions[i] for i in indices],
                  [(lookup[a],lookup[b]) for a,b in catalogue.conflicts if a in lookup and b in lookup],groups)
    return cat,[dict(r,decision_index=lookup[r['decision_index']]) for r in scores if r['decision_index'] in lookup]


def prepare(args):
    import torch
    from global_timing.feasibility import CandidateFeasibility
    from global_timing.mffc_model import MffcGpuSurrogate
    from global_timing.mffc_ste_search import ste_adam_search
    from global_timing.structure import make_partition,parse_aig
    from global_timing.surrogate import scalar_objective
    torch.set_num_threads(1)
    source=args.reference_run/'output'/f'case_{args.circuit:04d}_Ours'
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    context=read(source/'context.json')
    cat,scores,source_indices,score_path=load_subset(source)
    partition=make_partition(parse_aig(source/'input.aig'))
    cat.region_decisions={r:cat.region_decisions.get(r,[]) for r in range(len(partition.regions))}
    baseline=read(source/'work/original/result.json')
    library=ROOT/'projects/culs/src/algorithms/rewrite_library.inc'
    rule=CandidateFeasibility(partition.aig,cat,library)
    rule.validate([1]*len(cat.decisions))
    model=MffcGpuSurrogate(partition,cat,scores,baseline,context['tools']['native_library']['path'])
    model.requires_grad_(False)
    x=model.one_hot([0]*len(cat.decisions)).requires_grad_()
    po,area=model(x)
    loss=scalar_objective(po,area,model.delay_reference,model.area_reference)
    g=torch.autograd.grad(loss,x)[0]
    gain=(g[:,0]-g[:,1]).detach().cpu().tolist()
    order=sorted(range(len(gain)),key=lambda i:(-gain[i],source_indices[i]))
    chosen=[order[0]]
    rid=lambda i:cat.decisions[i]['decision_owner_region']
    others=[i for i in order if rid(i)!=rid(chosen[0])]
    if not others:raise ValueError('Archived best does not span two MFFCs; do not substitute a circuit silently.')
    chosen.append(others[0])
    neighbors={b for a,b in partition.edges if a in {rid(i) for i in chosen}}
    neighbors.update(a for a,b in partition.edges if b in {rid(i) for i in chosen})
    related=[i for i in order if i not in chosen and rid(i) in neighbors]
    if related:chosen.append(related[0])
    chosen.extend(i for i in order if i not in chosen)
    chosen=chosen[:4]
    model.native_plan.collect_profile()
    del model,x,po,area,loss,g
    cat,small_scores=restrict(cat,scores,chosen,len(partition.regions))
    rule=CandidateFeasibility(partition.aig,cat,library);rule.validate([1]*4)
    model=MffcGpuSurrogate(partition,cat,small_scores,baseline,context['tools']['native_library']['path'])
    model.requires_grad_(False)
    combinations=[]
    for mask in range(16):
        vector=[(mask>>i)&1 for i in range(4)];rule.validate(vector)
        with torch.no_grad():
            po,area=model(model.one_hot(vector))
            loss=scalar_objective(po,area,model.delay_reference,model.area_reference)
        combinations.append(dict(mask=mask,choices=vector,predicted_delay_ps=float(po.max()),
                                 predicted_area=float(area),predicted_loss=float(loss),predicted_po_ps=po.cpu().tolist()))
    best,trace=ste_adam_search(model,rule)
    save(out/'ste_trajectory.json',trace)
    save(out/'baseline.json',baseline)
    save(out/'catalogue.json',dict(decisions=cat.decisions,conflicts=cat.conflicts,region_decisions=cat.region_decisions))
    save(out/'scores.json',{'scores':small_scores})
    save(out/'frozen_predictions.json',{'schema':'joint_factorial_panel_v1','circuit':args.circuit,
        'source':str(source),'input_sha256':sha(source/'input.aig'),'scores_source':score_path,
        'pool_policy':'Archived selected nonempty combination only, all other decisions fixed no-op; four roots selected by full-DAG gradient, MFFC diversity and adjacency. This is a restricted diagnostic domain, not a full-domain optimizer evaluation.',
        'selected_source_indices':[source_indices[i] for i in chosen],
        'pool_source_indices':source_indices,'pool_directional_gains':gain,
        'combinations':combinations,'ste_best_mask':sum(c<<i for i,c in enumerate(best['choices'])),
        'ste_best':best,'native_profile':model.native_plan.collect_profile(),
        'preparation_s':time.perf_counter()-start,'new_mapping_calls':0,
        'all_predictions_precede_new_labels':True})
    print(json.dumps({'circuit':args.circuit,'phase':'prepared','decisions':4,'combinations':16}),flush=True)


def measure(args):
    from global_timing.candidates import Catalogue
    from global_timing.execution import replay_many
    from global_timing.evaluation import Evaluator
    out=args.output;frozen=out/'frozen_predictions.json'
    before=sha(frozen);payload=read(frozen)
    source=Path(payload['source']);context=read(source/'context.json')
    if sha(source/'input.aig')!=payload['input_sha256']:raise ValueError('Input changed.')
    if shutil.disk_usage(out).free<8*1024**3:raise RuntimeError('STORAGE_GUARD: less than 8 GiB free; preserving all outputs.')
    raw=read(out/'catalogue.json')
    cat=Catalogue(raw['decisions'],raw['conflicts'],{int(k):v for k,v in raw['region_decisions'].items()})
    # Every request is already fixed; this evaluator has no callback to search.
    vectors=[r['choices'] for r in payload['combinations'][1:]]
    destinations=[out/'measurements'/f'mask_{i:02d}'/'rewrite' for i in range(1,16)]
    for d in destinations:d.parent.mkdir(parents=True,exist_ok=False)
    os.environ['CULS_ALLOW_NONPOSITIVE_CANDIDATES']='1'
    records=replay_many(args.gpuls,source/'input.aig',source/'work/partition/owners.csv',cat,
                        vectors,out/'batch_replay',destinations,timeout=900,prepared=True,allow_nonpositive_gain=True)
    native_policy=read(out/'batch_replay/raw/replay_policy.json')
    if native_policy.get('candidate_rule')!='allow_nonpositive_gain':raise ValueError('Wrong native candidate mask policy.')
    evaluator=Evaluator(Path(context['tools']['abc']['path']),Path(context['tools']['library']['path']),source/'input.aig',900)
    results=[dict(mask=0,status='PASS',metrics=read(out/'baseline.json'),new_mapping=False,
                  provenance='Original saved paid mapping, not a serialized replay-noop or new runtime.')]
    for mask,execution in enumerate(records,1):
        row=dict(mask=mask,status=execution['status'],execution=execution,prediction_sha256=before)
        if execution['status']=='PASS':
            try:
                row['metrics']=evaluator.evaluate(Path(execution['aig']),out/'measurements'/f'mask_{mask:02d}'/'evaluation')
                row['area_feasible']=row['metrics']['area']<=read(out/'baseline.json')['area']*1.05
            except Exception as exc:row.update(status='FAIL',error=str(exc))
        save(out/'measurements'/f'mask_{mask:02d}'/'observation.json',row)
        results.append(row)
        print(json.dumps({'circuit':args.circuit,'phase':'measured','mask':mask,'status':row['status']}),flush=True)
    if sha(frozen)!=before:raise ValueError('Predictions changed during measurement.')
    save(out/'measurements.json',{'circuit':args.circuit,'frozen_predictions_sha256':before,
         'gpuls_sha256':sha(args.gpuls),'tools':context['tools'],'results':results,
         'new_mapping_calls':len(list((out/'measurements').rglob('mapping.command.json'))),
         'scope':'Offline factorial diagnostic; extra labels are never used by the optimizer.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['prepare','measure'],required=True)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--circuit',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpuls',type=Path)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):
        p.error('Committed /data source and Slurm required.')
    if args.phase=='measure' and not args.gpuls:p.error('--gpuls required for measurement.')
    if args.phase=='prepare':prepare(args)
    else:measure(args)


if __name__=='__main__':main()

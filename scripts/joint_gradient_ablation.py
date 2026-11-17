#!/usr/bin/env python3
"""Freeze proxy-only 2x2 and objective-alignment controls on factorial domains.

Reads original baseline and saved local scores only. Deliberately never opens
measurements.json. Each output can subsequently be joined with frozen labels.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint_gradient_panel import read,save,sha


def run(args):
    import torch
    from global_timing.candidates import Catalogue
    from global_timing.structure import make_partition,parse_aig
    from global_timing.feasibility import CandidateFeasibility
    from global_timing.mffc_model import MffcGpuSurrogate
    from global_timing.compositional_proxy import CompositionalTimingSurrogate
    from global_timing.joint_gradient import aligned_ste_search,joint_prox_search,critical_objective
    from global_timing.mffc_ste_search import ste_adam_search
    torch.set_num_threads(1)
    panel=read(args.panel/'frozen_predictions.json');source=Path(panel['source'])
    context=read(source/'context.json');raw=read(args.panel/'catalogue.json')
    if sha(source/'input.aig')!=panel['input_sha256']:raise ValueError('Input changed.')
    cat=Catalogue(raw['decisions'],raw['conflicts'],{int(k):v for k,v in raw['region_decisions'].items()})
    scores=read(args.panel/'scores.json')['scores'];baseline=read(args.panel/'baseline.json')
    partition=make_partition(parse_aig(source/'input.aig'))
    library=ROOT/'projects/culs/src/algorithms/rewrite_library.inc'
    rule=CandidateFeasibility(partition.aig,cat,library)
    started=time.perf_counter()
    old=MffcGpuSurrogate(partition,cat,scores,baseline,context['tools']['native_library']['path'])
    new=CompositionalTimingSurrogate(old,scores,library)
    result={'schema':'joint_gradient_ablation_v1','circuit':panel['circuit'],'panel_sha256':sha(args.panel/'frozen_predictions.json'),
            'input_sha256':panel['input_sha256'],'proxy_metadata':new.metadata,'models':{},'methods':{},
            'scope':'Restricted four binary decisions, development evidence only; no candidate measured labels read, no Mapping.',
            'tools':context['tools'],'new_mapping_calls':0}
    for name,model in [('old',old),('ports',new)]:
        model.requires_grad_(False);predictions=[]
        for mask in range(16):
            vector=[(mask>>i)&1 for i in range(4)];rule.validate(vector)
            with torch.no_grad():
                po,area=model(model.one_hot(vector))
                loss=critical_objective(po,area,model.delay_reference,model.area_reference)
            predictions.append({'mask':mask,'delay_ps':float(po.max()),'area':float(area),'objective':float(loss),
                                'po_ps':po.detach().cpu().tolist()})
        result['models'][name]=predictions
        for solver,fn in [('ste',ste_adam_search),('aligned_ste',aligned_ste_search),('joint',joint_prox_search)]:
            start=time.perf_counter();best,trace=fn(model,rule)
            torch.cuda.synchronize()
            result['methods'][name+'_'+solver]={'best':best,'mask':sum(c<<i for i,c in enumerate(best['choices'])),
                'search_wall_s':time.perf_counter()-start,'trace':trace}
            print(json.dumps({'circuit':panel['circuit'],'method':name+'_'+solver,'mask':result['methods'][name+'_'+solver]['mask']}),flush=True)
    result['proxy_only_total_s']=time.perf_counter()-started
    save(args.output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):p.error('Committed /data source and Slurm required.')
    run(a)


if __name__=='__main__':main()

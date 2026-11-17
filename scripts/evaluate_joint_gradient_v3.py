#!/usr/bin/env python3
"""Freeze v3 fixed-budget optimizer outputs on the existing development domain.

This process never reads candidate measurements. Its outputs may be joined
with the historical factorial labels only after the immutable file is written.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint_gradient_panel import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--variant',choices=['fixed','ports','global','combined'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ROOT.is_relative_to('/data/wushangge'):p.error('Committed /data source and Slurm required')
    import torch
    from global_timing.candidates import Catalogue
    from global_timing.structure import make_partition,parse_aig
    from global_timing.feasibility import CandidateFeasibility
    from global_timing.learned_ports import LearnedPortSurrogate,PortTimingParameters,make_port_reference
    from global_timing.joint_gradient import aligned_ste_search,joint_prox_search,critical_objective
    torch.set_num_threads(1)
    frozen=read(a.panel/'frozen_predictions.json');raw=read(a.panel/'catalogue.json');source=Path(frozen['source'])
    if sha(source/'input.aig')!=frozen['input_sha256']:raise ValueError('Changed input')
    cat=Catalogue(raw['decisions'],raw['conflicts'],{int(k):v for k,v in raw['region_decisions'].items()})
    partition=make_partition(parse_aig(source/'input.aig'));scores=read(a.panel/'scores.json')['scores'];baseline=read(a.panel/'baseline.json')
    library=ROOT/'projects/culs/src/algorithms/rewrite_library.inc'
    parameters=PortTimingParameters();checkpoint=torch.load(a.weights,map_location='cpu',weights_only=False)
    if checkpoint['variant']!=a.variant:raise ValueError('Ablation/checkpoint mismatch')
    parameters.load_state_dict(checkpoint['state_dict'])
    model=LearnedPortSurrogate(make_port_reference(partition,cat,scores,baseline),scores,library,parameters,a.variant).cuda()
    model.requires_grad_(False);rule=CandidateFeasibility(partition.aig,cat,library)
    predictions=[]
    with torch.no_grad():
        vectors=[r['choices'] for r in frozen['combinations']]
        for vector in vectors:rule.validate(vector)
        po,area=model(torch.stack([model.one_hot(v) for v in vectors]))
        objective=critical_objective(po,area,model.delay_reference,model.area_reference)
        for i in range(len(vectors)):
            predictions.append(dict(mask=i,delay_ps=float(po[i].max()),area=float(area[i]),objective=float(objective[i]),po_ps=po[i].cpu().tolist()))
    methods={}
    for name,solver in [('aligned_ste',aligned_ste_search),('joint',joint_prox_search)]:
        torch.cuda.synchronize();start=time.perf_counter();best,trace=solver(model,rule);torch.cuda.synchronize()
        methods[name]={'best':best,'mask':sum(c<<i for i,c in enumerate(best['choices'])),'trace':trace,'search_wall_s':time.perf_counter()-start}
    save(a.output,{'circuit':frozen['circuit'],'variant':a.variant,'model':model.metadata,'checkpoint_sha256':sha(a.weights),
        'panel_sha256':sha(a.panel/'frozen_predictions.json'),'predictions':predictions,'methods':methods,'new_mapping_calls':0,
        'scope':'Development factorial domain only; no candidate real timing read, no deployed-runtime claim'})


if __name__=='__main__':main()

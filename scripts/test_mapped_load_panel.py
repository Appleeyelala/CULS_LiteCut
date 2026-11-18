"""Evaluate the frozen load extension without using candidate timing to score choices."""
import argparse
import csv
from pathlib import Path
import statistics

import torch

from train_joint_gradient_ports import ROOT, read, save, sha
from global_timing.boolean_candidates import parse_library
from global_timing.candidates import Catalogue
from global_timing.learned_ports import LearnedPortSurrogate, PortTimingParameters, make_port_reference
from global_timing.mapped_load import corrected_arrivals, library_cells, load_impact
from global_timing.structure import make_partition, parse_aig

TOL=.001
SEEDS=(20261117,20261118,20261119)


def csv_file(path, rows):
    with path.open('x') as f:
        w=csv.DictWriter(f,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def run_case(entry, args, states, cells, lib, hashes):
    cid=entry['circuit_id']
    folder=args.data/f'case_{cid:04d}'
    frozen=read(folder/'frozen.json')
    base=read(folder/'baseline.json')
    source=Path(frozen['source'])
    mapped=source/'work/original/mapped.v'
    for path,digest in [(mapped,base['mapped_sha256']),(source/'input.aig',entry['input_sha256'])]:
        hashes[str(path)]=sha(path)
        assert hashes[str(path)]==digest
    for name in ('scores.json','catalogue.json','frozen.json','baseline.json','measurements.json'):
        hashes[str(folder/name)]=sha(folder/name)
    raw=read(folder/'catalogue.json')
    part=make_partition(parse_aig(source/'input.aig'))
    cat=Catalogue(raw['decisions'],raw['conflicts'],{i:raw['region_decisions'].get(str(i),[]) for i in range(len(part.regions))})
    scores=read(folder/'scores.json')['scores']
    ref=make_port_reference(part,cat,scores,base)
    model=LearnedPortSurrogate(ref,scores,lib,PortTimingParameters(),variant='global')
    basis,coverage=load_impact(model,mapped,cells,lib)
    combos=sorted(frozen['combinations'],key=lambda r:r['index'])
    assert [r['index'] for r in combos]==list(range(9))
    x=torch.stack([model.one_hot(r['choices']) for r in combos])
    records=[]
    endpoints=[]
    for seed,state in states.items():
        model.shared.load_state_dict(state)
        with torch.no_grad():
            old,_=model(x)
            new=corrected_arrivals(old,x,basis)
            assert torch.equal(old[0],new[0]) and torch.isfinite(new).all()
        for idx in range(1,9):
            records.append(dict(circuit_id=cid,seed=seed,index=idx,old=float(old[idx].max()),
                                new=float(new[idx].max()),old_peak=int(old[idx].argmax()),new_peak=int(new[idx].argmax())))
            for po in ({75:[90,95],179:[3,1035,1784]}.get(cid,[])):
                endpoints.append(dict(circuit_id=cid,seed=seed,index=idx,po=po,old=float(old[idx,po]),new=float(new[idx,po])))
    # Label join only after predictions are frozen. Never read candidate netlists.
    measured=read(folder/'measurements.json')
    assert measured['frozen_sha256']==hashes[str(folder/'frozen.json')]
    labels={r['index']:r for r in measured['results']}
    for row in records+endpoints:
        r=labels[row['index']]
        assert r['status']=='PASS' and r['area_feasible'] and r['choices']==combos[r['index']]['choices']
        m=r['metrics']
        assert m['cec']=='PASS' and m['timing_model']=='ABC_SCL_LIBERTY'
        row['actual']=m['po_arrivals_ps'][row['po']] if 'po' in row else m['delay_ps']
    return records,endpoints,dict(circuit_id=cid,**coverage)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','weights','liberty','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    manifest=ROOT/'analysis/joint_gradient_v3/learning_manifest.json'
    library=ROOT/'projects/culs/src/algorithms/rewrite_library.inc'
    hashes={str(f):sha(f) for f in [args.liberty,manifest,library]}
    cells=library_cells(args.liberty)
    lib=parse_library(library)
    states={}
    for seed in SEEDS:
        wp=args.weights/f'global_{seed}'/'weights.pt'
        hashes[str(wp)]=sha(wp)
        states[seed]=torch.load(wp,map_location='cpu',weights_only=False)['state_dict']
    # Check the differentiable contribution and its finite difference on a soft choice.
    xx=torch.tensor([[.35,.65]],dtype=torch.double,requires_grad=True)
    bb=torch.tensor([[[0.],[4.]]],dtype=torch.double)
    assert torch.autograd.gradcheck(lambda v:corrected_arrivals(torch.ones(1),v,bb),(xx,))
    combos,endpoints,coverage=[],[],[]
    for entry in read(manifest)['selected']:
        if entry['role']!='validation':
            continue
        rows,points,info=run_case(entry,args,states,cells,lib,hashes)
        combos.extend(rows);endpoints.extend(points);coverage.append(info)
        print(info,flush=True)
    assert len(combos)==288
    with (ROOT/'analysis/joint_gradient_v3/ranking_loss/results/all_combinations.csv').open() as f:
        archived={(int(r['circuit_id']),int(r['seed']),int(r['index'])):r for r in csv.DictReader(f)
                  if r['role']=='development' and r['model']=='v3_global'}
    for r in combos:
        old=archived[r['circuit_id'],r['seed'],r['index']]
        assert abs(float(old['predicted_delay_ps'])-r['old'])<TOL and abs(float(old['actual_delay_ps'])-r['actual'])<TOL
    selections=[]
    for cid,seed in sorted({(r['circuit_id'],r['seed']) for r in combos}):
        rows=[r for r in combos if r['circuit_id']==cid and r['seed']==seed]
        best=min(r['actual'] for r in rows)
        result=dict(circuit_id=cid,seed=seed,best_ps=best)
        for mode in ('old','new'):
            minimum=min(r[mode] for r in rows)
            choice=min((r for r in rows if r[mode]-minimum<=TOL),key=lambda r:r['index'])
            result.update({mode+'_choice':choice['index'],mode+'_actual':choice['actual'],
                           mode+'_regret':100*(choice['actual']/best-1),mode+'_hit':choice['actual']-best<=TOL})
        selections.append(result)
    for name,rows in [('combinations',combos),('endpoints',endpoints),('coverage',coverage),('selected',selections)]:
        csv_file(args.output/(name+'.csv'),rows)
    save(args.output/'inputs.json',hashes)
    for mode in ('old','new'):
        print(mode,statistics.mean(r[mode+'_regret'] for r in selections),sum(r[mode+'_hit'] for r in selections))


if __name__=='__main__':
    main()

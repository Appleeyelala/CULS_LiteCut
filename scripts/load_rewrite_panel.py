"""Freeze the new hard model on all archived legal vectors before reading delay labels."""
import argparse
import csv
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_timing.boolean_candidates import parse_library
from global_timing.candidates import Catalogue
from global_timing.feasibility import CandidateFeasibility
from global_timing.common import ROOT, digest, save_json
from global_timing.liberty_cover import Liberty
from global_timing.load_bank import LoadBank
from global_timing.structure import parse_aig
from abc_mode0_panel import write_csv

def read(path):
    return json.loads(path.read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('abc','library','tables','panel','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    tech=Liberty(a.tables,'cuda')
    lib=parse_library(ROOT/'projects/culs/src/algorithms/rewrite_library.inc')
    allrows=[]
    for f in sorted(a.panel.glob('case_*')):
        frozen=read(f/'frozen.json')
        if frozen['role']!='validation': continue
        i=frozen['circuit_id']
        original=Path(frozen['source'])/'input.aig'
        if digest(original)!=frozen['input_sha256']: raise ValueError('Original drift')
        aig=parse_aig(original)
        cat=Catalogue(**read(f/'catalogue.json'))
        rule=CandidateFeasibility(aig,cat,lib)
        # Use the entire supplied catalogue, all legal options compete for mapping.
        bank=LoadBank(aig,cat,rule,read(f/'scores.json')['scores'],lib,tech,a.abc,a.library,a.output/f.name)
        rows=[]
        for alpha in [0.,.5]:
            bank.set_alpha(alpha)
            for rec in frozen['combinations']:
                vec=rec['choices']
                rule.validate(vec)
                with torch.no_grad():
                    latency,area=bank.forward(bank.hard(vec))
                rows.append(dict(circuit=i,index=rec['index'],edits=sum(bool(x) for x in vec),
                                 alpha=alpha,score=float(latency),area=float(area)))
        save_json(a.output/f.name/'predictions.json',rows)
        allrows.extend(rows)
    write_csv(a.output/'predictions.csv',allrows)
    selected=[]
    for i in sorted({r['circuit'] for r in allrows}):
        folder=a.panel/f'case_{i:04d}'
        labels={r['index']:r['metrics'] for r in read(folder/'measurements.json')['results'] if r['status']=='PASS' and r['metrics']['cec']=='PASS'}
        assert set(range(1,9))<=labels.keys()
        best=min(labels[k]['delay_ps'] for k in range(1,9))
        for alpha in [0.,.5]:
            group=[v for v in allrows if v['circuit']==i and v['alpha']==alpha and v['index']]
            lowest=min(v['score'] for v in group)
            r=min((v for v in group if v['score']<=lowest+max(2e-5,2e-6*lowest)),key=lambda v:(v['edits'],v['index']))
            actual=labels[r['index']]['delay_ps']
            selected.append(dict(**r,real_ps=actual,best_ps=best,regret_pct=100*(actual/best-1),
                                 top1=abs(actual-best)<=max(2e-5,best*2e-6)))
    write_csv(a.output/'selected.csv',selected)
